#include "unified_llm_w4a16_hetero/npuSetup.hpp"
#include "unified_llm_w4a16_hetero/unified_llm_w4a16.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <torch/torch.h>
#include <unistd.h> // For access()
#include <vector>

// TEST_CASE options:
// 0 = Deterministic test with synthetic data
// 1 = Random test with synthetic data
// 2 = Real Llama-3-8B weights (q_proj for 4Kx4Kx4K, mlp.up_proj for 4Kx4Kx14336)
#define TEST_CASE 1
// Set to 1 to enable golden reference dequantization (enables slow reference path for correctness checks).
#define TEST_REFERENCE 0

// Helper to load Safetensors
struct SafetensorsTensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t offset_begin;
    uint64_t offset_end;
    bool valid;
    SafetensorsTensorInfo() : valid(false) {}
};

// Forward declare from unified_llm_w4a16.cpp
std::map<std::string, SafetensorsTensorInfo> parse_safetensors_header(const std::string &filename, uint64_t &header_size);
torch::Tensor load_tensor_from_safetensors(const std::string &filename, const SafetensorsTensorInfo &info, uint64_t header_size);
torch::Tensor unpack_awq_qweight(torch::Tensor qweight);
torch::Tensor unpack_awq_qzeros(torch::Tensor qzeros);

// Reference Dequantization Function (Golden Standard)
// Modified to accept unpacked weights (kInt8) directly
torch::Tensor reference_dequantize(torch::Tensor unpacked_weight_, torch::Tensor scale_, torch::Tensor zero_point_, int64_t in_features_,
                                   int64_t out_features_) {
    // torch::NoGradGuard no_grad; // Not needed in simple test
    auto device = unpacked_weight_.device();

    const int64_t in_features = in_features_;
    const int64_t out_features = out_features_;

    // 2) Apply scale / zero via broadcasting-friendly views.
    // Match debug_dequantize_weights logic which uses [In, Out] and [Groups, Out]

    // qweight passed in is [Out, In], but debug path expects [In, Out]
    auto debug_qweight = unpacked_weight_.contiguous().to(torch::kInt8);
    auto debug_scalefactor = scale_;
    auto debug_zeropoint = zero_point_;

    if (debug_scalefactor.dim() == 2) {
        auto apply_group = [&](const torch::Tensor &scales, const torch::Tensor &zeros) {
            // Scales are [Groups, Out] (User Request)
            int64_t n_groups = scales.size(0); // Dimm 0 is Groups
            int64_t group_size = in_features / n_groups;

            // w: [In, Out] -> [Groups, GroupSize, Out]
            auto w_view = debug_qweight.view({n_groups, group_size, out_features});

            // scales: [Groups, Out] -> [Groups, 1, Out] for broadcasting
            auto s_view = scales.unsqueeze(1);
            auto z_view = zeros.unsqueeze(1);

            // Convert Int8 to Int8 for subtraction (-15..15 range)
            // Result will be Int8
            auto w_sub_z = w_view.sub(z_view);

            // Convert to BF16 for multiplication with scales
            auto w_float = w_sub_z.to(torch::kBFloat16);

            // Result: [Groups, GroupSize, Out] -> [In, Out]
            return w_float.mul_(s_view).view({in_features, out_features});
        };

        return apply_group(debug_scalefactor, debug_zeropoint);
    } else {
        std::cout << "Per-channel dequantization: not implemented" << std::endl;
        std::exit(1);
    }
}

int main() {
    setbuf(stdout, NULL);
    std::cout << "Starting test_layout_npu_gemv..." << std::endl;
    try {
        if (!torch::cuda::is_available()) {
            std::cerr << "CUDA/HIP not available! Cannot run NPU test." << std::endl;
            return 1;
        }
        auto device = torch::kCUDA;

        // Initialize XDNA
        std::cout << "Initializing XDNA driver..." << std::endl;
        if (initialize_xdna_driver() != 0) {
            std::cerr << "Failed to initialize XDNA driver!" << std::endl;
            return 1;
        }
        init_npu();

        // Enable packed weights to use layer's internal packing logic
        use_packed_weights = true;

        // Dimensions
        int64_t M = 1;    // GEMV Case
        int64_t K = 4096; // In Features
        int64_t N = 4096; // Out Features (mlp.up_proj)
        int64_t group_size = 128;
        int64_t num_groups = K / group_size;

        std::cout << "Dimensions: M=" << M << ", K=" << K << ", N=" << N << ", Groups=" << num_groups << std::endl;

        torch::Tensor B_in_out;
        torch::Tensor scales;
        torch::Tensor zeros;
        torch::Tensor x_test;

        // qweight will be [Out, In] after processing
        torch::Tensor qweight;

        if (TEST_CASE == 1) {
            std::cout << "Starting Random Test (TEST_CASE 1)..." << std::endl;

            // Set seed for reproducibility
            torch::manual_seed(42);

            // Random Weights (uint8, 0-15)
            B_in_out = torch::randint(0, 16, {K, N}, torch::kUInt8);

            // Random Zeros (uint8, 0-15), [Groups, N]
            zeros = torch::randint(0, 16, {num_groups, N}, torch::kUInt8);

            // Random Scales (BF16 compatible), [Groups, N]
            auto scales_float = torch::rand({num_groups, N}, torch::kFloat32); // 0..1
            auto scales_bf16 = scales_float.to(torch::kBFloat16) * 0.1f;
            scales = scales_bf16.clone();

            // Random Input (BF16)
            x_test = torch::rand({M, K}, torch::kBFloat16) * 0.01f;

            // Convert to [Out, In] for packing (Standard layout)
            qweight = B_in_out.contiguous(); // [Out, In]

        } else if (TEST_CASE == 2) {
            // Load real Llama-3-8B weights from safetensors (Skipping for now to keep test simple self-contained or add back if needed)
            std::cout << "Real weights test not configured for this file yet." << std::endl;
            return 0; // Or fall back to random
        } else {
            std::cout << "Starting Deterministic Test (TEST_CASE 0)..." << std::endl;

            // B Matrix: Cyclic int4 values 0-15
            B_in_out = torch::zeros({K, N}, torch::kUInt8);
            for (int64_t i = 0; i < K; ++i) {
                uint8_t val = (i % 16);
                B_in_out[i].fill_(val);
            }

            // Scales: 0.1, 0.2, ... 0.8 (Per-Input)
            auto scales_per_group = torch::arange(1, num_groups + 1, torch::kFloat32) * 0.1f;
            scales = scales_per_group.unsqueeze(1).expand({num_groups, N}).clone().to(torch::kBFloat16);

            // Zeros: 0
            zeros = torch::zeros({num_groups, N}, torch::kUInt8);

            // X: 0.1
            x_test = torch::full({M, K}, 0.1, torch::kBFloat16);

            // Convert to [Out, In] for packing (Standard layout)
            qweight = B_in_out.contiguous(); // [Out, In]
        }
        std::cout << "qweight: shape=" << qweight.sizes() << ", type=" << qweight.dtype() << std::endl;
        std::cout << "scales:  shape=" << scales.sizes() << ", type=" << scales.dtype() << std::endl;
        std::cout << "zeros:   shape=" << zeros.sizes() << ", type=" << zeros.dtype() << std::endl;

        // Set up input tensor for NPU
        auto x_test_gpu = x_test.to(device);
        uint32_t handle_in = import_dma_buf_to_xdna(x_test_gpu.data_ptr(), x_test_gpu.numel(), 2);

        // Reference tensor for comparison (will be set based on TEST_REFERENCE)
        torch::Tensor y_ref;

        torch::Tensor w_npu_dequant;
#if TEST_REFERENCE
        // 3. Dequantize Reference
        std::cout << "Running Reference Dequantization..." << std::endl;

        // Not implemented fully for this snippet unless copied from test_layout_npu
        // For now assuming TEST_REFERENCE 0

#else
        std::cout << "Skipping reference dequantization test (TEST_REFERENCE=0)..." << std::endl;
#endif

        // Test QuantizedLinear (always run - this becomes the reference when TEST_REFERENCE=0)
        std::cout << "\nTesting QuantizedLinear (GPU/CPU Execution for Reference)..." << std::endl;
        QuantizedLinearImpl layer(K, N, false);
        // Transpose [Groups, N] -> [N, Groups] for the layer (No longer needed as layer expects [Groups, N] based on test_layout_npu)
        layer.set_quantized_weights(qweight, scales, zeros, torch::Tensor());

        // Use dequantize_weights_packed because use_packed_weights=true means standard dequantize_weights is not supported
        auto w_method = layer.dequantize_weights_packed().to(torch::kBFloat16);
        auto w_method_gpu = w_method.to(device);
        auto y_method = torch::matmul(x_test_gpu, w_method_gpu.t());

        std::cout << "First 8 values of y_method: ";
        std::cout << std::fixed << std::setprecision(2);
        for (int i = 0; i < 8; ++i) {
            std::cout << y_method[0][i].item<float>() << " ";
        }
        std::cout << std::endl;

#if !TEST_REFERENCE
        // Use method output as reference for NPU comparison
        y_ref = y_method.clone();
        std::cout << "Using QuantizedLinear output as reference for NPU comparison." << std::endl;
#endif

        // 8. NPU Execution (only if kernel exists for this dimension)
        std::cout << "\n============================================" << std::endl;
        std::cout << "Starting NPU Integration Test (GEMV Layout Verification)" << std::endl;
        std::cout << "============================================" << std::endl;

        // Check if we have a kernel for this dimension
        // Check if we have a kernel for this dimension
        const char *env_root = std::getenv("HETEROMOSAIC_ROOT");
        std::string root_dir = env_root ? env_root : "/home/greg/Desktop/heteroMosaic";
        // 1x4096x14336 kernel
        std::string xclbin_path =
            // root_dir + "/hw_bins/npu2/1x4096x14336/bf16_int4AWQ_bf16/final_1x4096x14336_128x64_8c_bf16_int4AWQ_bf16.pdi";
            root_dir + "/hw_bins/npu2/1x4096x14336/bf16_int4AWQ_bf16_K/final_1x4096x4096_128x64_8c_bf16_int4AWQ_bf16.pdi";

        std::string inst_path = root_dir + "/hw_bins/npu2/1x4096x14336/bf16_int4AWQ_bf16_K/insts_1x4096x4096_128x64_8c_bf16_int4AWQ_bf16.txt";

        // Check if files exist
        if (access(xclbin_path.c_str(), F_OK) == -1 || access(inst_path.c_str(), F_OK) == -1) {
            std::cout << "Skipping NPU Execution: Kernel files not found at expected paths." << std::endl;
            std::cout << "PDI: " << xclbin_path << std::endl;
            std::cout << "INST: " << inst_path << std::endl;
            return 0;
        }

        std::cout << "Loading PDI: " << xclbin_path << std::endl;
        // User requested 32 columns
        if (createHWctxt(xdna_drv_fd, hwctxt_array[0], xclbin_path.c_str(), 32) != 0) { // Using 1 column (GEMV default?)
            std::cerr << "Failed to create HW context!" << std::endl;
            return 1;
        }

        std::cout << "Loading Instructions: " << inst_path << std::endl;
        if (createInstctxt(xdna_drv_fd, instctxt_array[0], inst_path.c_str(), true) != 0) {
            std::cerr << "Failed to create Inst context!" << std::endl;
            return 1;
        }

        // Packed Weights for NPU
        std::cout << "Getting packed params from layer..." << std::endl;
        auto packed_tensor = layer.get_packed_params();
        if (!packed_tensor.defined()) {
            std::cerr << "Error: layer.get_packed_params() returned undefined tensor! Implementation failed to pack weights." << std::endl;
            return 1;
        }
        auto packed_cuda = packed_tensor.to(device);
        uint32_t handle_w = import_dma_buf_to_xdna(packed_cuda.data_ptr(), packed_cuda.numel(), 1);

        // Output Y
        auto y_npu = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));
        uint32_t handle_out = import_dma_buf_to_xdna(y_npu.data_ptr(), y_npu.numel(), 2);

        std::cout << "Running NPU GEMM (8 Iterations)..." << std::endl;
        // Flush caches for CPU written data before execution
        FlushCpuCache((const void *)instctxt_array[0].dpu_0_vaddr, 0, instctxt_array[0].num_dpu_0_insts * sizeof(uint32_t));
        FlushCpuCache((const void *)hwctxt_array[0].pdi_vaddr, 0, hwctxt_array[0].pdi_size);

        // Warmup Loop
        std::cout << "Running Warmup (4 Cycles)..." << std::endl;
        for (int i = 0; i < 4; ++i) {
            int ret = npuMatmul_zero(0, 0, y_npu.data_ptr(), x_test_gpu.data_ptr(), packed_cuda.data_ptr(), handle_out, handle_in, handle_w,
                                     (hipEvent_t) nullptr);
            if (ret != 0) {
                std::cerr << "Warmup Failed on iteration " << i << "!" << std::endl;
                return 1;
            }
        }

        int num_runs = 8;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_runs; ++i) {
            // Cast nullptr to hipEvent_t to resolve overload ambiguity
            int ret = npuMatmul_zero(0, 0, y_npu.data_ptr(), x_test_gpu.data_ptr(), packed_cuda.data_ptr(), handle_out, handle_in, handle_w,
                                     (hipEvent_t) nullptr);

            if (ret != 0) {
                std::cerr << "NPU Execution Failed on iteration " << i << "!" << std::endl;
                return 1;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "NPU Execution Successful!" << std::endl;

        std::chrono::duration<double> diff = end - start;
        double avg_latency = diff.count() / num_runs;

        // TOPS Calculation
        // Ops = 2 * M * N * K
        // TOPS = Ops / (Time * 1e12)
        double ops = 2.0 * M * K * N;
        double tops = ops / (avg_latency * 1e12);

        std::cout << "Average Latency: " << avg_latency * 1000.0 << " ms" << std::endl;
        std::cout << "Performance: " << tops << " TOPS" << std::endl;

        // Comparison
        std::cout << "Comparing NPU output with Reference..." << std::endl;
        auto y_npu_cpu = y_npu.cpu();

        // Check first 8 elements
        std::cout << "First 8 values of y_npu: ";
        std::cout << std::fixed << std::setprecision(2);
        for (int i = 0; i < 8; ++i) {
            std::cout << y_npu_cpu[0][i].item<float>() << " ";
        }
        std::cout << std::endl;

        bool match_npu = torch::allclose(y_ref.cpu(), y_npu_cpu, 0.2, 0.2); // Tolerance 0.2 roughly
        if (match_npu) {
            std::cout << "SUCCESS: y_npu matches y_ref!" << std::endl;
        } else {
            std::cout << "FAILURE: y_npu mismatch!" << std::endl;
            auto diff = (y_ref.cpu() - y_npu_cpu).abs();
            std::cout << "Max diff: " << diff.max().item<float>() << std::endl;
        }

    } catch (const c10::Error &e) {
        std::cerr << "Torch Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "STD Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown Error" << std::endl;
        return 1;
    }
    return 0;
}
