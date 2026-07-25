
#include "hipkernels/w4a16_gemm_packed.hpp"
#include "hipkernels/w4a16_gemm_packedGPU.hpp"
#include "hipkernels/w4a16_gemm_unpacked.hpp"
#include "hipkernels/w4a16_gemv_unpacked.hpp"
#include "hipkernels/w4a16_gemv_packedGPU.hpp"
#include "unified_llm_w4a16_hetero/unified_llm_w4a16.hpp"
#include <chrono>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <torch/torch.h>
#include <vector>


#define HIP_CHECK(status)                                                                                                                  \
    if (status != hipSuccess) {                                                                                                            \
        std::cerr << "HIP Error: " << hipGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl;                     \
        exit(1);                                                                                                                           \
    }

void hip_synchronize() { HIP_CHECK(hipDeviceSynchronize()); }

// Helper to calculate TOPS and run benchmark
// Helper to calculate TOPS and run benchmark
void run_benchmark(int M, int K, int N, const torch::Tensor &input_gpu, const torch::Tensor &packed_params, std::string label) {
    torch::Tensor output = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(input_gpu.device()));

    // Warmup (let GPU clocks boost)
    const int warmup_iters = 16;
    for (int w = 0; w < warmup_iters; ++w) {
        if (M == 1) {
            hipkernels::w4a16_gemv_packedGPU(output, input_gpu, packed_params, M, K, N);
        } else {
            hipkernels::w4a16_gemm_packedGPU(output, input_gpu, packed_params, M, K, N);
        }
    }
    hip_synchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        if (M == 1) {
            hipkernels::w4a16_gemv_packedGPU(output, input_gpu, packed_params, M, K, N);
        } else {
            hipkernels::w4a16_gemm_packedGPU(output, input_gpu, packed_params, M, K, N);
        }
    }
    hip_synchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    double avg_time = elapsed.count() / 100.0;
    double ops = 2.0 * (double)M * (double)K * (double)N;
    double tops = ops / (avg_time * 1e12);

    std::cout << label << " Avg Time: " << (avg_time * 1000.0) << " ms" << std::endl;
    std::cout << label << " Performance: " << tops << " TOPS" << std::endl;
}

// Helper for Unpacked Benchmark
void run_benchmark_unpacked(int M, int K, int N, const torch::Tensor &input_gpu, const torch::Tensor &qweights, const torch::Tensor &scales,
                            const torch::Tensor &zeros, int64_t group_size, std::string label) {
    torch::Tensor output = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(input_gpu.device()));

    // Warmup (let GPU clocks boost)
    const int warmup_iters = 16;
    for (int w = 0; w < warmup_iters; ++w) {
        hipkernels::w4a16_gemm_unpacked_fused(output, input_gpu, qweights, scales, zeros, K, N, group_size);
    }
    hip_synchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 8; ++i) {
        hipkernels::w4a16_gemm_unpacked_fused(output, input_gpu, qweights, scales, zeros, K, N, group_size);
    }
    hip_synchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    double avg_time = elapsed.count() / 8.0;
    double ops = 2.0 * (double)M * (double)K * (double)N;
    double tops = ops / (avg_time * 1e12);

    std::cout << label << " Avg Time: " << (avg_time * 1000.0) << " ms" << std::endl;
    std::cout << label << " Performance: " << tops << " TOPS" << std::endl;
}


// ----------------------------------------------------------------------------
// Quantized Linear Layer for GPU (Q4_K-style Layout)
// Moved here to avoid test suite ABI issues and support rapid iteration.
// ----------------------------------------------------------------------------
class QuantizedGPULinearImpl : public torch::nn::Module {
  public:
    QuantizedGPULinearImpl(int64_t in_features, int64_t out_features, bool bias = false, int64_t max_seq_len = 2048,
                           std::string layer_type = "")
        : in_features_(in_features), out_features_(out_features) {
        
        // Calculate buffer size
        // Layout: Block Interleaved
        // Block Size: 32 weights (implicit in packing 20 bytes: 0-1 scale, 2-3 zero, 4-19 weights)
        // Stride: 20 bytes 
        
        int64_t K = in_features;
        int64_t N = out_features;
        int64_t num_blocks_k = (K + 31) / 32;
        int64_t total_blocks = N * num_blocks_k;
        int64_t buffer_size = total_blocks * 20;

        packed_params_ = register_buffer("packed_params", torch::zeros({buffer_size}, torch::kUInt8));

        if (bias) {
            bias_ = register_buffer("bias", torch::zeros({out_features}, torch::kBFloat16));
        }
    }

    void set_quantized_weights(torch::Tensor qweight, torch::Tensor scale, torch::Tensor zero_point, torch::Tensor g_idx = torch::Tensor()) {
        torch::NoGradGuard no_grad;

        // One-time print to confirm method usage
        static bool _printed = false;
        if (!_printed) {
            std::cout << "QuantizedGPULinearImpl::set_quantized_weights: Packing AWQ/Generic weights into Q4_K layout..." << std::endl;
            _printed = true;
        }

        // qweight: [K, N] uint8 (unpacked indices) or [N, K]
        // scale: [Groups, N] bf16
        // zero_point: [Groups, N] bf16 (or int8 acting as value)
        
        torch::Tensor unpacked_w;

        if (qweight.size(0) == out_features_ && qweight.size(1) == in_features_) {
            // [Out, In]
            unpacked_w = qweight.to(torch::kInt8).contiguous();
        } else if (qweight.size(0) == in_features_ && qweight.size(1) == out_features_) {
            // [In, Out] -> Transpose
            unpacked_w = qweight.to(torch::kInt8).t().contiguous();
        } else {
            std::cerr << "Error: QuantizedGPULinearImpl::set_quantized_weights - unsupported shape " << qweight.sizes() << std::endl;
            return;
        }

        // 2. Prepare metadata
        // Check scale/zero shapes. 
        torch::Tensor s_expanded = scale.to(torch::kBFloat16);
        torch::Tensor z_expanded = zero_point.to(torch::kBFloat16);

        int64_t N = out_features_;
        int64_t K = in_features_;
        int64_t num_blocks_k = K / 32;

        // Expand to [N, num_blocks_k]
        // Logic handles group_size > 32 (e.g. 128) by replicating metadata
        if (s_expanded.numel() != N * num_blocks_k) {
             int64_t num_scales = s_expanded.numel();
             // Assume perfect division
             int64_t n_groups = num_scales / N; 
             int64_t group_size = K / n_groups;
             
             if (group_size % 32 != 0) {
                 std::cerr << "Error: Block size 32 must divide group_size " << group_size << std::endl;
                 return;
             }
             int64_t repeat = group_size / 32;
             
             // Ensure [N, Groups]
             if (s_expanded.size(0) != N) {
                 // Assume [Groups, N] -> [N, Groups]
                 s_expanded = s_expanded.view({n_groups, N}).t();
                 z_expanded = z_expanded.view({n_groups, N}).t();
             }
             
             s_expanded = s_expanded.unsqueeze(2).repeat({1, 1, repeat}).view({N, num_blocks_k});
             z_expanded = z_expanded.unsqueeze(2).repeat({1, 1, repeat}).view({N, num_blocks_k});
        } else {
             // Already correct total elements, ensure shape
             s_expanded = s_expanded.view({N, num_blocks_k});
             z_expanded = z_expanded.view({N, num_blocks_k});
        }

        // 3. Packing Loop
        auto packed_view = packed_params_.view({N, num_blocks_k, 20});

        // Write Scale (bytes 0-1)
        s_expanded = s_expanded.contiguous();
        auto s_u8 = torch::from_blob(s_expanded.data_ptr(), {N, num_blocks_k, 2}, torch::TensorOptions().dtype(torch::kUInt8).device(s_expanded.device()));
        packed_view.slice(2, 0, 2).copy_(s_u8);
        
        // Write Zero (bytes 2-3)
        z_expanded = z_expanded.contiguous();
        auto z_u8 = torch::from_blob(z_expanded.data_ptr(), {N, num_blocks_k, 2}, torch::TensorOptions().dtype(torch::kUInt8).device(z_expanded.device()));
        packed_view.slice(2, 2, 4).copy_(z_u8);

        // Write Weights (bytes 4-19)
        // unpacked_w is [N, K]. View as [N, num_blocks_k, 32].
        auto w_blocks = unpacked_w.view({N, num_blocks_k, 32});
        
        // Pack 2 weights per byte. 
        // Standard: (w[0] & 0xF) | (w[1] << 4)
        auto w_pairs = w_blocks.view({N, num_blocks_k, 16, 2});
        auto w_low = torch::bitwise_and(w_pairs.select(-1, 0).to(torch::kUInt8), 0x0F);
        auto w_high = torch::bitwise_and(w_pairs.select(-1, 1).to(torch::kUInt8), 0x0F);
        auto w_packed = torch::bitwise_or(w_low, torch::bitwise_left_shift(w_high, 4));

        packed_view.slice(2, 4, 20).copy_(w_packed);
    }

    torch::Tensor dequantize_weights_packedGPU() {
        torch::NoGradGuard no_grad;
        int64_t N = out_features_;
        int64_t K = in_features_;
        int64_t num_blocks_k = K / 32;

        auto packed_view = packed_params_.view({N, num_blocks_k, 20});

        // Extract metadata: Scale (bytes 0-1) and Zero (bytes 2-3)
        auto s_u8 = packed_view.slice(2, 0, 2).contiguous(); 
        auto z_u8 = packed_view.slice(2, 2, 4).contiguous();

        // Reinterpret [N, B, 2] uint8 as [N, B] bf16
        auto s = torch::from_blob(s_u8.data_ptr(), {N, num_blocks_k}, torch::TensorOptions().dtype(torch::kBFloat16).device(packed_view.device()));
        auto z = torch::from_blob(z_u8.data_ptr(), {N, num_blocks_k}, torch::TensorOptions().dtype(torch::kBFloat16).device(packed_view.device()));

        // Extract weights
        auto w_packed = packed_view.slice(2, 4, 20); // [N, B, 16]
        // Unpack
        auto w_low = torch::bitwise_and(w_packed, 0x0F);
        auto w_high = torch::bitwise_and(torch::bitwise_right_shift(w_packed, 4), 0x0F);

        std::vector<torch::Tensor> stack_tensors;
        stack_tensors.push_back(w_low);
        stack_tensors.push_back(w_high);
        auto w_stacked = torch::stack(stack_tensors, -1);
        auto w_unpacked = w_stacked.view({N, num_blocks_k, 32});

        // Dequantize: (w - z) * s
        auto s_exp = s.unsqueeze(-1); // [N, B, 1] broadcast to 32
        auto z_exp = z.unsqueeze(-1);
        
        auto w_val = w_unpacked.to(torch::kBFloat16);
        
        auto w_dq = (w_val - z_exp) * s_exp;

        return w_dq.view({N, K}); // [Out, In]
    }

    // Forward pass
    std::future<int> forward(torch::Tensor output_buffer, torch::Tensor input, std::string layer_type) {
        int64_t M = input.size(0);
        int64_t K = input.size(1);
        int64_t N = out_features_;

        if (M == 1 && false) { // disabled for now for testing
            // Use our new kernel
            hipkernels::w4a16_gemv_packedGPU(output_buffer, input, packed_params_, M, K, N);
        } else {
             // Fallback to dequantize + gemm for M > 1
             auto w = dequantize_weights_packedGPU(); // [Out, In]
             auto res = torch::matmul(input, w.t()); // [M, In] @ [In, Out] -> [M, Out]
             
             if (output_buffer.sizes() == res.sizes()) {
                 output_buffer.copy_(res);
             } else {
                 output_buffer.slice(1, 0, res.size(1)).copy_(res);
             }
        }
        
        if (bias_.defined()) {
            output_buffer.add_(bias_);
        }
        
        return std::future<int>();
    }

    // Accessors
    int64_t in_features() const { return in_features_; }
    int64_t out_features() const { return out_features_; }
    torch::Tensor get_packed_params() const { return packed_params_; }

  private:
    int64_t in_features_;
    int64_t out_features_;
    torch::Tensor packed_params_;    // Q4_K style packed buffer
    torch::Tensor bias_;             // Optional bias
};
TORCH_MODULE(QuantizedGPULinear);

int main(int argc, char **argv) {
    if (!torch::cuda::is_available()) {
        std::cerr << "HIP available check failed" << std::endl;
        return 1;
    }
    
    // Set seed for repeatability
    torch::manual_seed(42);
    
    auto device = torch::kCUDA;

    // Dimensions
    int64_t K = 4096;
    int64_t N = 14336;

    std::cout << "Testing QuantizedGPULinearImpl (Q4_K Layout) against QuantizedLinearImpl" << std::endl;
    std::cout << "Weights K=" << K << ", N=" << N << std::endl;

    // 1. Setup Layer and Weights
    // Keep reference layer on GPU (QuantizedLinearImpl should support device)
    // Assuming constructor has device arg or moves later? 
    // The current constructor signature: QuantizedLinearImpl(in, out, bias)
    // We can move it after.
    QuantizedLinearImpl layer(K, N, false);
    layer.to(device);
    
    int64_t group_size = 128; 
    int64_t num_groups = K / group_size;

    // Initialize weights directly on GPU? 
    // Usually weights are created on CPU for randomness consistency and then moved.
    // User requested "put all layers on the gpu from the beginning".
    // We will ensure `layer` and `layerGPU` are .to(device) immediately or as soon as possible.
    
    std::vector<int64_t> size_kn = {K, N};
    std::vector<int64_t> size_groups = {num_groups, N};
    auto qweight_cpu = torch::randint(0, 16, size_kn, torch::kUInt8);
    auto scales_cpu = (torch::rand(size_groups, torch::kFloat32).to(torch::kBFloat16) * 0.1f);
    auto zeros_cpu = torch::randint(0, 16, size_groups, torch::kInt8);

    auto qweight_gpu = qweight_cpu.to(device);
    auto scales_gpu = scales_cpu.to(device);
    auto zeros_gpu = zeros_cpu.to(device);

    std::cout << "Setting quantized weights for Reference Layer npuPacked..." << std::endl;
    layer.set_quantized_weights(qweight_gpu, scales_gpu, zeros_gpu, torch::Tensor());

    // Q4_K Layer: Setup on GPU
    QuantizedGPULinearImpl layerGPU(K, N, false);
    layerGPU.to(device);
    std::cout << "Setting quantized weights for gpuPacked Kayer" << std::endl;
    layerGPU.set_quantized_weights(qweight_cpu, scales_cpu, zeros_cpu, torch::Tensor()); // Method expects CPU weights for packing? 
    // `QuantizedGPULinearImpl::set_quantized_weights` implementation:
    // It takes qweight/scale/zero.
    // It creates `unpacked_w`, `s_expanded` etc.
    // If input is CPU, operations happen on CPU. If input is GPU, operations happen on GPU.
    // BUT `packed_params_` is registered as buffer. `layerGPU` was moved to `device`. So `packed_params_` is on GPU.
    // If we pass CPU tensors, we might have cross-device copy issues if not careful.
    // Let's pass the GPU tensors we created `qweight_gpu` etc. 
    // Exception: `set_quantized_weights` implementation might have `no_grad` and just `copy_`. 
    // Let's check `set_quantized_weights`:
    // `packed_view = packed_params_.view(...)` -> uses `packed_params_` device.
    // `s_expanded = scale...` -> adopts `scale` device.
    // `packed_view.copy_(s_u8)` -> copy from `s_u8` (scale device) to `packed_view` (param device).
    // Safest is to pass GPU tensors if layer is on GPU.
    
    // However, the user request specifically said "Setting quantized weights ... (CPU)..." in previous logs,
    // reflecting that previously inputs were CPU.
    // We will switch to using GPU inputs.
    
    layerGPU.set_quantized_weights(qweight_gpu, scales_gpu, zeros_gpu, torch::Tensor());

    // 2. Get Ground Truth (Reference)
    std::cout << "Generating Ground Truth..." << std::endl;
    auto w_ref = layer.dequantize_weights_packed().to(torch::kBFloat16); 
    
    std::cout << "Generating Candidate..." << std::endl;
    auto w_gpu = layerGPU.dequantize_weights_packedGPU().to(torch::kBFloat16);

    // Send tensors to GPU for GEMM/GEMV tests
    // Already there. 
    // auto device_gpu = torch::kCUDA;
    // auto qweight = qweight_cpu.to(device_gpu);
    // Move layerGPU to device for forward pass tests
    // Already moved.

    std::cout << "w_ref shape: " << w_ref.sizes() << std::endl;
    std::cout << "w_gpu shape: " << w_gpu.sizes() << std::endl;

    // Compare Weights directly
    bool match_w = torch::allclose(w_ref, w_gpu, 0, 0);
    if (match_w) {
        std::cout << "SUCCESS: Weight Dequantization matches!" << std::endl;
    } else {
        std::cout << "FAILURE: Weight Dequantization mismatch!" << std::endl;
        auto diff = (w_ref - w_gpu).abs();
        auto max_val = diff.max().item<float>();
        std::cout << "Max Diff: " << max_val << std::endl;
    }

    // Move reference weights to GPU for GEMM/GEMV comparison
    w_ref = w_ref.to(device);

    // Get packed params (used for GEMM + GEMV)
    auto packed_params = layerGPU.get_packed_params();

    // ------------------------------------------------
    // Test GEMM
    // ------------------------------------------------
    int64_t M_gemm = 128;
    std::cout << "\n=== Testing GEMM M=" << M_gemm << " ===" << std::endl;

    auto input_gemm = torch::rand({M_gemm, K}, torch::kBFloat16).to(device) * 0.1f;
    auto y_ref_gemm = torch::matmul(input_gemm, w_ref.t());

    auto output_gemm = torch::zeros({M_gemm, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));

    std::cout << "Running GEMM Kernel..." << std::endl;
    hipkernels::w4a16_gemm_packedGPU(output_gemm, input_gemm, packed_params, M_gemm, K, N);
    hip_synchronize();

    std::cout << "First 8 values of y_ref_gemm: ";
    {
        auto y_ref_cpu = y_ref_gemm.cpu();
        std::cout << std::fixed << std::setprecision(2);
        for (int i = 0; i < 8; ++i)
            std::cout << y_ref_cpu[0][i].item<float>() << " ";
        std::cout << std::endl;
    }

    std::cout << "First 8 values of output_gemm: ";
    {
        auto output_cpu = output_gemm.cpu();
        std::cout << std::fixed << std::setprecision(2);
        for (int i = 0; i < 8; ++i)
            std::cout << output_cpu[0][i].item<float>() << " ";
        std::cout << std::endl;
    }

    bool match_gemm = torch::allclose(y_ref_gemm, output_gemm, 0.02, 0.2);
    if (match_gemm) {
        std::cout << "SUCCESS: GEMM Output matches reference!" << std::endl;
    } else {
        std::cout << "FAILURE: GEMM Output mismatch!" << std::endl;
        auto diff = (y_ref_gemm - output_gemm).abs();
        auto max_val = diff.max();
        auto max_idx = diff.argmax();
        long flat_idx = max_idx.item<long>();
        long row = flat_idx / N;
        long col = flat_idx % N;
        std::cout << "Max Diff: " << max_val.item<float>() << " at [" << row << ", " << col << "]" << std::endl;
        std::cout << "Ref: " << y_ref_gemm[row][col].item<float>() << " Kernel: " << output_gemm[row][col].item<float>() << std::endl;
    }

    run_benchmark(M_gemm, K, N, input_gemm, packed_params, "GEMM");


    // ------------------------------------------------
    // Test GEMV (1x4096x16384)
    // ------------------------------------------------
    int64_t M_gemv = 1;
    std::cout << "\n=== Testing GEMV M=" << M_gemv << " ===" << std::endl;

    auto input_gemv = torch::rand({M_gemv, K}, torch::kBFloat16).to(device) * 0.1f;
    auto y_ref_gemv = torch::matmul(input_gemv, w_ref.t());

    auto output_gemv = torch::zeros({M_gemv, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));
    // auto input_2d = input_gemv.view({-1, K});
    // auto output_2d = output_gemv.view({-1, N});
    
    std::cout << "Running GEMV Kernel..." << std::endl;
    // DEBUG: Inspect packed_params at split boundary
    {
        int64_t num_blocks_k = (K + 31) / 32;
        size_t weight_bytes = (size_t)N * num_blocks_k * 16;
        auto params_cpu = packed_params.cpu();
        uint8_t* ptr = params_cpu.data_ptr<uint8_t>();
        std::cout << "[DEBUG] PackedParams Base: " << (void*)packed_params.data_ptr<uint8_t>() << std::endl;
        std::cout << "[DEBUG] Split Offset: " << weight_bytes << std::endl;
        std::cout << "[DEBUG] Meta Base (Host Calc): " << (void*)(packed_params.data_ptr<uint8_t>() + weight_bytes) << std::endl;
        
        std::cout << "[DEBUG] Bytes at Offset 0 (Weights): ";
        for(int i=0; i<4; ++i) printf("%02X ", ptr[i]);
        std::cout << std::endl;
        
        std::cout << "[DEBUG] Bytes at Split Offset (Meta): ";
        for(int i=0; i<4; ++i) printf("%02X ", ptr[weight_bytes + i]);
        std::cout << std::endl;
    }

    // hipkernels::w4a16_gemv_fused_packed(output_2d, input_2d, packed_params, K, N);
    hipkernels::w4a16_gemv_packedGPU(output_gemv, input_gemv, packed_params, M_gemv, K, N);
    hip_synchronize();

    std::cout << "First 8 values of y_ref_gemv: ";
    {
        auto y_ref_cpu = y_ref_gemv.cpu();
        std::cout << std::fixed << std::setprecision(2);
        for (int i = 0; i < 8; ++i)
            std::cout << y_ref_cpu[0][i].item<float>() << " ";
        std::cout << std::endl;
    }

    std::cout << "First 8 values of output_gemv: ";
    {
        auto output_cpu = output_gemv.cpu();
        std::cout << std::fixed << std::setprecision(2);
        for (int i = 0; i < 8; ++i)
            std::cout << output_cpu[0][i].item<float>() << " ";
        std::cout << std::endl;
    }

    bool match_gemv = torch::allclose(y_ref_gemv, output_gemv, 0.01, 0.1);
    if (match_gemv) {
        std::cout << "SUCCESS: GEMV Output matches reference!" << std::endl;
    } else {
        std::cout << "FAILURE: GEMV Output mismatch!" << std::endl;
        auto diff = (y_ref_gemv - output_gemv).abs();
        auto max_val = diff.max();
        auto max_idx = diff.argmax();
        long flat_idx = max_idx.item<long>();
        long col = flat_idx; // GEMV is [1, N], flat logic still works or explicitly [0, col]

        std::cout << "Max Diff: " << max_val.item<float>() << " at [0, " << col << "]" << std::endl;
        std::cout << "Ref: " << y_ref_gemv[0][col].item<float>() << " Kernel: " << output_gemv[0][col].item<float>() << std::endl;
    }

    run_benchmark(M_gemv, K, N, input_gemv, packed_params, "GEMV");

    return 0;
}

