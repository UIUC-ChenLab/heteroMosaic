
#include "hipkernels/w4a16_gemm_packed.hpp"
#include "hipkernels/w4a16_gemm_unpacked.hpp"
#include "hipkernels/w4a16_gemv_unpacked.hpp"
#include "unified_llm_w4a16_hetero/unified_llm_w4a16.hpp"
#include <chrono>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <torch/torch.h>
#include <vector>

#define TEST_UNPACKED_KERNEL 1

#define HIP_CHECK(status)                                                                                                                  \
    if (status != hipSuccess) {                                                                                                            \
        std::cerr << "HIP Error: " << hipGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl;                     \
        exit(1);                                                                                                                           \
    }

void hip_synchronize() { HIP_CHECK(hipDeviceSynchronize()); }

// Helper to calculate TOPS and run benchmark
void run_benchmark(int M, int K, int N, const torch::Tensor &input_gpu, const torch::Tensor &packed_params, std::string label) {
    torch::Tensor output = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(input_gpu.device()));

    // Warmup (let GPU clocks boost)
    const int warmup_iters = 16;
    auto input_2d = input_gpu.view({-1, K});
    auto output_2d = output.view({-1, N});
    for (int w = 0; w < warmup_iters; ++w) {
        if (M == 1) {
            hipkernels::w4a16_gemv_fused_packed(output_2d, input_2d, packed_params, K, N);
        } else {
            hipkernels::w4a16_gemm_fused_packed(output, input_gpu, packed_params, K, N);
        }
    }
    hip_synchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 8; ++i) {
        if (M == 1) {
            auto input_2d = input_gpu.view({-1, K});
            auto output_2d = output.view({-1, N});
            hipkernels::w4a16_gemv_fused_packed(output_2d, input_2d, packed_params, K, N);
        } else {
            hipkernels::w4a16_gemm_fused_packed(output, input_gpu, packed_params, K, N);
        }
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

int main(int argc, char **argv) {
    if (!torch::cuda::is_available()) {
        std::cerr << "HIP available check failed" << std::endl;
        return 1;
    }
    auto device = torch::kCUDA;

    // Dimensions
    int64_t K = 4096;
    int64_t N = 14336;

    std::cout << "Testing w4a16 HIP Kernels against QuantizedLinear (New Layout)" << std::endl;
    std::cout << "Weights K=" << K << ", N=" << N << std::endl;

    // 1. Setup Layer and Weights
#ifndef TEST_UNPACKED_KERNEL
    QuantizedLinearImpl layer(K, N, false);
    layer.to(device);
#endif

    int64_t group_size = 128;
    int64_t num_groups = K / group_size;

    torch::manual_seed(42);

    // Initialize random raw weights [K, N]
    auto qweight = torch::randint(0, 16, {K, N}, torch::kUInt8).to(device);

    // Initialize scales and zero points [Groups, N]
    auto scales = (torch::rand({num_groups, N}, torch::kFloat32).to(torch::kBFloat16) * 0.1f).to(device);
    auto zeros = torch::randint(0, 16, {num_groups, N}, torch::kInt8).to(device);

#ifndef TEST_UNPACKED_KERNEL
    std::cout << "Setting quantized weights..." << std::endl;
    // Note: set_quantized_weights is void, but sets layer.packed_params_
    layer.set_quantized_weights(qweight, scales, zeros, torch::Tensor());
#endif

    // 2. Get Ground Truth (Reference)
    torch::Tensor w_ref;
#ifdef TEST_UNPACKED_KERNEL
    std::cout << "Generating Ground Truth Manually (Unpacked)..." << std::endl;
    // qweight is [K, N]. Transpose to [N, K].
    auto w_int = qweight.t().contiguous().to(torch::kBFloat16);
    // scales/zeros are [Groups, N]. Transpose to [N, Groups].
    // Expand to [N, K].
    auto s_exp = scales.t().contiguous().repeat_interleave(group_size, 1);
    auto z_exp = zeros.t().contiguous().to(torch::kBFloat16).repeat_interleave(group_size, 1);

    w_ref = (w_int - z_exp) * s_exp; // [N, K]
#else
    std::cout << "Generating Ground Truth via layer.dequantize_weights_packed()..." << std::endl;
    w_ref = layer.dequantize_weights_packed().to(torch::kBFloat16); // [Out, In]
#endif
    std::cout << "w_ref shape: " << w_ref.sizes() << " device: " << w_ref.device() << std::endl;
    // Ground truth weights are [Out, In]
    // So returned is [Out, In]. Correct.

#ifndef TEST_UNPACKED_KERNEL
    auto packed_params = layer.get_packed_params();
#endif

    // ------------------------------------------------
    // Test GEMM (4096x4096x16384)
    // ------------------------------------------------
    int64_t M_gemm = 4096;
    std::cout << "\n=== Testing GEMM M=" << M_gemm << " ===" << std::endl;

    auto input_gemm = torch::rand({M_gemm, K}, torch::kBFloat16).to(device) * 0.1f;
    std::cout << "input_gemm shape: " << input_gemm.sizes() << " device: " << input_gemm.device() << std::endl;

    // Ref: y = x @ w.T
    // x: [M, K], w_ref: [Out, In] -> [N, K]
    // x @ w.T -> [M, K] @ [K, N] -> [M, N]
    auto y_ref_gemm = torch::matmul(input_gemm, w_ref.t());

    // Kernel
    auto output_gemm = torch::zeros({M_gemm, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));

#ifdef TEST_UNPACKED_KERNEL
    // --- Unpacked Kernel Setup ---
    std::cout << "Running UNPACKED GEMM Kernel..." << std::endl;
    // 1. Prepare Weights [N, K/2]
    // qweight is [K, N]. Need [N, K].
    auto w_nk = qweight.t().contiguous(); // [N, K]
    // Pack pairs (k, k+1) -> 1 byte
    // [N, K/2, 2]
    auto w_pairs = w_nk.view({N, K / 2, 2});
    auto w_low = w_pairs.select(2, 0);  // even cols
    auto w_high = w_pairs.select(2, 1); // odd cols
    // shift high << 4 | low
    auto qweights_gpu = (w_low & 0x0F) | torch::bitwise_left_shift(w_high & 0x0F, 4);
    qweights_gpu = qweights_gpu.to(torch::kUInt8).contiguous();

    // 2. Prepare Scales/Zeros [N, Groups]
    // scales is [Groups, N]. Need [N, Groups].
    auto scales_gpu = scales.t().contiguous(); // [N, Groups]
    // Zeros pass as Int8 (matching kernel signature update)
    auto zeros_gpu = zeros.t().contiguous().to(torch::kInt8); // [N, Groups]

    hipkernels::w4a16_gemm_unpacked_fused(output_gemm, input_gemm, qweights_gpu, scales_gpu, zeros_gpu, K, N, group_size);
#else
    std::cout << "Running PACKED GEMM Kernel..." << std::endl;
    hipkernels::w4a16_gemm_fused_packed(output_gemm, input_gemm, packed_params, K, N);
#endif

    hip_synchronize();

    auto y_out_gemm = output_gemm;

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
        auto y_out_cpu = y_out_gemm.cpu();
        std::cout << std::fixed << std::setprecision(2);
        for (int i = 0; i < 8; ++i)
            std::cout << y_out_cpu[0][i].item<float>() << " ";
        std::cout << std::endl;
    }

    bool match_gemm = torch::allclose(y_ref_gemm, y_out_gemm, 0.01, 0.1); // Loose tolerance for now
    if (match_gemm) {
        std::cout << "SUCCESS: GEMM Output matches reference!" << std::endl;
    } else {
        std::cout << "FAILURE: GEMM Output mismatch!" << std::endl;
        auto diff = (y_ref_gemm - y_out_gemm).abs();
        auto max_val = diff.max();
        auto max_idx = diff.argmax();
        // Convert flat index to 2D
        auto N_dim = y_ref_gemm.size(1);
        long flat_idx = max_idx.item<long>();
        long row = flat_idx / N_dim;
        long col = flat_idx % N_dim;

        std::cout << "Max Diff: " << max_val.item<float>() << " at [" << row << ", " << col << "]" << std::endl;
        std::cout << "Ref: " << y_ref_gemm[row][col].item<float>() << " Kernel: " << y_out_gemm[row][col].item<float>() << std::endl;
    }

#ifdef TEST_UNPACKED_KERNEL
    // Benchmark Unpacked
    run_benchmark_unpacked(M_gemm, K, N, input_gemm, qweights_gpu, scales_gpu, zeros_gpu, group_size, "UNPACKED GEMM");

    // ------------------------------------------------
    // Test GEMV (1x4096x16384)
    // ------------------------------------------------
    int64_t M_gemv = 1;
    std::cout << "\n=== Testing GEMV M=" << M_gemv << " ===" << std::endl;

    auto input_gemv = torch::rand({M_gemv, K}, torch::kBFloat16).to(device) * 0.1f;
    auto y_ref_gemv = torch::matmul(input_gemv, w_ref.t());

    auto output_gemv = torch::zeros({M_gemv, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));

    std::cout << "Running UNPACKED GEMV Kernel..." << std::endl;
    // Warmup
    for (int i = 0; i < 5; ++i)
        hipkernels::w4a16_gemv_unpacked_fused(output_gemv, input_gemv, qweights_gpu, scales_gpu, zeros_gpu, K, N, group_size);
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
        long col = flat_idx; // GEMV is [1, N]

        std::cout << "Max Diff: " << max_val.item<float>() << " at [0, " << col << "]" << std::endl;
        std::cout << "Ref: " << y_ref_gemv[0][col].item<float>() << " Kernel: " << output_gemv[0][col].item<float>() << std::endl;
    }

    // Benchmark GEMV
    auto start_gemv = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        hipkernels::w4a16_gemv_unpacked_fused(output_gemv, input_gemv, qweights_gpu, scales_gpu, zeros_gpu, K, N, group_size);
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto end_gemv = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff_gemv = end_gemv - start_gemv;
    double avg_time_gemv = (diff_gemv.count() / 100.0) * 1000.0; // ms
    double tops_gemv = (2.0 * M_gemv * N * K) / (avg_time_gemv * 1e-3) / 1e12;

    std::cout << "UNPACKED GEMV Avg Time: " << avg_time_gemv << " ms" << std::endl;
    std::cout << "UNPACKED GEMV Performance: " << tops_gemv << " TOPS" << std::endl;
#else
    run_benchmark(M_gemm, K, N, input_gemm, packed_params, "GEMM");
#endif

#ifndef TEST_UNPACKED_KERNEL
    // ------------------------------------------------
    // Test GEMV (1x4096x16384)
    // ------------------------------------------------
    int64_t M_gemv = 1;
    std::cout << "\n=== Testing GEMV M=" << M_gemv << " ===" << std::endl;

    auto input_gemv = torch::rand({M_gemv, K}, torch::kBFloat16).to(device) * 0.1f;
    auto y_ref_gemv = torch::matmul(input_gemv, w_ref.t());

    auto output_gemv = torch::zeros({M_gemv, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));
    auto input_2d = input_gemv.view({-1, K});
    auto output_2d = output_gemv.view({-1, N});

    std::cout << "Running GEMV Kernel..." << std::endl;
    hipkernels::w4a16_gemv_fused_packed(output_2d, input_2d, packed_params, K, N);
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
#endif

    return 0;
}
