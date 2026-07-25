#include "cpu_avx_kernels/w4a16_gemv_avx_packed.hpp"
#include "cpu_avx_kernels/w4a16_gemv_avx_unpacked.hpp"
#include "unified_llm_w4a16_hetero/unified_llm_w4a16.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread> // For sleep
#include <torch/torch.h>
#include <vector>

// #define TEST_UNPACKED_KERNEL 1

// Helper to calculate TOPS and run benchmark
// Helper to calculate TOPS and run benchmark
void run_benchmark_packed(int M, int K, int N, const torch::Tensor &input_cpu, const torch::Tensor &packed_params, std::string label,
                          int num_threads) {
    torch::Tensor output = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU));
    auto input_1d = input_cpu.view({-1});
    auto output_1d = output.view({-1});

    // Cool down before starting
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Warmup Burst (wake up CPU, fill cache, trigger turbo)
    for (int i = 0; i < 20; ++i) {
        w4a16_gemv_cpu_fused_packed(output_1d, input_1d, packed_params, K, N, nullptr, num_threads);
    }

    // Measurement Burst
    int num_iters = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iters; ++i) {
        w4a16_gemv_cpu_fused_packed(output_1d, input_1d, packed_params, K, N, nullptr, num_threads);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    double avg_time = elapsed.count() / num_iters;
    double tops = (2.0 * M * K * N) / (avg_time * 1e12);
    double gops = (2.0 * M * K * N) / (avg_time * 1e9);

    std::cout << label << " (Threads=" << num_threads << ") Avg Time: " << (avg_time * 1000.0) << " ms" << std::endl;
    std::cout << label << " (Threads=" << num_threads << ") Performance: " << tops << " TOPS (" << gops << " GOPS)" << std::endl;
}

void run_benchmark_unpacked(int M, int K, int N, const torch::Tensor &input_cpu, const torch::Tensor &qweights, const torch::Tensor &scales,
                            const torch::Tensor &zeros, int64_t group_size, std::string label, int num_threads) {
    torch::Tensor output = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU));

    // Cool down
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Warmup Burst
    for (int i = 0; i < 20; ++i) {
        w4a16_gemv_cpu_fused_unpacked(output, input_cpu, qweights, scales, zeros, K, N, group_size, nullptr, num_threads);
    }

    // Measurement Burst
    int num_iters = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iters; ++i) {
        w4a16_gemv_cpu_fused_unpacked(output, input_cpu, qweights, scales, zeros, K, N, group_size, nullptr, num_threads);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    double avg_time = elapsed.count() / num_iters;
    double tops = (2.0 * M * K * N) / (avg_time * 1e12);
    double gops = (2.0 * M * K * N) / (avg_time * 1e9);

    std::cout << label << " (Threads=" << num_threads << ") Avg Time: " << (avg_time * 1000.0) << " ms" << std::endl;
    std::cout << label << " (Threads=" << num_threads << ") Performance: " << tops << " TOPS (" << gops << " GOPS)" << std::endl;
}

#define DEFAULT_NUM_THREADS 1

int main(int argc, char **argv) {
    torch::manual_seed(42);
    auto device = torch::kCPU;

    int64_t K = 4096;
    int64_t N = 16384;

    std::cout << "Testing w4a16 CPU AVX Kernels" << std::endl;
    std::cout << "Weights K=" << K << ", N=" << N << std::endl;

    QuantizedLinearImpl layer(K, N, false);
    layer.to(device);

    int64_t group_size = 128;
    int64_t num_groups = K / group_size;

    auto qweight = torch::randint(0, 16, {K, N}, torch::kUInt8).to(device); // [K, N]
    auto scales = (torch::rand({num_groups, N}, torch::kFloat32).to(torch::kBFloat16) * 0.1f).to(device);
    // Use Int8 for Zeros (0..15 fits in Int8)
    auto zeros = torch::randint(0, 16, {num_groups, N}, torch::kInt8).to(device);

    std::cout << "Setting quantized weights..." << std::endl;
    layer.set_quantized_weights(qweight, scales, zeros, torch::Tensor());

    // 2. Reference
    std::cout << "Generating Ground Truth..." << std::endl;
    auto w_int = qweight.t().contiguous().to(torch::kBFloat16);                                // [N, K]
    auto s_exp = scales.t().contiguous().repeat_interleave(group_size, 1);                     // [N, K]
    auto z_exp = zeros.t().contiguous().to(torch::kBFloat16).repeat_interleave(group_size, 1); // [N, K]
    auto w_ref = (w_int - z_exp) * s_exp;                                                      // [N, K]

    // Packed Params
    auto packed_params = layer.get_packed_params();

    // Unpacked Params
    // Weights [N, K/2]
    auto w_nk = qweight.t().contiguous(); // [N, K]
    auto w_pairs = w_nk.view({N, K / 2, 2});
    auto w_low = w_pairs.select(2, 0);  // even cols
    auto w_high = w_pairs.select(2, 1); // odd cols
    auto qweights_unpacked = (w_low & 0x0F) | torch::bitwise_left_shift(w_high & 0x0F, 4);
    qweights_unpacked = qweights_unpacked.to(torch::kUInt8).contiguous();

    auto scales_unpacked = scales.t().contiguous(); // [N, Groups]
    // Zeros are now possibly Int8 for Unpacked Kernel (explicit to kInt8)
    auto zeros_unpacked = zeros.t().contiguous().to(torch::kInt8); // [N, Groups]

    // 3. Test GEMV M=1
    int64_t M_gemv = 1;
    std::cout << "\n=== Testing GEMV M=" << M_gemv << " ===" << std::endl;

    auto input_gemv = torch::rand({M_gemv, K}, torch::kBFloat16).to(device) * 0.1f;
    auto y_ref_gemv = torch::matmul(input_gemv, w_ref.t());

    auto output_gemv = torch::zeros({M_gemv, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));
    auto input_1d = input_gemv.view({-1});
    auto output_1d = output_gemv.view({-1});

    std::cout << "Running PACKED CPU Kernel..." << std::endl;
    // Basic Correctness Check (using 1 thread or default)
    w4a16_gemv_cpu_fused_packed(output_1d, input_1d, packed_params, K, N, nullptr, 1);

    bool match_packed = torch::allclose(y_ref_gemv, output_gemv, 0.01, 0.1);
    if (!match_packed) {
        std::cout << "FAILURE: PACKED GEMV Output mismatch!" << std::endl;
    } else {
        std::cout << "SUCCESS: PACKED GEMV Correct!" << std::endl;
    }

    // --- Thread Sweeping ---
    std::cout << "\n--- Thread Count Sweep ---" << std::endl;
    std::cout << "\n--- Thread Count Sweep (1 to 32) ---" << std::endl;
    // std::vector<int> thread_counts = {1, 2, 4, 8, 12, 16, 24, 32};
    int best_threads = 1;
    double max_gops = 0.0;

    for (int t = 1; t <= 32; ++t) {
        // Run a measurement (32 iterations)
        constexpr int NUM_ITERS = 32;
        auto start_t = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ITERS; ++i) {
            w4a16_gemv_cpu_fused_packed(output_1d, input_1d, packed_params, K, N, nullptr, t);
        }
        auto end_t = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_t = end_t - start_t;
        double avg_s = elapsed_t.count() / (double)NUM_ITERS;
        double avg_ms = avg_s * 1000.0;
        double gops = (2.0 * M_gemv * K * N) / (avg_s * 1e9);

        double total_bytes = (double)K * N * 0.5 + (double)K * 2 + (double)N * 2;
        double gbps = total_bytes / (avg_s * 1e9);

        std::cout << "Threads: " << std::setw(2) << t << " | Time: " << avg_ms << " ms | GOPS: " << gops << " | GB/s: " << gbps
                  << std::endl;

        if (gops > max_gops) {
            max_gops = gops;
            best_threads = t;
        }
    }
    std::cout << "--- Best Configuration: " << best_threads << " Threads (" << max_gops << " GOPS) ---\n" << std::endl;

    // Cool down
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Run Final Benchmark with Optimal Threads
    run_benchmark_packed(M_gemv, K, N, input_gemv, packed_params, "PACKED GEMV", best_threads);

    std::cout << std::endl;

    // Unpacked Test
    std::cout << "Running UNPACKED CPU Kernel..." << std::endl;
    auto output_unpacked = torch::zeros({M_gemv, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));

    w4a16_gemv_cpu_fused_unpacked(output_unpacked, input_gemv, qweights_unpacked, scales_unpacked, zeros_unpacked, K, N, group_size,
                                  nullptr, 1);

    bool match_unpacked = torch::allclose(y_ref_gemv, output_unpacked, 0.01, 0.1);
    if (match_unpacked) {
        std::cout << "SUCCESS: UNPACKED GEMV Output matches reference!" << std::endl;

        // --- UNPACKED Thread Sweeping ---
        std::cout << "\n--- UNPACKED Thread Count Sweep (1 to 32) ---" << std::endl;
        int best_threads_unpacked = 1;
        double max_gops_unpacked = 0.0;

        for (int t = 1; t <= 32; ++t) {
            // Run a measurement (32 iterations)
            constexpr int NUM_ITERS = 32;
            auto start_t = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < NUM_ITERS; ++i) {
                w4a16_gemv_cpu_fused_unpacked(output_unpacked, input_gemv, qweights_unpacked, scales_unpacked, zeros_unpacked, K, N,
                                              group_size, nullptr, t);
            }
            auto end_t = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_t = end_t - start_t;

            double avg_s = elapsed_t.count() / (double)NUM_ITERS;
            double avg_ms = avg_s * 1000.0;
            double gops = (2.0 * M_gemv * K * N) / (avg_s * 1e9);

            double total_bytes = (double)K * N * 0.5 + (double)K * 2 + (double)N * 2;
            double gbps = total_bytes / (avg_s * 1e9);

            std::cout << "Threads: " << std::setw(2) << t << " | Time: " << avg_ms << " ms | GOPS: " << gops << " | GB/s: " << gbps
                      << std::endl;

            if (gops > max_gops_unpacked) {
                max_gops_unpacked = gops;
                best_threads_unpacked = t;
            }
        }
        std::cout << "--- UNPACKED Best Configuration: " << best_threads_unpacked << " Threads (" << max_gops_unpacked << " GOPS) ---\n"
                  << std::endl;

        // Cool down
        std::this_thread::sleep_for(std::chrono::seconds(1));

        run_benchmark_unpacked(M_gemv, K, N, input_gemv, qweights_unpacked, scales_unpacked, zeros_unpacked, group_size, "UNPACKED GEMV",
                               best_threads_unpacked);

    } else {
        std::cout << "FAILURE: UNPACKED GEMV Output mismatch!" << std::endl;
        auto diff = (y_ref_gemv - output_unpacked).abs();
        auto max_val = diff.max();
        auto max_idx = diff.argmax();
        long flat_idx = max_idx.item<long>();
        std::cout << "Max Diff: " << max_val.item<float>() << " at [0, " << flat_idx << "]" << std::endl;
        std::cout << "Ref: " << y_ref_gemv[0][flat_idx].item<float>() << " Kernel: " << output_unpacked[0][flat_idx].item<float>()
                  << std::endl;
    }
    return 0;
}
