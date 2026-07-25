#include "unified_llm_w4a16_hetero/hetero_compute.hpp"
#include "unified_llm_w4a16_hetero/npuSetup.hpp"
#include "unified_llm_w4a16_hetero/unified_llm_w4a16.hpp"
#include <chrono>
#include <cmath>
#include <hip/hip_runtime.h> // Required for hipDeviceSynchronize
#include <iomanip>
#include <iostream>
#include <thread>
#include <torch/torch.h>
#include <vector>

// Defines to control which tests to run
#define TEST_PACKED
#define TEST_UNPACKED

// Declare global flag
extern bool use_packed_weights;

// Helper macro for HIP error checking
#ifndef HIP_CHECK
#define HIP_CHECK(error)                                                                                                                   \
    {                                                                                                                                      \
        hipError_t localError = error;                                                                                                     \
        if (localError != hipSuccess) {                                                                                                    \
            std::cerr << "HIP Error: " << hipGetErrorString(localError) << " at " << __FILE__ << ":" << __LINE__ << std::endl;             \
            exit(EXIT_FAILURE);                                                                                                            \
        }                                                                                                                                  \
    }
#endif

// Helper to confirm close match
bool check_close(const torch::Tensor &a, const torch::Tensor &b, float rtol = 0.05f, float atol = 3.0f) {
    if (a.sizes() != b.sizes()) {
        std::cerr << "Size mismatch: " << a.sizes() << " vs " << b.sizes() << std::endl;
        return false;
    }
    bool close = torch::allclose(a, b, rtol, atol);
    if (!close) {
        auto diff = (a - b).abs();
        float max_diff = diff.max().item<float>();
        float mean_diff = diff.mean().item<float>();
        auto max_idx = diff.argmax().item<long>();
        std::cerr << "Mismatch! Max Diff: " << max_diff << ", Mean Diff: " << mean_diff << std::endl;
        std::cerr << "At index " << max_idx << ": Ref " << b.flatten()[max_idx].item<float>() << " vs Out "
                  << a.flatten()[max_idx].item<float>() << std::endl;
    }
    return close;
}

void run_benchmark_packed(int M, int K, int N, const torch::Tensor &input, const torch::Tensor &packed_params,
                          const torch::Tensor &ref_output, int num_runs = 16) {
    std::cout << "\n=== Benchmarking Packed Hetero GEMV (M=" << M << ", K=" << K << ", N=" << N << ") ===" << std::endl;
    std::cout << std::left << std::setw(8) << "cpuN" << std::setw(8) << "threads" << std::setw(15) << "avg_time_ms" << std::setw(15)
              << "gops" << std::setw(15) << "tops" << std::setw(10) << "match" << std::endl;

    torch::Tensor output = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));

    double best_time_s = 1e9;
    int best_cpuN = -1;
    int best_threads = -1;

    for (int cpuN = 0; cpuN <= N; cpuN += 256) {
        for (int threads = 1; threads <= 24; ++threads) {
            double total_time_s = 0;
            bool match = false;

            for (int i = 0; i < num_runs; ++i) {
                try {
                    auto start = std::chrono::high_resolution_clock::now();
                    auto fut = hetero_matmul_out_gemv_packed_M(output, input, packed_params, K, N, 0, cpuN, threads, "bench_packed");
                    fut.wait();
                    // CRITICAL: Synchronize GPU to capture actual execution time
                    HIP_CHECK(hipDeviceSynchronize());
                    auto end = std::chrono::high_resolution_clock::now();

                    std::chrono::duration<double> elapsed = end - start;
                    total_time_s += elapsed.count();

                    if (i == num_runs - 1) {
                        match = check_close(output, ref_output, 0.01, 0.1);
                    }
                } catch (const std::exception &e) {
                    std::cerr << "Run failed: " << e.what() << std::endl;
                    total_time_s = -1;
                    match = false;
                    break;
                } catch (...) {
                    std::cerr << "Run failed: Unknown error" << std::endl;
                    total_time_s = -1;
                    break;
                }
            }

            if (total_time_s < 0)
                continue;

            double avg_time_s = total_time_s / num_runs;
            double gops = (2.0 * M * K * N) / (avg_time_s * 1e9);
            double tops = gops / 1000.0;

            std::cout << std::left << std::setw(8) << cpuN << std::setw(8) << threads << std::setw(15) << (avg_time_s * 1000.0)
                      << std::setw(15) << gops << std::setw(15) << tops << std::setw(10) << (match ? "PASS" : "FAIL") << std::endl;

            if (match && avg_time_s < best_time_s) {
                best_time_s = avg_time_s;
                best_cpuN = cpuN;
                best_threads = threads;
            }

            if (!match) {
                if (threads == 1) {
                    std::cout << "DEBUG Ref: " << ref_output.slice(0, 0, 1).slice(1, 0, 8) << std::endl;
                    std::cout << "DEBUG Out: " << output.slice(0, 0, 1).slice(1, 0, 8) << std::endl;
                }

                std::cerr << "Packed Benchmark Failed Correctness Check! Exiting." << std::endl;
                exit(1);
            }
        }
    }

    std::cout << "\nBest Packed Configuration: cpuN=" << best_cpuN << ", threads=" << best_threads << ", Time=" << (best_time_s * 1000.0)
              << " ms, GOPS=" << ((2.0 * M * K * N) / (best_time_s * 1e9)) << std::endl;
}

void run_benchmark_unpacked(int M, int K, int N, const torch::Tensor &input, const torch::Tensor &qweights, const torch::Tensor &scales,
                            const torch::Tensor &zeros, const torch::Tensor &ref_output, int num_runs = 16) {
    std::cout << "\n=== Benchmarking Unpacked Hetero GEMV (M=" << M << ", K=" << K << ", N=" << N << ") ===" << std::endl;
    std::cout << std::left << std::setw(8) << "cpuN" << std::setw(8) << "threads" << std::setw(15) << "avg_time_ms" << std::setw(15)
              << "gops" << std::setw(15) << "tops" << std::setw(10) << "match" << std::endl;

    torch::Tensor output = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));

    double best_time_s = 1e9;
    int best_cpuN = -1;
    int best_threads = -1;

    for (int cpuN = 0; cpuN <= N; cpuN += 256) {
        for (int threads = 1; threads <= 24; ++threads) {
            double total_time_s = 0;
            bool match = false;

            for (int i = 0; i < num_runs; ++i) {
                try {
                    auto start = std::chrono::high_resolution_clock::now();
                    auto fut =
                        hetero_matmul_out_gemv_unpacked(output, input, qweights, scales, zeros, K, N, cpuN, threads, "bench_unpacked");
                    fut.wait();
                    // CRITICAL: Synchronize GPU
                    HIP_CHECK(hipDeviceSynchronize());
                    auto end = std::chrono::high_resolution_clock::now();

                    std::chrono::duration<double> elapsed = end - start;
                    total_time_s += elapsed.count();

                    if (i == num_runs - 1) {
                        match = check_close(output, ref_output, 0.01, 0.1);
                    }
                } catch (const std::exception &e) {
                    std::cerr << "Run failed: " << e.what() << std::endl;
                    total_time_s = -1;
                    match = false;
                    break;
                } catch (...) {
                    std::cerr << "Run failed: Unknown error" << std::endl;
                    total_time_s = -1;
                    break;
                }
            }

            if (total_time_s < 0)
                continue;

            double avg_time_s = total_time_s / num_runs;
            double gops = (2.0 * M * K * N) / (avg_time_s * 1e9);
            double tops = gops / 1000.0;

            std::cout << std::left << std::setw(8) << cpuN << std::setw(8) << threads << std::setw(15) << (avg_time_s * 1000.0)
                      << std::setw(15) << gops << std::setw(15) << tops << std::setw(10) << (match ? "PASS" : "FAIL") << std::endl;

            if (match && avg_time_s < best_time_s) {
                best_time_s = avg_time_s;
                best_cpuN = cpuN;
                best_threads = threads;
            }

            if (!match) {
                if (threads == 1) {
                    std::cout << "DEBUG Ref: " << ref_output.slice(0, 0, 1).slice(1, 0, 8) << std::endl;
                    std::cout << "DEBUG Out: " << output.slice(0, 0, 1).slice(1, 0, 8) << std::endl;
                }

                std::cerr << "Unpacked Benchmark Failed Correctness Check! Exiting." << std::endl;
                exit(1);
            }
        }
    }

    std::cout << "\nBest Unpacked Configuration: cpuN=" << best_cpuN << ", threads=" << best_threads << ", Time=" << (best_time_s * 1000.0)
              << " ms, GOPS=" << ((2.0 * M * K * N) / (best_time_s * 1e9)) << std::endl;
}

int main(int argc, char **argv) {
    // Initialize NPU driver (required for import_dma_buf_to_xdna)
    if (initialize_xdna_driver() < 0) {
        std::cerr << "Warning: XDNA driver initialization failed. XDNA features will be disabled." << std::endl;
    }

    torch::manual_seed(42);
    // Use device GPU for inputs (as requested by User)
    // CPU slices will be handled by data movement in hetero_compute.cpp
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA/HIP not available!" << std::endl;
        return 1;
    }

    // Initialize Resources AFTER PyTorch (to avoid HIP context conflicts)
    init_gemm_resources();

    auto device = torch::kCUDA;

    int64_t M = 1;
    int64_t K = 4096;
    int64_t N = 14336; // Llama 3 MLP up/gate size

    std::cout << "Initializing Tensors (M=" << M << ", K=" << K << ", N=" << N << ")..." << std::endl;

    auto input = torch::rand({M, K}, torch::kBFloat16).to(device) * 0.1f;

    int64_t group_size = 128;
    int64_t num_groups = K / group_size;

    auto qweight_raw = torch::randint(0, 16, {K, N}, torch::kUInt8).to(device);
    std::cout << "Packing weights..." << std::endl;

    auto scales = (torch::rand({num_groups, N}, torch::kFloat32).to(torch::kBFloat16) * 0.1f).to(device);
    auto zeros = torch::randint(0, 16, {num_groups, N}, torch::kInt8).to(device);

    // --- Compute Reference using built-in dequantization ---
    std::cout << "Generating Ground Truth..." << std::endl;

    // 1. Force Unpacked logic to populate quantized_weight_, scale_, zero_point_
    use_packed_weights = false;

    // We create a temporary layer just to dequantize for ground truth
    QuantizedLinearImpl layer_ref(K, N, false);
    layer_ref.to(device);
    layer_ref.set_quantized_weights(qweight_raw, scales, zeros, torch::Tensor());

    // 2. Dequantize to get exact reference weights (bf16)
    auto w_dequant = layer_ref.dequantize_weights();
    std::cout << "Dequantized Weight Shape: " << w_dequant.sizes() << std::endl;

    // 3. Compute Reference (Input @ W.T)
    // Use Float32 for accumulation to match Kernel precision (Kernel uses FP32 acc)
    auto input_fp32 = input.to(torch::kFloat32);
    auto w_dequant_fp32 = w_dequant.to(torch::kFloat32);
    auto ref_output_fp32 = torch::matmul(input_fp32, w_dequant_fp32.t());
    auto ref_output = ref_output_fp32.to(torch::kBFloat16);

    // Transform weights for Unpacked Path (Manually for custom kernel inputs)
    auto w_nk = qweight_raw.t().contiguous(); // [N, K]
    auto w_pairs = w_nk.view({N, K / 2, 2});
    auto w_low = w_pairs.select(2, 0);
    auto w_high = w_pairs.select(2, 1);
    auto qweights_unpacked = (w_low & 0x0F) | torch::bitwise_left_shift(w_high & 0x0F, 4);
    qweights_unpacked = qweights_unpacked.to(torch::kUInt8).contiguous();

    auto scales_unpacked = scales.t().contiguous();
    // Unpacked Kernel now expects Int8 zeros (matching the hetero backend logic)
    auto zeros_unpacked = zeros.to(torch::kInt8).t().contiguous();

    // 1. Packed Benchmark (Needs Packed Params)
    std::cout << "Preparing Packed Benchmark..." << std::endl;
    use_packed_weights = true;

    // Re-set weights to populate packed_params_
    QuantizedLinearImpl layer_packed(K, N, false);
    layer_packed.to(device);
    layer_packed.set_quantized_weights(qweight_raw, scales, zeros, torch::Tensor());
    layer_packed.import_weights_to_xdna();

    {
#ifdef TEST_PACKED
        if (layer_packed.get_packed_params().defined()) {
            std::cout << "Packed Params Device: " << layer_packed.get_packed_params().device() << std::endl;
            // Explicitly move to device just in case
            auto packed_gpu = layer_packed.get_packed_params().to(device);
            // auto packed_cpu = layer_packed.get_packed_params_cpu(); // Use cached CPU weights
            run_benchmark_packed(M, K, N, input, packed_gpu, ref_output, 1);
        } else {
            std::cout << "Skipping Packed Benchmark: packed_params_ not defined" << std::endl;
        }
#endif
    }

    // 2. Unpacked Benchmark
    std::cout << "Preparing Unpacked Benchmark..." << std::endl;
    use_packed_weights = false;

    // Explicitly set weights for Unpacked mode as requested
    QuantizedLinearImpl layer_unpacked(K, N, false);
    layer_unpacked.to(device);
    std::cout << "Setting quantized weights for unpacked layer..." << std::endl;
    layer_unpacked.set_quantized_weights(qweight_raw, scales, zeros, torch::Tensor());
    std::cout << "Weights set. Accessing buffers..." << std::endl;

    auto q_w = layer_unpacked.get_quantized_weights();
    auto s = layer_unpacked.get_scales();
    auto z = layer_unpacked.get_zeros();

    std::cout << "Buffers accessed. q_w defined: " << q_w.defined() << std::endl;

#ifdef TEST_UNPACKED
    run_benchmark_unpacked(M, K, N, input, q_w, s, z, ref_output, 1);
#endif

    return 0;
}
