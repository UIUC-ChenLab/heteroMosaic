#include <chrono>
#include <hip/hip_runtime.h>
#include <iostream>
#include <torch/torch.h>
#include <vector>

#define HIP_CHECK(status)                                                                                                                  \
    do {                                                                                                                                   \
        hipError_t error = (status);                                                                                                       \
        if (error != hipSuccess) {                                                                                                         \
            std::cerr << "HIP Error: " << hipGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl;                  \
            std::exit(EXIT_FAILURE);                                                                                                       \
        }                                                                                                                                  \
    } while (0)

// Llama 3 8B Config
const int64_t SEQ_LEN = 4096;
const int64_t NUM_HEADS = 32;
const int64_t HEAD_DIM = 128; // 4096 / 32
const int64_t BATCH_SIZE = 1;

void run_benchmark(torch::Device device, const std::string &device_name) {
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Running Benchmark on " << device_name << " (" << device << ")" << std::endl;
    std::cout << "Dimensions: [Batch=" << BATCH_SIZE << ", SeqLen=" << SEQ_LEN << ", Heads=" << NUM_HEADS << ", HeadDim=" << HEAD_DIM << "]"
              << std::endl;

    // Create tensors
    // Q: [Batch, Seq, Heads, Dim] -> Transposed for SDPA usually: [Batch, Heads, Seq, Dim]
    // The requested call signature implies inputs are already in the expected format for SDPA or matching the code context.
    // In unified_llm_w4a16.cpp:
    // q = q.transpose(1, 2); -> [bsz, heads, seq, dim]

    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

    auto q = torch::randn({BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM}, options);
    auto k = torch::randn({BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM}, options);
    auto v = torch::randn({BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM}, options);

    // Warmup
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < 5; ++i) {
        torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true, std::nullopt, false);
    }

    if (device.is_cuda() || device.is_hip()) {
        HIP_CHECK(hipDeviceSynchronize());
    }

    // Benchmark
    int num_iters = 50;
    std::cout << "Benchmarking " << num_iters << " iterations..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iters; ++i) {
        // "First prompt pass: use native causal optimization"
        // return torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true, std::nullopt, false);
        auto out = torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true, std::nullopt, false);
    }

    if (device.is_cuda() || device.is_hip()) {
        HIP_CHECK(hipDeviceSynchronize());
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    double total_ms = diff.count() * 1000.0;
    double avg_ms = total_ms / num_iters;

    std::cout << "Total Time: " << total_ms << " ms" << std::endl;
    std::cout << "Average Latency: " << avg_ms << " ms" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
}

int main() {
    // 1. CPU Benchmark
    if (torch::cuda::is_available()) {
        // To ensure fair comparison if possible, or just standard CPU
    }

    try {
        run_benchmark(torch::kCPU, "CPU");
    } catch (const std::exception &e) {
        std::cerr << "CPU Benchmark Failed: " << e.what() << std::endl;
    }

    // 2. GPU Benchmark
    if (torch::cuda::is_available()) {
        try {
            run_benchmark(torch::kCUDA, "GPU (ROCm/HIP)");
        } catch (const std::exception &e) {
            std::cerr << "GPU Benchmark Failed: " << e.what() << std::endl;
        }
    } else {
        std::cout << "No CUDA/ROCm device found, skipping GPU benchmark." << std::endl;
    }

    return 0;
}
