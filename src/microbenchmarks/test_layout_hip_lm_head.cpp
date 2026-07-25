#include "hipkernels/lm_head.hpp"
#include <chrono>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <torch/torch.h>
#include <vector>

// Helper check
#define HIP_CHECK(status)                                                                                                                  \
    if (status != hipSuccess) {                                                                                                            \
        std::cerr << "HIP Error: " << hipGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl;                     \
        exit(1);                                                                                                                           \
    }

int main(int argc, char **argv) {
    if (!torch::cuda::is_available()) {
        std::cerr << "HIP available check failed" << std::endl;
        return 1;
    }
    auto device = torch::kCUDA;

    // Llama 3 8B Dimensions
    int64_t K = 4096;      // Hidden Size
    int64_t V = 128256;    // Vocab Size
    int64_t Batch = 1;

    std::cout << "Testing LM Head Kernel (BF16)" << std::endl;
    std::cout << "Dimensions: Batch=" << Batch << ", Hidden(K)=" << K << ", Vocab(V)=" << V << std::endl;

    torch::manual_seed(42);

    // 1. Initialize Tensors
    // Weight [V, K] - simulating nn.Linear.weight
    auto weight = torch::randn({V, K}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));
    
    // Input [Batch, K]
    auto input = torch::randn({Batch, K}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));

    // Output tensors
    auto output_ref = torch::zeros({Batch, V}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto output_kernel = torch::zeros({Batch, V}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // 2. Reference Implementation (PyTorch MatMul / Linear)
    // LM Head is Linear(in_features=K, out_features=V, bias=False)
    // forward(x) = x @ weight.T
    // x: [1, K], weight: [V, K] -> weight.T: [K, V]
    // Result: [1, V]
    std::cout << "Computing Reference (PyTorch)..." << std::endl;
    output_ref = torch::matmul(input, weight.t()).to(torch::kFloat32);

    // 3. Custom Kernel Implementation
    std::cout << "Running Custom Kernel..." << std::endl;
    
    // warm up
    for(int i=0; i<5; ++i) {
        hipkernels::launch_lm_head_forward(
            output_kernel.data_ptr<float>(),
            (const hip_bfloat16*)input.data_ptr<at::BFloat16>(),
            (const hip_bfloat16*)weight.data_ptr<at::BFloat16>(),
            Batch,
            K,
            V,
            0 // or 0
        );
    }
    HIP_CHECK(hipDeviceSynchronize());

    // 4. Comparison
    auto output_kernel_cpu = output_kernel.cpu();
    auto output_ref_cpu = output_ref.cpu();

    bool match = torch::allclose(output_ref_cpu, output_kernel_cpu, 1e-2, 1e-2);
    
    std::cout << "First 8 values Ref:    ";
    for(int i=0; i<8; ++i) std::cout << std::fixed << std::setprecision(4) << output_ref_cpu[0][i].item<float>() << " ";
    std::cout << std::endl;

    std::cout << "First 8 values Kernel: ";
    for(int i=0; i<8; ++i) std::cout << std::fixed << std::setprecision(4) << output_kernel_cpu[0][i].item<float>() << " ";
    std::cout << std::endl;

    if (match) {
        std::cout << "SUCCESS: Outputs match!" << std::endl;
    } else {
        std::cout << "FAILURE: Outputs mismatch!" << std::endl;
        auto diff = (output_ref_cpu - output_kernel_cpu).abs();
        auto max_diff = diff.max().item<float>();
        std::cout << "Max Diff: " << max_diff << std::endl;
    }

    // 5. Benchmark
    std::cout << "\nBenchmarking (100 iterations)..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    for(int i=0; i<100; ++i) {
        hipkernels::launch_lm_head_forward(
            output_kernel.data_ptr<float>(),
            (const hip_bfloat16*)input.data_ptr<at::BFloat16>(),
            (const hip_bfloat16*)weight.data_ptr<at::BFloat16>(),
            Batch,
            K,
            V,
            0
        );
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    double avg_ms = (elapsed.count() * 1000.0) / 100.0;
    
    std::cout << "Average Latency: " << avg_ms << " ms" << std::endl;
    
    // Bandwidth Calc:
    // Read: Weight (V*K*2 bytes) + Input (K*2 bytes)
    // Write: Output (V*4 bytes) -> Actually output is float32
    double total_bytes = (double)V * K * 2.0 + (double)K * 2.0 + (double)V * 4.0;
    double gb_s = (total_bytes / 1e9) / (avg_ms / 1000.0);
    
    std::cout << "Effective Bandwidth: " << gb_s << " GB/s" << std::endl;

    return 0;
}
