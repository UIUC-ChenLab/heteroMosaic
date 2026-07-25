#pragma once

#include "third_party/BSpool/BS_thread_pool.hpp"
#include <future>
#include <hip/hip_runtime.h>
#include <torch/torch.h>

// Global thread pool shared across modules
extern BS::thread_pool<> g_thread_pool;

// Custom GEMM wrapper that calls torch::matmul_out
// This allows us to add custom logic before/after the matmul operation
// in_features/out_features: explicit dimensions, useful when weights are packed (opaque shape)
// Hetero Path (M > 1): Hybrid GPU + NPU
// Hetero Path (M > 1): Hybrid GPU + NPU
std::future<int> hetero_matmul_out_gemm_packed_M(torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &weights,
                                                 int64_t in_features = -1, int64_t out_features = -1, std::string layer_type = "",
                                                 int chunk_id = -1);

// Function for GEMM (packed weights) - K Split
std::future<int> hetero_matmul_out_gemm_packed_K(torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &weights0,
                                                 const torch::Tensor &weights1, int64_t in_features = -1, int64_t out_features = -1,
                                                 int64_t split_k = -1, std::string layer_type = "", int chunk_id = -1,
                                                 bool force_gpu_only = false);

// Custom GEMV wrapper (M = 1) - Packed
std::future<int> hetero_matmul_out_gemv_packed_M(torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &weights,
                                                 int64_t in_features, int64_t out_features, std::string layer_type, int chunk_id = -1);

// Custom GEMV wrapper (M = 1) - Packed - K Split
std::future<int> hetero_matmul_out_gemv_packed_K(torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &weights0,
                                                 const torch::Tensor &weights1, int64_t in_features, int64_t out_features,
                                                 int64_t split_k, std::string layer_type, int chunk_id = -1,
                                                 bool force_gpu_only = false);

// Custom GEMV wrapper (M = 1) - Packed - Explicit Config
std::future<int> hetero_matmul_out_gemv_packed_M(torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &weights,
                                                 int64_t in_features, int64_t out_features, int npuN, int cpuN, int cpuThreads,
                                                 std::string layer_type, int chunk_id = -1);

// Custom GEMV wrapper (M = 1) - Unpacked
// Custom GEMV wrapper (M = 1) - Unpacked
std::future<int> hetero_matmul_out_gemv_unpacked_M(torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &qweights,
                                                   const torch::Tensor &scales, const torch::Tensor &zeros, int64_t in_features,
                                                   int64_t out_features, std::string layer_type, int chunk_id = -1);

// Custom GEMV wrapper (M = 1) - Unpacked - Explicit Config
std::future<int> hetero_matmul_out_gemv_unpacked(torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &qweights,
                                                 const torch::Tensor &scales, const torch::Tensor &zeros, int64_t in_features,
                                                 int64_t out_features, int cpuN, int cpuThreads, std::string layer_type,
                                                 hipEvent_t hip_event = nullptr, int chunk_id = -1);

// NPU Strict Path (M > 1): Pure NPU
std::future<int> npu_top_matmul_out_gemm_packed(torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &weights,
                                                int64_t in_features = -1, int64_t out_features = -1, std::string layer_type = "",
                                                int chunk_id = -1);

// NPU Strict Path (M = 1): Single token (CPU AVX fallback)
std::future<int> npu_top_matmul_out_gemv_packed(torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &weights,
                                                int64_t in_features = -1, int64_t out_features = -1, std::string layer_type = "",
                                                int chunk_id = -1);

// Initialize GEMM resources (HIP events, etc.)
void init_gemm_resources();
