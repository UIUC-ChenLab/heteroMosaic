#include "unified_llm_w4a16_hetero/hetero_compute.hpp"
#include "hipkernels/w4a16_gemm_packed.hpp"
#include "hipkernels/w4a16_gemv_unpacked.hpp"
#include "npu_matmul_func/npu_util.hpp"
#include "unified_llm_w4a16_hetero/hipblasltSetup.hpp"
#include "unified_llm_w4a16_hetero/npuSetup.hpp"
#include "unified_llm_w4a16_hetero/unified_llm_w4a16.hpp"
#include <ATen/Parallel.h>
#include <algorithm>
#include <atomic>
#include <c10/hip/HIPStream.h>
#include <cmath>
#include <cstdlib>
#include <future>
#include <hip/hip_runtime.h>
#include <iostream>
#include <mutex>
#include <torch/torch.h>
#include <unistd.h>
#include <vector>

#ifdef __AVX512F__
#include <immintrin.h>
#endif
#include <omp.h>

#include "cpu_avx_kernels/w4a16_gemv_avx_packed.hpp"
#include "cpu_avx_kernels/w4a16_gemv_avx_unpacked.hpp"

// Global Thread Pool
BS::thread_pool<> g_thread_pool;

// HIP Event Pool
#define EVENT_POOL_SIZE 16
std::vector<hipEvent_t> event_pool_storage; // Holds all created events for cleanup
std::vector<hipEvent_t> free_events;        // Stack of available events
std::mutex pool_mutex;
bool event_pool_initialized = false;

namespace {
int64_t round_up_packed_split_dim(const std::string &layer_type, int64_t dim) {
    if (pad_packed_weights <= 0) {
        return dim;
    }
    auto pad_spec = get_packed_weight_pad_alignment(layer_type);
    if (pad_spec[0] <= 0) {
        return dim;
    }
    return ((dim + pad_spec[0] - 1) / pad_spec[0]) * pad_spec[0];
}
} // namespace

void init_gemm_resources() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    if (event_pool_initialized)
        return;

    event_pool_storage.reserve(EVENT_POOL_SIZE);
    free_events.reserve(EVENT_POOL_SIZE);

    for (int i = 0; i < EVENT_POOL_SIZE; ++i) {
        hipEvent_t evt;
        HIP_CHECK(hipEventCreate(&evt));
        event_pool_storage.push_back(evt);
        free_events.push_back(evt);
    }

    event_pool_initialized = true;
    if (debug_verbosity >= 2) {
        std::cout << "Initialized GEMM resources (HIP event pool size: " << EVENT_POOL_SIZE << ")" << std::endl;
    }
}

hipEvent_t acquire_event() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    if (!event_pool_initialized) {
        // Fallback or error if not initialized
    }

    if (free_events.empty()) {
        if (debug_verbosity >= 1)
            std::cerr << "Warning: HIP event pool exhausted! Creating temporary event." << std::endl;
        hipEvent_t evt;
        HIP_CHECK(hipEventCreate(&evt));
        return evt;
    }

    hipEvent_t evt = free_events.back();
    free_events.pop_back();
    return evt;
}

void release_event(hipEvent_t evt) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    free_events.push_back(evt);
}

inline std::map<NPUKey, NPUValue>::const_iterator find_config_entry(int M, int K, int N, int layer_id, int chunk_id) {
    return config_map.find({M, K, N, layer_id, chunk_id});

    // auto exact_it = config_map.find({M, K, N, layer_id, chunk_id});
    // if (exact_it != config_map.end()) {
    //     return exact_it;
    // }
    // return config_map.find({M, K, N, layer_id, -1});
}

// --- PACKED IMPLEMENTATIONS ---

// 1. GEMM (M-Split)
std::future<int> hetero_matmul_out_gemm_packed_M(torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &weights,
                                                 int64_t in_features, int64_t out_features, std::string layer_type, int chunk_id) {
    hipStream_t call_stream = c10::hip::getCurrentHIPStream().stream();
    // Ensure GPU has finished writing to shared buffers (x, output) before NPU reads/writes
    hipEvent_t hip_event = acquire_event();
    HIP_CHECK(hipEventRecord(hip_event, call_stream));

    // Handle batch dimension: 2D (seq_len, hidden) or 3D (batch, seq_len, hidden)
    int M, K;
    if (x.dim() == 2) {
        M = x.size(0);
        K = x.size(1);
    } else if (x.dim() == 3) {
        M = x.size(1); // seq_len
        K = x.size(2); // hidden_dim
    } else {
        // Fallback to GPU-only for unexpected dimensions
        torch::matmul_out(output, x, weights);
        return std::future<int>();
    }

    // Always assume Row Major [K, N]
    int N = (out_features != -1) ? out_features : weights.size(1);

    void *x_ptr = x.data_ptr();
    void *w_ptr = weights.data_ptr();
    void *out_ptr = output.data_ptr();

    // Check if handles exist in cache
    auto it_x = ptr_to_handle_map.find(x_ptr);
    auto it_w = ptr_to_handle_map.find(w_ptr);
    auto it_out = ptr_to_handle_map.find(out_ptr);

    bool handles_exist = (it_x != ptr_to_handle_map.end()) && (it_w != ptr_to_handle_map.end()) && (it_out != ptr_to_handle_map.end());

    // Check if configuration exists (exact chunk_id match)
    int layer_id = get_layer_id(layer_type);
    auto cfg_it = find_config_entry(M, K, N, layer_id, chunk_id);
    bool config_exists = (cfg_it != config_map.end());

    if (debug_verbosity >= 2) {
        std::cout << "[" << "M=" << M << " K=" << K << " N=" << N << "]" << " weight_layout=" << "Packed" << " layer_type=" << layer_type
                  << " chunk_id=" << chunk_id << std::endl;
        std::cout << "Handles exist: " << handles_exist << " (x=" << (it_x != ptr_to_handle_map.end())
                  << ", w=" << (it_w != ptr_to_handle_map.end()) << ", out=" << (it_out != ptr_to_handle_map.end()) << ")" << std::endl;

        if (it_x == ptr_to_handle_map.end()) {
            std::cout << "  x_ptr: " << x_ptr << " not found in map (size: " << ptr_to_handle_map.size() << ")" << std::endl;
            // Print first few keys in map to check
            int count = 0;
            for (const auto &pair : ptr_to_handle_map) {
                if (count++ < 5)
                    std::cout << "    Map key: " << pair.first << std::endl;
            }
        }
        std::cout << "Config exists: " << config_exists << std::endl;
        if (config_exists && debug_verbosity >= 3) {
            const auto &val = cfg_it->second;
            std::cout << "  Config value: hw_idx=" << val.hw_idx << " inst_idx=" << val.inst_idx << " npuM=" << val.npuM
                      << " npuK=" << val.npuK << " npuN=" << val.npuN << " config=" << val.config << std::endl;
        }
        if (!config_exists && debug_verbosity >= 3) {
            std::cout << "Available configs:" << std::endl;
            for (const auto &kv : config_map) {
                std::cout << "  {" << kv.first[0] << ", " << kv.first[1] << ", " << kv.first[2] << ", " << kv.first[3] << ", "
                          << kv.first[4] << "}" << std::endl;
            }
        }
    }

    if (hw_target == "hetero" && config_exists && handles_exist) {
        if (debug_verbosity >= 2) {
            std::cout << "Hetero GPU + NPU Path" << std::endl;
        }

        // Found NPU config
        NPUValue val = cfg_it->second;

        // Calculate split
        int npu_M = val.npuM; // The M supported by the NPU kernel (e.g. 512)
        int gpuM = M - npu_M;
        if (debug_verbosity >= 2) {
            std::cout << "Split: gpuM=" << gpuM << " npuM=" << npu_M << std::endl;
        }

        // Check if we need to split
        if (npu_M > M) {
            npu_M = M;
        }

        if (gpuM < 0) {
            gpuM = 0;
            npu_M = M;
        }

        // Slice tensors
        // GPU takes [0, gpuM)
        // NPU takes [gpuM, M)
        if (gpuM > 0) {
            // GPU part
            auto x_gpu = x.slice(x.dim() == 3 ? 1 : 0, 0, gpuM);
            auto out_gpu = output.slice(x.dim() == 3 ? 1 : 0, 0, gpuM);

            // Use optimized fused kernel (vectorized + register blocked)
            auto input_2d = x_gpu.view({-1, in_features});
            auto output_2d = out_gpu.view({-1, out_features});

            if (debug_verbosity >= 2) {
                std::cout << "DEBUG: Calling GEMM (Split GPU part)" << std::endl;
            }
            hipkernels::w4a16_gemm_fused_packed(output_2d, input_2d, weights, in_features, out_features, call_stream);
        }

        if (npu_M <= 0) {
            if (debug_verbosity >= 2) {
                std::cout << "Config requests npuM=0; skipping NPU launch for this GEMM." << std::endl;
            }
            release_event(hip_event);
            return std::future<int>();
        }

        // NPU part: Prepare pointers (bf16)
        uint16_t *x_npu = (uint16_t *)x_ptr + gpuM * K;
        uint16_t *w_npu = (uint16_t *)w_ptr;
        uint16_t *out_npu = (uint16_t *)out_ptr + gpuM * N;

        // Use cached handles
        uint32_t x_handle = ptr_to_handle_map[x_ptr];
        uint32_t out_handle = ptr_to_handle_map[out_ptr];
        uint32_t w_handle = ptr_to_handle_map[w_ptr];

        // Async launch
        std::future<int> fut = std::async(std::launch::async, [val, out_npu, x_npu, w_npu, out_handle, x_handle, w_handle, hip_event]() {
            // Execute NPU kernel (handles synchronization internally)
            // Cast pointers to void* for npuMatmul_zero
            int ret = npuMatmul_zero(val.hw_idx, val.inst_idx, (void *)out_npu, (void *)x_npu, (void *)w_npu, out_handle, x_handle,
                                     w_handle, hip_event);

            // Release event back to pool
            release_event(hip_event);

            return ret;
        });

        if (debug_verbosity >= 2) {
            std::cout << "Launched NPU async" << std::endl;
        }

        return fut;
    }

    // GPU default Path
    {
        if (debug_verbosity >= 2) {
            std::cout << "Hetero GPU No Config Path (Optimized Fused Kernel)" << std::endl;
        }
        // Use optimized fused kernel (vectorized + register blocked)
        auto input_2d = x.view({-1, in_features});
        auto output_2d = output.view({-1, out_features});

        if (debug_verbosity >= 2) {
            std::cout << "DEBUG: Calling GEMM" << std::endl;
        }
        hipkernels::w4a16_gemm_fused_packed(output_2d, input_2d, weights, in_features, out_features, call_stream);

        release_event(hip_event);
        return std::future<int>();
    }
}

// 2. GEMM (K-Split)
std::future<int> hetero_matmul_out_gemm_packed_K(torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &weights0,
                                                 const torch::Tensor &weights1, int64_t in_features, int64_t out_features,
                                                 int64_t split_k, std::string layer_type, int chunk_id, bool force_gpu_only) {
    hipStream_t call_stream = c10::hip::getCurrentHIPStream().stream();
    hipEvent_t hip_event = acquire_event();

    if (debug_verbosity >= 2)
        std::cout << "K-Split GEMM Path - Layer: " << layer_type << " Split K=" << split_k << std::endl;

    int M;
    if (x.dim() == 2) {
        M = x.size(0);
    } else if (x.dim() == 3) {
        M = x.size(1);
    } else {
        return std::future<int>();
    }

    int64_t K = in_features;
    int64_t K0 = split_k;
    int64_t K1 = K - K0;
    int64_t N = out_features;

    int64_t K0_internal = K0;
    int64_t K1_internal = K1;

    if (pad_packed_weights) {
        K0_internal = round_up_packed_split_dim(layer_type, K0);
        K1_internal = round_up_packed_split_dim(layer_type, K1);
    }

    auto input_2d = x.view({-1, x.size(-1)});
    auto output_2d = output.view({-1, N});

    int64_t M_padded = (M + 127) / 128 * 128;
    auto split_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(x.device());
    auto input_buf_0 = torch::empty({M_padded, K0}, split_opts);
    auto input_buf_1 = torch::empty({M_padded, K1}, split_opts);
    auto temp_output = torch::empty({M_padded, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(x.device()));

    input_buf_0.slice(0, 0, M).copy_(input_2d.slice(1, 0, K0));
    HIP_CHECK(hipEventRecord(hip_event, call_stream)); // wait until npu input buf is valid
    input_buf_1.slice(0, 0, M).copy_(input_2d.slice(1, K0, K));
    // Launch K1 contribution as early as possible to maximize overlap with
    // config lookup, NPU K0 launch, and K0 GPU fallback path.
    auto temp_output_2d = temp_output.slice(0, 0, M);
    hipkernels::w4a16_gemm_fused_packed(temp_output_2d, input_buf_1.slice(0, 0, M), weights1, K1_internal, N, call_stream);

    bool npu_launched = false;
    std::future<int> npu_fut;

    int npu_rows = 0;
    int gpu_rows_k0 = M;

    int layer_id = get_layer_id(layer_type);
    auto cfg_it = find_config_entry(M, static_cast<int>(K), static_cast<int>(N), layer_id, chunk_id);

    void *x_full_ptr = input_2d.data_ptr();
    void *x_ptr = x_full_ptr;
    void *w_ptr = weights0.data_ptr();
    void *out_ptr = output.data_ptr();

    // Iterator lookups
    auto it_x = ptr_to_handle_map.find(x_ptr);
    auto it_w = ptr_to_handle_map.find(w_ptr);
    auto it_out = ptr_to_handle_map.find(out_ptr);

    bool handles_exist = (it_x != ptr_to_handle_map.end()) && (it_w != ptr_to_handle_map.end()) && (it_out != ptr_to_handle_map.end());
    bool config_exists = (cfg_it != config_map.end());

    if (debug_verbosity >= 2) {
        std::cout << "[" << "M=" << M << " K=" << K << " N=" << N << "]" << " weight_layout=" << "Packed" << " layer_type=" << layer_type
                  << std::endl;
        std::cout << "Handles exist: " << handles_exist << " (x=" << (it_x != ptr_to_handle_map.end())
                  << ", w=" << (it_w != ptr_to_handle_map.end()) << ", out=" << (it_out != ptr_to_handle_map.end()) << ")" << std::endl;

        if (it_x == ptr_to_handle_map.end()) {
            std::cout << "  x_ptr: " << x_ptr << " not found in map (size: " << ptr_to_handle_map.size() << ")" << std::endl;
            // Print first few keys in map to check
            int count = 0;
            for (const auto &pair : ptr_to_handle_map) {
                if (count++ < 5)
                    std::cout << "    Map key: " << pair.first << std::endl;
            }
        }
        std::cout << "Config exists: " << config_exists << std::endl;
        if (config_exists && debug_verbosity >= 3) {
            const auto &val = cfg_it->second;
            std::cout << "  Config value: hw_idx=" << val.hw_idx << " inst_idx=" << val.inst_idx << " npuM=" << val.npuM
                      << " npuK=" << val.npuK << " npuN=" << val.npuN << " config=" << val.config << std::endl;
        }
        if (!config_exists && debug_verbosity >= 3) {
            std::cout << "Available configs:" << std::endl;
            for (const auto &kv : config_map) {
                std::cout << "  {" << kv.first[0] << ", " << kv.first[1] << ", " << kv.first[2] << ", " << kv.first[3] << ", "
                          << kv.first[4] << "}" << std::endl;
            }
        }
    }

    if (!force_gpu_only && hw_target == "hetero" && config_exists && handles_exist) {
        NPUValue val = cfg_it->second;

        // Split M for K0 if needed (e.g. M=4096, npuM=2048)
        npu_rows = val.npuM;
        if (npu_rows > M)
            npu_rows = M;
        if (npu_rows < 0)
            npu_rows = 0;
        gpu_rows_k0 = M - npu_rows;

        if (npu_rows > 0) {
            // K-split NPU kernels may accumulate into output instead of hard-overwriting.
            // Clear the destination rows before launch so the NPU contribution is deterministic.
            output_2d.slice(0, gpu_rows_k0, M).zero_();

            const bool using_full_x = (x_ptr == x_full_ptr);
            const int64_t npu_x_row_stride = using_full_x ? K : K0;
            uint16_t *x_npu = (uint16_t *)x_ptr + static_cast<size_t>(gpu_rows_k0) * static_cast<size_t>(npu_x_row_stride);
            uint16_t *w_npu = (uint16_t *)w_ptr;
            uint16_t *out_npu = (uint16_t *)out_ptr + static_cast<size_t>(gpu_rows_k0) * static_cast<size_t>(N);

            uint32_t x_handle = ptr_to_handle_map[x_ptr];
            uint32_t w_handle = ptr_to_handle_map[w_ptr];
            uint32_t out_handle = ptr_to_handle_map[out_ptr];

            npu_launched = true;
            npu_fut = std::async(std::launch::async, [val, out_npu, x_npu, w_npu, out_handle, x_handle, w_handle, hip_event]() {
                int ret = npuMatmul_zero(val.hw_idx, val.inst_idx, (void *)out_npu, (void *)x_npu, (void *)w_npu, out_handle, x_handle,
                                         w_handle, hip_event);
                release_event(hip_event);
                return ret;
            });

            if (debug_verbosity >= 2)
                std::cout << "Launched K-Split NPU path (K=" << K0 << ", M[" << gpu_rows_k0 << ".." << M
                          << "], x_stride=" << npu_x_row_stride << ")" << std::endl;
        } else if (debug_verbosity >= 2) {
            std::cout << "K-Split config requests npuM=0; skipping NPU launch for this chunk." << std::endl;
        }
    }

    if (npu_launched) {
        if (gpu_rows_k0 > 0) {
            if (debug_verbosity >= 2)
                std::cout << "Split NPU portion M and K wise" << std::endl;
            auto out_slice = output_2d.slice(0, 0, gpu_rows_k0);
            auto in_slice = input_buf_0.slice(0, 0, gpu_rows_k0);
            hipkernels::w4a16_gemm_fused_packed(out_slice, in_slice, weights0, K0_internal, N, call_stream);
        }
    } else {
        if (debug_verbosity >= 2)
            std::cout << (force_gpu_only ? "Split K GPU-Only Path" : "Split K GPU Fallback of NPU Portion") << std::endl;
        hipkernels::w4a16_gemm_fused_packed(output_2d, input_buf_0.slice(0, 0, M), weights0, K0_internal, N, call_stream);
    }

    if (npu_launched) {
        npu_fut.get();
    } else {
        release_event(hip_event);
    }

    output_2d.add_(temp_output_2d);
    return std::future<int>();
}

// 3. GEMV (M-Split)
std::future<int> hetero_matmul_out_gemv_packed_M(torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &weights,
                                                 int64_t in_features, int64_t out_features, std::string layer_type, int chunk_id) {

    // Lookup NPU config for M=1 GEMV
    int layer_id = get_layer_id(layer_type);
    auto cfg_it = find_config_entry(1, static_cast<int>(in_features), static_cast<int>(out_features), layer_id, chunk_id);

    int cpuN = 0;
    int npuN = 0;
    int cpuThreads = 1;

    // CPU path
    if (cpu_decode) {
        if (debug_verbosity >= 2) {
            std::cout << "DEBUG: CPU Threads" << std::endl;
        }

        // Pure CPU Path - Execute entirely on CPU (No GPU/NPU)
        hipStream_t call_stream = c10::hip::getCurrentHIPStream().stream();
        hipEvent_t hip_event = acquire_event();
        HIP_CHECK(hipEventRecord(hip_event, call_stream));

        auto input_2d = x.view({-1, in_features});
        auto output_2d = output.view({-1, out_features});

        return std::async(std::launch::async, [=]() mutable {
            w4a16_gemv_cpu_fused_packed(output_2d, input_2d, weights, in_features, out_features, hip_event, cpuThreads, 0, out_features);
            release_event(hip_event);
            return 0;
        });
    }

    // Check global config map
    if (cfg_it != config_map.end()) {
        const NPUValue &val = cfg_it->second;
        npuN = val.npuN;
        cpuN = val.cpuN;
        cpuThreads = val.cpuThreads > 0 ? val.cpuThreads : 1;
    } else {
        if (debug_verbosity >= 2) {
            std::cout << "DEBUG: GEMV Config not found for 1x" << in_features << "x" << out_features << " "
                      << layer_type
                      /*<< " - using GPU fallback direct" */
                      << std::endl;
        }
        // hipkernels::w4a16_gemv_fused_packed(output, x, weights, in_features, out_features, -1, 0, call_stream);
        // return std::future<int>();
    }

    return hetero_matmul_out_gemv_packed_M(output, x, weights, in_features, out_features, npuN, cpuN, cpuThreads, layer_type, chunk_id);
}

// 3b. GEMV (M-Split) Explicit Config
inline std::future<int> hetero_matmul_out_gemv_packed_M(torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &weights,
                                                        int64_t in_features, int64_t out_features, int npuN, int cpuN, int cpuThreads,
                                                        std::string layer_type, int chunk_id) {
    hipStream_t call_stream = c10::hip::getCurrentHIPStream().stream();

    // Calculate GPU remainder
    int gpuN = out_features - npuN - cpuN;
    if (gpuN < 0) {
        std::cerr << "NPU+CPU N (" << npuN + cpuN << ") exceeds total N (" << out_features << ")" << std::endl;
        npuN = 0;
        cpuN = 0;
        gpuN = out_features;
    }

    // View inputs for raw access
    auto input_2d = x.view({-1, in_features});
    auto output_2d = output.view({-1, out_features});

    // Ensure GPU has finished writing to shared buffers (x, output)
    hipEvent_t hip_event = acquire_event();
    HIP_CHECK(hipEventRecord(hip_event, call_stream));

    // Lauch GPU thread as fast as possible
    if (gpuN > 0) {
        int start_gpu = npuN + cpuN;
        if (debug_verbosity >= 2)
            std::cout << "GEMV: Launching GPU in Child Function (" << start_gpu << ".." << out_features << ")" << std::endl;

        auto output_slice = output_2d.slice(1, start_gpu, out_features);
        hipkernels::w4a16_gemv_fused_packed(output_slice, input_2d, weights, in_features, gpuN, out_features, start_gpu, call_stream);
    }

    // Default CPU logic constraint (multiples of 64 columns for AVX/layout)
    if (cpuN > 0 && cpuN % 64 != 0) {
        if (debug_verbosity >= 1)
            std::cerr << "Warning: cpuN (" << cpuN << ") is not a multiple of 64. Disabling CPU part." << std::endl;
        cpuN = 0;
    }

    // Unified Host Compute (NPU + CPU)
    // Optimization: Pure CPU internal path (avoid NPU logic overhead)
    if (cpuN > 0 && npuN == 0) {
        return g_thread_pool.submit_task([output_2d, input_2d, weights, in_features, out_features, hip_event, cpuThreads, cpuN]() mutable {
            if (cpuN > 0) {
                if (debug_verbosity >= 2)
                    std::cout << "GEMV: Launching CPU (" << 0 << ".." << cpuN << ") [Sync]" << std::endl;

                torch::Tensor weights_cpu_kernel = weights;
                w4a16_gemv_cpu_fused_packed(output_2d, input_2d, weights_cpu_kernel, in_features, out_features, hip_event, cpuThreads, 0,
                                            cpuN);
            }
            release_event(hip_event);
            return 0;
        });
    }

    // Full Hybrid Path (NPU + CPU) behavior
    return g_thread_pool.submit_task([=]() mutable {
        std::future<int> npu_fut;
        int ret = 0;

        // 1. Launch NPU (Async parallel to CPU)
        if (npuN > 0) {
            void *x_ptr = x.data_ptr();
            void *w_ptr = weights.data_ptr();
            void *out_ptr = output.data_ptr();

            int layer_id = get_layer_id(layer_type);

            // We need to access global maps. Capture by value [=] captures 'this' if member, but these are global.
            // Global variables (config_map, ptr_to_handle_map) are accessible.
            // However, concurrent access to maps? They are read-only here (config_map) or assumed thread-safe/stable (handles).

            auto cfg_it = find_config_entry(1, static_cast<int>(in_features), static_cast<int>(out_features), layer_id, chunk_id);
            bool config_exists = (cfg_it != config_map.end());
            bool handles_exist = (ptr_to_handle_map.count(x_ptr) && ptr_to_handle_map.count(w_ptr) && ptr_to_handle_map.count(out_ptr));

            if (config_exists && handles_exist) {
                if (debug_verbosity >= 2)
                    std::cout << "GEMV: Launching NPU (0.." << npuN << ") [Async] HW: " << cfg_it->second.hw_idx
                              << " Inst: " << cfg_it->second.inst_idx << std::endl;

                NPUValue val = cfg_it->second;

                uint16_t *x_npu = (uint16_t *)x_ptr;
                uint16_t *w_npu = (uint16_t *)w_ptr;
                uint16_t *out_npu = (uint16_t *)out_ptr;

                uint32_t x_handle = ptr_to_handle_map[x_ptr];
                uint32_t w_handle = ptr_to_handle_map[w_ptr];
                uint32_t out_handle = ptr_to_handle_map[out_ptr];

                // Spawn NPU thread
                npu_fut = std::async(std::launch::async, [val, out_npu, x_npu, w_npu, out_handle, x_handle, w_handle, hip_event]() {
                    return npuMatmul_zero(val.hw_idx, val.inst_idx, (void *)out_npu, (void *)x_npu, (void *)w_npu, out_handle, x_handle,
                                          w_handle, hip_event);
                });

            } else {
                if (debug_verbosity >= 1) {
                    std::cerr << "Warning: NPU requested but config/handles missing. Fallback to GPU (Delayed)." << std::endl;
                    std::exit(1);
                }

                // Fallback GPU Kernel (Synchronous regarding this thread, but stream-based)
                auto output_slice = output_2d.slice(1, 0, npuN);
                hipkernels::w4a16_gemv_fused_packed(output_slice, input_2d, weights, in_features, npuN, out_features, 0, call_stream);
            }
        }

        // 2. Launch CPU (Synchronous in this thread)
        if (cpuN > 0) {
            if (debug_verbosity >= 2)
                std::cout << "GEMV: Launching CPU (" << npuN << ".." << npuN + cpuN << ") [Sync]" << std::endl;

            torch::Tensor weights_cpu_kernel = weights;
            w4a16_gemv_cpu_fused_packed(output_2d, input_2d, weights_cpu_kernel, in_features, out_features, hip_event, cpuThreads, npuN,
                                        npuN + cpuN);
        }

        // 3. Wait for NPU
        if (npu_fut.valid()) {
            ret = npu_fut.get();
        }

        release_event(hip_event);
        return ret;
    });
}

// 4. GEMV (K-Split)
std::future<int> hetero_matmul_out_gemv_packed_K(torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &weights0,
                                                 const torch::Tensor &weights1, int64_t in_features, int64_t out_features,
                                                 int64_t split_k, std::string layer_type, int chunk_id,
                                                 bool force_gpu_only) {
    hipStream_t call_stream = c10::hip::getCurrentHIPStream().stream();
    if (debug_verbosity >= 2)
        std::cout << "K-Split GEMV Path - Layer: " << layer_type << " Split K=" << split_k << std::endl;

    int64_t K = in_features;
    int64_t K0 = (split_k > 0) ? split_k : (K / 2);
    int64_t K1 = K - K0;
    int64_t N = out_features;
    int64_t K0_internal = K0;
    int64_t K1_internal = K1;

    if (pad_packed_weights) {
        K0_internal = round_up_packed_split_dim(layer_type, K0);
        K1_internal = round_up_packed_split_dim(layer_type, K1);
    }

    auto input_2d = x.view({-1, x.size(-1)});
    auto output_2d = output.view({-1, N});
    auto temp_output = torch::empty({1, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(x.device()));

    torch::Tensor input_0 = input_2d.slice(1, 0, K0);
    torch::Tensor input_1 = input_2d.slice(1, K0, K);

    // Acquire event for Hetero sync
    hipEvent_t hip_event = acquire_event();
    HIP_CHECK(hipEventRecord(hip_event, call_stream));

    // --- Hetero Partitioning for Weights0 (K0) ---
    // Lookup config using TOTAL dims (1, K, N)
    int layer_id = get_layer_id(layer_type);
    auto cfg_it = find_config_entry(1, static_cast<int>(K), static_cast<int>(N), layer_id, chunk_id);

    int cpuN = 0;
    int npuN = 0;
    int cpuThreads = 1;
    bool enable_hetero = false;

    // Verify Handles
    void *x_ptr = x.data_ptr();
    void *w_ptr = weights0.data_ptr();
    void *out_ptr = output.data_ptr();

    auto it_x = ptr_to_handle_map.find(x_ptr);
    auto it_w = ptr_to_handle_map.find(w_ptr);
    auto it_out = ptr_to_handle_map.find(out_ptr);

    bool handles_exist = (it_x != ptr_to_handle_map.end()) && (it_w != ptr_to_handle_map.end()) && (it_out != ptr_to_handle_map.end());
    bool config_exists = (cfg_it != config_map.end());

    if (debug_verbosity >= 2) {
        std::cout << "[" << "M=" << 1 << " K=" << K << " N=" << N << "]" << " weight_layout=" << "Packed" << " layer_type=" << layer_type
                  << std::endl;
        std::cout << "Handles exist: " << handles_exist << " (x=" << (it_x != ptr_to_handle_map.end())
                  << ", w=" << (it_w != ptr_to_handle_map.end()) << ", out=" << (it_out != ptr_to_handle_map.end()) << ")" << std::endl;

        if (it_x == ptr_to_handle_map.end()) {
            std::cout << "  x_ptr: " << x_ptr << " not found in map (size: " << ptr_to_handle_map.size() << ")" << std::endl;
            // Print first few keys in map to check
            int count = 0;
            for (const auto &pair : ptr_to_handle_map) {
                if (count++ < 5)
                    std::cout << "    Map key: " << pair.first << std::endl;
            }
        }
        std::cout << "Config exists: " << config_exists << std::endl;
        if (config_exists && debug_verbosity >= 3) {
            const auto &val = cfg_it->second;
            std::cout << "  Config value: hw_idx=" << val.hw_idx << " inst_idx=" << val.inst_idx << " npuM=" << val.npuM
                      << " npuK=" << val.npuK << " npuN=" << val.npuN << " config=" << val.config << std::endl;
        }
        if (!config_exists && debug_verbosity >= 3) {
            std::cout << "Available configs:" << std::endl;
            for (const auto &kv : config_map) {
                std::cout << "  {" << kv.first[0] << ", " << kv.first[1] << ", " << kv.first[2] << ", " << kv.first[3] << ", "
                          << kv.first[4] << "}" << std::endl;
            }
        }
    }

    if (!force_gpu_only && hw_target == "hetero" && config_exists && handles_exist) {
        const NPUValue &val = cfg_it->second;
        if (val.npuK == K0 || val.npuK == K0_internal) {
            npuN = val.npuN;
            cpuN = val.cpuN;
            cpuThreads = val.cpuThreads > 0 ? val.cpuThreads : 1;
            enable_hetero = true;
        }
    }

    if (enable_hetero) {
        // K-split GEMV policy: when npuK != forK, NPU must cover full N.
        if (npuN != N) {
            std::cerr << "Error: Invalid K-split GEMV config for layer " << layer_type << ": npuN=" << npuN << " but forN=" << N
                      << ". K-split requires npuN == forN." << std::endl;
            release_event(hip_event);
            return std::future<int>();
        }

        if (debug_verbosity >= 2)
            std::cout << "K-Split Hetro Path: K0_NPU=" << npuN << " K1_CPU=" << cpuN << std::endl;

        // K0 is fully covered by NPU (npuN == N). For K1, CPU may cover [0, cpuN) and GPU handles [cpuN, N).
        int gpuN_1 = N - cpuN;
        if (gpuN_1 < 0)
            gpuN_1 = 0;

        // Dispatch K1 remainder to GPU.
        if (gpuN_1 > 0) {
            int start_gpu_1 = cpuN;
            auto temp_out_slice = temp_output.view({-1, N}).slice(1, start_gpu_1, N);
            hipkernels::w4a16_gemv_fused_packed(temp_out_slice, input_1, weights1, K1_internal, gpuN_1, N, start_gpu_1, call_stream);
        }

        // 3. Launch CPU/NPU - Async Thread
        // Capture copies of lightweight objects, tensors (views) by value
        NPUValue val = (npuN > 0) ? cfg_it->second : NPUValue();

        return g_thread_pool.submit_task([=]() mutable {
            std::future<int> npu_fut;

            // A. Launch NPU Async (Processing K0, output 0..npuN)
            if (npuN > 0) {
                void *x_ptr = input_0.data_ptr();
                void *w_ptr = weights0.data_ptr();
                void *out_ptr = output.data_ptr();

                uint32_t x_handle = ptr_to_handle_map[x_ptr];
                uint32_t w_handle = ptr_to_handle_map[w_ptr];
                uint32_t out_handle = ptr_to_handle_map[out_ptr];

                npu_fut = std::async(std::launch::async, [val, out_ptr, x_ptr, w_ptr, out_handle, x_handle, w_handle, hip_event]() {
                    return npuMatmul_zero(val.hw_idx, val.inst_idx, out_ptr, x_ptr, w_ptr, out_handle, x_handle, w_handle, hip_event);
                });
            }

            // B. Run CPU Sync (Processing K1, output 0..cpuN)
            // CPU writes to temp_output
            if (cpuN > 0) {
                w4a16_gemv_cpu_fused_packed(temp_output, input_1, weights1, K1_internal, N, hip_event, cpuThreads, 0, cpuN);
            }

            // C. Wait NPU
            if (npu_fut.valid()) {
                npu_fut.get();
            }

            release_event(hip_event);

            // output += temp_output.
            // temp_output contains the full K1 result (CPU part + GPU part)
            // output contains the full K0 result (NPU part + GPU part)
            output_2d.add_(temp_output);

            return 0;
        });

    } else {
        // Fallback GPU
        hipkernels::w4a16_gemv_fused_packed(output_2d, input_0, weights0, K0_internal, N, -1, 0, call_stream);
        hipkernels::w4a16_gemv_fused_packed(temp_output, input_1, weights1, K1_internal, N, -1, 0, call_stream);

        output_2d.add_(temp_output);
        release_event(hip_event);
        return std::future<int>();
    }
}

// --- UNPACKED IMPLEMENTATIONS ---

// Custom GEMV wrapper (M = 1) - Unpacked
std::future<int> hetero_matmul_out_gemv_unpacked_M(torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &qweights,
                                                   const torch::Tensor &scales, const torch::Tensor &zeros, int64_t in_features,
                                                   int64_t out_features, std::string layer_type, int chunk_id) {
    // Ensure GPU has finished writing to shared buffers (x, output)
    hipStream_t call_stream = c10::hip::getCurrentHIPStream().stream();
    hipEvent_t hip_event = acquire_event();
    HIP_CHECK(hipEventRecord(hip_event, call_stream));

    // Lookup NPU config for M=1 GEMV
    int layer_id = get_layer_id(layer_type);
    auto cfg_it = find_config_entry(1, static_cast<int>(in_features), static_cast<int>(out_features), layer_id, chunk_id);

    int cpuN = 0;
    int cpuThreads = 1;

    // Check global config map
    if (cfg_it != config_map.end()) {
        const NPUValue &val = cfg_it->second;
        cpuN = val.cpuN;
        cpuThreads = val.cpuThreads > 0 ? val.cpuThreads : 1;
    } else {
        if (debug_verbosity >= 2) {
            std::cout << "DEBUG: GEMV Unpacked Config not found for 1x" << in_features << "x" << out_features << " " << layer_type
                      << " - using GPU fallback" << std::endl;
        }
    }

    return hetero_matmul_out_gemv_unpacked(output, x, qweights, scales, zeros, in_features, out_features, cpuN, cpuThreads, layer_type,
                                           hip_event, chunk_id);
}

// Custom GEMV wrapper (M = 1) - Unpacked - Explicit Config
std::future<int> hetero_matmul_out_gemv_unpacked(torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &qweights,
                                                 const torch::Tensor &scales, const torch::Tensor &zeros, int64_t in_features,
                                                 int64_t out_features, int cpuN, int cpuThreads, std::string layer_type,
                                                 hipEvent_t hip_event, int chunk_id) {
    hipStream_t call_stream = c10::hip::getCurrentHIPStream().stream();

    // If hip_event is null (not passed from wrapper), acquire one
    if (hip_event == nullptr) {
        hip_event = acquire_event();
        HIP_CHECK(hipEventRecord(hip_event, call_stream));
    }

    // Default CPU logic constraint
    if (cpuN > 0 && cpuN % 64 != 0) {
        if (debug_verbosity >= 1)
            std::cerr << "Warning: cpuN (" << cpuN << ") is not a multiple of 64. Disabling CPU part." << std::endl;
        cpuN = 0;
    }

    // Calculate GPU remainder
    // Unpacked split: CPU takes [0, cpuN), GPU takes [cpuN, out_features)
    int gpuN = out_features - cpuN;
    if (gpuN < 0) {
        cpuN = 0;
        gpuN = out_features;
    }

    // View inputs for raw access
    auto input_2d = x.view({-1, in_features});
    auto output_2d = output.view({-1, out_features});

    // 1. Launch GPU Kernel first (Async launch on main stream)
    if (gpuN > 0) {
        int start_gpu = cpuN;
        if (debug_verbosity >= 2)
            std::cout << "GEMV Unpacked (" << layer_type << "): Launching GPU (" << start_gpu << ".." << out_features << ") [Async]"
                      << std::endl;

        // Slice tensors for GPU
        // Slicing output (dim 1, columns)
        auto output_slice = output_2d.slice(1, start_gpu, out_features);

        // qweights are [Out, In/2], so slice dim 0
        auto qweights_slice = qweights.slice(0, start_gpu, out_features).contiguous();

        // scales/zeros are [Out, Groups] (transposed by set_quantized_weights), so slice dim 0.
        // GPU kernel now handles strides (pass non-contiguous view directly)
        // But ensures 16-byte alignment etc.
        auto scales_slice = scales.slice(0, start_gpu, out_features).contiguous();
        auto zeros_slice = zeros.slice(0, start_gpu, out_features).contiguous();
        auto input_contig = input_2d.contiguous();

        // Calculate group size from scales shape logic
        int64_t num_groups = scales.size(1);
        int64_t group_size = in_features / num_groups;

        hipkernels::w4a16_gemv_unpacked_fused(output_slice, input_contig, qweights_slice, scales_slice, zeros_slice, in_features, gpuN,
                                              group_size, call_stream);
    }

    // 2. Execute CPU Kernel (Async - concurrent with GPU)
    if (cpuN > 0) {
        if (debug_verbosity >= 2)
            std::cout << "GEMV Unpacked (" << layer_type << "): Launching CPU (0.." << cpuN << ") [Async]" << std::endl;

        // Slice inputs for CPU (0 to cpuN) - capture by value for async
        auto input_cpu = input_2d;
        auto output_slice_cpu = output_2d.slice(1, 0, cpuN);
        auto qweights_slice_cpu = qweights.slice(0, 0, cpuN);
        auto scales_slice_cpu = scales.slice(0, 0, cpuN);
        auto zeros_slice_cpu = zeros.slice(0, 0, cpuN);

        // Async launch using Thread Pool
        return g_thread_pool.submit_task([output_slice_cpu, input_cpu, qweights_slice_cpu, scales_slice_cpu, zeros_slice_cpu, in_features,
                                          cpuN, hip_event, cpuThreads]() mutable {
            int64_t num_groups = scales_slice_cpu.size(1);
            int64_t group_size = in_features / num_groups;

            // Execute CPU kernel on the slice
            w4a16_gemv_cpu_fused_unpacked(output_slice_cpu, input_cpu, qweights_slice_cpu, scales_slice_cpu, zeros_slice_cpu, in_features,
                                          cpuN, group_size, hip_event, cpuThreads);

            // Release event back to pool
            release_event(hip_event);
            return 0;
        });
    }

    // If only GPU ran, we still need to return a future that handles event cleanup
    // Release the event immediately as we are not launching a dependent CPU task
    release_event(hip_event);
    return std::future<int>();

    release_event(hip_event);
    return std::async(std::launch::deferred, [] { return 0; });
}

// Stricter NPU-only GEMM wrapper (M > 1)
std::future<int> npu_top_matmul_out_gemm_packed(torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &weights,
                                                int64_t in_features, int64_t out_features, std::string layer_type, int chunk_id) {
    // Ensure GPU has finished writing to shared buffers (x, output) before NPU reads/writes
    hipStream_t call_stream = c10::hip::getCurrentHIPStream().stream();
    hipEvent_t hip_event = acquire_event();
    HIP_CHECK(hipEventRecord(hip_event, call_stream));

    // Handle batch dimension: 2D (seq_len, hidden) or 3D (batch, seq_len, hidden)
    int M, K;
    if (x.dim() == 2) {
        M = x.size(0);
        K = x.size(1);
    } else if (x.dim() == 3) {
        M = x.size(1); // seq_len
        K = x.size(2); // hidden_dim
    } else {
        std::cerr << "Error: Unexpected dimensions for NPU execution: " << x.sizes() << std::endl;
        exit(1);
    }

    // Always assume Row Major [K, N]
    int N = (out_features != -1) ? out_features : weights.size(1);

    void *x_ptr = x.data_ptr();
    void *w_ptr = weights.data_ptr();
    void *out_ptr = output.data_ptr();

    // Check if handles exist in cache
    auto it_x = ptr_to_handle_map.find(x_ptr);
    auto it_w = ptr_to_handle_map.find(w_ptr);
    auto it_out = ptr_to_handle_map.find(out_ptr);

    bool handles_exist = (it_x != ptr_to_handle_map.end()) && (it_w != ptr_to_handle_map.end()) && (it_out != ptr_to_handle_map.end());

    // Check if configuration exists (exact chunk_id match)
    int layer_id = get_layer_id(layer_type);
    auto cfg_it = find_config_entry(M, K, N, layer_id, chunk_id);
    bool config_exists = (cfg_it != config_map.end());

    if (debug_verbosity >= 2) {
        std::cout << "[" << "M=" << M << " K=" << K << " N=" << N << "]" << " weight_layout=" << "Packed" << " layer_type=" << layer_type
                  << std::endl;
        std::cout << "Handles exist: " << handles_exist << " (x=" << (it_x != ptr_to_handle_map.end())
                  << ", w=" << (it_w != ptr_to_handle_map.end()) << ", out=" << (it_out != ptr_to_handle_map.end()) << ")" << std::endl;

        if (it_x == ptr_to_handle_map.end()) {
            std::cout << "  x_ptr: " << x_ptr << " not found in map (size: " << ptr_to_handle_map.size() << ")" << std::endl;
        }
    }

    if (config_exists) {
        const NPUValue val = cfg_it->second;
        if (val.config > 0) {
            std::future<int> fut = std::async(std::launch::async, [val, hip_event]() {
                double gops = (double)val.config;
                // GEMM: Flops = 2 * M * K * N
                double flops = 2.0 * (double)val.npuM * (double)val.npuK * (double)val.npuN;
                // AI 7 350 Llama3-8B prefill fit by prompt M.
                const double m_log2 = std::log2(static_cast<double>(val.npuM) / 1024.0);
                const double proportional_scale =
                    2.2001571005769756 + 0.04439597331737696 * m_log2 + 0.13277359458798946 * m_log2 * m_log2;
                constexpr useconds_t reconfig_bias_us = 0;
                double time_sec = flops / (gops * 1e9) * proportional_scale;
                if (debug_verbosity >= 2)
                    std::cout << "Simulating NPU GEMM (Strict): " << time_sec * 1000.0 << " ms (GOPS=" << gops
                              << ", scale=" << proportional_scale << ", bias_us=" << reconfig_bias_us << ")" << std::endl;
                usleep(static_cast<useconds_t>(time_sec * 1e6) + reconfig_bias_us);
                release_event(hip_event);
                return 0;
            });
            return fut;
        }
    }

    if (!handles_exist || !config_exists) {
        if (debug_verbosity >= 1) {
            std::cerr << "Warning: NPU execution failed! Missing handles or configuration. Falling back to GPU." << std::endl;
            std::cerr << "Layer: " << layer_type << " [M=" << M << " K=" << K << " N=" << N << "]" << std::endl;
            std::cerr << "Handles exist: " << handles_exist << " (x=" << (it_x != ptr_to_handle_map.end())
                      << ", w=" << (it_w != ptr_to_handle_map.end()) << ", out=" << (it_out != ptr_to_handle_map.end()) << ")" << std::endl;
            std::cerr << "Config exists: " << config_exists << std::endl;
        }

        auto input_2d = x.view({-1, in_features});
        auto output_2d = output.view({-1, out_features});
        hipkernels::w4a16_gemm_fused_packed(output_2d, input_2d, weights, in_features, out_features, call_stream);
        release_event(hip_event);
        return std::future<int>();

        // exit(1);
    }

    // Found NPU config
    NPUValue val = cfg_it->second;

    // NPU Execution (Assuming full offload as per strict NPU path)

    uint16_t *x_npu = (uint16_t *)x_ptr;
    uint16_t *w_npu = (uint16_t *)w_ptr;
    uint16_t *out_npu = (uint16_t *)out_ptr;

    uint32_t x_handle = ptr_to_handle_map[x_ptr];
    uint32_t out_handle = ptr_to_handle_map[out_ptr];
    uint32_t w_handle = ptr_to_handle_map[w_ptr];

    // Async launch
    std::future<int> fut = std::async(std::launch::async, [val, out_npu, x_npu, w_npu, out_handle, x_handle, w_handle, hip_event]() {
        // Execute NPU kernel (handles synchronization internally)
        int ret = npuMatmul_zero(val.hw_idx, val.inst_idx, (void *)out_npu, (void *)x_npu, (void *)w_npu, out_handle, x_handle, w_handle,
                                 hip_event);

        // Release event back to pool
        release_event(hip_event);

        return ret;
    });

    if (debug_verbosity >= 2) {
        std::cout << "Launched NPU async (Top NPU Path)" << std::endl;
    }

    return fut;
}

// Stricter NPU-only GEMV wrapper (M = 1)
std::future<int> npu_top_matmul_out_gemv_packed(torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &weights,
                                                int64_t in_features, int64_t out_features, std::string layer_type, int chunk_id) {
    // Ensure GPU has finished writing to shared buffers (x, output)
    hipStream_t call_stream = c10::hip::getCurrentHIPStream().stream();
    hipEvent_t hip_event = acquire_event();
    HIP_CHECK(hipEventRecord(hip_event, call_stream));

    // M=1 for generation
    int M = 1;
    int K = in_features;
    int N = out_features != -1 ? out_features : weights.size(1);

    void *x_ptr = x.data_ptr();
    void *w_ptr = weights.data_ptr();
    void *out_ptr = output.data_ptr();

    // Check if handles exist
    auto it_x = ptr_to_handle_map.find(x_ptr);
    auto it_w = ptr_to_handle_map.find(w_ptr);
    auto it_out = ptr_to_handle_map.find(out_ptr);
    bool handles_exist = (it_x != ptr_to_handle_map.end()) && (it_w != ptr_to_handle_map.end()) && (it_out != ptr_to_handle_map.end());

    // Check configuration (exact chunk_id match)
    int layer_id = get_layer_id(layer_type);
    auto cfg_it = find_config_entry(1, K, N, layer_id, chunk_id);
    bool config_exists = (cfg_it != config_map.end());

    if (debug_verbosity >= 2) {
        std::cout << "[" << "M=" << M << " K=" << K << " N=" << N << "]" << " weight_layout=" << "Packed" << " layer_type=" << layer_type
                  << std::endl;
        std::cout << "Handles exist: " << handles_exist << " (x=" << (it_x != ptr_to_handle_map.end())
                  << ", w=" << (it_w != ptr_to_handle_map.end()) << ", out=" << (it_out != ptr_to_handle_map.end()) << ")" << std::endl;

        if (it_x == ptr_to_handle_map.end()) {
            std::cout << "  x_ptr: " << x_ptr << " not found in map (size: " << ptr_to_handle_map.size() << ")" << std::endl;
            // Print first few keys in map to check
            int count = 0;
            for (const auto &pair : ptr_to_handle_map) {
                if (count++ < 5)
                    std::cout << "    Map key: " << pair.first << std::endl;
            }
        }
        std::cout << "Config exists: " << config_exists << std::endl;
        if (config_exists && debug_verbosity >= 3) {
            const auto &val = cfg_it->second;
            std::cout << "  Config value: hw_idx=" << val.hw_idx << " inst_idx=" << val.inst_idx << " npuM=" << val.npuM
                      << " npuK=" << val.npuK << " npuN=" << val.npuN << " config=" << val.config << std::endl;
        }
        if (!config_exists && debug_verbosity >= 3) {
            std::cout << "Available configs:" << std::endl;
            for (const auto &kv : config_map) {
                std::cout << "  {" << kv.first[0] << ", " << kv.first[1] << ", " << kv.first[2] << ", " << kv.first[3] << ", "
                          << kv.first[4] << "}" << std::endl;
            }
        }
    }

    if (config_exists) {
        const NPUValue val = cfg_it->second;
        if (val.config > 0) {
            std::future<int> fut = std::async(std::launch::async, [val, hip_event, K]() {
                double gops = (double)val.config;
                // GEMV: M=1. Flops = 2 * K * N
                double flops = 2.0 * (double)K * (double)val.npuN;
                double time_sec = flops / (gops * 1e9);
                if (debug_verbosity >= 2)
                    std::cout << "Simulating NPU GEMV (Strict): " << time_sec * 1000.0 << " ms (GOPS=" << gops << ")" << std::endl;
                usleep(static_cast<useconds_t>(time_sec * 1e6));
                release_event(hip_event);
                return 0;
            });
            return fut;
        }
    }

    if (handles_exist && config_exists) {
        if (debug_verbosity >= 2) {
            std::cout << "NPU Strict Path (GEMV): Found config for [M=1 K=" << K << " N=" << N << "]" << std::endl;
        }

        NPUValue val = cfg_it->second;

        uint16_t *x_npu = (uint16_t *)x_ptr;
        uint16_t *w_npu = (uint16_t *)w_ptr;
        uint16_t *out_npu = (uint16_t *)out_ptr;

        uint32_t x_handle = ptr_to_handle_map[x_ptr];
        uint32_t out_handle = ptr_to_handle_map[out_ptr];
        uint32_t w_handle = ptr_to_handle_map[w_ptr];

        std::future<int> fut = std::async(std::launch::async, [val, out_npu, x_npu, w_npu, out_handle, x_handle, w_handle, hip_event, K]() {
            int ret = npuMatmul_zero(val.hw_idx, val.inst_idx, (void *)out_npu, (void *)x_npu, (void *)w_npu, out_handle, x_handle,
                                     w_handle, hip_event);
            release_event(hip_event);
            return ret;
        });
        return fut;
    }

    // Fix for padded inputs (e.g. Qwen 1536 -> 2048)
    int64_t actual_K = x.size(-1);
    if (x.dim() == 1) {
        actual_K = x.numel();
    }
    auto input_2d = x.view({-1, actual_K});
    auto output_2d = output.view({-1, out_features});

    hipkernels::w4a16_gemv_fused_packed(output_2d, input_2d, weights, actual_K, out_features, -1, 0, call_stream);

    release_event(hip_event);
    return std::future<int>();
}
