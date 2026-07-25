#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <type_traits>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// =============================================================================
// HIP Helpers
// =============================================================================

#define HIP_CHECK(call)                                                                                                                    \
    do {                                                                                                                                   \
        hipError_t _e = (call);                                                                                                            \
        if (_e != hipSuccess) {                                                                                                            \
            std::cerr << "HIP error: " << hipGetErrorString(_e) << " at " << __FILE__ << ":" << __LINE__ << std::endl;                     \
            std::abort();                                                                                                                  \
        }                                                                                                                                  \
    } while (0)

namespace {
struct FlashAttnDecodeWorkspace {
    void *partials = nullptr;
    void *meta = nullptr;
    size_t partials_bytes = 0;
    size_t meta_bytes = 0;
    int device = -1;
};

std::mutex g_flash_attn_decode_mutex;
FlashAttnDecodeWorkspace g_flash_attn_decode_ws;

FlashAttnDecodeWorkspace acquire_flash_attn_decode_workspace(int device, size_t partials_bytes, size_t meta_bytes) {
    std::lock_guard<std::mutex> lock(g_flash_attn_decode_mutex);

    if (g_flash_attn_decode_ws.device != device) {
        if (g_flash_attn_decode_ws.partials) {
            HIP_CHECK(hipFree(g_flash_attn_decode_ws.partials));
        }
        if (g_flash_attn_decode_ws.meta) {
            HIP_CHECK(hipFree(g_flash_attn_decode_ws.meta));
        }
        g_flash_attn_decode_ws = FlashAttnDecodeWorkspace{};
        g_flash_attn_decode_ws.device = device;
    }

    if (partials_bytes > g_flash_attn_decode_ws.partials_bytes) {
        if (g_flash_attn_decode_ws.partials) {
            HIP_CHECK(hipFree(g_flash_attn_decode_ws.partials));
        }
        HIP_CHECK(hipMalloc(&g_flash_attn_decode_ws.partials, partials_bytes));
        g_flash_attn_decode_ws.partials_bytes = partials_bytes;
    }

    if (meta_bytes > g_flash_attn_decode_ws.meta_bytes) {
        if (g_flash_attn_decode_ws.meta) {
            HIP_CHECK(hipFree(g_flash_attn_decode_ws.meta));
        }
        HIP_CHECK(hipMalloc(&g_flash_attn_decode_ws.meta, meta_bytes));
        g_flash_attn_decode_ws.meta_bytes = meta_bytes;
    }

    return g_flash_attn_decode_ws;
}
} // namespace

// =============================================================================
// Constants - llama.cpp fattn-vec style for RDNA
// =============================================================================

#define WARP_SIZE 32
#define NWARPS 4
#define NTHREADS (WARP_SIZE * NWARPS) // 128 threads total
#define D_HEAD 128

// RDNA optimization: 2 threads collaborate on each KQ dot product
#define NTHREADS_KQ 2
// Tokens processed per iteration (each thread pair handles one token)
#define TOKENS_PER_ITER (NTHREADS / NTHREADS_KQ) // 64
// Decode kernel processes two tokens per thread-pair each iteration to amortize syncs.
#define TOKENS_PER_ITER_DECODE (TOKENS_PER_ITER * 2) // 128

// Prefill-specific launch geometry (kept aliasable for future tuning).
#define NWARPS_PREFILL 4
#define NTHREADS_PREFILL (WARP_SIZE * NWARPS_PREFILL)
#define TOKENS_PER_ITER_PREFILL_HALF (NTHREADS_PREFILL / NTHREADS_KQ)
#define TOKENS_PER_ITER_PREFILL (TOKENS_PER_ITER_PREFILL_HALF * 2)
#define Q_COLS_PREFILL_TILE 2

#define FATTN_KQ_MAX_OFFSET (3.0f * 0.6931f)

// =============================================================================
// Helper Functions
// =============================================================================

// RDNA3 Intrinsic for dot2_f32_f16
static __device__ __forceinline__ float dot2_f32_f16(half2 a, half2 b, float c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__)
    // v_dot2_f32_f16 accumulates a * b into c
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(c) : "v"(a), "v"(b));
    return c;
#else
    float2 af = __half22float2(a);
    float2 bf = __half22float2(b);
    return c + af.x * bf.x + af.y * bf.y;
#endif
}

template <int width = WARP_SIZE> static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor(x, offset, width);
    }
    return x;
}

template <int width = WARP_SIZE> static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor(x, offset, width));
    }
    return x;
}

template <typename T> __device__ __forceinline__ T convert_out(float v);

template <> __device__ __forceinline__ half convert_out<half>(float v) { return __float2half(v); }

template <> __device__ __forceinline__ __hip_bfloat16 convert_out<__hip_bfloat16>(float v) { return __float2bfloat16(v); }

// Load 4 elements as float
template <typename T> __device__ __forceinline__ void load_vec4_as_float(const void *ptr, float *out);

template <> __device__ __forceinline__ void load_vec4_as_float<half>(const void *ptr, float *out) {
    const half2 *h2 = (const half2 *)ptr;
    float2 f0 = __half22float2(h2[0]);
    float2 f1 = __half22float2(h2[1]);
    out[0] = f0.x;
    out[1] = f0.y;
    out[2] = f1.x;
    out[3] = f1.y;
}

template <> __device__ __forceinline__ void load_vec4_as_float<__hip_bfloat16>(const void *ptr, float *out) {
    const __hip_bfloat16 *bf = (const __hip_bfloat16 *)ptr;
    out[0] = __bfloat162float(bf[0]);
    out[1] = __bfloat162float(bf[1]);
    out[2] = __bfloat162float(bf[2]);
    out[3] = __bfloat162float(bf[3]);
}

template <typename T> __device__ __forceinline__ void store_float_as_vec4(void *ptr, const float *in);

template <> __device__ __forceinline__ void store_float_as_vec4<half>(void *ptr, const float *in) {
    half2 *h2 = (half2 *)ptr;
    h2[0] = __float22half2_rn(make_float2(in[0], in[1]));
    h2[1] = __float22half2_rn(make_float2(in[2], in[3]));
}

template <> __device__ __forceinline__ void store_float_as_vec4<__hip_bfloat16>(void *ptr, const float *in) {
    __hip_bfloat16 *bf = (__hip_bfloat16 *)ptr;
    bf[0] = __float2bfloat16(in[0]);
    bf[1] = __float2bfloat16(in[1]);
    bf[2] = __float2bfloat16(in[2]);
    bf[3] = __float2bfloat16(in[3]);
}

template <typename T> __device__ __forceinline__ float load_scalar_as_float(const char *ptr);
template <> __device__ __forceinline__ float load_scalar_as_float<half>(const char *ptr) { return __half2float(*(const half *)ptr); }
template <>
__device__ __forceinline__ float load_scalar_as_float<__hip_bfloat16>(const char *ptr) {
    return __bfloat162float(*(const __hip_bfloat16 *)ptr);
}

template <typename T> __device__ __forceinline__ void store_scalar_from_float(char *ptr, float v);
template <> __device__ __forceinline__ void store_scalar_from_float<half>(char *ptr, float v) { *(half *)ptr = __float2half(v); }
template <>
__device__ __forceinline__ void store_scalar_from_float<__hip_bfloat16>(char *ptr, float v) {
    *(__hip_bfloat16 *)ptr = __float2bfloat16(v);
}

template <typename T> __device__ __forceinline__ float2 load_pair_as_float2(const char *ptr);
template <> __device__ __forceinline__ float2 load_pair_as_float2<half>(const char *ptr) {
    return __half22float2(*(const half2 *)ptr);
}
template <>
__device__ __forceinline__ float2 load_pair_as_float2<__hip_bfloat16>(const char *ptr) {
    return __bfloat1622float2(*(const __hip_bfloat162 *)ptr);
}

template <typename T> __device__ __forceinline__ void store_pair_from_float2(char *ptr, const float2 &v);
template <> __device__ __forceinline__ void store_pair_from_float2<half>(char *ptr, const float2 &v) {
    *(half2 *)ptr = __float22half2_rn(v);
}
template <>
__device__ __forceinline__ void store_pair_from_float2<__hip_bfloat16>(char *ptr, const float2 &v) {
    *(__hip_bfloat162 *)ptr = __float22bfloat162_rn(v);
}

__device__ __forceinline__ float block_reduce_sum_256(float x) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;

    x = warp_reduce_sum<WARP_SIZE>(x);
    __shared__ float warp_sums[8];
    if (lane == 0) {
        warp_sums[warp] = x;
    }
    __syncthreads();

    float total = 0.0f;
    if (warp == 0 && lane < 8) {
        total = warp_sums[lane];
    }
    if (warp == 0) {
        total = warp_reduce_sum<WARP_SIZE>(total);
    }
    return total;
}

__device__ __forceinline__ float block_reduce_sum_128(float x) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;

    x = warp_reduce_sum<WARP_SIZE>(x);
    __shared__ float warp_sums[4];
    if (lane == 0) {
        warp_sums[warp] = x;
    }
    __syncthreads();

    float total = 0.0f;
    if (warp == 0 && lane < 4) {
        total = warp_sums[lane];
    }
    if (warp == 0) {
        total = warp_reduce_sum<WARP_SIZE>(total);
    }
    return total;
}

// =============================================================================
// Flash Attention Decode Kernel - half2 optimized with 128-bit loads
// =============================================================================

// Specialized for FP16 to use half2 intrinsics
template <bool WRITE_PARTIALS>
__global__ __launch_bounds__(NTHREADS, 1) void flash_attn_vec_f16(const char *__restrict__ Q_base, const char *__restrict__ K_base,
                                                                  const char *__restrict__ V_base, half *__restrict__ dst,
                                                                  float *__restrict__ dst_partials, float2 *__restrict__ dst_meta,
                                                                  const float scale,
                                                                  const int ne11,                                 // seq_len_kv
                                                                  const int ne02,                                 // n_heads_Q
                                                                  const int ne12,                                 // n_heads_KV
                                                                  const int nb01, const int nb02, const int nb03, // Q strides
                                                                  const int nb11, const int nb12, const int nb13, // K strides
                                                                  const int nb21, const int nb22, const int nb23, // V strides
                                                                  const int batch_size) {

    // Thread indexing
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;

    // Token pair indexing
    const int token_idx = tid / NTHREADS_KQ;    // 0-63
    const int lane_in_pair = tid % NTHREADS_KQ; // 0 or 1

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z % ne02;
    if (sequence >= batch_size)
        return;

    const int gqa_ratio = ne02 / ne12;
    const int head_kv = head / gqa_ratio;

    const char *Q_ptr = Q_base + (int64_t)nb03 * sequence + (int64_t)nb02 * head;
    const char *K_ptr = K_base + (int64_t)nb13 * sequence + (int64_t)nb12 * head_kv;
    const char *V_ptr = V_base + (int64_t)nb23 * sequence + (int64_t)nb22 * head_kv;

    // =========================================================================
    // Load Q as half2 - using 128-bit loads (4x half2 per load)
    // =========================================================================

    constexpr int D_PER_THREAD = D_HEAD / NTHREADS_KQ; // 64 floats
    constexpr int H2_PER_THREAD = D_PER_THREAD / 2;    // 32 half2s

    half2 Q_h2[H2_PER_THREAD];
    {
        const int4 *Q_I4 = (const int4 *)Q_ptr;
        // Each int4 = 16 bytes = 4x half2
        // H2_PER_THREAD = 32 half2s = 8x int4s
        const int base_i4 = lane_in_pair * 8;

        half2 scale_h2 = __float2half2_rn(scale);

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int4 val_i4 = Q_I4[base_i4 + i];
            // Unpack int4 -> 4x half2
            half2 *h2_ptr = (half2 *)&val_i4;

            Q_h2[i * 4 + 0] = __hmul2(h2_ptr[0], scale_h2);
            Q_h2[i * 4 + 1] = __hmul2(h2_ptr[1], scale_h2);
            Q_h2[i * 4 + 2] = __hmul2(h2_ptr[2], scale_h2);
            Q_h2[i * 4 + 3] = __hmul2(h2_ptr[3], scale_h2);
        }
    }

    // =========================================================================
    // VKQ accumulator - half2
    // =========================================================================

    half2 VKQ[H2_PER_THREAD];
#pragma unroll
    for (int i = 0; i < H2_PER_THREAD; ++i)
        VKQ[i] = __float2half2_rn(0.0f); // Init to 0

    float KQ_max = -FLT_MAX / 2.0f;
    float KQ_sum = 0.0f;

    __shared__ float KQ_smem[TOKENS_PER_ITER_DECODE]; // 128
    __shared__ float reduce_smem[NWARPS];      // 4

    // =========================================================================
    // Main loop
    // =========================================================================

    for (int k0 = blockIdx.y * TOKENS_PER_ITER_DECODE; k0 < ne11; k0 += gridDim.y * TOKENS_PER_ITER_DECODE) {
        const int k_idx0 = k0 + token_idx;
        const int k_idx1 = k_idx0 + TOKENS_PER_ITER;

        // ---------------------------------------------------------------------
        // 1. Compute Q*K dot product (Mixed Precision)
        // ---------------------------------------------------------------------
        float KQ_val0 = -FLT_MAX;
        float KQ_val1 = -FLT_MAX;
        const bool in_bounds0 = (k_idx0 < ne11);
        const bool in_bounds1 = (k_idx1 < ne11);

        if (in_bounds0) {
            const int4 *K_I4_row = (const int4 *)(K_ptr + k_idx0 * nb11);
            const int base_i4 = lane_in_pair * 8;

            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int4 k_val_i4 = K_I4_row[base_i4 + i];
                half2 *k_h2 = (half2 *)&k_val_i4;

                // Process 4 half2s
                dot = dot2_f32_f16(Q_h2[i * 4 + 0], k_h2[0], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 1], k_h2[1], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 2], k_h2[2], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 3], k_h2[3], dot);
            }

            // Reduce across pair (lane 0 and 1)
            dot += __shfl_xor(dot, 1, WARP_SIZE);
            KQ_val0 = dot;
        }

        if (in_bounds1) {
            const int4 *K_I4_row = (const int4 *)(K_ptr + k_idx1 * nb11);
            const int base_i4 = lane_in_pair * 8;

            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int4 k_val_i4 = K_I4_row[base_i4 + i];
                half2 *k_h2 = (half2 *)&k_val_i4;

                // Process 4 half2s
                dot = dot2_f32_f16(Q_h2[i * 4 + 0], k_h2[0], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 1], k_h2[1], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 2], k_h2[2], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 3], k_h2[3], dot);
            }

            // Reduce across pair (lane 0 and 1)
            dot += __shfl_xor(dot, 1, WARP_SIZE);
            KQ_val1 = dot;
        }
        if (lane_in_pair == 0) {
            KQ_smem[token_idx] = in_bounds0 ? KQ_val0 : -FLT_MAX;
            KQ_smem[token_idx + TOKENS_PER_ITER] = in_bounds1 ? KQ_val1 : -FLT_MAX;
        }
        __syncthreads();

        // ---------------------------------------------------------------------
        // 2. Find tile max
        // ---------------------------------------------------------------------
        float tile_max = -FLT_MAX;
        for (int i = tid; i < TOKENS_PER_ITER_DECODE; i += NTHREADS) {
            tile_max = fmaxf(tile_max, KQ_smem[i]);
        }
        tile_max = warp_reduce_max(tile_max);

        if (lane_id == 0)
            reduce_smem[warp_id] = tile_max;
        __syncthreads();

        if (tid == 0) {
            float m = reduce_smem[0];
            for (int w = 1; w < NWARPS; ++w)
                m = fmaxf(m, reduce_smem[w]);
            reduce_smem[0] = m + FATTN_KQ_MAX_OFFSET;
        }
        __syncthreads();
        float block_max = reduce_smem[0];

        // ---------------------------------------------------------------------
        // 3. Rescale previous accumulator
        // ---------------------------------------------------------------------
        float scale_prev = __expf(KQ_max - block_max);
        KQ_max = block_max;
        KQ_sum *= scale_prev;

        half2 scale_h2 = __float2half2_rn(scale_prev);
#pragma unroll
        for (int i = 0; i < H2_PER_THREAD; ++i) {
            VKQ[i] = __hmul2(VKQ[i], scale_h2);
        }

        // ---------------------------------------------------------------------
        // 4. Compute softmax and accumulate V
        // ---------------------------------------------------------------------
        float my_score0 = KQ_smem[token_idx];
        float my_score1 = KQ_smem[token_idx + TOKENS_PER_ITER];
        float KQ_exp0 = in_bounds0 ? __expf(my_score0 + FATTN_KQ_MAX_OFFSET - KQ_max) : 0.0f;
        float KQ_exp1 = in_bounds1 ? __expf(my_score1 + FATTN_KQ_MAX_OFFSET - KQ_max) : 0.0f;

        // Sum reduction (only lane 0 of each pair contributes)
        float tile_sum = (lane_in_pair == 0) ? (KQ_exp0 + KQ_exp1) : 0.0f;
        tile_sum = warp_reduce_sum(tile_sum);

        if (lane_id == 0)
            reduce_smem[warp_id] = tile_sum;
        __syncthreads();

        if (tid == 0) {
            float s = 0.0f;
            for (int w = 0; w < NWARPS; ++w)
                s += reduce_smem[w];
            reduce_smem[0] = s;
        }
        __syncthreads();
        KQ_sum += reduce_smem[0];

        // Accumulate V (FP16 accumulation) using 128-bit loads
        if (in_bounds0) {
            const int4 *V_I4_row = (const int4 *)(V_ptr + k_idx0 * nb21);
            const int base_i4 = lane_in_pair * 8;

            half2 prob_h2 = __float2half2_rn(KQ_exp0);

#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int4 v_val_i4 = V_I4_row[base_i4 + i];
                half2 *v_h2 = (half2 *)&v_val_i4;

                VKQ[i * 4 + 0] = __hfma2(v_h2[0], prob_h2, VKQ[i * 4 + 0]);
                VKQ[i * 4 + 1] = __hfma2(v_h2[1], prob_h2, VKQ[i * 4 + 1]);
                VKQ[i * 4 + 2] = __hfma2(v_h2[2], prob_h2, VKQ[i * 4 + 2]);
                VKQ[i * 4 + 3] = __hfma2(v_h2[3], prob_h2, VKQ[i * 4 + 3]);
            }
        }
        if (in_bounds1) {
            const int4 *V_I4_row = (const int4 *)(V_ptr + k_idx1 * nb21);
            const int base_i4 = lane_in_pair * 8;

            half2 prob_h2 = __float2half2_rn(KQ_exp1);

#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int4 v_val_i4 = V_I4_row[base_i4 + i];
                half2 *v_h2 = (half2 *)&v_val_i4;

                VKQ[i * 4 + 0] = __hfma2(v_h2[0], prob_h2, VKQ[i * 4 + 0]);
                VKQ[i * 4 + 1] = __hfma2(v_h2[1], prob_h2, VKQ[i * 4 + 1]);
                VKQ[i * 4 + 2] = __hfma2(v_h2[2], prob_h2, VKQ[i * 4 + 2]);
                VKQ[i * 4 + 3] = __hfma2(v_h2[3], prob_h2, VKQ[i * 4 + 3]);
            }
        }
        __syncthreads();
    }

// =========================================================================
// Final: Reduce VKQ across token pairs using warp shuffles
// =========================================================================

// Step 1: Sum within warp (pair reduction)
#pragma unroll
    for (int i = 0; i < H2_PER_THREAD; ++i) {
        half2 val = VKQ[i];
        int v_int = *(int *)&val;
        int remote = __shfl_xor(v_int, 2, WARP_SIZE);
        val = __hadd2(val, *(half2 *)&remote);
        v_int = *(int *)&val;
        remote = __shfl_xor(v_int, 4, WARP_SIZE);
        val = __hadd2(val, *(half2 *)&remote);
        v_int = *(int *)&val;
        remote = __shfl_xor(v_int, 8, WARP_SIZE);
        val = __hadd2(val, *(half2 *)&remote);
        v_int = *(int *)&val;
        remote = __shfl_xor(v_int, 16, WARP_SIZE);
        val = __hadd2(val, *(half2 *)&remote);
        VKQ[i] = val;
    }

    // Step 2: Sum across warps
    __shared__ half2 VKQ_smem_h2[NWARPS][D_HEAD / 2]; // D_HEAD/2 because half2

    if (lane_id < 2) {
        const int base = lane_id * H2_PER_THREAD;
#pragma unroll
        for (int i = 0; i < H2_PER_THREAD; ++i) {
            VKQ_smem_h2[warp_id][base + i] = VKQ[i];
        }
    }
    __syncthreads();

    // Thread 0 sums across warps
    if (tid < D_HEAD / 2) {
        half2 val = __float2half2_rn(0.0f);
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) {
            val = __hadd2(val, VKQ_smem_h2[w][tid]);
        }

        if constexpr (WRITE_PARTIALS) {
            const float2 val_f = __half22float2(val);
            const int base = (blockIdx.z * gridDim.y + blockIdx.y) * D_HEAD;
            dst_partials[base + tid * 2 + 0] = val_f.x;
            dst_partials[base + tid * 2 + 1] = val_f.y;
        } else {
            float inv_sum = 1.0f / KQ_sum;
            half2 inv_sum_h2 = __float2half2_rn(inv_sum);
            val = __hmul2(val, inv_sum_h2);

            // Write output
            ((half2 *)(dst + blockIdx.z * D_HEAD))[tid] = val;
        }
    }

    if constexpr (WRITE_PARTIALS) {
        if (tid == 0) {
            dst_meta[blockIdx.z * gridDim.y + blockIdx.y] = make_float2(KQ_max, KQ_sum);
        }
    }
}

// Generic fallback for BF16 or other types
template <typename T, bool WRITE_PARTIALS>
__global__ __launch_bounds__(NTHREADS, 1) void flash_attn_vec_generic(const char *__restrict__ Q_base, const char *__restrict__ K_base,
                                                                      const char *__restrict__ V_base, T *__restrict__ dst,
                                                                      float *__restrict__ dst_partials, float2 *__restrict__ dst_meta,
                                                                      const float scale,
                                                                      const int ne11,                                 // seq_len_kv
                                                                      const int ne02,                                 // n_heads_Q
                                                                      const int ne12,                                 // n_heads_KV
                                                                      const int nb01, const int nb02, const int nb03, // Q strides
                                                                      const int nb11, const int nb12, const int nb13, // K strides
                                                                      const int nb21, const int nb22, const int nb23, // V strides
                                                                      const int batch_size) {

    // Thread indexing
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;

    // Token pair indexing: threads 0-1 handle token 0, threads 2-3 handle token 1, etc.
    const int token_idx = tid / NTHREADS_KQ;    // 0-63: which token in tile
    const int lane_in_pair = tid % NTHREADS_KQ; // 0 or 1: which half of dimensions

    // Sequence and head indexing
    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z % ne02;
    if (sequence >= batch_size)
        return;

    const int gqa_ratio = ne02 / ne12;
    const int head_kv = head / gqa_ratio;

    // Pointers
    const char *Q_ptr = Q_base + (int64_t)nb03 * sequence + (int64_t)nb02 * head;
    const char *K_ptr = K_base + (int64_t)nb13 * sequence + (int64_t)nb12 * head_kv;
    const char *V_ptr = V_base + (int64_t)nb23 * sequence + (int64_t)nb22 * head_kv;

    // =========================================================================
    // Load Q - each thread loads D_HEAD/2 = 64 elements (16 vec4s)
    // =========================================================================

    constexpr int D_PER_THREAD = D_HEAD / NTHREADS_KQ; // 64

    float Q_f[D_PER_THREAD];
    {
        const T *Q_T = (const T *)Q_ptr;
        const int base = lane_in_pair * D_PER_THREAD;

#pragma unroll
        for (int i = 0; i < D_PER_THREAD; i += 4) {
            load_vec4_as_float<T>(&Q_T[base + i], &Q_f[i]);
        }

#pragma unroll
        for (int i = 0; i < D_PER_THREAD; ++i) {
            Q_f[i] *= scale;
        }
    }

    // =========================================================================
    // VKQ accumulator - each thread accumulates D_HEAD/2 dimensions
    // =========================================================================

    float VKQ[D_PER_THREAD] = {0.0f};

    float KQ_max = -FLT_MAX / 2.0f;
    float KQ_sum = 0.0f;

    // Shared memory for KQ scores and cross-warp reduction
    __shared__ float KQ_smem[TOKENS_PER_ITER_DECODE]; // 128
    __shared__ float reduce_smem[NWARPS];      // 4

    // =========================================================================
    // Main loop over KV cache tiles
    // =========================================================================

    for (int k0 = blockIdx.y * TOKENS_PER_ITER_DECODE; k0 < ne11; k0 += gridDim.y * TOKENS_PER_ITER_DECODE) {
        const int k_idx0 = k0 + token_idx;           // Global token index (first)
        const int k_idx1 = k_idx0 + TOKENS_PER_ITER; // Global token index (second)

        // ---------------------------------------------------------------------
        // 1. Compute Q*K dot product
        // ---------------------------------------------------------------------
        float KQ_val0 = -FLT_MAX;
        float KQ_val1 = -FLT_MAX;
        const bool in_bounds0 = (k_idx0 < ne11);
        const bool in_bounds1 = (k_idx1 < ne11);

        if (in_bounds0) {
            const T *K_row = (const T *)(K_ptr + k_idx0 * nb11);
            const int base = lane_in_pair * D_PER_THREAD;

            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < D_PER_THREAD; i += 4) {
                float k_f[4];
                load_vec4_as_float<T>(&K_row[base + i], k_f);
                dot += Q_f[i + 0] * k_f[0];
                dot += Q_f[i + 1] * k_f[1];
                dot += Q_f[i + 2] * k_f[2];
                dot += Q_f[i + 3] * k_f[3];
            }

            // Reduce across pair (lane 0 and 1)
            dot += __shfl_xor(dot, 1, WARP_SIZE);
            KQ_val0 = dot;
        }

        if (in_bounds1) {
            const T *K_row = (const T *)(K_ptr + k_idx1 * nb11);
            const int base = lane_in_pair * D_PER_THREAD;

            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < D_PER_THREAD; i += 4) {
                float k_f[4];
                load_vec4_as_float<T>(&K_row[base + i], k_f);
                dot += Q_f[i + 0] * k_f[0];
                dot += Q_f[i + 1] * k_f[1];
                dot += Q_f[i + 2] * k_f[2];
                dot += Q_f[i + 3] * k_f[3];
            }

            // Reduce across pair (lane 0 and 1)
            dot += __shfl_xor(dot, 1, WARP_SIZE);
            KQ_val1 = dot;
        }
        // Store to shared memory (only lane 0 of each pair)
        if (lane_in_pair == 0) {
            KQ_smem[token_idx] = in_bounds0 ? KQ_val0 : -FLT_MAX;
            KQ_smem[token_idx + TOKENS_PER_ITER] = in_bounds1 ? KQ_val1 : -FLT_MAX;
        }
        __syncthreads();

        // ---------------------------------------------------------------------
        // 2. Find tile max (parallel reduction)
        // ---------------------------------------------------------------------
        float tile_max = -FLT_MAX;
        for (int i = tid; i < TOKENS_PER_ITER_DECODE; i += NTHREADS) {
            tile_max = fmaxf(tile_max, KQ_smem[i]);
        }
        tile_max = warp_reduce_max(tile_max);

        if (lane_id == 0)
            reduce_smem[warp_id] = tile_max;
        __syncthreads();

        if (tid == 0) {
            float m = reduce_smem[0];
            for (int w = 1; w < NWARPS; ++w)
                m = fmaxf(m, reduce_smem[w]);
            reduce_smem[0] = m + FATTN_KQ_MAX_OFFSET;
        }
        __syncthreads();
        float block_max = reduce_smem[0];

        // ---------------------------------------------------------------------
        // 3. Rescale previous accumulator
        // ---------------------------------------------------------------------
        float scale_prev = __expf(KQ_max - block_max);
        KQ_max = block_max;
        KQ_sum *= scale_prev;

#pragma unroll
        for (int i = 0; i < D_PER_THREAD; ++i) {
            VKQ[i] *= scale_prev;
        }

        // ---------------------------------------------------------------------
        // 4. Compute softmax and accumulate V
        // ---------------------------------------------------------------------
        float my_score0 = KQ_smem[token_idx];
        float my_score1 = KQ_smem[token_idx + TOKENS_PER_ITER];
        float KQ_exp0 = in_bounds0 ? __expf(my_score0 + FATTN_KQ_MAX_OFFSET - KQ_max) : 0.0f;
        float KQ_exp1 = in_bounds1 ? __expf(my_score1 + FATTN_KQ_MAX_OFFSET - KQ_max) : 0.0f;

        // Sum reduction (only lane 0 of each pair contributes)
        float tile_sum = (lane_in_pair == 0) ? (KQ_exp0 + KQ_exp1) : 0.0f;
        tile_sum = warp_reduce_sum(tile_sum);

        if (lane_id == 0)
            reduce_smem[warp_id] = tile_sum;
        __syncthreads();

        if (tid == 0) {
            float s = 0.0f;
            for (int w = 0; w < NWARPS; ++w)
                s += reduce_smem[w];
            reduce_smem[0] = s;
        }
        __syncthreads();
        KQ_sum += reduce_smem[0];

        // Accumulate V (each thread accumulates its D_PER_THREAD dimensions)
        if (in_bounds0) {
            const T *V_row = (const T *)(V_ptr + k_idx0 * nb21);
            const int base = lane_in_pair * D_PER_THREAD;

#pragma unroll
            for (int i = 0; i < D_PER_THREAD; i += 4) {
                float v_f[4];
                load_vec4_as_float<T>(&V_row[base + i], v_f);
                VKQ[i + 0] += v_f[0] * KQ_exp0;
                VKQ[i + 1] += v_f[1] * KQ_exp0;
                VKQ[i + 2] += v_f[2] * KQ_exp0;
                VKQ[i + 3] += v_f[3] * KQ_exp0;
            }
        }
        if (in_bounds1) {
            const T *V_row = (const T *)(V_ptr + k_idx1 * nb21);
            const int base = lane_in_pair * D_PER_THREAD;

#pragma unroll
            for (int i = 0; i < D_PER_THREAD; i += 4) {
                float v_f[4];
                load_vec4_as_float<T>(&V_row[base + i], v_f);
                VKQ[i + 0] += v_f[0] * KQ_exp1;
                VKQ[i + 1] += v_f[1] * KQ_exp1;
                VKQ[i + 2] += v_f[2] * KQ_exp1;
                VKQ[i + 3] += v_f[3] * KQ_exp1;
            }
        }
        __syncthreads();
    }

// =========================================================================
// Final: Reduce VKQ across token pairs using warp shuffles
// =========================================================================

// Step 1: Sum within warp (pair reduction)
#pragma unroll
    for (int i = 0; i < D_PER_THREAD; ++i) {
        // Sum within warp: stride by 2 to sum same lane_in_pair threads
        VKQ[i] += __shfl_xor(VKQ[i], 2, WARP_SIZE);
        VKQ[i] += __shfl_xor(VKQ[i], 4, WARP_SIZE);
        VKQ[i] += __shfl_xor(VKQ[i], 8, WARP_SIZE);
        VKQ[i] += __shfl_xor(VKQ[i], 16, WARP_SIZE);
    }

    // Step 2: Sum across warps using shared memory
    __shared__ float VKQ_smem[NWARPS][D_HEAD];

    // Thread 0 and 1 from each warp write to smem
    if (lane_id < 2) { // lane 0 = lower half, lane 1 = upper half
        const int base = lane_id * D_PER_THREAD;
#pragma unroll
        for (int i = 0; i < D_PER_THREAD; ++i) {
            VKQ_smem[warp_id][base + i] = VKQ[i];
        }
    }
    __syncthreads();

    // Thread 0 sums across warps and writes output
    if (tid < D_HEAD) {
        float val = 0.0f;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) {
            val += VKQ_smem[w][tid];
        }

        if constexpr (WRITE_PARTIALS) {
            const int base = (blockIdx.z * gridDim.y + blockIdx.y) * D_HEAD;
            dst_partials[base + tid] = val;
        } else {
            // Normalize
            float inv_sum = 1.0f / KQ_sum;

            // Write output
            T *dst_head = dst + blockIdx.z * D_HEAD;
            if constexpr (sizeof(T) == 2) {
                if (tid % 4 == 0) {
                    float out_f[4];
                    out_f[0] = val * inv_sum;
                    for (int j = 1; j < 4 && tid + j < D_HEAD; ++j) {
                        out_f[j] = 0.0f;
                        for (int w = 0; w < NWARPS; ++w) {
                            out_f[j] += VKQ_smem[w][tid + j];
                        }
                        out_f[j] *= inv_sum;
                    }
                    store_float_as_vec4<T>(&dst_head[tid], out_f);
                }
            }
        }
    }

    if constexpr (WRITE_PARTIALS) {
        if (tid == 0) {
            dst_meta[blockIdx.z * gridDim.y + blockIdx.y] = make_float2(KQ_max, KQ_sum);
        }
    }
}

// =============================================================================
// Flash Attention Prefill Kernels - Q length > 1 (no split-K by default)
// =============================================================================

template <bool WRITE_PARTIALS>
__global__ __launch_bounds__(NTHREADS_PREFILL, 1) void flash_attn_prefill_vec_f16(
    const char *__restrict__ Q_base, const char *__restrict__ K_base, const char *__restrict__ V_base, half *__restrict__ dst,
    float *__restrict__ dst_partials, float2 *__restrict__ dst_meta, const float scale, const int ne11, // seq_len_kv
    const int q_start,
    const int ne02,                                     // n_heads_Q
    const int ne12,                                     // n_heads_KV
    const int nb01, const int nb02, const int nb03,     // Q strides
    const int nb11, const int nb12, const int nb13,     // K strides
    const int nb21, const int nb22, const int nb23,     // V strides
    const int nbO1, const int nbO2, const int nbO3,     // O strides
    const int batch_size) {

    const int q_idx = blockIdx.x;
    const int effective_seq_len = min(ne11, q_start + q_idx + 1);

    // Thread indexing
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;

    // Token pair indexing
    const int token_idx = tid / NTHREADS_KQ;    // 0-127
    const int lane_in_pair = tid % NTHREADS_KQ; // 0 or 1

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z % ne02;
    if (sequence >= batch_size)
        return;

    const int gqa_ratio = ne02 / ne12;
    const int head_kv = head / gqa_ratio;

    const char *Q_ptr = Q_base + (int64_t)nb03 * sequence + (int64_t)nb02 * head + (int64_t)nb01 * q_idx;
    const char *K_ptr = K_base + (int64_t)nb13 * sequence + (int64_t)nb12 * head_kv;
    const char *V_ptr = V_base + (int64_t)nb23 * sequence + (int64_t)nb22 * head_kv;
    char *O_ptr = (char *)dst + (int64_t)nbO3 * sequence + (int64_t)nbO2 * head + (int64_t)nbO1 * q_idx;

    // =========================================================================
    // Load Q as half2 - using 128-bit loads (4x half2 per load)
    // =========================================================================

    constexpr int D_PER_THREAD = D_HEAD / NTHREADS_KQ; // 64 floats
    constexpr int H2_PER_THREAD = D_PER_THREAD / 2;    // 32 half2s

    half2 Q_h2[H2_PER_THREAD];
    {
        const int4 *Q_I4 = (const int4 *)Q_ptr;
        const int base_i4 = lane_in_pair * 8;

        half2 scale_h2 = __float2half2_rn(scale);

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int4 val_i4 = Q_I4[base_i4 + i];
            half2 *h2_ptr = (half2 *)&val_i4;

            Q_h2[i * 4 + 0] = __hmul2(h2_ptr[0], scale_h2);
            Q_h2[i * 4 + 1] = __hmul2(h2_ptr[1], scale_h2);
            Q_h2[i * 4 + 2] = __hmul2(h2_ptr[2], scale_h2);
            Q_h2[i * 4 + 3] = __hmul2(h2_ptr[3], scale_h2);
        }
    }

    // =========================================================================
    // VKQ accumulator - half2
    // =========================================================================

    half2 VKQ[H2_PER_THREAD];
#pragma unroll
    for (int i = 0; i < H2_PER_THREAD; ++i)
        VKQ[i] = __float2half2_rn(0.0f);

    float KQ_max = -FLT_MAX / 2.0f;
    float KQ_sum = 0.0f;

    __shared__ float KQ_smem[TOKENS_PER_ITER_PREFILL];
    __shared__ float reduce_smem[NWARPS_PREFILL];

    // =========================================================================
    // Main loop
    // =========================================================================

    for (int k0 = blockIdx.y * TOKENS_PER_ITER_PREFILL; k0 < effective_seq_len; k0 += gridDim.y * TOKENS_PER_ITER_PREFILL) {
        const int k_idx0 = k0 + token_idx;
        const int k_idx1 = k_idx0 + TOKENS_PER_ITER_PREFILL_HALF;

        float KQ_val0 = -FLT_MAX;
        float KQ_val1 = -FLT_MAX;
        const bool in_bounds0 = (k_idx0 < effective_seq_len);
        const bool in_bounds1 = (k_idx1 < effective_seq_len);

        if (in_bounds0) {
            const int4 *K_I4_row = (const int4 *)(K_ptr + k_idx0 * nb11);
            const int base_i4 = lane_in_pair * 8;

            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int4 k_val_i4 = K_I4_row[base_i4 + i];
                half2 *k_h2 = (half2 *)&k_val_i4;

                dot = dot2_f32_f16(Q_h2[i * 4 + 0], k_h2[0], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 1], k_h2[1], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 2], k_h2[2], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 3], k_h2[3], dot);
            }

            dot += __shfl_xor(dot, 1, WARP_SIZE);
            KQ_val0 = dot;
        }

        if (in_bounds1) {
            const int4 *K_I4_row = (const int4 *)(K_ptr + k_idx1 * nb11);
            const int base_i4 = lane_in_pair * 8;

            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int4 k_val_i4 = K_I4_row[base_i4 + i];
                half2 *k_h2 = (half2 *)&k_val_i4;

                dot = dot2_f32_f16(Q_h2[i * 4 + 0], k_h2[0], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 1], k_h2[1], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 2], k_h2[2], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 3], k_h2[3], dot);
            }

            dot += __shfl_xor(dot, 1, WARP_SIZE);
            KQ_val1 = dot;
        }

        if (lane_in_pair == 0) {
            KQ_smem[token_idx] = in_bounds0 ? KQ_val0 : -FLT_MAX;
            KQ_smem[token_idx + TOKENS_PER_ITER_PREFILL_HALF] = in_bounds1 ? KQ_val1 : -FLT_MAX;
        }
        __syncthreads();

        float tile_max = -FLT_MAX;
        for (int i = tid; i < TOKENS_PER_ITER_PREFILL; i += NTHREADS_PREFILL) {
            tile_max = fmaxf(tile_max, KQ_smem[i]);
        }
        tile_max = warp_reduce_max(tile_max);

        if (lane_id == 0)
            reduce_smem[warp_id] = tile_max;
        __syncthreads();

        if (tid == 0) {
            float m = reduce_smem[0];
            for (int w = 1; w < NWARPS_PREFILL; ++w)
                m = fmaxf(m, reduce_smem[w]);
            reduce_smem[0] = m + FATTN_KQ_MAX_OFFSET;
        }
        __syncthreads();
        float block_max = reduce_smem[0];

        float scale_prev = __expf(KQ_max - block_max);
        KQ_max = block_max;
        KQ_sum *= scale_prev;

        half2 scale_h2 = __float2half2_rn(scale_prev);
#pragma unroll
        for (int i = 0; i < H2_PER_THREAD; ++i) {
            VKQ[i] = __hmul2(VKQ[i], scale_h2);
        }

        float my_score0 = KQ_smem[token_idx];
        float my_score1 = KQ_smem[token_idx + TOKENS_PER_ITER_PREFILL_HALF];
        float KQ_exp0 = in_bounds0 ? __expf(my_score0 + FATTN_KQ_MAX_OFFSET - KQ_max) : 0.0f;
        float KQ_exp1 = in_bounds1 ? __expf(my_score1 + FATTN_KQ_MAX_OFFSET - KQ_max) : 0.0f;

        float tile_sum = (lane_in_pair == 0) ? (KQ_exp0 + KQ_exp1) : 0.0f;
        tile_sum = warp_reduce_sum(tile_sum);

        if (lane_id == 0)
            reduce_smem[warp_id] = tile_sum;
        __syncthreads();

        if (tid == 0) {
            float s = 0.0f;
            for (int w = 0; w < NWARPS_PREFILL; ++w)
                s += reduce_smem[w];
            reduce_smem[0] = s;
        }
        __syncthreads();
        KQ_sum += reduce_smem[0];

        if (in_bounds0) {
            const int4 *V_I4_row = (const int4 *)(V_ptr + k_idx0 * nb21);
            const int base_i4 = lane_in_pair * 8;
            half2 prob_h2 = __float2half2_rn(KQ_exp0);

#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int4 v_val_i4 = V_I4_row[base_i4 + i];
                half2 *v_h2 = (half2 *)&v_val_i4;

                VKQ[i * 4 + 0] = __hfma2(v_h2[0], prob_h2, VKQ[i * 4 + 0]);
                VKQ[i * 4 + 1] = __hfma2(v_h2[1], prob_h2, VKQ[i * 4 + 1]);
                VKQ[i * 4 + 2] = __hfma2(v_h2[2], prob_h2, VKQ[i * 4 + 2]);
                VKQ[i * 4 + 3] = __hfma2(v_h2[3], prob_h2, VKQ[i * 4 + 3]);
            }
        }

        if (in_bounds1) {
            const int4 *V_I4_row = (const int4 *)(V_ptr + k_idx1 * nb21);
            const int base_i4 = lane_in_pair * 8;
            half2 prob_h2 = __float2half2_rn(KQ_exp1);

#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int4 v_val_i4 = V_I4_row[base_i4 + i];
                half2 *v_h2 = (half2 *)&v_val_i4;

                VKQ[i * 4 + 0] = __hfma2(v_h2[0], prob_h2, VKQ[i * 4 + 0]);
                VKQ[i * 4 + 1] = __hfma2(v_h2[1], prob_h2, VKQ[i * 4 + 1]);
                VKQ[i * 4 + 2] = __hfma2(v_h2[2], prob_h2, VKQ[i * 4 + 2]);
                VKQ[i * 4 + 3] = __hfma2(v_h2[3], prob_h2, VKQ[i * 4 + 3]);
            }
        }
    }

    // Final reduction
#pragma unroll
    for (int i = 0; i < H2_PER_THREAD; ++i) {
        half2 val = VKQ[i];
        int v_int = *(int *)&val;
        int remote = __shfl_xor(v_int, 2, WARP_SIZE);
        val = __hadd2(val, *(half2 *)&remote);
        v_int = *(int *)&val;
        remote = __shfl_xor(v_int, 4, WARP_SIZE);
        val = __hadd2(val, *(half2 *)&remote);
        v_int = *(int *)&val;
        remote = __shfl_xor(v_int, 8, WARP_SIZE);
        val = __hadd2(val, *(half2 *)&remote);
        v_int = *(int *)&val;
        remote = __shfl_xor(v_int, 16, WARP_SIZE);
        val = __hadd2(val, *(half2 *)&remote);
        VKQ[i] = val;
    }

    __shared__ half2 VKQ_smem_h2[NWARPS_PREFILL][D_HEAD / 2];

    if (lane_id < 2) {
        const int base = lane_id * H2_PER_THREAD;
#pragma unroll
        for (int i = 0; i < H2_PER_THREAD; ++i) {
            VKQ_smem_h2[warp_id][base + i] = VKQ[i];
        }
    }
    __syncthreads();

    if (tid < D_HEAD / 2) {
        half2 val = __float2half2_rn(0.0f);
#pragma unroll
        for (int w = 0; w < NWARPS_PREFILL; ++w) {
            val = __hadd2(val, VKQ_smem_h2[w][tid]);
        }

        if constexpr (WRITE_PARTIALS) {
            const float2 val_f = __half22float2(val);
            const int base = ((blockIdx.z * gridDim.x + q_idx) * gridDim.y + blockIdx.y) * D_HEAD;
            dst_partials[base + tid * 2 + 0] = val_f.x;
            dst_partials[base + tid * 2 + 1] = val_f.y;
        } else {
            float inv_sum = 1.0f / KQ_sum;
            half2 inv_sum_h2 = __float2half2_rn(inv_sum);
            val = __hmul2(val, inv_sum_h2);

            ((half2 *)O_ptr)[tid] = val;
        }
    }

    if constexpr (WRITE_PARTIALS) {
        if (tid == 0) {
            dst_meta[(blockIdx.z * gridDim.x + q_idx) * gridDim.y + blockIdx.y] = make_float2(KQ_max, KQ_sum);
        }
    }
}

template <typename T, bool WRITE_PARTIALS>
__global__ __launch_bounds__(NTHREADS_PREFILL, 1) void flash_attn_prefill_vec_generic(
    const char *__restrict__ Q_base, const char *__restrict__ K_base, const char *__restrict__ V_base, T *__restrict__ dst,
    float *__restrict__ dst_partials, float2 *__restrict__ dst_meta, const float scale, const int ne11, // seq_len_kv
    const int q_start,
    const int ne02,                                     // n_heads_Q
    const int ne12,                                     // n_heads_KV
    const int nb01, const int nb02, const int nb03,     // Q strides
    const int nb11, const int nb12, const int nb13,     // K strides
    const int nb21, const int nb22, const int nb23,     // V strides
    const int nbO1, const int nbO2, const int nbO3,     // O strides
    const int batch_size) {

    const int q_idx = blockIdx.x;
    const int effective_seq_len = min(ne11, q_start + q_idx + 1);

    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;

    const int token_idx = tid / NTHREADS_KQ;
    const int lane_in_pair = tid % NTHREADS_KQ;

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z % ne02;
    if (sequence >= batch_size)
        return;

    const int gqa_ratio = ne02 / ne12;
    const int head_kv = head / gqa_ratio;

    const char *Q_ptr = Q_base + (int64_t)nb03 * sequence + (int64_t)nb02 * head + (int64_t)nb01 * q_idx;
    const char *K_ptr = K_base + (int64_t)nb13 * sequence + (int64_t)nb12 * head_kv;
    const char *V_ptr = V_base + (int64_t)nb23 * sequence + (int64_t)nb22 * head_kv;
    char *O_ptr = (char *)dst + (int64_t)nbO3 * sequence + (int64_t)nbO2 * head + (int64_t)nbO1 * q_idx;

    constexpr int D_PER_THREAD = D_HEAD / NTHREADS_KQ;

    float Q_f[D_PER_THREAD];
    {
        const T *Q_T = (const T *)Q_ptr;
        const int base = lane_in_pair * D_PER_THREAD;

#pragma unroll
        for (int i = 0; i < D_PER_THREAD; i += 4) {
            load_vec4_as_float<T>(&Q_T[base + i], &Q_f[i]);
        }

#pragma unroll
        for (int i = 0; i < D_PER_THREAD; ++i) {
            Q_f[i] *= scale;
        }
    }

    float VKQ[D_PER_THREAD] = {0.0f};

    float KQ_max = -FLT_MAX / 2.0f;
    float KQ_sum = 0.0f;

    __shared__ float KQ_smem[TOKENS_PER_ITER_PREFILL];
    __shared__ float reduce_smem[NWARPS_PREFILL];

    for (int k0 = blockIdx.y * TOKENS_PER_ITER_PREFILL; k0 < effective_seq_len; k0 += gridDim.y * TOKENS_PER_ITER_PREFILL) {
        const int k_idx0 = k0 + token_idx;
        const int k_idx1 = k_idx0 + TOKENS_PER_ITER_PREFILL_HALF;

        float KQ_val0 = -FLT_MAX;
        float KQ_val1 = -FLT_MAX;
        const bool in_bounds0 = (k_idx0 < effective_seq_len);
        const bool in_bounds1 = (k_idx1 < effective_seq_len);

        if (in_bounds0) {
            const T *K_row = (const T *)(K_ptr + k_idx0 * nb11);
            const int base = lane_in_pair * D_PER_THREAD;

            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < D_PER_THREAD; i += 4) {
                float k_f[4];
                load_vec4_as_float<T>(&K_row[base + i], k_f);
                dot += Q_f[i + 0] * k_f[0];
                dot += Q_f[i + 1] * k_f[1];
                dot += Q_f[i + 2] * k_f[2];
                dot += Q_f[i + 3] * k_f[3];
            }

            dot += __shfl_xor(dot, 1, WARP_SIZE);
            KQ_val0 = dot;
        }

        if (in_bounds1) {
            const T *K_row = (const T *)(K_ptr + k_idx1 * nb11);
            const int base = lane_in_pair * D_PER_THREAD;

            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < D_PER_THREAD; i += 4) {
                float k_f[4];
                load_vec4_as_float<T>(&K_row[base + i], k_f);
                dot += Q_f[i + 0] * k_f[0];
                dot += Q_f[i + 1] * k_f[1];
                dot += Q_f[i + 2] * k_f[2];
                dot += Q_f[i + 3] * k_f[3];
            }

            dot += __shfl_xor(dot, 1, WARP_SIZE);
            KQ_val1 = dot;
        }

        if (lane_in_pair == 0) {
            KQ_smem[token_idx] = in_bounds0 ? KQ_val0 : -FLT_MAX;
            KQ_smem[token_idx + TOKENS_PER_ITER_PREFILL_HALF] = in_bounds1 ? KQ_val1 : -FLT_MAX;
        }
        __syncthreads();

        float tile_max = -FLT_MAX;
        for (int i = tid; i < TOKENS_PER_ITER_PREFILL; i += NTHREADS_PREFILL) {
            tile_max = fmaxf(tile_max, KQ_smem[i]);
        }
        tile_max = warp_reduce_max(tile_max);

        if (lane_id == 0)
            reduce_smem[warp_id] = tile_max;
        __syncthreads();

        if (tid == 0) {
            float m = reduce_smem[0];
            for (int w = 1; w < NWARPS_PREFILL; ++w)
                m = fmaxf(m, reduce_smem[w]);
            reduce_smem[0] = m + FATTN_KQ_MAX_OFFSET;
        }
        __syncthreads();
        float block_max = reduce_smem[0];

        float scale_prev = __expf(KQ_max - block_max);
        KQ_max = block_max;
        KQ_sum *= scale_prev;

#pragma unroll
        for (int i = 0; i < D_PER_THREAD; ++i) {
            VKQ[i] *= scale_prev;
        }

        float my_score0 = KQ_smem[token_idx];
        float my_score1 = KQ_smem[token_idx + TOKENS_PER_ITER_PREFILL_HALF];
        float KQ_exp0 = in_bounds0 ? __expf(my_score0 + FATTN_KQ_MAX_OFFSET - KQ_max) : 0.0f;
        float KQ_exp1 = in_bounds1 ? __expf(my_score1 + FATTN_KQ_MAX_OFFSET - KQ_max) : 0.0f;

        float tile_sum = (lane_in_pair == 0) ? (KQ_exp0 + KQ_exp1) : 0.0f;
        tile_sum = warp_reduce_sum(tile_sum);

        if (lane_id == 0)
            reduce_smem[warp_id] = tile_sum;
        __syncthreads();

        if (tid == 0) {
            float s = 0.0f;
            for (int w = 0; w < NWARPS_PREFILL; ++w)
                s += reduce_smem[w];
            reduce_smem[0] = s;
        }
        __syncthreads();
        KQ_sum += reduce_smem[0];

        if (in_bounds0) {
            const T *V_row = (const T *)(V_ptr + k_idx0 * nb21);
            const int base = lane_in_pair * D_PER_THREAD;

#pragma unroll
            for (int i = 0; i < D_PER_THREAD; i += 4) {
                float v_f[4];
                load_vec4_as_float<T>(&V_row[base + i], v_f);
                VKQ[i + 0] += v_f[0] * KQ_exp0;
                VKQ[i + 1] += v_f[1] * KQ_exp0;
                VKQ[i + 2] += v_f[2] * KQ_exp0;
                VKQ[i + 3] += v_f[3] * KQ_exp0;
            }
        }

        if (in_bounds1) {
            const T *V_row = (const T *)(V_ptr + k_idx1 * nb21);
            const int base = lane_in_pair * D_PER_THREAD;

#pragma unroll
            for (int i = 0; i < D_PER_THREAD; i += 4) {
                float v_f[4];
                load_vec4_as_float<T>(&V_row[base + i], v_f);
                VKQ[i + 0] += v_f[0] * KQ_exp1;
                VKQ[i + 1] += v_f[1] * KQ_exp1;
                VKQ[i + 2] += v_f[2] * KQ_exp1;
                VKQ[i + 3] += v_f[3] * KQ_exp1;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < D_PER_THREAD; ++i) {
        VKQ[i] += __shfl_xor(VKQ[i], 2, WARP_SIZE);
        VKQ[i] += __shfl_xor(VKQ[i], 4, WARP_SIZE);
        VKQ[i] += __shfl_xor(VKQ[i], 8, WARP_SIZE);
        VKQ[i] += __shfl_xor(VKQ[i], 16, WARP_SIZE);
    }

    __shared__ float VKQ_smem[NWARPS_PREFILL][D_HEAD];

    if (lane_id < 2) {
        const int base = lane_id * D_PER_THREAD;
#pragma unroll
        for (int i = 0; i < D_PER_THREAD; ++i) {
            VKQ_smem[warp_id][base + i] = VKQ[i];
        }
    }
    __syncthreads();

    if (tid < D_HEAD) {
        float val = 0.0f;
#pragma unroll
        for (int w = 0; w < NWARPS_PREFILL; ++w) {
            val += VKQ_smem[w][tid];
        }

        if constexpr (WRITE_PARTIALS) {
            const int base = ((blockIdx.z * gridDim.x + q_idx) * gridDim.y + blockIdx.y) * D_HEAD;
            dst_partials[base + tid] = val;
        } else {
            float inv_sum = 1.0f / KQ_sum;

            T *dst_head = (T *)O_ptr;
            if constexpr (sizeof(T) == 2) {
                if (tid % 4 == 0) {
                    float out_f[4];
                    out_f[0] = val * inv_sum;
                    for (int j = 1; j < 4 && tid + j < D_HEAD; ++j) {
                        out_f[j] = 0.0f;
                        for (int w = 0; w < NWARPS_PREFILL; ++w) {
                            out_f[j] += VKQ_smem[w][tid + j];
                        }
                        out_f[j] *= inv_sum;
                    }
                    store_float_as_vec4<T>(&dst_head[tid], out_f);
                }
            }
        }
    }

    if constexpr (WRITE_PARTIALS) {
        if (tid == 0) {
            dst_meta[(blockIdx.z * gridDim.x + q_idx) * gridDim.y + blockIdx.y] = make_float2(KQ_max, KQ_sum);
        }
    }
}

template <typename T, bool WRITE_PARTIALS>
__global__ __launch_bounds__(NTHREADS_PREFILL, 1) void flash_attn_prefill_vec_generic_tile2(
    const char *__restrict__ Q_base, const char *__restrict__ K_base, const char *__restrict__ V_base, T *__restrict__ dst,
    float *__restrict__ dst_partials, float2 *__restrict__ dst_meta, const float scale, const int ne11, // seq_len_kv
    const int q_start, const int seq_len_q, const int ne02,                         // n_heads_Q
    const int ne12,                                                                  // n_heads_KV
    const int nb01, const int nb02, const int nb03,                                  // Q strides
    const int nb11, const int nb12, const int nb13,                                  // K strides
    const int nb21, const int nb22, const int nb23,                                  // V strides
    const int nbO1, const int nbO2, const int nbO3,                                  // O strides
    const int batch_size) {

    constexpr int D_PER_THREAD = D_HEAD / NTHREADS_KQ;
    constexpr int Q_TILE = Q_COLS_PREFILL_TILE;

    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int token_idx = tid / NTHREADS_KQ;
    const int lane_in_pair = tid % NTHREADS_KQ;

    const int q_tile_start = blockIdx.x * Q_TILE;
    const int q_idx0 = q_tile_start + 0;
    const int q_idx1 = q_tile_start + 1;
    const bool q_valid0 = q_idx0 < seq_len_q;
    const bool q_valid1 = q_idx1 < seq_len_q;

    const int effective_seq_len0 = q_valid0 ? min(ne11, q_start + q_idx0 + 1) : 0;
    const int effective_seq_len1 = q_valid1 ? min(ne11, q_start + q_idx1 + 1) : 0;

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z % ne02;
    if (sequence >= batch_size)
        return;

    const int gqa_ratio = ne02 / ne12;
    const int head_kv = head / gqa_ratio;

    const char *K_ptr = K_base + (int64_t)nb13 * sequence + (int64_t)nb12 * head_kv;
    const char *V_ptr = V_base + (int64_t)nb23 * sequence + (int64_t)nb22 * head_kv;
    const char *Q_ptr_base = Q_base + (int64_t)nb03 * sequence + (int64_t)nb02 * head;

    float Q0_f[D_PER_THREAD] = {0.0f};
    float Q1_f[D_PER_THREAD] = {0.0f};

    if (q_valid0) {
        const T *Q0 = (const T *)(Q_ptr_base + (int64_t)nb01 * q_idx0);
        const int base = lane_in_pair * D_PER_THREAD;
#pragma unroll
        for (int i = 0; i < D_PER_THREAD; i += 4) {
            load_vec4_as_float<T>(&Q0[base + i], &Q0_f[i]);
        }
#pragma unroll
        for (int i = 0; i < D_PER_THREAD; ++i) {
            Q0_f[i] *= scale;
        }
    }

    if (q_valid1) {
        const T *Q1 = (const T *)(Q_ptr_base + (int64_t)nb01 * q_idx1);
        const int base = lane_in_pair * D_PER_THREAD;
#pragma unroll
        for (int i = 0; i < D_PER_THREAD; i += 4) {
            load_vec4_as_float<T>(&Q1[base + i], &Q1_f[i]);
        }
#pragma unroll
        for (int i = 0; i < D_PER_THREAD; ++i) {
            Q1_f[i] *= scale;
        }
    }

    float VKQ0[D_PER_THREAD] = {0.0f};
    float VKQ1[D_PER_THREAD] = {0.0f};

    float KQ_max0 = -FLT_MAX / 2.0f;
    float KQ_max1 = -FLT_MAX / 2.0f;
    float KQ_sum0 = 0.0f;
    float KQ_sum1 = 0.0f;

    __shared__ float KQ_smem0[TOKENS_PER_ITER_PREFILL];
    __shared__ float KQ_smem1[TOKENS_PER_ITER_PREFILL];
    __shared__ float reduce_smem[NWARPS_PREFILL];

    for (int k0 = blockIdx.y * TOKENS_PER_ITER_PREFILL; k0 < ne11; k0 += gridDim.y * TOKENS_PER_ITER_PREFILL) {
        const int k_idx0 = k0 + token_idx;
        const int k_idx1 = k_idx0 + TOKENS_PER_ITER_PREFILL_HALF;

        const bool in00 = q_valid0 && (k_idx0 < effective_seq_len0);
        const bool in10 = q_valid0 && (k_idx1 < effective_seq_len0);
        const bool in01 = q_valid1 && (k_idx0 < effective_seq_len1);
        const bool in11 = q_valid1 && (k_idx1 < effective_seq_len1);

        float KQ00 = -FLT_MAX;
        float KQ10 = -FLT_MAX;
        float KQ01 = -FLT_MAX;
        float KQ11 = -FLT_MAX;

        if (in00 || in01) {
            const T *K_row = (const T *)(K_ptr + k_idx0 * nb11);
            const int base = lane_in_pair * D_PER_THREAD;
            float dot0 = 0.0f;
            float dot1 = 0.0f;
#pragma unroll
            for (int i = 0; i < D_PER_THREAD; i += 4) {
                float k_f[4];
                load_vec4_as_float<T>(&K_row[base + i], k_f);
                if (in00) {
                    dot0 += Q0_f[i + 0] * k_f[0];
                    dot0 += Q0_f[i + 1] * k_f[1];
                    dot0 += Q0_f[i + 2] * k_f[2];
                    dot0 += Q0_f[i + 3] * k_f[3];
                }
                if (in01) {
                    dot1 += Q1_f[i + 0] * k_f[0];
                    dot1 += Q1_f[i + 1] * k_f[1];
                    dot1 += Q1_f[i + 2] * k_f[2];
                    dot1 += Q1_f[i + 3] * k_f[3];
                }
            }
            if (in00) {
                dot0 += __shfl_xor(dot0, 1, WARP_SIZE);
                KQ00 = dot0;
            }
            if (in01) {
                dot1 += __shfl_xor(dot1, 1, WARP_SIZE);
                KQ01 = dot1;
            }
        }

        if (in10 || in11) {
            const T *K_row = (const T *)(K_ptr + k_idx1 * nb11);
            const int base = lane_in_pair * D_PER_THREAD;
            float dot0 = 0.0f;
            float dot1 = 0.0f;
#pragma unroll
            for (int i = 0; i < D_PER_THREAD; i += 4) {
                float k_f[4];
                load_vec4_as_float<T>(&K_row[base + i], k_f);
                if (in10) {
                    dot0 += Q0_f[i + 0] * k_f[0];
                    dot0 += Q0_f[i + 1] * k_f[1];
                    dot0 += Q0_f[i + 2] * k_f[2];
                    dot0 += Q0_f[i + 3] * k_f[3];
                }
                if (in11) {
                    dot1 += Q1_f[i + 0] * k_f[0];
                    dot1 += Q1_f[i + 1] * k_f[1];
                    dot1 += Q1_f[i + 2] * k_f[2];
                    dot1 += Q1_f[i + 3] * k_f[3];
                }
            }
            if (in10) {
                dot0 += __shfl_xor(dot0, 1, WARP_SIZE);
                KQ10 = dot0;
            }
            if (in11) {
                dot1 += __shfl_xor(dot1, 1, WARP_SIZE);
                KQ11 = dot1;
            }
        }

        if (lane_in_pair == 0) {
            KQ_smem0[token_idx] = in00 ? KQ00 : -FLT_MAX;
            KQ_smem0[token_idx + TOKENS_PER_ITER_PREFILL_HALF] = in10 ? KQ10 : -FLT_MAX;
            KQ_smem1[token_idx] = in01 ? KQ01 : -FLT_MAX;
            KQ_smem1[token_idx + TOKENS_PER_ITER_PREFILL_HALF] = in11 ? KQ11 : -FLT_MAX;
        }
        __syncthreads();

        float KQ_exp00 = 0.0f;
        float KQ_exp10 = 0.0f;
        float KQ_exp01 = 0.0f;
        float KQ_exp11 = 0.0f;

        if (q_valid0) {
            float tile_max = -FLT_MAX;
            for (int i = tid; i < TOKENS_PER_ITER_PREFILL; i += NTHREADS_PREFILL) {
                tile_max = fmaxf(tile_max, KQ_smem0[i]);
            }
            tile_max = warp_reduce_max(tile_max);

            if (lane_id == 0)
                reduce_smem[warp_id] = tile_max;
            __syncthreads();

            if (tid == 0) {
                float m = reduce_smem[0];
                for (int w = 1; w < NWARPS_PREFILL; ++w)
                    m = fmaxf(m, reduce_smem[w]);
                reduce_smem[0] = m + FATTN_KQ_MAX_OFFSET;
            }
            __syncthreads();

            float block_max = reduce_smem[0];
            float scale_prev = __expf(KQ_max0 - block_max);
            KQ_max0 = block_max;
            KQ_sum0 *= scale_prev;
#pragma unroll
            for (int i = 0; i < D_PER_THREAD; ++i) {
                VKQ0[i] *= scale_prev;
            }

            KQ_exp00 = in00 ? __expf(KQ_smem0[token_idx] + FATTN_KQ_MAX_OFFSET - KQ_max0) : 0.0f;
            KQ_exp10 = in10 ? __expf(KQ_smem0[token_idx + TOKENS_PER_ITER_PREFILL_HALF] + FATTN_KQ_MAX_OFFSET - KQ_max0) : 0.0f;

            float tile_sum = (lane_in_pair == 0) ? (KQ_exp00 + KQ_exp10) : 0.0f;
            tile_sum = warp_reduce_sum(tile_sum);
            if (lane_id == 0)
                reduce_smem[warp_id] = tile_sum;
            __syncthreads();

            if (tid == 0) {
                float s = 0.0f;
                for (int w = 0; w < NWARPS_PREFILL; ++w)
                    s += reduce_smem[w];
                reduce_smem[0] = s;
            }
            __syncthreads();
            KQ_sum0 += reduce_smem[0];
        } else {
            __syncthreads();
            __syncthreads();
            __syncthreads();
            __syncthreads();
        }

        if (q_valid1) {
            float tile_max = -FLT_MAX;
            for (int i = tid; i < TOKENS_PER_ITER_PREFILL; i += NTHREADS_PREFILL) {
                tile_max = fmaxf(tile_max, KQ_smem1[i]);
            }
            tile_max = warp_reduce_max(tile_max);

            if (lane_id == 0)
                reduce_smem[warp_id] = tile_max;
            __syncthreads();

            if (tid == 0) {
                float m = reduce_smem[0];
                for (int w = 1; w < NWARPS_PREFILL; ++w)
                    m = fmaxf(m, reduce_smem[w]);
                reduce_smem[0] = m + FATTN_KQ_MAX_OFFSET;
            }
            __syncthreads();

            float block_max = reduce_smem[0];
            float scale_prev = __expf(KQ_max1 - block_max);
            KQ_max1 = block_max;
            KQ_sum1 *= scale_prev;
#pragma unroll
            for (int i = 0; i < D_PER_THREAD; ++i) {
                VKQ1[i] *= scale_prev;
            }

            KQ_exp01 = in01 ? __expf(KQ_smem1[token_idx] + FATTN_KQ_MAX_OFFSET - KQ_max1) : 0.0f;
            KQ_exp11 = in11 ? __expf(KQ_smem1[token_idx + TOKENS_PER_ITER_PREFILL_HALF] + FATTN_KQ_MAX_OFFSET - KQ_max1) : 0.0f;

            float tile_sum = (lane_in_pair == 0) ? (KQ_exp01 + KQ_exp11) : 0.0f;
            tile_sum = warp_reduce_sum(tile_sum);
            if (lane_id == 0)
                reduce_smem[warp_id] = tile_sum;
            __syncthreads();

            if (tid == 0) {
                float s = 0.0f;
                for (int w = 0; w < NWARPS_PREFILL; ++w)
                    s += reduce_smem[w];
                reduce_smem[0] = s;
            }
            __syncthreads();
            KQ_sum1 += reduce_smem[0];
        } else {
            __syncthreads();
            __syncthreads();
            __syncthreads();
            __syncthreads();
        }

        if ((in00 && q_valid0) || (in01 && q_valid1)) {
            const T *V_row = (const T *)(V_ptr + k_idx0 * nb21);
            const int base = lane_in_pair * D_PER_THREAD;
#pragma unroll
            for (int i = 0; i < D_PER_THREAD; i += 4) {
                float v_f[4];
                load_vec4_as_float<T>(&V_row[base + i], v_f);
                if (in00 && q_valid0) {
                    VKQ0[i + 0] += v_f[0] * KQ_exp00;
                    VKQ0[i + 1] += v_f[1] * KQ_exp00;
                    VKQ0[i + 2] += v_f[2] * KQ_exp00;
                    VKQ0[i + 3] += v_f[3] * KQ_exp00;
                }
                if (in01 && q_valid1) {
                    VKQ1[i + 0] += v_f[0] * KQ_exp01;
                    VKQ1[i + 1] += v_f[1] * KQ_exp01;
                    VKQ1[i + 2] += v_f[2] * KQ_exp01;
                    VKQ1[i + 3] += v_f[3] * KQ_exp01;
                }
            }
        }

        if ((in10 && q_valid0) || (in11 && q_valid1)) {
            const T *V_row = (const T *)(V_ptr + k_idx1 * nb21);
            const int base = lane_in_pair * D_PER_THREAD;
#pragma unroll
            for (int i = 0; i < D_PER_THREAD; i += 4) {
                float v_f[4];
                load_vec4_as_float<T>(&V_row[base + i], v_f);
                if (in10 && q_valid0) {
                    VKQ0[i + 0] += v_f[0] * KQ_exp10;
                    VKQ0[i + 1] += v_f[1] * KQ_exp10;
                    VKQ0[i + 2] += v_f[2] * KQ_exp10;
                    VKQ0[i + 3] += v_f[3] * KQ_exp10;
                }
                if (in11 && q_valid1) {
                    VKQ1[i + 0] += v_f[0] * KQ_exp11;
                    VKQ1[i + 1] += v_f[1] * KQ_exp11;
                    VKQ1[i + 2] += v_f[2] * KQ_exp11;
                    VKQ1[i + 3] += v_f[3] * KQ_exp11;
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < D_PER_THREAD; ++i) {
        VKQ0[i] += __shfl_xor(VKQ0[i], 2, WARP_SIZE);
        VKQ0[i] += __shfl_xor(VKQ0[i], 4, WARP_SIZE);
        VKQ0[i] += __shfl_xor(VKQ0[i], 8, WARP_SIZE);
        VKQ0[i] += __shfl_xor(VKQ0[i], 16, WARP_SIZE);
        VKQ1[i] += __shfl_xor(VKQ1[i], 2, WARP_SIZE);
        VKQ1[i] += __shfl_xor(VKQ1[i], 4, WARP_SIZE);
        VKQ1[i] += __shfl_xor(VKQ1[i], 8, WARP_SIZE);
        VKQ1[i] += __shfl_xor(VKQ1[i], 16, WARP_SIZE);
    }

    __shared__ float VKQ_smem[NWARPS_PREFILL][D_HEAD];

    for (int q_local = 0; q_local < Q_TILE; ++q_local) {
        const bool q_valid = (q_local == 0) ? q_valid0 : q_valid1;
        const int q_idx_global = q_tile_start + q_local;
        float *VKQ_cur = (q_local == 0) ? VKQ0 : VKQ1;
        const float KQ_sum_cur = (q_local == 0) ? KQ_sum0 : KQ_sum1;
        const float KQ_max_cur = (q_local == 0) ? KQ_max0 : KQ_max1;

        if (lane_id < 2) {
            const int base = lane_id * D_PER_THREAD;
#pragma unroll
            for (int i = 0; i < D_PER_THREAD; ++i) {
                VKQ_smem[warp_id][base + i] = q_valid ? VKQ_cur[i] : 0.0f;
            }
        }
        __syncthreads();

        if (q_valid && tid < D_HEAD) {
            float val = 0.0f;
#pragma unroll
            for (int w = 0; w < NWARPS_PREFILL; ++w) {
                val += VKQ_smem[w][tid];
            }

            if constexpr (WRITE_PARTIALS) {
                const int base = ((blockIdx.z * seq_len_q + q_idx_global) * gridDim.y + blockIdx.y) * D_HEAD;
                dst_partials[base + tid] = val;
            } else {
                float inv_sum = 1.0f / KQ_sum_cur;
                T *dst_head = (T *)((char *)dst + (int64_t)nbO3 * sequence + (int64_t)nbO2 * head + (int64_t)nbO1 * q_idx_global);

                if constexpr (sizeof(T) == 2) {
                    if (tid % 4 == 0) {
                        float out_f[4];
                        out_f[0] = val * inv_sum;
                        for (int j = 1; j < 4 && tid + j < D_HEAD; ++j) {
                            out_f[j] = 0.0f;
                            for (int w = 0; w < NWARPS_PREFILL; ++w) {
                                out_f[j] += VKQ_smem[w][tid + j];
                            }
                            out_f[j] *= inv_sum;
                        }
                        store_float_as_vec4<T>(&dst_head[tid], out_f);
                    }
                }
            }
        }

        if constexpr (WRITE_PARTIALS) {
            if (q_valid && tid == 0) {
                dst_meta[(blockIdx.z * seq_len_q + q_idx_global) * gridDim.y + blockIdx.y] = make_float2(KQ_max_cur, KQ_sum_cur);
            }
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void flash_attn_decode_combine(const float *__restrict__ partials, const float2 *__restrict__ meta, T *__restrict__ dst,
                                          int parallel_blocks) {
    const int tid = threadIdx.x;
    if (tid >= D_HEAD)
        return;

    const int head_idx = blockIdx.x;
    const int base = head_idx * parallel_blocks;

    float kqmax = meta[base].x;
    for (int i = 1; i < parallel_blocks; ++i) {
        kqmax = fmaxf(kqmax, meta[base + i].x);
    }

    float numerator = 0.0f;
    float denominator = 0.0f;
    for (int i = 0; i < parallel_blocks; ++i) {
        const float2 m = meta[base + i];
        const float scale = __expf(m.x - kqmax);
        numerator += scale * partials[(base + i) * D_HEAD + tid];
        denominator += scale * m.y;
    }

    const float out = denominator > 0.0f ? (numerator / denominator) : 0.0f;
    dst[head_idx * D_HEAD + tid] = convert_out<T>(out);
}

template <typename T>
__global__ void flash_attn_prefill_combine(const float *__restrict__ partials, const float2 *__restrict__ meta, T *__restrict__ dst,
                                           int parallel_blocks, int seq_len_q, int n_heads_Q, int stride_O_seq, int stride_O_head,
                                           int stride_O_batch) {
    const int tid = threadIdx.x;
    if (tid >= D_HEAD) {
        return;
    }

    const int q_idx = blockIdx.x;
    const int head_idx = blockIdx.y; // flattened [batch, n_heads_Q]
    const int sequence = head_idx / n_heads_Q;
    const int head = head_idx % n_heads_Q;

    const int base = (head_idx * seq_len_q + q_idx) * parallel_blocks;

    float kqmax = meta[base].x;
    for (int i = 1; i < parallel_blocks; ++i) {
        kqmax = fmaxf(kqmax, meta[base + i].x);
    }

    float numerator = 0.0f;
    float denominator = 0.0f;
    for (int i = 0; i < parallel_blocks; ++i) {
        const float2 m = meta[base + i];
        const float scale = __expf(m.x - kqmax);
        numerator += scale * partials[(base + i) * D_HEAD + tid];
        denominator += scale * m.y;
    }

    const float out = denominator > 0.0f ? (numerator / denominator) : 0.0f;
    char *O_ptr = (char *)dst + (int64_t)stride_O_batch * sequence + (int64_t)stride_O_head * head + (int64_t)stride_O_seq * q_idx;
    ((T *)O_ptr)[tid] = convert_out<T>(out);
}

// =============================================================================
// Launcher
// =============================================================================

extern "C" void launch_flash_attn_decode_hip(const void *Q, const void *K, const void *V, const void *mask, void *dst, int batch_size,
                                             int n_heads_Q, int n_heads_KV, int head_dim, int seq_len_kv, float scale, int stride_Q_seq,
                                             int stride_Q_head, int stride_Q_batch, int stride_K_seq, int stride_K_head, int stride_K_batch,
                                             int stride_V_seq, int stride_V_head, int stride_V_batch, int stride_mask_seq, bool is_bf16,
                                             hipStream_t stream) {
    dim3 block(WARP_SIZE, NWARPS); // 2D: (32, 4) = 128 threads
    size_t smem_size = 0;          // kernel uses static shared memory only

    const int num_k_tiles = (seq_len_kv + TOKENS_PER_ITER_DECODE - 1) / TOKENS_PER_ITER_DECODE;
    int parallel_blocks = 1;

    if (num_k_tiles > 1) {
        int max_blocks_per_sm = 1;
        if (is_bf16) {
            HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm, flash_attn_vec_generic<__hip_bfloat16, true>, block.x * block.y * block.z, smem_size));
        } else {
            HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm, flash_attn_vec_f16<true>, block.x * block.y * block.z, smem_size));
        }

        int device = 0;
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDevice(&device));
        HIP_CHECK(hipGetDeviceProperties(&props, device));

        const int blocks_per_head = batch_size * n_heads_Q;
        const int target_blocks = props.multiProcessorCount * max_blocks_per_sm;
        const int desired_parallel = std::max(1, (target_blocks + blocks_per_head - 1) / blocks_per_head);

        int min_parallel = 1;
        if (num_k_tiles >= 32) {
            min_parallel = 8;
        } else if (num_k_tiles >= 16) {
            min_parallel = 4;
        } else if (num_k_tiles >= 8) {
            min_parallel = 2;
        }

        const int kMaxParallelBlocks = 32;
        parallel_blocks = std::min(num_k_tiles, std::min(std::max(desired_parallel, min_parallel), kMaxParallelBlocks));
    }

    if (is_bf16) {
        // For BF16, small sequences run best without split-K.
        // For long sequences, a light split-K (2) improves throughput.
        const int bf16_split_k_threshold = 32 * TOKENS_PER_ITER; // 2048 tokens
        if (seq_len_kv >= bf16_split_k_threshold) {
            parallel_blocks = std::min(parallel_blocks, 2);
            if (parallel_blocks < 2) {
                parallel_blocks = 2;
            }
        } else {
            parallel_blocks = 1;
        }
    }

    const char *parallel_override = std::getenv("FLASH_ATTN_DECODE_PARALLEL_BLOCKS");
    if (parallel_override && std::atoi(parallel_override) > 0) {
        const int override_val = std::atoi(parallel_override);
        parallel_blocks = std::min(num_k_tiles, override_val);
    }

    const char *disable_parallel = std::getenv("FLASH_ATTN_DECODE_DISABLE_PARALLEL");
    if (disable_parallel && std::atoi(disable_parallel) != 0) {
        parallel_blocks = 1;
    }

    if (parallel_blocks <= 1) {
        dim3 grid(1, 1, batch_size * n_heads_Q);
        if (is_bf16) {
            flash_attn_vec_generic<__hip_bfloat16, false>
                <<<grid, block, smem_size, stream>>>((const char *)Q, (const char *)K, (const char *)V, (__hip_bfloat16 *)dst, nullptr,
                                                     nullptr, scale, seq_len_kv, n_heads_Q, n_heads_KV, stride_Q_seq, stride_Q_head,
                                                     stride_Q_batch, stride_K_seq, stride_K_head, stride_K_batch, stride_V_seq,
                                                     stride_V_head, stride_V_batch, batch_size);
        } else {
            // Use optimized FP16 kernel
            flash_attn_vec_f16<false><<<grid, block, smem_size, stream>>>(
                (const char *)Q, (const char *)K, (const char *)V, (half *)dst, nullptr, nullptr, scale, seq_len_kv, n_heads_Q, n_heads_KV,
                stride_Q_seq, stride_Q_head, stride_Q_batch, stride_K_seq, stride_K_head, stride_K_batch, stride_V_seq, stride_V_head,
                stride_V_batch, batch_size);
        }
        return;
    }

    // Parallel blocks path for long KV: compute partials + meta, then combine.
    const size_t partials_bytes = static_cast<size_t>(batch_size) * n_heads_Q * parallel_blocks * D_HEAD * sizeof(float);
    const size_t meta_bytes = static_cast<size_t>(batch_size) * n_heads_Q * parallel_blocks * sizeof(float2);

    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    FlashAttnDecodeWorkspace ws = acquire_flash_attn_decode_workspace(device, partials_bytes, meta_bytes);
    float *partials = static_cast<float *>(ws.partials);
    float2 *meta = static_cast<float2 *>(ws.meta);

    dim3 grid(1, parallel_blocks, batch_size * n_heads_Q);
    if (is_bf16) {
        flash_attn_vec_generic<__hip_bfloat16, true>
            <<<grid, block, smem_size, stream>>>((const char *)Q, (const char *)K, (const char *)V, (__hip_bfloat16 *)dst, partials, meta,
                                                 scale, seq_len_kv, n_heads_Q, n_heads_KV, stride_Q_seq, stride_Q_head, stride_Q_batch,
                                                 stride_K_seq, stride_K_head, stride_K_batch, stride_V_seq, stride_V_head, stride_V_batch,
                                                 batch_size);
        dim3 combine_grid(batch_size * n_heads_Q, 1, 1);
        dim3 combine_block(D_HEAD, 1, 1);
        flash_attn_decode_combine<__hip_bfloat16><<<combine_grid, combine_block, 0, stream>>>(partials, meta,
                                                                                             (__hip_bfloat16 *)dst, parallel_blocks);
    } else {
        flash_attn_vec_f16<true><<<grid, block, smem_size, stream>>>(
            (const char *)Q, (const char *)K, (const char *)V, (half *)dst, partials, meta, scale, seq_len_kv, n_heads_Q, n_heads_KV,
            stride_Q_seq, stride_Q_head, stride_Q_batch, stride_K_seq, stride_K_head, stride_K_batch, stride_V_seq, stride_V_head,
            stride_V_batch, batch_size);
        dim3 combine_grid(batch_size * n_heads_Q, 1, 1);
        dim3 combine_block(D_HEAD, 1, 1);
        flash_attn_decode_combine<half><<<combine_grid, combine_block, 0, stream>>>(partials, meta, (half *)dst, parallel_blocks);
    }
}

template <typename T>
__global__ void flash_attn_decode_hd256_kernel(const char *__restrict__ Q_base, const char *__restrict__ K_base,
                                                const char *__restrict__ V_base, const char *__restrict__ mask_base,
                                                T *__restrict__ dst, float scale, int seq_len_kv, int n_heads_Q, int n_heads_KV,
                                                int stride_Q_seq, int stride_Q_head, int stride_Q_batch, int stride_K_seq, int stride_K_head,
                                                int stride_K_batch, int stride_V_seq, int stride_V_head, int stride_V_batch,
                                                int stride_mask_seq, int batch_size) {
    constexpr int D_HEAD_HD256 = 256;
    constexpr int PAIR_WIDTH = 2;
    constexpr int PAIRS_PER_HEAD = D_HEAD_HD256 / PAIR_WIDTH;
    const int tid = threadIdx.x;
    const int head_idx = blockIdx.x;
    const int sequence = head_idx / n_heads_Q;
    const int head = head_idx % n_heads_Q;

    if (sequence >= batch_size || tid >= PAIRS_PER_HEAD) {
        return;
    }

    const int gqa_ratio = (n_heads_KV > 0) ? ((n_heads_Q / n_heads_KV) > 0 ? (n_heads_Q / n_heads_KV) : 1) : 1;
    const int head_kv = head / gqa_ratio;

    const char *Q_ptr = Q_base + (int64_t)stride_Q_batch * sequence + (int64_t)stride_Q_head * head;
    const char *K_ptr = K_base + (int64_t)stride_K_batch * sequence + (int64_t)stride_K_head * head_kv;
    const char *V_ptr = V_base + (int64_t)stride_V_batch * sequence + (int64_t)stride_V_head * head_kv;

    __shared__ float inv_denom;
    extern __shared__ float scores[];

    const int64_t pair_byte_offset = static_cast<int64_t>(tid * PAIR_WIDTH) * static_cast<int64_t>(sizeof(T));
    float2 q_pair = load_pair_as_float2<T>(Q_ptr + pair_byte_offset);
    q_pair.x *= scale;
    q_pair.y *= scale;

    for (int kv = 0; kv < seq_len_kv; ++kv) {
        const char *k_token_ptr = K_ptr + (int64_t)stride_K_seq * kv;
        float2 k_pair = load_pair_as_float2<T>(k_token_ptr + pair_byte_offset);
        float dot_partial = q_pair.x * k_pair.x + q_pair.y * k_pair.y;
        float dot_sum = block_reduce_sum_128(dot_partial);

        if (tid == 0) {
            float score = dot_sum;
            if (mask_base != nullptr && stride_mask_seq > 0) {
                score += *((const float *)(mask_base + (int64_t)stride_mask_seq * kv));
            }
            scores[kv] = score;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float kq_max = -FLT_MAX;
        for (int kv = 0; kv < seq_len_kv; ++kv) {
            kq_max = fmaxf(kq_max, scores[kv]);
        }
        float denom = 0.0f;
        for (int kv = 0; kv < seq_len_kv; ++kv) {
            float w = __expf(scores[kv] - kq_max);
            scores[kv] = w;
            denom += w;
        }
        inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;
    }
    __syncthreads();

    float2 out_pair = make_float2(0.0f, 0.0f);
    for (int kv = 0; kv < seq_len_kv; ++kv) {
        const char *v_token_ptr = V_ptr + (int64_t)stride_V_seq * kv;
        const float2 v_pair = load_pair_as_float2<T>(v_token_ptr + pair_byte_offset);
        const float weight = scores[kv] * inv_denom;
        out_pair.x += weight * v_pair.x;
        out_pair.y += weight * v_pair.y;
    }

    char *dst_ptr = (char *)dst + ((((int64_t)sequence * n_heads_Q + head) * D_HEAD_HD256) + (int64_t)tid * PAIR_WIDTH) * sizeof(T);
    store_pair_from_float2<T>(dst_ptr, out_pair);
}

extern "C" void launch_flash_attn_decode_hip_hd256(const void *Q, const void *K, const void *V, const void *mask, void *dst,
                                                   int batch_size, int n_heads_Q, int n_heads_KV, int head_dim, int seq_len_kv, float scale,
                                                   int stride_Q_seq, int stride_Q_head, int stride_Q_batch, int stride_K_seq,
                                                   int stride_K_head, int stride_K_batch, int stride_V_seq, int stride_V_head,
                                                   int stride_V_batch, int stride_mask_seq, bool is_bf16, hipStream_t stream) {
    (void)stride_Q_seq;

    if (head_dim != 256) {
        std::cerr << "launch_flash_attn_decode_hip_hd256 requires head_dim=256, got " << head_dim << std::endl;
        return;
    }

    dim3 block(128, 1, 1);
    dim3 grid(batch_size * n_heads_Q, 1, 1);
    size_t smem_bytes = static_cast<size_t>(std::max(1, seq_len_kv)) * sizeof(float);

    if (is_bf16) {
        flash_attn_decode_hd256_kernel<__hip_bfloat16><<<grid, block, smem_bytes, stream>>>(
            (const char *)Q, (const char *)K, (const char *)V, (const char *)mask, (__hip_bfloat16 *)dst, scale, seq_len_kv, n_heads_Q,
            n_heads_KV, stride_Q_seq, stride_Q_head, stride_Q_batch, stride_K_seq, stride_K_head, stride_K_batch, stride_V_seq,
            stride_V_head, stride_V_batch, stride_mask_seq, batch_size);
    } else {
        flash_attn_decode_hd256_kernel<half><<<grid, block, smem_bytes, stream>>>(
            (const char *)Q, (const char *)K, (const char *)V, (const char *)mask, (half *)dst, scale, seq_len_kv, n_heads_Q, n_heads_KV,
            stride_Q_seq, stride_Q_head, stride_Q_batch, stride_K_seq, stride_K_head, stride_K_batch, stride_V_seq, stride_V_head,
            stride_V_batch, stride_mask_seq, batch_size);
    }
}

extern "C" void launch_flash_attn_prefill_hip(const void *Q, const void *K, const void *V, void *dst, int batch_size, int n_heads_Q,
                                              int n_heads_KV, int head_dim, int seq_len_q, int seq_len_kv, float scale, int stride_Q_seq,
                                              int stride_Q_head, int stride_Q_batch, int stride_K_seq, int stride_K_head, int stride_K_batch,
                                              int stride_V_seq, int stride_V_head, int stride_V_batch, int stride_O_seq, int stride_O_head,
                                              int stride_O_batch, int q_start, bool is_bf16, hipStream_t stream) {
    dim3 block(WARP_SIZE, NWARPS_PREFILL);
    size_t smem_size = 0;

    if (head_dim != D_HEAD) {
        std::cerr << "FlashAttn prefill only supports head_dim=" << D_HEAD << std::endl;
        return;
    }

    const int num_k_tiles = (seq_len_kv + TOKENS_PER_ITER_PREFILL - 1) / TOKENS_PER_ITER_PREFILL;
    int parallel_blocks = 1;

    (void)num_k_tiles;

    const char *parallel_override = std::getenv("FLASH_ATTN_PREFILL_PARALLEL_BLOCKS");
    if (parallel_override && std::atoi(parallel_override) > 0) {
        const int override_val = std::atoi(parallel_override);
        parallel_blocks = std::min(num_k_tiles, override_val);
    }

    const char *disable_parallel = std::getenv("FLASH_ATTN_PREFILL_DISABLE_PARALLEL");
    if (disable_parallel && std::atoi(disable_parallel) != 0) {
        parallel_blocks = 1;
    }

    const int prefill_grid_x = (seq_len_q + Q_COLS_PREFILL_TILE - 1) / Q_COLS_PREFILL_TILE;
    const char *tile2_env = std::getenv("FLASH_ATTN_PREFILL_TILE2");
    const bool enable_prefill_tile2 = (tile2_env != nullptr) && (std::atoi(tile2_env) != 0);

    if (parallel_blocks <= 1) {
        if (is_bf16) {
            if (enable_prefill_tile2 && seq_len_q >= Q_COLS_PREFILL_TILE) {
                dim3 grid(prefill_grid_x, 1, batch_size * n_heads_Q);
                flash_attn_prefill_vec_generic_tile2<__hip_bfloat16, false><<<grid, block, smem_size, stream>>>(
                    (const char *)Q, (const char *)K, (const char *)V, (__hip_bfloat16 *)dst, nullptr, nullptr, scale, seq_len_kv, q_start,
                    seq_len_q, n_heads_Q, n_heads_KV, stride_Q_seq, stride_Q_head, stride_Q_batch, stride_K_seq, stride_K_head,
                    stride_K_batch, stride_V_seq, stride_V_head, stride_V_batch, stride_O_seq, stride_O_head, stride_O_batch, batch_size);
            } else {
                dim3 grid(seq_len_q, 1, batch_size * n_heads_Q);
                flash_attn_prefill_vec_generic<__hip_bfloat16, false>
                    <<<grid, block, smem_size, stream>>>((const char *)Q, (const char *)K, (const char *)V, (__hip_bfloat16 *)dst, nullptr,
                                                         nullptr, scale, seq_len_kv, q_start, n_heads_Q, n_heads_KV, stride_Q_seq,
                                                         stride_Q_head, stride_Q_batch, stride_K_seq, stride_K_head, stride_K_batch,
                                                         stride_V_seq, stride_V_head, stride_V_batch, stride_O_seq, stride_O_head,
                                                         stride_O_batch, batch_size);
            }
        } else {
            dim3 grid(seq_len_q, 1, batch_size * n_heads_Q);
            flash_attn_prefill_vec_f16<false>
                <<<grid, block, smem_size, stream>>>((const char *)Q, (const char *)K, (const char *)V, (half *)dst, nullptr, nullptr,
                                                     scale, seq_len_kv, q_start, n_heads_Q, n_heads_KV, stride_Q_seq, stride_Q_head,
                                                     stride_Q_batch, stride_K_seq, stride_K_head, stride_K_batch, stride_V_seq, stride_V_head,
                                                     stride_V_batch, stride_O_seq, stride_O_head, stride_O_batch, batch_size);
        }
        return;
    }

    const size_t partials_bytes =
        static_cast<size_t>(batch_size) * n_heads_Q * seq_len_q * parallel_blocks * D_HEAD * sizeof(float);
    const size_t meta_bytes = static_cast<size_t>(batch_size) * n_heads_Q * seq_len_q * parallel_blocks * sizeof(float2);

    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    FlashAttnDecodeWorkspace ws = acquire_flash_attn_decode_workspace(device, partials_bytes, meta_bytes);
    float *partials = static_cast<float *>(ws.partials);
    float2 *meta = static_cast<float2 *>(ws.meta);

    dim3 combine_grid(seq_len_q, batch_size * n_heads_Q, 1);
    dim3 combine_block(D_HEAD, 1, 1);

    if (is_bf16) {
        if (enable_prefill_tile2 && seq_len_q >= Q_COLS_PREFILL_TILE) {
            dim3 grid(prefill_grid_x, parallel_blocks, batch_size * n_heads_Q);
            flash_attn_prefill_vec_generic_tile2<__hip_bfloat16, true><<<grid, block, smem_size, stream>>>(
                (const char *)Q, (const char *)K, (const char *)V, (__hip_bfloat16 *)dst, partials, meta, scale, seq_len_kv, q_start,
                seq_len_q, n_heads_Q, n_heads_KV, stride_Q_seq, stride_Q_head, stride_Q_batch, stride_K_seq, stride_K_head, stride_K_batch,
                stride_V_seq, stride_V_head, stride_V_batch, stride_O_seq, stride_O_head, stride_O_batch, batch_size);
        } else {
            dim3 grid(seq_len_q, parallel_blocks, batch_size * n_heads_Q);
            flash_attn_prefill_vec_generic<__hip_bfloat16, true>
                <<<grid, block, smem_size, stream>>>((const char *)Q, (const char *)K, (const char *)V, (__hip_bfloat16 *)dst, partials,
                                                     meta, scale, seq_len_kv, q_start, n_heads_Q, n_heads_KV, stride_Q_seq, stride_Q_head,
                                                     stride_Q_batch, stride_K_seq, stride_K_head, stride_K_batch, stride_V_seq, stride_V_head,
                                                     stride_V_batch, stride_O_seq, stride_O_head, stride_O_batch, batch_size);
        }
        flash_attn_prefill_combine<__hip_bfloat16><<<combine_grid, combine_block, 0, stream>>>(
            partials, meta, (__hip_bfloat16 *)dst, parallel_blocks, seq_len_q, n_heads_Q, stride_O_seq, stride_O_head, stride_O_batch);
    } else {
        dim3 grid(seq_len_q, parallel_blocks, batch_size * n_heads_Q);
        flash_attn_prefill_vec_f16<true>
            <<<grid, block, smem_size, stream>>>((const char *)Q, (const char *)K, (const char *)V, (half *)dst, partials, meta, scale,
                                                 seq_len_kv, q_start, n_heads_Q, n_heads_KV, stride_Q_seq, stride_Q_head, stride_Q_batch,
                                                 stride_K_seq, stride_K_head, stride_K_batch, stride_V_seq, stride_V_head, stride_V_batch,
                                                 stride_O_seq, stride_O_head, stride_O_batch, batch_size);
        flash_attn_prefill_combine<half><<<combine_grid, combine_block, 0, stream>>>(
            partials, meta, (half *)dst, parallel_blocks, seq_len_q, n_heads_Q, stride_O_seq, stride_O_head, stride_O_batch);
    }
}
