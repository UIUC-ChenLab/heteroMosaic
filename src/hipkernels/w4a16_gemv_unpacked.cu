#include "hipkernels/w4a16_gemv_unpacked.hpp"
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

using bfloat16_t = hip_bfloat16;

namespace hipkernels {

__device__ __forceinline__ float bf16_to_float_dev(bfloat16_t val) { return static_cast<float>(val); }
__device__ __forceinline__ bfloat16_t float_to_bf16_dev(float val) { return static_cast<bfloat16_t>(val); }

// Kernel: One Warp per Output (N). 8 Outputs per Block (Block 256).
// Optimization: v23 (Hybrid Scale Caching + NegZS + Branch).
template <int MAX_GROUPS>
__global__ void __launch_bounds__(256)
    w4a16_gemv_unpacked_kernel(bfloat16_t *__restrict__ output, const bfloat16_t *__restrict__ input,
                               const uint8_t *__restrict__ qweights,  // [N, K/2]
                               const bfloat16_t *__restrict__ scales, // [Groups, N_total] or [N_total, Groups]
                               const uint8_t *__restrict__ zeros,     // [Groups, N_total] or [N_total, Groups]
                               const int M,                           // 1
                               const int K, const int N, const int group_shift, const int stride_groups, const int stride_n) {
    const int bx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;

    const int global_n = bx * 8 + warp_id;
    bool active = (global_n < N);

    __shared__ __half2 s_input[16 * 257];
    __shared__ __half2 s_scales[MAX_GROUPS * 8];
    __shared__ __half2 s_neg_zs[MAX_GROUPS * 8];

    const int stride = 257;

    const uint4 *input_vecs = reinterpret_cast<const uint4 *>(input);
    const uint4 *w_row_vec = nullptr;
    if (active) {
        const uint8_t *w_base = &qweights[global_n * (K / 2)];
        w_row_vec = reinterpret_cast<const uint4 *>(w_base);
    }

    // Pre-load Scales/Zeros & Pre-compute NegZS
    const int num_groups = K >> group_shift;
    const int groups_to_cache = (num_groups < MAX_GROUPS) ? num_groups : MAX_GROUPS;

    if (active) {
        // Load scales/zeros for all groups (up to MAX_GROUPS cached)
        if constexpr (MAX_GROUPS == 128) {
#pragma unroll
            for (int iter = 0; iter < 4; ++iter) {
                int g_idx = lane_id + iter * 32;
                if (g_idx < groups_to_cache) {
                    int idx_s = g_idx * stride_groups + global_n * stride_n;
                    unsigned short s_raw = __ldg((const unsigned short *)&scales[idx_s]);
                    uint8_t z_raw = __ldg(&zeros[idx_s]);
                    float s_f = bf16_to_float_dev(*reinterpret_cast<bfloat16_t *>(&s_raw));
                    float z_f = (float)((int8_t)z_raw);

                    __half2 s2 = __half2half2(__float2half(s_f));
                    __half2 z2 = __half2half2(__float2half(z_f));

                    s_scales[warp_id * MAX_GROUPS + g_idx] = s2;
                    s_neg_zs[warp_id * MAX_GROUPS + g_idx] = __hneg2(__hmul2(z2, s2));
                }
            }
        } else {
            for (int g_idx = lane_id; g_idx < groups_to_cache; g_idx += 32) {
                int idx_s = g_idx * stride_groups + global_n * stride_n;
                unsigned short s_raw = __ldg((const unsigned short *)&scales[idx_s]);
                uint8_t z_raw = __ldg(&zeros[idx_s]);
                float s_f = bf16_to_float_dev(*reinterpret_cast<bfloat16_t *>(&s_raw));
                float z_f = (float)((int8_t)z_raw);

                __half2 s2 = __half2half2(__float2half(s_f));
                __half2 z2 = __half2half2(__float2half(z_f));

                s_scales[warp_id * MAX_GROUPS + g_idx] = s2;
                s_neg_zs[warp_id * MAX_GROUPS + g_idx] = __hneg2(__hmul2(z2, s2));
            }
        }
    }
    __syncthreads();

    __half2 acc0 = __float2half2_rn(0.0f);
    __half2 acc1 = __float2half2_rn(0.0f);
    __half2 acc2 = __float2half2_rn(0.0f);
    __half2 acc3 = __float2half2_rn(0.0f);

    int current_group = -1;
    __half2 s2 = __float2half2_rn(1.0f);
    __half2 neg_zs = __float2half2_rn(0.0f);

    const int TILE_SIZE = 8192;
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    int stagger_idx = (global_n & 31);
    int stagger_offset = (stagger_idx * 8);

    for (int t = 0; t < num_tiles; ++t) {
        int k_tile_base = t * TILE_SIZE;

        // 1. Vectorized Input Load (v18)
        int vec_base_k = k_tile_base / 8;
        int max_vec_k = K / 8;

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int vec_offset = i * 256 + tid;
            int vec_idx = vec_base_k + vec_offset;
            if (vec_idx < max_vec_k) {
                uint4 val4 = input_vecs[vec_idx];
                int *vals = reinterpret_cast<int *>(&val4);

                int base_pair = vec_offset * 4;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int pair_offset_rel = base_pair + j;
                    int val = vals[j];
                    bfloat16_t *v = reinterpret_cast<bfloat16_t *>(&val);
                    float f0 = bf16_to_float_dev(v[0]);
                    float f1 = bf16_to_float_dev(v[1]);
                    __half2 h2 = __float22half2_rn(make_float2(f0, f1));
                    int r_s = pair_offset_rel >> 4;
                    int c_s = pair_offset_rel & 15;
                    s_input[c_s * stride + r_s] = h2;
                }
            }
        }
        __syncthreads();

        if (active) {

            uint4 w_vals[8];
            int chunk_ids[8];

            // Prefetch Weights (with bounds check for smaller K)
            int max_vecs = (K / 2) / 16; // Total uint4 vectors per row
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int chunk_id_raw = i * 32 + lane_id;
                int chunk_id = (chunk_id_raw + stagger_offset) & 255;
                chunk_ids[i] = chunk_id;
                int vec_idx = t * 256 + chunk_id;
                if (vec_idx < max_vecs) {
                    w_vals[i] = w_row_vec[vec_idx];
                } else {
                    w_vals[i] = {0, 0, 0, 0}; // Zero out-of-bounds
                }
            }

// Math Loop
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int chunk_id = chunk_ids[i];
                uint4 w_val = w_vals[i];

                // Register Cache
                __half2 in_reg[16];
                int r = chunk_id;
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    in_reg[k] = s_input[k * stride + r];
                }

                int k_abs_start = k_tile_base + chunk_id * 32;

                if (k_abs_start < K) {
                    int g = k_abs_start >> group_shift;
                    // Branched Read: Only update on group change
                    // Reduces SMEM Conflicts
                    if (g != current_group) {
                        current_group = g;
                        if (g < MAX_GROUPS) {
                            s2 = s_scales[warp_id * MAX_GROUPS + g];
                            neg_zs = s_neg_zs[warp_id * MAX_GROUPS + g];
                        } else {
                            // Fallback: load directly from global if groups exceed cache
                            int idx_s = g * stride_groups + global_n * stride_n;
                            unsigned short s_raw = __ldg((const unsigned short *)&scales[idx_s]);
                            uint8_t z_raw = __ldg(&zeros[idx_s]);
                            float s_f = bf16_to_float_dev(*reinterpret_cast<bfloat16_t *>(&s_raw));
                            float z_f = (float)((int8_t)z_raw);
                            __half2 s2_tmp = __half2half2(__float2half(s_f));
                            __half2 z2_tmp = __half2half2(__float2half(z_f));
                            s2 = s2_tmp;
                            neg_zs = __hneg2(__hmul2(z2_tmp, s2_tmp));
                        }
                    }

#define PROCESS_PAIR(w_byte, c_idx, acc)                                                                                                   \
    {                                                                                                                                      \
        uint8_t p = w_byte;                                                                                                                \
        int w0_i = p & 0x0F;                                                                                                               \
        int w1_i = (p >> 4) & 0x0F;                                                                                                        \
        __half w0 = __float2half((float)w0_i);                                                                                             \
        __half w1 = __float2half((float)w1_i);                                                                                             \
        __half2 w_pair = __halves2half2(w0, w1);                                                                                           \
        __half2 term = __hfma2(w_pair, s2, neg_zs);                                                                                        \
        acc = __hfma2(term, in_reg[c_idx], acc);                                                                                           \
    }

                    uint32_t wx = w_val.x;
                    PROCESS_PAIR((wx & 0xFF), 0, acc0);
                    wx >>= 8;
                    PROCESS_PAIR((wx & 0xFF), 1, acc1);
                    wx >>= 8;
                    PROCESS_PAIR((wx & 0xFF), 2, acc2);
                    wx >>= 8;
                    PROCESS_PAIR((wx & 0xFF), 3, acc3);

                    uint32_t wy = w_val.y;
                    PROCESS_PAIR((wy & 0xFF), 4, acc0);
                    wy >>= 8;
                    PROCESS_PAIR((wy & 0xFF), 5, acc1);
                    wy >>= 8;
                    PROCESS_PAIR((wy & 0xFF), 6, acc2);
                    wy >>= 8;
                    PROCESS_PAIR((wy & 0xFF), 7, acc3);

                    uint32_t wz = w_val.z;
                    PROCESS_PAIR((wz & 0xFF), 8, acc0);
                    wz >>= 8;
                    PROCESS_PAIR((wz & 0xFF), 9, acc1);
                    wz >>= 8;
                    PROCESS_PAIR((wz & 0xFF), 10, acc2);
                    wz >>= 8;
                    PROCESS_PAIR((wz & 0xFF), 11, acc3);

                    uint32_t ww = w_val.w;
                    PROCESS_PAIR((ww & 0xFF), 12, acc0);
                    ww >>= 8;
                    PROCESS_PAIR((ww & 0xFF), 13, acc1);
                    ww >>= 8;
                    PROCESS_PAIR((ww & 0xFF), 14, acc2);
                    ww >>= 8;
                    PROCESS_PAIR((ww & 0xFF), 15, acc3);

#undef PROCESS_PAIR
                }
            }
        }
        __syncthreads();
    }

    acc0 = __hadd2(acc0, acc1);
    acc2 = __hadd2(acc2, acc3);
    acc0 = __hadd2(acc0, acc2);
    __half acc_sum = __hadd(acc0.x, acc0.y);
    float final_acc = __half2float(acc_sum);

#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        final_acc += __shfl_xor(final_acc, offset);

    if (active && lane_id == 0) {
        output[global_n] = float_to_bf16_dev(final_acc);
    }
}

void w4a16_gemv_unpacked_fused(torch::Tensor &output, const torch::Tensor &input, const torch::Tensor &qweights,
                               const torch::Tensor &scales, const torch::Tensor &zeros, int64_t in_features, int64_t out_features,
                               int64_t group_size, hipStream_t stream) {
    const int K = in_features;
    const int N = out_features;
    const int M = 1;

    int stride_groups, stride_n;
    if (scales.size(0) == out_features) {
        stride_n = scales.stride(0);
        stride_groups = scales.stride(1);
    } else {
        stride_groups = scales.stride(0);
        stride_n = scales.stride(1);
    }

    dim3 block(256);
    dim3 grid((N + 7) / 8);

    int group_shift = 0;
    int gs = group_size;
    while (gs > 1) {
        gs >>= 1;
        group_shift++;
    }

    int num_groups = K >> group_shift;
    if (num_groups <= 128) {
        w4a16_gemv_unpacked_kernel<128><<<grid, block, 0, stream>>>(
            (bfloat16_t *)output.data_ptr(), (const bfloat16_t *)input.data_ptr(), qweights.data_ptr<uint8_t>(),
            (const bfloat16_t *)scales.data_ptr(), (const uint8_t *)zeros.data_ptr(), M, K, N, group_shift, stride_groups, stride_n);
    } else {
        w4a16_gemv_unpacked_kernel<256><<<grid, block, 0, stream>>>(
            (bfloat16_t *)output.data_ptr(), (const bfloat16_t *)input.data_ptr(), qweights.data_ptr<uint8_t>(),
            (const bfloat16_t *)scales.data_ptr(), (const uint8_t *)zeros.data_ptr(), M, K, N, group_shift, stride_groups, stride_n);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP GEMV Unpacked error: ") + hipGetErrorString(err));
    }
}

} // namespace hipkernels
