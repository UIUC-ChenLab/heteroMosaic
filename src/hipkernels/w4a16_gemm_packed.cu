#undef __HIP_NO_HALF_CONVERSIONS__
#include <hip/hip_fp16.h>

#include "hipkernels/w4a16_gemm_packed.hpp"
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <hipblas/hipblas.h>
#include <iostream>
#include <limits>
#include <mutex>
#include <unordered_map>

#define BF16_COMPUTE 1 // 0 = Int8 Native ASM, 1 = BF16 Baseline
#define AUTOTUNE 0     // 0 = disable kernel autotuning, 1 = benchmark/select per shape

#if BF16_COMPUTE
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;
#endif

using bfloat16_t = hip_bfloat16;

#define HIP_CHECK(call)                                                                                                               \
    {                                                                                                                                 \
        hipError_t e = (call);                                                                                                         \
        if (e != hipSuccess) {                                                                                                         \
            std::cerr << "HIP:" << hipGetErrorString(e) << "\n";                                                                      \
            std::exit(1);                                                                                                              \
        }                                                                                                                             \
    }

#define TILE_K 128
#define TILE_N 64
#define PACKED_TILE_SIZE 4352
#define WEIGHT_BYTES 4096
#define SCALE_BYTES 128
#define ZERO_BYTES 128
#define PAD 16

__device__ __forceinline__ float bf16_to_float(bfloat16_t val) { return static_cast<float>(val); }
__device__ __forceinline__ bfloat16_t float_to_bf16(float val) { return hip_bfloat16(val); }

typedef int v4i32 __attribute__((ext_vector_type(4)));
typedef int v8i32 __attribute__((ext_vector_type(8)));
typedef float v8f32 __attribute__((ext_vector_type(8)));

namespace hipkernels {

enum class PackedKernelVariant : uint8_t {
    Generic128 = 0,
    K2048Special = 1,
    N64General = 2,
};

struct PackedShapeKey {
    int m;
    int k;
    int n;
    bool aligned_mn;

    bool operator==(const PackedShapeKey &other) const {
        return m == other.m && k == other.k && n == other.n && aligned_mn == other.aligned_mn;
    }
};

struct PackedShapeKeyHash {
    std::size_t operator()(const PackedShapeKey &key) const {
        std::size_t h = static_cast<std::size_t>(key.m);
        h = (h * 1315423911u) ^ static_cast<std::size_t>(key.k);
        h = (h * 2654435761u) ^ static_cast<std::size_t>(key.n);
        h = (h * 1099511628211ull) ^ static_cast<std::size_t>(key.aligned_mn ? 1 : 0);
        return h;
    }
};

#if AUTOTUNE
static std::unordered_map<PackedShapeKey, PackedKernelVariant, PackedShapeKeyHash> g_packed_variant_cache;
static std::mutex g_packed_variant_cache_mu;
#endif

// ============================================================================
// User's Original Int8 ASM Kernel (64x64 Tiles, K=128)
// ============================================================================
#if !BF16_COMPUTE
extern "C" __global__ void __launch_bounds__(256) w4a16_gemm_int8_asm(
    bfloat16_t *__restrict__ output,
    const bfloat16_t *__restrict__ input,
    const uint8_t *__restrict__ packed_params,
    const int M, const int K, const int N,
    const int num_tiles_k, const int num_tiles_n)
{
    // Shared Memory (M=64)
    __shared__ int8_t smem_A[8704];  // 64 * (128 + 8)
    const int stride_a = 128 + 8;

    // N=64
    __shared__ int8_t smem_B[8704];  // 64 * (128 + 8)
    const int stride_b = 128 + 8;

    __shared__ float smem_scale_A[64];
    __shared__ int smem_sum_A[64];
    __shared__ float smem_scale_W[64];
    __shared__ uint8_t smem_zero_W[64];

    v8i32 acc0 = {0};
    v8i32 acc1 = {0};

    v8f32 val_C0, val_C1;
    for (int i = 0; i < 8; ++i) {
        val_C0[i] = 0.0f;
        val_C1[i] = 0.0f;
    }

    const int tid = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Wave Mapping for 64x64 Tile: 4 row-strips (16 rows each).
    // 8 Waves. 2 waves per row-strip.
    int wave_id = tid / 32;
    int lane_id = tid % 32;
    int w_row = wave_id / 2; // 0..3 (Stripe M index)
    int w_col_start = (wave_id % 2) * 2; // 0 or 2 (Stripe N index base)

    const int global_m_start = by * 64;
    const int global_n_start = bx * 64;

    for (int k_outer = 0; k_outer < K; k_outer += 128) {
        __syncthreads();

        // Zero accumulators for this tile
        acc0 = (v8i32){0, 0, 0, 0, 0, 0, 0, 0};
        acc1 = (v8i32){0, 0, 0, 0, 0, 0, 0, 0};

        // === 1. Load A (BF16 -> Int8 with per-row scaling) ===
        // Load 64 rows. 256 threads.
        // tid/4 covers 0..63.
        {
            int row = tid / 4;
            int col_base = (tid % 4) * 16;
            int g_row = global_m_start + row;
            bool active = (row < 64 && g_row < M);

            if (active) {
                const bfloat16_t *row_ptr = &input[g_row * K + k_outer];

                v4i32 v_c0_0 = *(v4i32 *)&row_ptr[col_base];
                v4i32 v_c0_1 = *(v4i32 *)&row_ptr[col_base + 8];
                v4i32 v_c1_0 = *(v4i32 *)&row_ptr[col_base + 64];
                v4i32 v_c1_1 = *(v4i32 *)&row_ptr[col_base + 64 + 8];

                float vmax = 0.0f;
                float vals_0[16];
                float vals_1[16];

                uint16_t *u = (uint16_t *)&v_c0_0;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float f = bf16_to_float(*(bfloat16_t *)&u[i]);
                    vals_0[i] = f;
                    vmax = fmaxf(vmax, fabsf(f));
                }
                u = (uint16_t *)&v_c0_1;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float f = bf16_to_float(*(bfloat16_t *)&u[i]);
                    vals_0[i + 8] = f;
                    vmax = fmaxf(vmax, fabsf(f));
                }

                u = (uint16_t *)&v_c1_0;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float f = bf16_to_float(*(bfloat16_t *)&u[i]);
                    vals_1[i] = f;
                    vmax = fmaxf(vmax, fabsf(f));
                }
                u = (uint16_t *)&v_c1_1;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float f = bf16_to_float(*(bfloat16_t *)&u[i]);
                    vals_1[i + 8] = f;
                    vmax = fmaxf(vmax, fabsf(f));
                }

                float m = vmax;
                m = fmaxf(m, __shfl_xor(m, 1));
                m = fmaxf(m, __shfl_xor(m, 2));
                float scale = m / 127.0f;
                float inv_s = (scale > 1e-6f) ? (1.0f / scale) : 0.0f;

                if ((tid % 4) == 0 && row < 64) {
                    smem_scale_A[row] = scale;
                }
                int l_sum = 0;

                int8_t q_buf[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    int8_t q = (int8_t)(vals_0[i] * inv_s);
                    q_buf[i] = q;
                    l_sum += q;
                }
                *(v4i32 *)&smem_A[row * stride_a + col_base] = *(v4i32 *)q_buf;

#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    int8_t q = (int8_t)(vals_1[i] * inv_s);
                    q_buf[i] = q;
                    l_sum += q;
                }
                *(v4i32 *)&smem_A[row * stride_a + col_base + 64] = *(v4i32 *)q_buf;

                int s = l_sum;
                s += __shfl_xor(s, 1);
                s += __shfl_xor(s, 2);
                if ((tid % 4) == 0 && row < 64) {
                    smem_sum_A[row] = s;
                }

            } else if (row < 64) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    *(v4i32 *)&smem_A[row * stride_a + col_base + i * 16] = (v4i32){0, 0, 0, 0};
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    *(v4i32 *)&smem_A[row * stride_a + col_base + 64 + i * 16] = (v4i32){0, 0, 0, 0};
                }
            }
        }

        // === 2. Load B ===
        {
            int blk = tid / 64;
            int off_base = ((tid * 4) % 256) * 4;

            int k_idx_base = k_outer / 32;
            int n_idx_base = bx;
            int col_group = n_idx_base % 8;
            int col_off = n_idx_base / 8; // Fixed typo
            int tbg = 0;
#pragma unroll
            for (int g = 0; g < col_group; ++g) {
                tbg += ((num_tiles_n + 7 - g) / 8) * num_tiles_k;
            }
            int base_z = tbg + col_off * num_tiles_k;

            int k_super = k_outer / 128;
            size_t tile_start = (size_t)(base_z + k_super) * PACKED_TILE_SIZE;
            int k_sub_blk = (k_outer % 128) / 32;

            int w_off = (k_sub_blk + blk) * 1024 + off_base;

            uint4 raw_vec = *(const uint4 *)(packed_params + tile_start + w_off);
            uint32_t raws[4] = {raw_vec.x, raw_vec.y, raw_vec.z, raw_vec.w};

#pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t raw = raws[i];
                int curr_off = off_base + i * 4;
                int rem = curr_off % 32;
                int SR_SC = curr_off / 32;
                int SR = SR_SC / 8;
                int SC = SR_SC % 8;
                int IR = rem / 4;
                int k_in = SR * 8 + IR;
                int n_in = SC * 8;
                int k_final = blk * 32 + k_in;
                int n_final = n_in;

#pragma unroll
                for (int nib = 0; nib < 8; ++nib) {
                    int b_idx = nib / 2;
                    int shift = (nib % 2) ? 4 : 0;
                    int8_t q = (int8_t)((raw >> (b_idx * 8 + shift)) & 0x0F);
                    smem_B[(n_final + nib) * stride_b + k_final] = q;
                }
            }
            if (tid < 64) {
                const bfloat16_t *ts = (const bfloat16_t *)(packed_params + tile_start + WEIGHT_BYTES);
                const uint8_t *tz = packed_params + tile_start + WEIGHT_BYTES + SCALE_BYTES;
                smem_scale_W[tid] = bf16_to_float(ts[tid]);
                // Fix: Load zeros with correct stride
                smem_zero_W[tid] = tz[(tid / 8) * 16 + (tid % 8)];
            }
        }
        __syncthreads();

        // === 3. Compute (8x Unroll, 2 Accumulators) ===
#pragma unroll 8
        for (int ki = 0; ki < 8; ++ki) {
            int k_base = ki * 16;

            // Load A reuse for 2 B loads
            int r_idx = (w_row * 16) + (lane_id / 2); // User logic: / 2
            v4i32 va = *(v4i32 *)&smem_A[r_idx * stride_a + k_base];

            v4i32 vb0, vb1;
            // Load B for cols w_col_start
            {
                int b_n = (w_col_start * 16) + (lane_id / 2);
                vb0 = *(v4i32 *)&smem_B[b_n * stride_b + k_base];
            }
            asm volatile("v_wmma_i32_16x16x16_iu8 %0, %1, %2, %0 neg_lo:[1,1,0] clamp" : "+v"(acc0) : "v"(va), "v"(vb0));

            // Load B for cols w_col_start + 1
            {
                int b_n = ((w_col_start + 1) * 16) + (lane_id / 2);
                vb1 = *(v4i32 *)&smem_B[b_n * stride_b + k_base];
            }
            asm volatile("v_wmma_i32_16x16x16_iu8 %0, %1, %2, %0 neg_lo:[1,1,0] clamp" : "+v"(acc1) : "v"(va), "v"(vb1));
        }

        // === 4. Dequantize ===
        int row0 = w_row * 16;
        int col0 = w_col_start * 16;
        int col1 = (w_col_start + 1) * 16;
        int local_r = lane_id / 2;
        int local_c = (lane_id % 2) * 8;

#define DEQUANT_AND_ACC(ACC, VAL_C, C_BASE)                                                                                            \
    for (int i = 0; i < 8; ++i) {                                                                                                       \
        int r = row0 + local_r;                                                                                                         \
        int c = C_BASE + local_c + i;                                                                                                   \
        float sa = smem_scale_A[r];                                                                                                     \
        int suma = smem_sum_A[r];                                                                                                       \
        float sw = smem_scale_W[c];                                                                                                     \
        int zw = (int)smem_zero_W[c];                                                                                                   \
        int32_t val_i32 = ACC[i];                                                                                                       \
        float val_f = sw * ((float)val_i32 * sa - (float)zw * (float)suma * sa);                                                       \
        VAL_C[i] += val_f;                                                                                                              \
    }

        DEQUANT_AND_ACC(acc0, val_C0, col0);
        DEQUANT_AND_ACC(acc1, val_C1, col1);
#undef DEQUANT_AND_ACC
    }

    // === 5. Store ===
    int row0 = w_row * 16;
    int col0 = w_col_start * 16;
    int col1 = (w_col_start + 1) * 16;
    int local_r = lane_id / 2;
    int local_c = (lane_id % 2) * 8;

#define STORE_C(VAL_C, C_BASE)                                                                                                          \
    for (int i = 0; i < 8; ++i) {                                                                                                       \
        int r = row0 + local_r;                                                                                                         \
        int c = C_BASE + local_c + i;                                                                                                   \
        int gr = global_m_start + r;                                                                                                    \
        int gc = global_n_start + c;                                                                                                    \
        if (gr < M && gc < N)                                                                                                           \
            output[gr * N + gc] = float_to_bf16(VAL_C[i]);                                                                              \
    }

    STORE_C(val_C0, col0);
    STORE_C(val_C1, col1);
#undef STORE_C
}
#endif

// ============================================================================
// BF16 Baseline (rocWMMA)
// ============================================================================
#if BF16_COMPUTE
using FragA = fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major>;
using FragB_Row = fragment<matrix_b, 16, 16, 16, bfloat16_t, row_major>;
using FragC = fragment<accumulator, 16, 16, 16, float>;

template <bool AlignedMN>
__global__ void __launch_bounds__(256) w4a16_gemm_rocwmma_packed(
    bfloat16_t *__restrict__ output, const bfloat16_t *__restrict__ input,
    const uint8_t *__restrict__ packed_params, const int M, const int K,
    const int N, const int num_tiles_k, const int num_tiles_n)
{
    // Process K in 128 chunks to reduce loop + sync overhead vs 64-chunk loop.
    __shared__ __align__(16) bfloat16_t smem_B[128][128 + PAD];
    __shared__ __align__(16) uint8_t smem_W_packed[8192];
    __shared__ float smem_scales[2][128];
    __shared__ uint8_t smem_zeros[2][128];

    const int tid = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int wave_id = tid / 32;
    const int wave_y = wave_id / 2;
    const int wave_x = wave_id % 2;

    const int n_tile_idx = tid / 128;
    const int local_tid = tid & 127;
    const int tile_n_idx_global = (bx * 2) + n_tile_idx;
    const int col_group = tile_n_idx_global % 8;
    const int col_offset = tile_n_idx_global / 8;
    int tbg = 0;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        if (g < col_group) {
            tbg += ((num_tiles_n + 7 - g) / 8) * num_tiles_k;
        }
    }
    const int base_z = tbg + col_offset * num_tiles_k;
    const int dst_base = n_tile_idx * 4096;

    const int global_m_start = by * 128;
    const int global_n_start = bx * 128;

    FragC acc[2][4];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            fill_fragment(acc[i][j], 0.0f);
        }
    }

    for (int k_outer = 0; k_outer < K; k_outer += 128) {
        const int k_tile_idx = k_outer / 128;
        const size_t tile_offset = (size_t)(base_z + k_tile_idx) * PACKED_TILE_SIZE;
        const uint8_t *w_base = packed_params + tile_offset;

        // Load packed weights for K=128 (4x 32-row blocks) into shared
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            const uint2 *src_ptr = (const uint2 *)(w_base + kt * 1024);
            uint2 *dst_ptr = (uint2 *)(smem_W_packed);
            const int dst_off = dst_base + kt * 1024;
            dst_ptr[dst_off / 8 + local_tid] = src_ptr[local_tid];

            // Load scales/zeros once per tile
            if (kt == 0 && local_tid < 64) {
                const bfloat16_t *ts = (const bfloat16_t *)(w_base + WEIGHT_BYTES);
                const uint8_t *tz = w_base + WEIGHT_BYTES + SCALE_BYTES;
                smem_scales[n_tile_idx][local_tid] = bf16_to_float(ts[local_tid]);
                smem_zeros[n_tile_idx][local_tid] = tz[(local_tid / 8) * 16 + (local_tid % 8)];
            }
        }
        __syncthreads();

#pragma unroll
        for (int iter = 0; iter < 8; ++iter) {
            int grp_idx = tid * 8 + iter;
            int idx_base = grp_idx << 3;               // * 8
            int k_in = idx_base >> 7;                  // / 128
            int n_in = idx_base & 127;                 // % 128
            int t = n_in >> 6;                         // / 64
            int n_local = n_in & 63;                   // % 64
            int k_sub = k_in & 31;                     // % 32
            int k_block = k_in >> 5;                   // / 32
            int SR = k_sub >> 3;
            int IR = k_sub & 7;
            int SC = n_local >> 3;
            int block_idx = (SR << 3) | SC;
            int byte_off = (block_idx << 5) | (IR << 2);
            int smem_base = (t << 12) + (k_block << 10); // t*4096 + k_block*1024

            const uint32_t packed = *reinterpret_cast<const uint32_t *>(
                &smem_W_packed[smem_base + byte_off]);
            const uint2 z_packed = *reinterpret_cast<const uint2 *>(&smem_zeros[t][n_local]);
            const float4 s0 = *reinterpret_cast<const float4 *>(&smem_scales[t][n_local]);
            const float4 s1 = *reinterpret_cast<const float4 *>(&smem_scales[t][n_local + 4]);

            float s_arr[8];
            *(float4 *)&s_arr[0] = s0;
            *(float4 *)&s_arr[4] = s1;

            uint8_t z_arr[8];
            *(uint2 *)z_arr = z_packed;

            union {
                bfloat16_t res[8];
                float packed_res[4];
            } u;

#pragma unroll
            for (int x = 0; x < 4; ++x) {
                uint8_t p_byte = (packed >> (x * 8)) & 0xFF;
                int q0 = p_byte & 0x0F;
                int q1 = (p_byte >> 4) & 0x0F;

                int z0 = z_arr[x * 2];
                int z1 = z_arr[x * 2 + 1];

                float val0 = (float)(q0 - z0) * s_arr[x * 2];
                float val1 = (float)(q1 - z1) * s_arr[x * 2 + 1];

                float2 f2;
                f2.x = val0;
                f2.y = val1;
                __bf16_2 bf2 = __float22bfloat162_rn(f2);

                union {
                    __bf16_2 bf;
                    float f;
                } converter;
                converter.bf = bf2;
                u.packed_res[x] = converter.f;
            }

            *reinterpret_cast<uint4 *>(&smem_B[k_in][n_in]) = *reinterpret_cast<uint4 *>(u.res);
        }
        __syncthreads();

        FragA fA;
        FragB_Row fB[4];
#pragma unroll
        for (int ki = 0; ki < 128; ki += 16) {
            // Load B fragments once (shared for both i=0/1)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                load_matrix_sync(fB[j], &smem_B[ki][wave_x * 64 + j * 16], 128 + PAD);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int r_offset = global_m_start + wave_y * 32 + i * 16;
                load_matrix_sync(fA, input + r_offset * K + (k_outer + ki), K);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    mma_sync(acc[i][j], fA, fB[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    float *smem_F32 = (float *)&smem_B[0][0];
    // Batch 4 wave tiles per phase to reduce synchronization overhead.
    for (int base_w = 0; base_w < 8; base_w += 4) {
        if (wave_id >= base_w && wave_id < base_w + 4) {
            const int slot = wave_id - base_w;
            float *tile_ptr = &smem_F32[slot * 2048];
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 4; ++j) {
                    store_matrix_sync(&tile_ptr[(i * 16) * 64 + j * 16], acc[i][j], 64, mem_row_major);
                }
            }
        }
        __syncthreads();

        int offset = tid * 8;
        if (offset < 2048) {
            for (int slot = 0; slot < 4; ++slot) {
                int w = base_w + slot;
                int r_base = global_m_start + (w / 2) * 32;
                int c_base = global_n_start + (w % 2) * 64;
                const float *tile_ptr = &smem_F32[slot * 2048];
                if constexpr (AlignedMN) {
                    for (int k = 0; k < 8; k += 2) {
                        int lr = (offset + k) / 64;
                        int lc = (offset + k) % 64;
                        int gr = r_base + lr;
                        int gc = c_base + lc;
                        float2 f2;
                        f2.x = tile_ptr[lr * 64 + lc];
                        f2.y = tile_ptr[lr * 64 + lc + 1];
                        __bf16_2 bf2 = __float22bfloat162_rn(f2);
                        *reinterpret_cast<__bf16_2 *>(&output[gr * N + gc]) = bf2;
                    }
                } else {
                    for (int k = 0; k < 8; ++k) {
                        int lr = (offset + k) / 64;
                        int lc = (offset + k) % 64;
                        int gr = r_base + lr;
                        int gc = c_base + lc;
                        if (gr < M && gc < N) {
                            output[gr * N + gc] = float_to_bf16(tile_ptr[lr * 64 + lc]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <bool AlignedMN>
__global__ void __launch_bounds__(256, 2) w4a16_gemm_rocwmma_packed_k2048(
    bfloat16_t *__restrict__ output, const bfloat16_t *__restrict__ input,
    const uint8_t *__restrict__ packed_params, const int M, const int K,
    const int N, const int num_tiles_k, const int num_tiles_n)
{
    constexpr int PAD_K2048 = 8;
    // K=2048 tuned variant:
    // - process each K=128 tile in two K=64 sub-tiles
    // - much lower LDS footprint than full K=128 decode buffer
    struct K2048ComputeLds {
        bfloat16_t B[64][128 + PAD_K2048];
        uint8_t W[4096];
        float scales[2][64];
        uint8_t zeros[2][64];
    };
    union K2048SharedLds {
        K2048ComputeLds lds;
        float store_f32[8192];
    };
    __shared__ __align__(16) K2048SharedLds smem;

    const int tid = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int wave_id = tid / 32;
    const int wave_y = wave_id / 2;
    const int wave_x = wave_id % 2;

    const int n_tile_idx = tid / 128;
    const int local_tid = tid & 127;
    const int tile_n_idx_global = (bx * 2) + n_tile_idx;
    const int col_group = tile_n_idx_global % 8;
    const int col_offset = tile_n_idx_global / 8;
    int tbg = 0;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        if (g < col_group) {
            tbg += ((num_tiles_n + 7 - g) / 8) * num_tiles_k;
        }
    }
    const int base_z = tbg + col_offset * num_tiles_k;
    const int dst_base = n_tile_idx * 4096;

    const int global_m_start = by * 128;
    const int global_n_start = bx * 128;

    FragC acc[2][4];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            fill_fragment(acc[i][j], 0.0f);
        }
    }

    // Specialized path for K == 2048 (16 tiles of 128), each split into two K=64 chunks.
#pragma unroll 1
    for (int k_tile_idx = 0; k_tile_idx < num_tiles_k; ++k_tile_idx) {
        const size_t tile_offset = (size_t)(base_z + k_tile_idx) * PACKED_TILE_SIZE;
        const uint8_t *w_base = packed_params + tile_offset;

        if (local_tid < 64) {
            const bfloat16_t *ts = (const bfloat16_t *)(w_base + WEIGHT_BYTES);
            const uint8_t *tz = w_base + WEIGHT_BYTES + SCALE_BYTES;
            smem.lds.scales[n_tile_idx][local_tid] = bf16_to_float(ts[local_tid]);
            smem.lds.zeros[n_tile_idx][local_tid] = tz[(local_tid / 8) * 16 + (local_tid % 8)];
        }
        __syncthreads();

#pragma unroll
        for (int k_half = 0; k_half < 2; ++k_half) {
            const int k_block_base = k_half * 2;
            const int dst_base = n_tile_idx * 2048;

#pragma unroll
            for (int kt = 0; kt < 2; ++kt) {
                const uint2 *src_ptr = (const uint2 *)(w_base + (k_block_base + kt) * 1024);
                uint2 *dst_ptr = (uint2 *)(smem.lds.W);
                const int dst_off = dst_base + kt * 1024;
                dst_ptr[dst_off / 8 + local_tid] = src_ptr[local_tid];
            }
            __syncthreads();

#pragma unroll
            for (int iter = 0; iter < 4; ++iter) {
                int grp_idx = tid * 4 + iter;
                int idx_base = grp_idx << 3;
                int k_in = idx_base >> 7;
                int n_in = idx_base & 127;
                int t = n_in >> 6;
                int n_local = n_in & 63;
                int k_sub = k_in & 31;
                int k_block = k_in >> 5;
                int SR = k_sub >> 3;
                int IR = k_sub & 7;
                int SC = n_local >> 3;
                int block_idx = (SR << 3) | SC;
                int byte_off = (block_idx << 5) | (IR << 2);
                int smem_base = (t << 11) + (k_block << 10);

                const uint32_t packed = *reinterpret_cast<const uint32_t *>(&smem.lds.W[smem_base + byte_off]);
                const uint2 z_packed = *reinterpret_cast<const uint2 *>(&smem.lds.zeros[t][n_local]);
                const float4 s0 = *reinterpret_cast<const float4 *>(&smem.lds.scales[t][n_local]);
                const float4 s1 = *reinterpret_cast<const float4 *>(&smem.lds.scales[t][n_local + 4]);

                float s_arr[8];
                *(float4 *)&s_arr[0] = s0;
                *(float4 *)&s_arr[4] = s1;

                uint8_t z_arr[8];
                *(uint2 *)z_arr = z_packed;

                union {
                    bfloat16_t res[8];
                    float packed_res[4];
                } u;

#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    uint8_t p_byte = (packed >> (x * 8)) & 0xFF;
                    int q0 = p_byte & 0x0F;
                    int q1 = (p_byte >> 4) & 0x0F;
                    int z0 = z_arr[x * 2];
                    int z1 = z_arr[x * 2 + 1];
                    float val0 = (float)(q0 - z0) * s_arr[x * 2];
                    float val1 = (float)(q1 - z1) * s_arr[x * 2 + 1];

                    float2 f2;
                    f2.x = val0;
                    f2.y = val1;
                    __bf16_2 bf2 = __float22bfloat162_rn(f2);
                    union {
                        __bf16_2 bf;
                        float f;
                    } converter;
                    converter.bf = bf2;
                    u.packed_res[x] = converter.f;
                }

                *reinterpret_cast<uint4 *>(&smem.lds.B[k_in][n_in]) = *reinterpret_cast<uint4 *>(u.res);
            }
            __syncthreads();

            FragA fA;
            FragB_Row fB[4];
#pragma unroll
            for (int ki = 0; ki < 64; ki += 16) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    load_matrix_sync(fB[j], &smem.lds.B[ki][wave_x * 64 + j * 16], 128 + PAD_K2048);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    int r_offset = global_m_start + wave_y * 32 + i * 16;
                    load_matrix_sync(fA, input + r_offset * K + (k_tile_idx * 128 + k_half * 64 + ki), K);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        mma_sync(acc[i][j], fA, fB[j], acc[i][j]);
                    }
                }
            }
            __syncthreads();
        }
    }

    float *smem_F32 = smem.store_f32;
    for (int base_w = 0; base_w < 8; base_w += 4) {
        if (wave_id >= base_w && wave_id < base_w + 4) {
            const int slot = wave_id - base_w;
            float *tile_ptr = &smem_F32[slot * 2048];
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 4; ++j) {
                    store_matrix_sync(&tile_ptr[(i * 16) * 64 + j * 16], acc[i][j], 64, mem_row_major);
                }
            }
        }
        __syncthreads();

        int offset = tid * 8;
        if (offset < 2048) {
            for (int slot = 0; slot < 4; ++slot) {
                int w = base_w + slot;
                int r_base = global_m_start + (w / 2) * 32;
                int c_base = global_n_start + (w % 2) * 64;
                const float *tile_ptr = &smem_F32[slot * 2048];
                for (int k = 0; k < 8; ++k) {
                    int lr = (offset + k) / 64;
                    int lc = (offset + k) % 64;
                    int gr = r_base + lr;
                    int gc = c_base + lc;
                    if constexpr (AlignedMN) {
                        output[gr * N + gc] = float_to_bf16(tile_ptr[lr * 64 + lc]);
                    } else {
                        if (gr < M && gc < N) {
                            output[gr * N + gc] = float_to_bf16(tile_ptr[lr * 64 + lc]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <bool AlignedMN>
__global__ void __launch_bounds__(128) w4a16_gemm_rocwmma_packed_n64(
    bfloat16_t *__restrict__ output, const bfloat16_t *__restrict__ input,
    const uint8_t *__restrict__ packed_params, const int M, const int K,
    const int N, const int num_tiles_k, const int num_tiles_n)
{
    __shared__ __align__(16) bfloat16_t smem_B[128][64 + PAD];
    __shared__ __align__(16) uint8_t smem_W_packed[4096];
    __shared__ float smem_scales[64];
    __shared__ uint8_t smem_zeros[64];

    const int tid = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int wave_id = tid / 32; // 0..3
    const int wave_y = wave_id;   // one N tile (64 cols), four M stripes

    const int tile_n_idx_global = bx;
    const int col_group = tile_n_idx_global % 8;
    const int col_offset = tile_n_idx_global / 8;
    int tbg = 0;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        if (g < col_group) {
            tbg += ((num_tiles_n + 7 - g) / 8) * num_tiles_k;
        }
    }
    const int base_z = tbg + col_offset * num_tiles_k;

    const int global_m_start = by * 128;
    const int global_n_start = bx * 64;

    FragC acc[2][4];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            fill_fragment(acc[i][j], 0.0f);
        }
    }

    for (int k_outer = 0; k_outer < K; k_outer += 128) {
        const int k_tile_idx = k_outer / 128;
        const size_t tile_offset = (size_t)(base_z + k_tile_idx) * PACKED_TILE_SIZE;
        const uint8_t *w_base = packed_params + tile_offset;

        for (int idx = tid; idx < 4096; idx += 128) {
            smem_W_packed[idx] = w_base[idx];
        }
        if (tid < 64) {
            const bfloat16_t *ts = (const bfloat16_t *)(w_base + WEIGHT_BYTES);
            const uint8_t *tz = w_base + WEIGHT_BYTES + SCALE_BYTES;
            smem_scales[tid] = bf16_to_float(ts[tid]);
            smem_zeros[tid] = tz[(tid / 8) * 16 + (tid % 8)];
        }
        __syncthreads();

#pragma unroll
        for (int iter = 0; iter < 8; ++iter) {
            int grp_idx = tid * 8 + iter; // 0..1023
            int idx_base = grp_idx << 3;  // *8
            int k_in = idx_base >> 6;     // /64
            int n_local = idx_base & 63;  // %64
            int k_sub = k_in & 31;
            int k_block = k_in >> 5;

            int SR = k_sub >> 3;
            int IR = k_sub & 7;
            int SC = n_local >> 3;
            int block_idx = (SR << 3) | SC;
            int byte_off = (block_idx << 5) | (IR << 2);
            int smem_base = (k_block << 10); // k_block * 1024

            const uint32_t packed = *reinterpret_cast<const uint32_t *>(&smem_W_packed[smem_base + byte_off]);
            const uint2 z_packed = *reinterpret_cast<const uint2 *>(&smem_zeros[n_local]);
            const float4 s0 = *reinterpret_cast<const float4 *>(&smem_scales[n_local]);
            const float4 s1 = *reinterpret_cast<const float4 *>(&smem_scales[n_local + 4]);

            float s_arr[8];
            *(float4 *)&s_arr[0] = s0;
            *(float4 *)&s_arr[4] = s1;

            uint8_t z_arr[8];
            *(uint2 *)z_arr = z_packed;

            union {
                bfloat16_t res[8];
                float packed_res[4];
            } u;

#pragma unroll
            for (int x = 0; x < 4; ++x) {
                uint8_t p_byte = (packed >> (x * 8)) & 0xFF;
                int q0 = p_byte & 0x0F;
                int q1 = (p_byte >> 4) & 0x0F;
                int z0 = z_arr[x * 2];
                int z1 = z_arr[x * 2 + 1];
                float val0 = (float)(q0 - z0) * s_arr[x * 2];
                float val1 = (float)(q1 - z1) * s_arr[x * 2 + 1];

                float2 f2;
                f2.x = val0;
                f2.y = val1;
                __bf16_2 bf2 = __float22bfloat162_rn(f2);
                union {
                    __bf16_2 bf;
                    float f;
                } converter;
                converter.bf = bf2;
                u.packed_res[x] = converter.f;
            }

            *reinterpret_cast<uint4 *>(&smem_B[k_in][n_local]) = *reinterpret_cast<uint4 *>(u.res);
        }
        __syncthreads();

        FragA fA;
        FragB_Row fB[4];
#pragma unroll
        for (int ki = 0; ki < 128; ki += 16) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                load_matrix_sync(fB[j], &smem_B[ki][j * 16], 64 + PAD);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int r_offset = global_m_start + wave_y * 32 + i * 16;
                load_matrix_sync(fA, input + r_offset * K + (k_outer + ki), K);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    mma_sync(acc[i][j], fA, fB[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    float *smem_F32 = (float *)&smem_B[0][0];
    for (int w = 0; w < 4; ++w) {
        if (wave_id == w) {
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 4; ++j) {
                    store_matrix_sync(&smem_F32[(i * 16) * 64 + j * 16], acc[i][j], 64, mem_row_major);
                }
            }
        }
        __syncthreads();
        int r_base = global_m_start + w * 32;
        int c_base = global_n_start;
        int offset = tid * 16;
        if (offset < 2048) {
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                int lr = (offset + k) / 64;
                int lc = (offset + k) % 64;
                int gr = r_base + lr;
                int gc = c_base + lc;
                if constexpr (AlignedMN) {
                    output[gr * N + gc] = float_to_bf16(smem_F32[lr * 64 + lc]);
                } else {
                    if (gr < M && gc < N) {
                        output[gr * N + gc] = float_to_bf16(smem_F32[lr * 64 + lc]);
                    }
                }
            }
        }
        __syncthreads();
    }
}
#endif

// ============================================================================
// Host Wrapper
// ============================================================================
void w4a16_gemm_fused_packed(torch::Tensor &output, const torch::Tensor &input,
                             const torch::Tensor &packed_params, int64_t in_features,
                             int64_t out_features, hipStream_t stream) {
    const int M = input.numel() / in_features;
    const int K = in_features;
    const int N = out_features;

    const int num_tiles_k = (K + 128 - 1) / 128;
    const int num_tiles_n = (N + 64 - 1) / 64;

#if BF16_COMPUTE
    static bool cache_config_set = false;
    if (!cache_config_set) {
        (void)hipFuncSetCacheConfig((const void *)w4a16_gemm_rocwmma_packed<true>, hipFuncCachePreferShared);
        (void)hipFuncSetCacheConfig((const void *)w4a16_gemm_rocwmma_packed<false>, hipFuncCachePreferShared);
        (void)hipFuncSetCacheConfig((const void *)w4a16_gemm_rocwmma_packed_k2048<true>, hipFuncCachePreferL1);
        (void)hipFuncSetCacheConfig((const void *)w4a16_gemm_rocwmma_packed_k2048<false>, hipFuncCachePreferL1);
        (void)hipFuncSetCacheConfig((const void *)w4a16_gemm_rocwmma_packed_n64<true>, hipFuncCachePreferShared);
        (void)hipFuncSetCacheConfig((const void *)w4a16_gemm_rocwmma_packed_n64<false>, hipFuncCachePreferShared);
        cache_config_set = true;
    }

    const bool aligned_mn = ((M & 127) == 0) && ((N & 127) == 0);
    auto launch_generic128 = [&]() {
        dim3 block(256);
        if (aligned_mn) {
            dim3 grid(N / 128, M / 128);
            hipLaunchKernelGGL((w4a16_gemm_rocwmma_packed<true>), grid, block, 0, stream,
                               (bfloat16_t *)output.data_ptr(), (const bfloat16_t *)input.data_ptr(),
                               packed_params.data_ptr<uint8_t>(), M, K, N, num_tiles_k, num_tiles_n);
        } else {
            dim3 grid((N + 127) / 128, (M + 127) / 128);
            hipLaunchKernelGGL((w4a16_gemm_rocwmma_packed<false>), grid, block, 0, stream,
                               (bfloat16_t *)output.data_ptr(), (const bfloat16_t *)input.data_ptr(),
                               packed_params.data_ptr<uint8_t>(), M, K, N, num_tiles_k, num_tiles_n);
        }
    };

    auto launch_k2048_special = [&]() {
        dim3 block(256);
        if (aligned_mn) {
            dim3 grid(N / 128, M / 128);
            hipLaunchKernelGGL((w4a16_gemm_rocwmma_packed_k2048<true>), grid, block, 0, stream,
                               (bfloat16_t *)output.data_ptr(), (const bfloat16_t *)input.data_ptr(),
                               packed_params.data_ptr<uint8_t>(), M, K, N, num_tiles_k, num_tiles_n);
        } else {
            dim3 grid((N + 127) / 128, (M + 127) / 128);
            hipLaunchKernelGGL((w4a16_gemm_rocwmma_packed_k2048<false>), grid, block, 0, stream,
                               (bfloat16_t *)output.data_ptr(), (const bfloat16_t *)input.data_ptr(),
                               packed_params.data_ptr<uint8_t>(), M, K, N, num_tiles_k, num_tiles_n);
        }
    };

    auto launch_n64_general = [&]() {
        dim3 block(128);
        if (aligned_mn && ((M & 127) == 0)) {
            dim3 grid(N / 64, M / 128);
            hipLaunchKernelGGL((w4a16_gemm_rocwmma_packed_n64<true>), grid, block, 0, stream,
                               (bfloat16_t *)output.data_ptr(), (const bfloat16_t *)input.data_ptr(),
                               packed_params.data_ptr<uint8_t>(), M, K, N, num_tiles_k, num_tiles_n);
        } else {
            dim3 grid((N + 63) / 64, (M + 127) / 128);
            hipLaunchKernelGGL((w4a16_gemm_rocwmma_packed_n64<false>), grid, block, 0, stream,
                               (bfloat16_t *)output.data_ptr(), (const bfloat16_t *)input.data_ptr(),
                               packed_params.data_ptr<uint8_t>(), M, K, N, num_tiles_k, num_tiles_n);
        }
    };

    // Keep existing small-K N=64 path for very small K only.
    if ((K <= 1024) && ((N & 63) == 0)) {
        launch_n64_general();
    } else if ((K >= 2048) && ((K & 127) == 0)) {
#if AUTOTUNE
        const PackedShapeKey shape_key{M, K, N, aligned_mn};
        PackedKernelVariant selected = PackedKernelVariant::Generic128;
        bool has_cached = false;

        {
            std::lock_guard<std::mutex> lock(g_packed_variant_cache_mu);
            auto it = g_packed_variant_cache.find(shape_key);
            if (it != g_packed_variant_cache.end()) {
                selected = it->second;
                has_cached = true;
            }
        }

        auto launch_variant = [&](PackedKernelVariant variant) {
            if (variant == PackedKernelVariant::K2048Special) {
                launch_k2048_special();
            } else if (variant == PackedKernelVariant::N64General) {
                launch_n64_general();
            } else {
                launch_generic128();
            }
        };

        if (!has_cached) {
            auto benchmark_variant_ms = [&](PackedKernelVariant variant) -> float {
                hipEvent_t ev_start = nullptr;
                hipEvent_t ev_stop = nullptr;
                if (hipEventCreate(&ev_start) != hipSuccess || hipEventCreate(&ev_stop) != hipSuccess) {
                    if (ev_start) {
                        (void)hipEventDestroy(ev_start);
                    }
                    if (ev_stop) {
                        (void)hipEventDestroy(ev_stop);
                    }
                    return 1.0e30f;
                }

                constexpr int warmup_iters = 2;
                constexpr int timed_iters = 6;
                for (int i = 0; i < warmup_iters; ++i) {
                    launch_variant(variant);
                }

                float elapsed_ms = 1.0e30f;
                if (hipEventRecord(ev_start, stream) == hipSuccess) {
                    for (int i = 0; i < timed_iters; ++i) {
                        launch_variant(variant);
                    }
                    if (hipEventRecord(ev_stop, stream) == hipSuccess && hipEventSynchronize(ev_stop) == hipSuccess) {
                        float total_ms = 0.0f;
                        if (hipEventElapsedTime(&total_ms, ev_start, ev_stop) == hipSuccess) {
                            elapsed_ms = total_ms / static_cast<float>(timed_iters);
                        }
                    }
                }

                (void)hipEventDestroy(ev_start);
                (void)hipEventDestroy(ev_stop);
                return elapsed_ms;
            };

            float t_generic = benchmark_variant_ms(PackedKernelVariant::Generic128);
            float t_k2048 = benchmark_variant_ms(PackedKernelVariant::K2048Special);
            float t_n64 = ((N & 63) == 0) ? benchmark_variant_ms(PackedKernelVariant::N64General) : 1.0e30f;
            selected = PackedKernelVariant::Generic128;
            float best_ms = t_generic;
            if (t_k2048 < best_ms) {
                selected = PackedKernelVariant::K2048Special;
                best_ms = t_k2048;
            }
            if (t_n64 < best_ms) {
                selected = PackedKernelVariant::N64General;
                best_ms = t_n64;
            }

            {
                std::lock_guard<std::mutex> lock(g_packed_variant_cache_mu);
                g_packed_variant_cache[shape_key] = selected;
            }

            std::cout << "[w4a16_gemm_fused_packed] autotune K=" << K << " M=" << M << " N=" << N
                      << " generic=" << t_generic << "ms k2048=" << t_k2048 << "ms n64=" << t_n64 << "ms selected="
                      << ((selected == PackedKernelVariant::K2048Special) ? "k2048"
                                                                           : (selected == PackedKernelVariant::N64General) ? "n64"
                                                                                                                             : "generic")
                      << std::endl;
        }

        launch_variant(selected);
#else
        // Gemma MLP shapes on this target benchmark best on the generic
        // 128x128 path, including K=2048 gate/up and K=16384 down.
        launch_generic128();
#endif
    } else {
        launch_generic128();
    }
#else
    // Int8 64x64 Tile
    dim3 block(256), grid((N + 63) / 64, (M + 63) / 64);
    hipLaunchKernelGGL(w4a16_gemm_int8_asm, grid, block, 0, stream,
                       (bfloat16_t *)output.data_ptr(), (const bfloat16_t *)input.data_ptr(),
                       packed_params.data_ptr<uint8_t>(), M, K, N, num_tiles_k, num_tiles_n);
#endif

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err));
}

bool is_hip_available() {
    int c = 0;
    return hipGetDeviceCount(&c) == hipSuccess && c > 0;
}

void hip_synchronize() { HIP_CHECK(hipDeviceSynchronize()); }

} // namespace hipkernels
