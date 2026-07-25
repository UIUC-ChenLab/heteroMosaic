/**
 * @file w4a16_gemm_unpacked.cu
 * @brief Fused W4A16 GEMM kernel for Unpacked (Manual) weights layout using rocWMMA
 */

#undef __HIP_NO_HALF_CONVERSIONS__
#include <hip/hip_fp16.h>

#include "hipkernels/w4a16_gemm_unpacked.hpp"
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;

// Use hip_bfloat16 type for AMD
using bfloat16_t = hip_bfloat16;

#include <cstdlib>
#include <hipblas/hipblas.h>
#include <iostream>

// Macro Function to handle HIP errors
#define HIP_CHECK(call)                                                                                                                    \
    {                                                                                                                                      \
        hipError_t err = (call);                                                                                                           \
        if (err != hipSuccess) {                                                                                                           \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": " << hipGetErrorString(err) << " (" << err << ")"            \
                      << std::endl;                                                                                                        \
            std::exit(EXIT_FAILURE);                                                                                                       \
        }                                                                                                                                  \
    }

namespace hipkernels {

__device__ __forceinline__ float bf16_to_float(bfloat16_t val) { return static_cast<float>(val); }
__device__ __forceinline__ bfloat16_t float_to_bf16(float val) { return static_cast<bfloat16_t>(val); }

// --------------------------------------------------------------------------
// rocWMMA Kernel with F32 Accumulator for Unpacked Layout
// --------------------------------------------------------------------------

// Fragments
using FragA = fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major>;
// B is effectively W^T. W is stored as [N, K]. W^T is [K, N].
// W has layout [N, K]. Elements W(n, k) and W(n, k+1) are contiguous.
// W^T(k, n) = W(n, k). W^T(k+1, n) = W(n, k+1).
// So traversing K (rows of W^T) means traversing K (cols of W).
// Traversing N (cols of W^T) means traversing N (rows of W).
// Since W is Row Major [N, K], traversing K is contiguous.
// Thus W^T is Column Major [K, N].
using FragB_Col = fragment<matrix_b, 16, 16, 16, bfloat16_t, col_major>;
// Accumulator: F32
using FragC = fragment<accumulator, 16, 16, 16, float>;

// SMEM Padding
// Ensure LDM (stride) is multiple of 16 bytes (8 bf16).
// 128 + 16 = 144 elements = 288 bytes (16-byte aligned).
#define PAD 16

// Kernel assumes Block M=128, N=128, K_step=128
// Grid dimensions: (N / 128), (M / 128)
__global__ void __launch_bounds__(256, 1) w4a16_gemm_rocwmma_unpacked(bfloat16_t *__restrict__ output, const bfloat16_t *__restrict__ input,
                                                                   const uint8_t *__restrict__ qweights,  // [N, K/2]
                                                                   const bfloat16_t *__restrict__ scales, // [N, Groups]
                                                                   const uint8_t *__restrict__ zeros,     // [N, Groups]
                                                                   const int M, const int K, const int N, const int group_size) {
    // SMEM Layouts
    // smem_B is [TileN, TileK] with K-contiguous storage
    __shared__ __align__(16) bfloat16_t smem_B[128][128 + PAD];

    const int tid = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Wave ID (0..7)
    const int wave_id = tid / 32;
    const int wave_y = wave_id / 2; // 0..3 (4 rows of waves)
    const int wave_x = wave_id % 2; // 0..1 (2 cols of waves)

    // Grid: bx maps to N, by maps to M (match packed kernel layout)
    const int global_n_start = bx * 128;
    const int global_m_start = by * 128;

    // Accumulators
    FragC acc[2][4];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            fill_fragment(acc[i][j], 0.0f);

    const int groups = K / group_size;

    // K Loop (128-step to reduce overhead)
    for (int k_outer = 0; k_outer < K; k_outer += 128) {

        // --- 2. Load Packed Weights [128, 32] (Logical) ---
        // W stored as [N, K/2] uint8.
        // We need W[global_n_start : +128, k_outer : +32].
        // This is 128 rows of 32 logical elements (16 bytes).
        // Total bytes = 128 * 16 = 2048 bytes.
        // 256 threads. Each thread loads 8 bytes (16 elements type uint8 -> 16 el bf16).

        // Also load Scales/Zeros.
        // Scales/Zeros are [N, Groups]. k_outer determines group.
        int group_idx = k_outer / group_size;

        // --- 2. Load Packed Weights into smem_B[N][K] ---
        // W is [N, K/2] in global. We want smem_B[N][K].
        // 256 threads: 2 threads per N-row, each loads 32 bytes -> 64 values.
        int w_row = tid >> 1;     // 0..127 (N-index)
        int w_blk = tid & 1;      // 0..1 (64 K-values each)

        if (w_row < 128) {
            int gn = global_n_start + w_row;
            int k_base = w_blk * 64;
            if (gn < N) {
                float scale = 0.0f;
                int zero = 0;
                if (w_blk == 0) {
                    int idx = gn * groups + group_idx;
                    scale = bf16_to_float(scales[idx]);
                    zero = (int8_t)zeros[idx];
                }
                float scale_peer = __shfl_xor(scale, 1);
                int zero_peer = __shfl_xor(zero, 1);
                if (w_blk == 1) {
                    scale = scale_peer;
                    zero = zero_peer;
                }

                const uint8_t *w_src_base = &qweights[gn * (K / 2) + (k_outer / 2) + w_blk * 32];
                const uint4 *src_u4 = reinterpret_cast<const uint4 *>(w_src_base);
                uint4 v0 = src_u4[0];
                uint4 v1 = src_u4[1];

                float sz = scale * (float)zero;

#define DEQUANT_STORE_RAW(PACKED, OFF)                                                                                                      \
    do {                                                                                                                                     \
        uint32_t packed = (PACKED);                                                                                                          \
        union {                                                                                                                              \
            bfloat16_t res[8];                                                                                                               \
            float packed_res[4];                                                                                                             \
        } u;                                                                                                                                 \
        _Pragma("unroll") for (int b = 0; b < 4; ++b) {                                                                                      \
            uint8_t p = (packed >> (b * 8)) & 0xFF;                                                                                          \
            int q0 = p & 0x0F;                                                                                                               \
            int q1 = (p >> 4) & 0x0F;                                                                                                        \
            float val0 = (float)q0 * scale - sz;                                                                                             \
            float val1 = (float)q1 * scale - sz;                                                                                             \
            float2 f2;                                                                                                                       \
            f2.x = val0;                                                                                                                     \
            f2.y = val1;                                                                                                                     \
            __bf16_2 bf2 = __float22bfloat162_rn(f2);                                                                                        \
            union {                                                                                                                          \
                __bf16_2 bf;                                                                                                                 \
                float f;                                                                                                                     \
            } converter;                                                                                                                     \
            converter.bf = bf2;                                                                                                              \
            u.packed_res[b] = converter.f;                                                                                                   \
        }                                                                                                                                    \
        *reinterpret_cast<uint4 *>(&smem_B[w_row][k_base + (OFF)]) = *reinterpret_cast<uint4 *>(u.res);                                      \
    } while (0)

                DEQUANT_STORE_RAW(v0.x, 0);
                DEQUANT_STORE_RAW(v0.y, 8);
                DEQUANT_STORE_RAW(v0.z, 16);
                DEQUANT_STORE_RAW(v0.w, 24);
                DEQUANT_STORE_RAW(v1.x, 32);
                DEQUANT_STORE_RAW(v1.y, 40);
                DEQUANT_STORE_RAW(v1.z, 48);
                DEQUANT_STORE_RAW(v1.w, 56);

#undef DEQUANT_STORE_RAW
            } else {
                const uint4 zero4 = {0, 0, 0, 0};
#pragma unroll
                for (int x = 0; x < 8; ++x) {
                    *reinterpret_cast<uint4 *>(&smem_B[w_row][k_base + x * 8]) = zero4;
                }
            }
        }
        __syncthreads();

        // --- 3. Compute ---
        // A is [128, 32]. B is [128, 32] (Logical W).
        // Actual GEMM is A * B^T.
        // rocWMMA B fragment is B^T.
        // If we load B as col_major, WMMA treats it as [32, 128].
        // Elements in col are stride 32+PAD? No, row_major storage [128][32+PAD].
        // smem_B[0][0], smem_B[0][1]...
        // Transpose view: [32, 128].
        // Col-Major access of [32, 128] means accessing (0,0), (1,0)...
        // (1,0) in [32,128] is smem_B[0][1]. (0,1) is smem_B[1][0]. Wait.
        // If matrix B logical is [K=32, N=128].
        // Row 0 is [b00, b01... b0_127].
        // smem_B stores [128][32].
        // smem_B[n][k].
        // We want B_logical[k][n].
        // B_logical[k][n] corresponds to smem_B[n][k].
        // So B_logical is smem_B transposed.
        // If we tell WMMA that B is col_major [K, N], it expects B[k, n] and B[k+1, n] to be contiguous.
        // Contiguous in smem_B are smem_B[n][k] and smem_B[n][k+1].
        // This corresponds to B_logical[k][n] and B_logical[k+1][n].
        // So contiguous memory in smem_B corresponds to contiguous K in B_logical.
        // This MATCHES col_major expectation for B_logical [K, N].
        // So we can point WMMA Fragment B (col_major) to smem_B.

        FragA fA;
        FragB_Col fB; // Col Major

        #pragma unroll
        for (int ki = 0; ki < 128; ki += 16) {
            // Load A: [TileM, 16] sub-tile
            #pragma unroll
            for (int i = 0; i < 2; ++i) { // 2 sub-waves M
                int r_offset = global_m_start + wave_y * 32 + i * 16;
                load_matrix_sync(fA, input + r_offset * K + (k_outer + ki), K);

                #pragma unroll
                for (int j = 0; j < 4; ++j) { // 4 sub-waves N
                    int c_offset = wave_x * 64 + j * 16;
                    load_matrix_sync(fB, &smem_B[c_offset][ki], 128 + PAD);
                    mma_sync(acc[i][j], fA, fB, acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // --- Store ---
    // Reuse smem_A as float buffer (128*32 = 4096 elements).
    // We need 128*128 = 16384 elements buffer.
    // smem_A is too small.
    // Just write directly to global if possible?
    // Or tile the write.
    // 256 threads. 128x128 elements.
    // We can't hold all ACC in registers if we spill.
    // But acc is register. 2x4 frags = 8 * 256 el = 2048 el per thread?? No.
    // FragC is 16x16=256 elements. Distributed across 32 threads. 8 per thread.
    // 8 frags * 8 el = 64 elements/thread.
    // Total 256 th * 64 = 16384. Matches block.

    // Direct store
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            // Store fragment to global.
            // Warning: non-coalesced store is slow.
            // But implementing shmem swizzle for store is complex.
            // Let's try direct store first, if functionality works.
            // Ideally we'd stage to smem.
            // We have smem_B [128][32+PAD] (bf16) = 128*32*2 = 8KB.
            // smem_A [128][32+PAD] (bf16) = 8KB.
            // Total 16KB.
            // Output tile 128x128 bf16 = 32KB.
            // Cannot fit in smem.
            // We can do it in chunks.
            // Store first half of waves, then second?
            // Easier: just store directly. WMMA store puts elements in memory linearly-ish?
            // "store_matrix_sync" stores to memory.
            // Access: [wave_y*32 + i*16][wave_x*64 + j*16]
            int r_start = global_m_start + wave_y * 32 + i * 16;
            int c_start = global_n_start + wave_x * 64 + j * 16;

            if (r_start < M && c_start < N) {
                // We need a pointer to global [r_start, c_start].
                int offset = r_start * N + c_start;
                // store_matrix_sync writes a 16x16 tile.
                // It handles stride N.
                // Need to convert float acc to bf16 before store.
                // Map F32 acc to BF16 fragment?
                // No, store F32, then load/cvt? No.
                // Manually convert frag?
                // loop over elements of fragment (opaque)?
                // rocWMMA provides store_matrix_sync for accumulator (float) -> float ptr.
                // We want bf16 output.
                // Use frag cast?
                // auto frag_bf16 = dst_fragment<bfloat16_t>(acc[i][j]); ?
                // No.
                // Let's store to smem chunk by chunk.
                // We have enough smem to store one 16x16 tile? Yes.
                // We have enough smem to store 32x64 (one wave)?
                // 32*64*4 bytes = 8KB. Yes!
                // Serialize waves.
            }
        }
    }

    // Serialization strategy using smem_B (recast as float)
    // smem_B size: 128*34*2 = 8704 bytes.
    // One wave output: 32x64 bf16 = 4096 bytes.
    // One wave output float = 8192 bytes.
    // Fits!

    // Re-declare smem ptr
    float *smem_wb = (float *)&smem_B[0][0];

    for (int w = 0; w < 8; ++w) {
        __syncthreads(); // Wait for SMEM to be free
        if (wave_id == w) {
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 4; ++j) {
                    // Tile inside the wave (32x64)
                    int tr = i * 16;
                    int tc = j * 16;
                    store_matrix_sync(&smem_wb[tr * 64 + tc], acc[i][j], 64, mem_row_major);
                }
            }
        }
        __syncthreads(); // Wait for store to complete

        // All threads help write 32x64 tile from SMEM to Global (converting F32->BF16)
        // 2048 elements. 256 threads. 8 per thread.

        int target_wy = w / 2;
        int target_wx = w % 2;
        int r_base = global_m_start + target_wy * 32;
        int c_base = global_n_start + target_wx * 64;

        int tid_offset = tid * 8;
        if (tid_offset < 2048) {
            for (int k = 0; k < 8; ++k) {
                int idx = tid_offset + k;
                int r = idx / 64; // 0..31
                int c = idx % 64; // 0..63

                float val = smem_wb[r * 64 + c];

                int gr = r_base + r;
                int gc = c_base + c;

                if (gr < M && gc < N) {
                    output[gr * N + gc] = float_to_bf16(val);
                }
            }
        }
    }
}

void w4a16_gemm_unpacked_fused(torch::Tensor &output, const torch::Tensor &input, const torch::Tensor &qweights,
                               const torch::Tensor &scales, const torch::Tensor &zeros, int64_t in_features, int64_t out_features,
                               int64_t group_size, hipStream_t stream) {
    const int M = input.numel() / in_features;
    const int K = in_features;
    const int N = out_features;

    // Ensure grid covers M, N with 128x128 blocks (grid.x = N, grid.y = M)
    dim3 block(256);
    dim3 grid((N + 127) / 128, (M + 127) / 128);

    static bool cache_configured = false;
    if (!cache_configured) {
        hipError_t cache_err = hipFuncSetCacheConfig((const void *)w4a16_gemm_rocwmma_unpacked, hipFuncCachePreferL1);
        if (cache_err != hipSuccess) {
            throw std::runtime_error(std::string("HIP GEMM unpacked cache config error: ") + hipGetErrorString(cache_err));
        }
        cache_configured = true;
    }

    hipLaunchKernelGGL(w4a16_gemm_rocwmma_unpacked, grid, block, 0, stream, (bfloat16_t *)output.data_ptr(),
                       (const bfloat16_t *)input.data_ptr(), qweights.data_ptr<uint8_t>(), (const bfloat16_t *)scales.data_ptr(),
                       (const uint8_t *)zeros.data_ptr(), M, K, N, group_size);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP kernel error: ") + hipGetErrorString(err));
    }
}

} // namespace hipkernels
