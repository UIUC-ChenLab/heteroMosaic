#undef __HIP_NO_HALF_CONVERSIONS__
#include "hipkernels/w4a16_gemm_packedGPU.hpp"
#include "hipkernels/w4a16_gemm_packed.hpp"
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <iostream>
#include <tuple>
#include <vector>

using bfloat16_t = hip_bfloat16;
using namespace rocwmma;

#define PAD 16
#define LARGE_TILE_SIZE_ROW 128
#define LARGE_TILE_SIZE_COL 64

namespace hipkernels {

using FragA = fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major>;
using FragB = fragment<matrix_b, 16, 16, 16, bfloat16_t, row_major>;
using FragC = fragment<accumulator, 16, 16, 16, float>;

static std::tuple<torch::Tensor, int64_t, int64_t> get_tile_indices_host(int64_t rows, int64_t cols) {
    int64_t num_tiles_row = (rows + LARGE_TILE_SIZE_ROW - 1) / LARGE_TILE_SIZE_ROW;
    int64_t num_tiles_col = (cols + LARGE_TILE_SIZE_COL - 1) / LARGE_TILE_SIZE_COL;

    std::vector<int64_t> tile_indices;
    tile_indices.reserve(num_tiles_row * num_tiles_col);

    for (int col_mod = 0; col_mod < 8; col_mod++) {
        for (int c = col_mod; c < num_tiles_col; c += 8) {
            for (int r = 0; r < num_tiles_row; r++) {
                tile_indices.push_back(r * num_tiles_col + c);
            }
        }
    }

    return std::make_tuple(torch::tensor(tile_indices, torch::kLong), num_tiles_row, num_tiles_col);
}

static torch::Tensor pack_stride20_to_tile(const torch::Tensor &packed_params, int64_t K, int64_t N) {
    auto device = packed_params.device();

    const int64_t num_blocks_k = (K + 31) / 32;
    auto packed_view = packed_params.view({N, num_blocks_k, 20});

    auto s_u8 = packed_view.slice(2, 0, 2).contiguous();
    auto z_u8 = packed_view.slice(2, 2, 4).contiguous();
    auto w_packed = packed_view.slice(2, 4, 20).contiguous(); // [N, Bk, 16]

    auto w_low = torch::bitwise_and(w_packed, 0x0F);
    auto w_high = torch::bitwise_and(torch::bitwise_right_shift(w_packed, 4), 0x0F);
    auto w_stacked = torch::stack({w_low, w_high}, -1);
    auto w_unpacked = w_stacked.view({N, num_blocks_k, 32});
    auto qw = w_unpacked.view({N, num_blocks_k * 32}).to(torch::kInt8);
    auto qw_t = qw.t().contiguous(); // [K, N]

    auto scales = torch::from_blob(
        s_u8.data_ptr(),
        {N, num_blocks_k},
        torch::TensorOptions().dtype(torch::kBFloat16).device(device)).contiguous();
    auto zeros_bf16 = torch::from_blob(
        z_u8.data_ptr(),
        {N, num_blocks_k},
        torch::TensorOptions().dtype(torch::kBFloat16).device(device)).contiguous();
    auto zeros = zeros_bf16.to(torch::kInt8);

    const int64_t group_size = 128;
    const int64_t blocks_per_group = group_size / 32;
    const int64_t num_groups = (K + group_size - 1) / group_size;

    auto scales_grouped = scales.view({N, num_groups, blocks_per_group});
    auto zeros_grouped = zeros.view({N, num_groups, blocks_per_group});
    auto scales_gn = scales_grouped.select(2, 0).transpose(0, 1).contiguous(); // [G, N]
    auto zeros_gn = zeros_grouped.select(2, 0).transpose(0, 1).contiguous();    // [G, N]

    const int64_t num_tiles_row = (K + LARGE_TILE_SIZE_ROW - 1) / LARGE_TILE_SIZE_ROW;
    const int64_t num_tiles_col = (N + LARGE_TILE_SIZE_COL - 1) / LARGE_TILE_SIZE_COL;
    const int64_t K_padded = num_tiles_row * LARGE_TILE_SIZE_ROW;
    const int64_t N_padded = num_tiles_col * LARGE_TILE_SIZE_COL;
    const int64_t num_groups_padded = (K_padded + group_size - 1) / group_size;

    auto qw_padded = torch::zeros({K_padded, N_padded}, torch::TensorOptions().dtype(torch::kInt8).device(device));
    qw_padded.slice(0, 0, K).slice(1, 0, N).copy_(qw_t);

    auto scales_padded = torch::zeros({num_groups_padded, N_padded}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));
    scales_padded.slice(0, 0, num_groups).slice(1, 0, N).copy_(scales_gn);
    auto zeros_padded = torch::zeros({num_groups_padded, N_padded}, torch::TensorOptions().dtype(torch::kInt8).device(device));
    zeros_padded.slice(0, 0, num_groups).slice(1, 0, N).copy_(zeros_gn);

    auto w_tiles = qw_padded.view({num_tiles_row, 128, num_tiles_col, 64});
    w_tiles = w_tiles.permute({0, 2, 1, 3}).contiguous().view({-1, 128, 64});

    auto [tile_indices, _r, _c] = get_tile_indices_host(K_padded, N_padded);
    tile_indices = tile_indices.to(device).to(torch::kLong);
    auto w_ordered = w_tiles.index_select(0, tile_indices);

    const int64_t total_tiles = w_ordered.size(0);
    auto w_reshaped = w_ordered.view({total_tiles, 16, 8, 8, 8});
    auto w_permuted = w_reshaped.permute({0, 1, 3, 2, 4}).contiguous();
    auto w_even = w_permuted.index({"...", torch::indexing::Slice(0, torch::indexing::None, 2)});
    auto w_odd = w_permuted.index({"...", torch::indexing::Slice(1, torch::indexing::None, 2)});
    auto w_packed_blk =
        torch::bitwise_or(torch::bitwise_and(w_even, 0x0F), torch::bitwise_left_shift(torch::bitwise_and(w_odd, 0x0F), 4))
            .to(torch::kUInt8);
    auto packed_weights_flat = w_packed_blk.view({-1, 4096});

    auto scale_flat = scales_padded.contiguous().view({-1});
    auto zero_flat = zeros_padded.contiguous().view({-1});
    const int64_t num_scales = scale_flat.size(0);

    auto tile_rows = torch::div(tile_indices, num_tiles_col, "floor");
    auto tile_cols = torch::remainder(tile_indices, num_tiles_col);
    auto col_offsets = torch::arange(64, torch::TensorOptions().device(device)).unsqueeze(0);
    auto global_col_indices = tile_cols.unsqueeze(1) * 64 + col_offsets;
    auto global_rows_start = tile_rows.unsqueeze(1) * 128;
    auto group_idx = torch::div(global_rows_start, group_size, "floor");
    auto scale_indices = group_idx * N_padded + global_col_indices;
    scale_indices = torch::clamp(scale_indices, 0, num_scales - 1).to(torch::kLong);

    auto gathered_scales = scale_flat.index_select(0, scale_indices.view({-1})).view({-1, 64});
    auto gathered_zeros = zero_flat.index_select(0, scale_indices.view({-1})).view({-1, 64});

    auto scales_uint8 = gathered_scales.view(torch::kUInt8).view({-1, 128});
    auto zeros_dup = gathered_zeros.view(torch::kUInt8).view({-1, 8, 8}).repeat_interleave(2, 1).view({-1, 128});

    auto packed_final = torch::cat({packed_weights_flat, scales_uint8, zeros_dup}, 1);
    return packed_final.view({-1});
}

template <bool Aligned>
__global__ void __launch_bounds__(256)
w4a16_gemm_packedGPU_rocwmma(
    bfloat16_t *__restrict__ output,
    const bfloat16_t *__restrict__ input,
    const uint8_t *__restrict__ packed_params,
    int M,
    int K,
    int N,
    int num_blocks_k
) {
    __shared__ __align__(16) bfloat16_t smem_B[128][128 + PAD];

    const int tid = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int wave_id = tid / 32;
    const int wave_y = wave_id / 2;
    const int wave_x = wave_id % 2;

    const int global_m_start = by * 128;
    const int global_n_start = bx * 128;

    const size_t row_stride = (size_t)num_blocks_k * 20;

    FragC acc[2][4];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            fill_fragment(acc[i][j], 0.0f);
        }
    }

    for (int k_outer = 0; k_outer < K; k_outer += 128) {
        const int k_outer_block = k_outer / 32;

        for (int blk = tid; blk < 512; blk += 256) {
            const int n_in = blk >> 2;    // 0..127
            const int k_block = blk & 3;  // 0..3 (32 weights)
            const int global_n = global_n_start + n_in;
            const int k_base = k_block * 32;
            const int k_global_base = k_outer + k_base;

            if constexpr (Aligned) {
                const size_t block_offset = (size_t)global_n * row_stride + (size_t)(k_outer_block + k_block) * 20;
                const uint8_t *block_ptr = packed_params + block_offset;

                const uint32_t sz = *reinterpret_cast<const uint32_t *>(block_ptr);
                const bfloat16_t *sz_bf16 = reinterpret_cast<const bfloat16_t *>(&sz);
                const float scale = static_cast<float>(sz_bf16[0]);
                const float zero = static_cast<float>(sz_bf16[1]);

                const uint4 w4 = *reinterpret_cast<const uint4 *>(block_ptr + 4);
                const uint32_t w_words[4] = {w4.x, w4.y, w4.z, w4.w};

                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    uint32_t val = w_words[w];
                    #pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        uint8_t qbyte = (val >> (b * 8)) & 0xFF;
                        int q0 = qbyte & 0x0F;
                        int q1 = (qbyte >> 4) & 0x0F;
                        int idx = w * 8 + b * 2;
                        float v0 = (static_cast<float>(q0) - zero) * scale;
                        float v1 = (static_cast<float>(q1) - zero) * scale;
                        smem_B[k_base + idx][n_in] = static_cast<bfloat16_t>(v0);
                        smem_B[k_base + idx + 1][n_in] = static_cast<bfloat16_t>(v1);
                    }
                }
            } else {
                if (global_n < N && k_global_base < K) {
                    const size_t block_offset = (size_t)global_n * row_stride + (size_t)(k_outer_block + k_block) * 20;
                    const uint8_t *block_ptr = packed_params + block_offset;

                    const uint32_t sz = *reinterpret_cast<const uint32_t *>(block_ptr);
                    const bfloat16_t *sz_bf16 = reinterpret_cast<const bfloat16_t *>(&sz);
                    const float scale = static_cast<float>(sz_bf16[0]);
                    const float zero = static_cast<float>(sz_bf16[1]);

                    const uint4 w4 = *reinterpret_cast<const uint4 *>(block_ptr + 4);
                    const uint32_t w_words[4] = {w4.x, w4.y, w4.z, w4.w};

                    if (k_global_base + 31 < K) {
                        #pragma unroll
                        for (int w = 0; w < 4; ++w) {
                            uint32_t val = w_words[w];
                            #pragma unroll
                            for (int b = 0; b < 4; ++b) {
                                uint8_t qbyte = (val >> (b * 8)) & 0xFF;
                                int q0 = qbyte & 0x0F;
                                int q1 = (qbyte >> 4) & 0x0F;
                                int idx = w * 8 + b * 2;
                                float v0 = (static_cast<float>(q0) - zero) * scale;
                                float v1 = (static_cast<float>(q1) - zero) * scale;
                                smem_B[k_base + idx][n_in] = static_cast<bfloat16_t>(v0);
                                smem_B[k_base + idx + 1][n_in] = static_cast<bfloat16_t>(v1);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int w = 0; w < 4; ++w) {
                            uint32_t val = w_words[w];
                            #pragma unroll
                            for (int b = 0; b < 4; ++b) {
                                uint8_t qbyte = (val >> (b * 8)) & 0xFF;
                                int q0 = qbyte & 0x0F;
                                int q1 = (qbyte >> 4) & 0x0F;
                                int idx = w * 8 + b * 2;
                                int kg0 = k_global_base + idx;
                                int kg1 = k_global_base + idx + 1;
                                if (kg0 < K) {
                                    float v0 = (static_cast<float>(q0) - zero) * scale;
                                    smem_B[k_base + idx][n_in] = static_cast<bfloat16_t>(v0);
                                } else {
                                    smem_B[k_base + idx][n_in] = static_cast<bfloat16_t>(0.0f);
                                }
                                if (kg1 < K) {
                                    float v1 = (static_cast<float>(q1) - zero) * scale;
                                    smem_B[k_base + idx + 1][n_in] = static_cast<bfloat16_t>(v1);
                                } else {
                                    smem_B[k_base + idx + 1][n_in] = static_cast<bfloat16_t>(0.0f);
                                }
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        smem_B[k_base + i][n_in] = static_cast<bfloat16_t>(0.0f);
                    }
                }
            }
        }
        __syncthreads();

        FragA fA;
        FragB fB[4];
        #pragma unroll
        for (int ki = 0; ki < 128; ki += 16) {
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

    float *smem_F32 = reinterpret_cast<float *>(&smem_B[0][0]);
    for (int w = 0; w < 8; ++w) {
        if (wave_id == w) {
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 4; ++j) {
                    store_matrix_sync(&smem_F32[(i * 16) * 64 + j * 16], acc[i][j], 64, mem_row_major);
                }
            }
        }
        __syncthreads();
        int r_base = global_m_start + (w / 2) * 32;
        int c_base = global_n_start + (w % 2) * 64;
        int offset = tid * 8;
        if (offset < 2048) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                int lr = (offset + k) / 64;
                int lc = (offset + k) % 64;
                int gr = r_base + lr;
                int gc = c_base + lc;
                if constexpr (Aligned) {
                    output[gr * N + gc] = static_cast<bfloat16_t>(smem_F32[lr * 64 + lc]);
                } else {
                    if (gr < M && gc < N) {
                        output[gr * N + gc] = static_cast<bfloat16_t>(smem_F32[lr * 64 + lc]);
                    }
                }
            }
        }
        __syncthreads();
    }
}

void w4a16_gemm_packedGPU(torch::Tensor &output, const torch::Tensor &input, const torch::Tensor &packed_params, int M, int K, int N) {
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    int num_blocks_k = (K + 31) / 32;
    const int64_t stride20_size = (int64_t)N * (int64_t)num_blocks_k * 20;
    const int64_t num_tiles_row = (K + 127) / 128;
    const int64_t num_tiles_col = (N + 63) / 64;
    const int64_t tile_packed_size = num_tiles_row * num_tiles_col * 4352;

    torch::Tensor input_bf16 = input;
    if (input.scalar_type() != torch::kBFloat16) {
        input_bf16 = input.to(torch::kBFloat16);
    }

    if (packed_params.numel() == tile_packed_size) {
        w4a16_gemm_fused_packed(output, input_bf16, packed_params, K, N, 0);
        return;
    }

    if (packed_params.numel() == stride20_size && (num_blocks_k % 4 == 0)) {
        static thread_local torch::Tensor packed_cache;
        static thread_local const void *cached_ptr = nullptr;
        static thread_local int64_t cached_numel = 0;
        static thread_local int64_t cached_K = 0;
        static thread_local int64_t cached_N = 0;

        if (!packed_cache.defined() || cached_ptr != packed_params.data_ptr() || cached_numel != packed_params.numel() ||
            cached_K != K || cached_N != N) {
            packed_cache = pack_stride20_to_tile(packed_params, K, N);
            cached_ptr = packed_params.data_ptr();
            cached_numel = packed_params.numel();
            cached_K = K;
            cached_N = N;
        }

        w4a16_gemm_fused_packed(output, input_bf16, packed_cache, K, N, 0);
        return;
    }

    dim3 block(256);
    dim3 grid((N + 127) / 128, (M + 127) / 128);

    const bool aligned = (M % 128 == 0) && (N % 128 == 0) && (K % 128 == 0);
    if (aligned) {
        hipLaunchKernelGGL((w4a16_gemm_packedGPU_rocwmma<true>), grid, block, 0, 0,
                           (bfloat16_t *)output.data_ptr(),
                           (const bfloat16_t *)input_bf16.data_ptr(),
                           packed_params.data_ptr<uint8_t>(),
                           M, K, N, num_blocks_k);
    } else {
        hipLaunchKernelGGL((w4a16_gemm_packedGPU_rocwmma<false>), grid, block, 0, 0,
                           (bfloat16_t *)output.data_ptr(),
                           (const bfloat16_t *)input_bf16.data_ptr(),
                           packed_params.data_ptr<uint8_t>(),
                           M, K, N, num_blocks_k);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "HIP Error in w4a16_gemm_packedGPU: " << hipGetErrorString(err) << std::endl;
    }
}

} // namespace hipkernels
