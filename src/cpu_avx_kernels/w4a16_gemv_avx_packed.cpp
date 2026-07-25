#include "cpu_avx_kernels/w4a16_gemv_avx_packed.hpp"
#include "unified_llm_w4a16_hetero/npuSetup.hpp"
#include <ATen/Parallel.h>
#include <atomic>
#include <iostream>
#include <torch/torch.h>
#include <vector>

#ifdef __AVX512F__
#include <immintrin.h>
#endif
#include <omp.h>

#include <hip/hip_runtime.h>

// Helper to broadcast BF16 pair to 512-bit vector
// val0 at even indices, val1 at odd indices
#ifdef __AVX512F__
inline __m512bh broadcast_bf16_pair(at::BFloat16 val0, at::BFloat16 val1) {
    // Cast to uint16_t relative bit patterns
    uint16_t u0 = *reinterpret_cast<uint16_t *>(&val0);
    uint16_t u1 = *reinterpret_cast<uint16_t *>(&val1);
    uint32_t combined = (static_cast<uint32_t>(u1) << 16) | u0;
    return (__m512bh)_mm512_set1_epi32(combined);
}
#endif

void w4a16_gemv_cpu_fused_packed(at::Tensor &output, const at::Tensor &input, const at::Tensor &packed_params, int64_t in_features,
                                 int64_t out_features, hipEvent_t hip_event, int num_threads, int64_t start_col, int64_t end_col) {

    // Spin-wait for GPU event
    while (true) {
        bool gpu_ready = (hip_event == nullptr) || (hipEventQuery(hip_event) == hipSuccess);
        if (gpu_ready) {
            if (debug_verbosity >= 3) {
                std::cout << "Hip Event Query Success" << std::endl;
            }
            break;
        }
        _mm_pause();
    }
    std::atomic_thread_fence(std::memory_order_acquire);

    const int PACKED_TILE_SIZE = 4352;
    const int WEIGHT_BYTES = 4096;
    const int SCALE_BYTES = 128;

    int num_tiles_k = (in_features + 127) / 128;
    int num_tiles_n = (out_features + 63) / 64;

    if (num_threads > 1) {
        at::set_num_threads(num_threads);
    }

    if (end_col == -1) {
        end_col = out_features;
    }

    int64_t start_tile = start_col / 64;
    int64_t end_tile = (end_col + 63) / 64;

    if (start_tile < 0)
        start_tile = 0;
    if (end_tile > num_tiles_n)
        end_tile = num_tiles_n;

    auto out_ptr = output.data_ptr<at::BFloat16>();
    auto in_ptr = input.data_ptr<at::BFloat16>();
    const uint8_t *packed_base = packed_params.data_ptr<uint8_t>();

    at::parallel_for(start_tile, end_tile, 1, [&](int64_t start, int64_t end) {
        for (int64_t bx = start; bx < end; ++bx) {
            int col_group = bx % 8;
            int col_offset = bx / 8;
            int tbg = 0;
            for (int g = 0; g < col_group; ++g) {
                tbg += ((num_tiles_n + 7 - g) / 8) * num_tiles_k;
            }
            int base_z = tbg + col_offset * num_tiles_k;

            alignas(64) float acc[64] = {0};

            for (int k_tile = 0; k_tile < num_tiles_k; ++k_tile) {
                const uint8_t *w_tile_base = packed_base + (size_t)(base_z + k_tile) * PACKED_TILE_SIZE;
                const at::BFloat16 *scale_ptr_bf16 = (const at::BFloat16 *)(w_tile_base + WEIGHT_BYTES);
                const uint8_t *zero_ptr_u8 = w_tile_base + WEIGHT_BYTES + SCALE_BYTES;

#ifdef __AVX512F__
                // Optimized BF16 Path
                // 1. Calculate Sum(Input) for the current tile (128 rows) - VECTORIZED
                int row_start = k_tile * 128;
                int r_limit = (row_start + 128 > in_features) ? (in_features - row_start) : 128;

                // Vectorized input sum using AVX512
                __m512 v_sum = _mm512_setzero_ps();
                int r = 0;
                for (; r + 16 <= r_limit; r += 16) {
                    __m256i v_in_raw = _mm256_loadu_si256((const __m256i *)&in_ptr[row_start + r]);
                    __m512 v_in_f = _mm512_cvtpbh_ps((__m256bh)v_in_raw);
                    v_sum = _mm512_add_ps(v_sum, v_in_f);
                }
                float input_sum = _mm512_reduce_add_ps(v_sum);
                // Handle remainder scalar
                for (; r < r_limit; ++r) {
                    input_sum += static_cast<float>(in_ptr[row_start + r]);
                }
                __m512 v_input_sum = _mm512_set1_ps(input_sum);

                // 2. Process Columns in 4 chunks (16 cols each)
                for (int c_chunk = 0; c_chunk < 4; ++c_chunk) {
                    int c_start = c_chunk * 16;
                    if (bx * 64 + c_start >= out_features)
                        continue;

                    __m256bh v_s_bh = (__m256bh)_mm256_loadu_si256((const __m256i *)&scale_ptr_bf16[c_start]);
                    __m512 v_s = _mm512_cvtpbh_ps(v_s_bh);

                    const uint8_t *zero_ptr_u8_chunk = zero_ptr_u8;
                    alignas(32) uint16_t zeros_bf16_arr[16];
                    for (int i = 0; i < 16; ++i) {
                        int c = c_start + i;
                        int zero_idx = (c / 8) * 16 + (c % 8);
                        uint8_t z_u8 = zero_ptr_u8_chunk[zero_idx];
                        float z_f = (float)z_u8;
                        at::BFloat16 z_bf = static_cast<at::BFloat16>(z_f);
                        zeros_bf16_arr[i] = *reinterpret_cast<uint16_t *>(&z_bf);
                    }
                    __m256bh v_z_bh = (__m256bh)_mm256_load_si256((const __m256i *)zeros_bf16_arr);
                    __m512 v_z = _mm512_cvtpbh_ps(v_z_bh);

                    // 4 Accumulators needed to hide FMA latency (Pipeline depth ~4)
                    __m512 v_acc0 = _mm512_setzero_ps();
                    __m512 v_acc1 = _mm512_setzero_ps();
                    __m512 v_acc2 = _mm512_setzero_ps();
                    __m512 v_acc3 = _mm512_setzero_ps();

                    int sc0 = c_chunk * 2;
                    int sc1 = c_chunk * 2 + 1;

                    // Iterate SR (blocks of 8 rows)
                    for (int sr = 0; sr < 16; ++sr) {
                        int r_base = sr * 8;
                        if (r_base >= r_limit)
                            break;

                        const uint8_t *p0 = w_tile_base + (sr * 8 + sc0) * 32;
                        const uint8_t *p1 = w_tile_base + (sr * 8 + sc1) * 32;

                        _mm_prefetch((const char *)(p0 + 64), _MM_HINT_T0);
                        _mm_prefetch((const char *)(p1 + 64), _MM_HINT_T0);

                        // Process 2 Quads (8 rows total)
                        for (int q = 0; q < 2; ++q) {
                            int r_quad = q * 4; // 0 or 4
                            if (r_base + r_quad >= r_limit)
                                break;

                            // Load 128-bit blocks (4 rows x 8 cols)
                            // Optimization: Load 4 rows at once instead of scalar loads
                            __m128i v0 = _mm_loadu_si128((const __m128i *)(p0 + r_quad * 4));
                            __m128i v1 = _mm_loadu_si128((const __m128i *)(p1 + r_quad * 4));

                            // Unpack to get Rows
                            // Low = Rows 0, 1 (Full 16 cols)
                            // High = Rows 2, 3 (Full 16 cols)
                            __m128i v_pair0 = _mm_unpacklo_epi32(v0, v1); // [R1_Hi, R1_Lo, R0_Hi, R0_Lo] -> [Row1, Row0]
                            __m128i v_pair1 = _mm_unpackhi_epi32(v0, v1); // [R3_Hi, R3_Lo, R2_Hi, R2_Lo] -> [Row3, Row2]

                            // --- Process Pair 0 (Row 0, 1) ---
                            if (r_base + r_quad < r_limit) {
                                // Unpack Nibbles for Pair 0
                                __m128i v_mask = _mm_set1_epi8(0x0F);
                                __m128i v_lo = _mm_and_si128(v_pair0, v_mask);
                                __m128i v_hi = _mm_and_si128(_mm_srli_epi16(v_pair0, 4), v_mask);

                                __m128i v_u8_lo = _mm_unpacklo_epi8(v_lo, v_hi);
                                __m128i v_u8_hi = _mm_unpackhi_epi8(v_lo, v_hi);

                                // Row 0
                                int ir = r_quad + 0;
                                float in0 = static_cast<float>(in_ptr[row_start + r_base + ir]);
                                __m512 v_in0 = _mm512_set1_ps(in0);
                                __m512 v_f_k = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm256_cvtepu8_epi16(v_u8_lo)));

                                if (q == 0)
                                    v_acc0 = _mm512_fmadd_ps(v_f_k, v_in0, v_acc0);
                                else
                                    v_acc2 = _mm512_fmadd_ps(v_f_k, v_in0, v_acc2); // Reuse Acc2 for Row 4

                                // Row 1
                                if (r_base + ir + 1 < r_limit) {
                                    float in1 = static_cast<float>(in_ptr[row_start + r_base + ir + 1]);
                                    __m512 v_in1 = _mm512_set1_ps(in1);
                                    __m512 v_f_k1 = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm256_cvtepu8_epi16(v_u8_hi)));

                                    if (q == 0)
                                        v_acc1 = _mm512_fmadd_ps(v_f_k1, v_in1, v_acc1);
                                    else
                                        v_acc3 = _mm512_fmadd_ps(v_f_k1, v_in1, v_acc3); // Reuse Acc3 for Row 5
                                }
                            }

                            // --- Process Pair 1 (Row 2, 3) ---
                            if (r_base + r_quad + 2 < r_limit) {
                                // Unpack Nibbles for Pair 1
                                __m128i v_mask = _mm_set1_epi8(0x0F);
                                __m128i v_lo = _mm_and_si128(v_pair1, v_mask);
                                __m128i v_hi = _mm_and_si128(_mm_srli_epi16(v_pair1, 4), v_mask);

                                __m128i v_u8_lo = _mm_unpacklo_epi8(v_lo, v_hi);
                                __m128i v_u8_hi = _mm_unpackhi_epi8(v_lo, v_hi);

                                // Row 2
                                int ir = r_quad + 2;
                                int acc_idx = (q == 0) ? 2 : 0; // Use Acc2 for Row 2, Acc0 for Row 6 (Wrap reuse)

                                float in0 = static_cast<float>(in_ptr[row_start + r_base + ir]);
                                __m512 v_in0 = _mm512_set1_ps(in0);
                                __m512 v_f_k = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm256_cvtepu8_epi16(v_u8_lo)));

                                if (q == 0)
                                    v_acc2 = _mm512_fmadd_ps(v_f_k, v_in0, v_acc2);
                                else
                                    v_acc0 = _mm512_fmadd_ps(v_f_k, v_in0, v_acc0); // Reuse Acc0

                                // Row 3 (Offset 3)
                                if (r_base + ir + 1 < r_limit) {
                                    float in1 = static_cast<float>(in_ptr[row_start + r_base + ir + 1]);
                                    __m512 v_in1 = _mm512_set1_ps(in1);
                                    __m512 v_f_k1 = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm256_cvtepu8_epi16(v_u8_hi)));

                                    if (q == 0)
                                        v_acc3 = _mm512_fmadd_ps(v_f_k1, v_in1, v_acc3);
                                    else
                                        v_acc1 = _mm512_fmadd_ps(v_f_k1, v_in1, v_acc1); // Reuse Acc1
                                }
                            }
                        }
                    }

                    // Sum accumulators
                    __m512 v_acc_unscaled = _mm512_add_ps(_mm512_add_ps(v_acc0, v_acc1), _mm512_add_ps(v_acc2, v_acc3));

                    // Finalize Accumulation for this tile
                    // Result = (Acc_unscaled - Z * InputSum) * S + Acc_existing
                    __m512 v_term = _mm512_mul_ps(v_z, v_input_sum);
                    __m512 v_diff = _mm512_sub_ps(v_acc_unscaled, v_term);
                    __m512 v_scaled = _mm512_mul_ps(v_diff, v_s);

                    __m512 v_acc_existing = _mm512_load_ps(&acc[c_start]);
                    __m512 v_result = _mm512_add_ps(v_acc_existing, v_scaled);

                    _mm512_store_ps(&acc[c_start], v_result);
                }
#else
                // Scalar Fallback (unchanged)
                float tile_scales[64];
                float tile_zeros[64];
                for (int i = 0; i < 64; ++i) {
                    tile_scales[i] = static_cast<float>(scale_ptr_bf16[i]);
                    int zero_idx = (i / 8) * 16 + (i % 8);
                    tile_zeros[i] = (float)zero_ptr_u8[zero_idx];
                }
                int row_start = k_tile * 128;
                int r_limit = (row_start + 128 > in_features) ? (in_features - row_start) : 128;
                for (int r = 0; r < r_limit; ++r) {
                    float in_val = static_cast<float>(in_ptr[row_start + r]);
                    int SR = r >> 3;
                    int IR = r & 7;
                    for (int c = 0; c < 64; ++c) {
                        if (bx * 64 + c >= out_features) continue;
                        int SC = c >> 3;
                        int IC = c & 7;
                        int byte_off = ((SR * 8 + SC) << 5) + (IR << 2) + (IC >> 1);
                        uint8_t packed = w_tile_base[byte_off];
                        int q4 = (IC & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
                        float w_val = ((float)q4 - tile_zeros[c]) * tile_scales[c];
                        acc[c] += w_val * in_val;
                    }
                }
#endif
            }

            int out_start = bx * 64;
            for (int c = 0; c < 64; ++c) {
                if (out_start + c < out_features) {
                    out_ptr[out_start + c] = static_cast<at::BFloat16>(acc[c]);
                }
            }
        }
    });
}
