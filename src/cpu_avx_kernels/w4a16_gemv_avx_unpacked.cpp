#include "cpu_avx_kernels/w4a16_gemv_avx_unpacked.hpp"
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

// Helper for bf16 to float
inline float bf16_to_fp32(uint16_t val) {
    uint32_t val32 = static_cast<uint32_t>(val) << 16;
    float f;
    std::memcpy(&f, &val32, sizeof(float));
    return f;
}

void w4a16_gemv_cpu_fused_unpacked(at::Tensor &output, const at::Tensor &input, const at::Tensor &qweights, const at::Tensor &scales,
                                   const at::Tensor &zeros, int64_t in_features, int64_t out_features, int64_t group_size,
                                   hipEvent_t hip_event, int num_threads) {

    // Spin-wait for GPU event
    while (true) {
        bool gpu_ready = (hip_event == nullptr) || (hipEventQuery(hip_event) == hipSuccess);
        if (gpu_ready)
            break;
        _mm_pause();
    }
    std::atomic_thread_fence(std::memory_order_acquire);

    if (num_threads > 1) {
        at::set_num_threads(num_threads);
    }

    auto out_ptr = output.data_ptr<at::BFloat16>();
    auto in_ptr = input.data_ptr<at::BFloat16>();

    const uint8_t *qw_ptr = qweights.data_ptr<uint8_t>();             // [N, K/2]
    const at::BFloat16 *scales_ptr = scales.data_ptr<at::BFloat16>(); // [N, Groups]
    const int8_t *zeros_ptr = zeros.data_ptr<int8_t>();               // [N, Groups]

    int64_t num_groups = in_features / group_size;

    // Precompute Input Sums per Group
    // This allows us to simplify the inner loop: Sum((w-z)*in) = Sum(w*in) - z*Sum(in)
    // Removing the subtraction from the inner loop saves K*N operations.
    std::vector<float> group_input_sums(num_groups);

// Efficiently compute sums using AVX512 if available
#ifdef __AVX512F__
    for (int64_t g = 0; g < num_groups; ++g) {
        int64_t k_start = g * group_size;
        int64_t k_end = k_start + group_size;
        __m512 v_g_sum = _mm512_setzero_ps();

        for (int64_t k = k_start; k < k_end; k += 32) {
            // Load 32 inputs (16 + 16)
            // Convert to float and accumulate
            __m512i v_in_bf16 = _mm512_loadu_si512((const void *)&in_ptr[k]);

            __m512i v_lo_bf16 = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(v_in_bf16));
            __m512i v_lo_u32 = _mm512_slli_epi32(v_lo_bf16, 16);
            __m512 v_in_f_lo = _mm512_castsi512_ps(v_lo_u32);

            __m256i v_hi_256 = _mm512_extracti64x4_epi64(v_in_bf16, 1);
            __m512i v_hi_bf16 = _mm512_cvtepu16_epi32(v_hi_256);
            __m512i v_hi_u32 = _mm512_slli_epi32(v_hi_bf16, 16);
            __m512 v_in_f_hi = _mm512_castsi512_ps(v_hi_u32);

            v_g_sum = _mm512_add_ps(v_g_sum, v_in_f_lo);
            v_g_sum = _mm512_add_ps(v_g_sum, v_in_f_hi);
        }
        group_input_sums[g] = _mm512_reduce_add_ps(v_g_sum);
    }
#else
    // Fallback
    for (int64_t g = 0; g < num_groups; ++g) {
        float s = 0;
        for (int64_t k = g * group_size; k < (g + 1) * group_size; ++k) {
            s += static_cast<float>(in_ptr[k]);
        }
        group_input_sums[g] = s;
    }
#endif

    // Parallel Output Loop
    int64_t K = in_features;
    int64_t N = out_features;
    int64_t K_div_2 = K / 2;

    at::parallel_for(0, N, 1, [&](int64_t start, int64_t end) {
        for (int64_t n = start; n < end; ++n) {
            float global_acc = 0.0f;

            // Pointers for this row
            const uint8_t *row_w = qw_ptr + n * K_div_2;
            const at::BFloat16 *row_s = scales_ptr + n * num_groups;
            const int8_t *row_z = zeros_ptr + n * num_groups;

#ifdef __AVX512F__
            // Unrolling 4 groups per iteration
            int64_t g = 0;
            for (; g < num_groups - 3; g += 4) {
                // Prefetch scales/zeros for 4 groups
                float s0 = static_cast<float>(row_s[g]);
                float z0_val = (float)row_z[g];
                float i_sum0 = group_input_sums[g];

                float s1 = static_cast<float>(row_s[g + 1]);
                float z1_val = (float)row_z[g + 1];
                float i_sum1 = group_input_sums[g + 1];

                float s2 = static_cast<float>(row_s[g + 2]);
                float z2_val = (float)row_z[g + 2];
                float i_sum2 = group_input_sums[g + 2];

                float s3 = static_cast<float>(row_s[g + 3]);
                float z3_val = (float)row_z[g + 3];
                float i_sum3 = group_input_sums[g + 3];

                int64_t k_start0 = g * group_size;
                int64_t k_w_start0 = k_start0 / 2;

                int64_t k_start1 = (g + 1) * group_size;
                int64_t k_w_start1 = k_start1 / 2;

                int64_t k_start2 = (g + 2) * group_size;
                int64_t k_w_start2 = k_start2 / 2;

                int64_t k_start3 = (g + 3) * group_size;
                int64_t k_w_start3 = k_start3 / 2;

                // Prefetch next 4 groups of weights (4 * 64 bytes = 256 bytes ahead)
                // Assuming next iteration g+4 is valid.
                // We access row_w + k_w_start0 + 256
                _mm_prefetch((const char *)(row_w + k_w_start0 + 256), _MM_HINT_T0);
                _mm_prefetch((const char *)(row_w + k_w_start1 + 256), _MM_HINT_T0);
                _mm_prefetch((const char *)(row_w + k_w_start2 + 256), _MM_HINT_T0);
                _mm_prefetch((const char *)(row_w + k_w_start3 + 256), _MM_HINT_T0);

                // Accumulators (Split for Latency Hiding)
                __m512 v_sum0_lo = _mm512_setzero_ps();
                __m512 v_sum0_hi = _mm512_setzero_ps();
                __m512 v_sum1_lo = _mm512_setzero_ps();
                __m512 v_sum1_hi = _mm512_setzero_ps();
                __m512 v_sum2_lo = _mm512_setzero_ps();
                __m512 v_sum2_hi = _mm512_setzero_ps();
                __m512 v_sum3_lo = _mm512_setzero_ps();
                __m512 v_sum3_hi = _mm512_setzero_ps();

                for (int i = 0; i < 4; ++i) {
// Helper Macro: NO SUBTRACTION HERE
// Just W * In
#define PROCESS_GROUP(IDX, K_W_START, K_START, V_SUM_LO, V_SUM_HI)                                                                         \
    int w_offset##IDX = K_W_START + i * 16;                                                                                                \
    __m128i v_w_packed##IDX = _mm_loadu_si128((const __m128i *)(row_w + w_offset##IDX));                                                   \
    __m128i v_w_lo##IDX = _mm_and_si128(v_w_packed##IDX, _mm_set1_epi8(0x0F));                                                             \
    __m128i v_w_hi##IDX = _mm_and_si128(_mm_srli_epi16(v_w_packed##IDX, 4), _mm_set1_epi8(0x0F));                                          \
    __m128i v_w_u8_lo##IDX = _mm_unpacklo_epi8(v_w_lo##IDX, v_w_hi##IDX);                                                                  \
    __m128i v_w_u8_hi##IDX = _mm_unpackhi_epi8(v_w_lo##IDX, v_w_hi##IDX);                                                                  \
    __m512i v_w_i32_lo##IDX = _mm512_cvtepu8_epi32(v_w_u8_lo##IDX);                                                                        \
    __m512i v_w_i32_hi##IDX = _mm512_cvtepu8_epi32(v_w_u8_hi##IDX);                                                                        \
    __m512 v_w_f_lo##IDX = _mm512_cvtepi32_ps(v_w_i32_lo##IDX);                                                                            \
    __m512 v_w_f_hi##IDX = _mm512_cvtepi32_ps(v_w_i32_hi##IDX);                                                                            \
    int64_t k_base##IDX = K_START + i * 32;                                                                                                \
    __m512i v_in_bf16_##IDX = _mm512_loadu_si512((const void *)&in_ptr[k_base##IDX]);                                                      \
    __m512 v_in_f_lo##IDX, v_in_f_hi##IDX;                                                                                                 \
    {                                                                                                                                      \
        __m512i v_lo_bf16 = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(v_in_bf16_##IDX));                                                \
        __m512i v_lo_u32 = _mm512_slli_epi32(v_lo_bf16, 16);                                                                               \
        v_in_f_lo##IDX = _mm512_castsi512_ps(v_lo_u32);                                                                                    \
        __m256i v_hi_256 = _mm512_extracti64x4_epi64(v_in_bf16_##IDX, 1);                                                                  \
        __m512i v_hi_bf16 = _mm512_cvtepu16_epi32(v_hi_256);                                                                               \
        __m512i v_hi_u32 = _mm512_slli_epi32(v_hi_bf16, 16);                                                                               \
        v_in_f_hi##IDX = _mm512_castsi512_ps(v_hi_u32);                                                                                    \
    }                                                                                                                                      \
    V_SUM_LO = _mm512_fmadd_ps(v_w_f_lo##IDX, v_in_f_lo##IDX, V_SUM_LO);                                                                   \
    V_SUM_HI = _mm512_fmadd_ps(v_w_f_hi##IDX, v_in_f_hi##IDX, V_SUM_HI);

                    PROCESS_GROUP(0, k_w_start0, k_start0, v_sum0_lo, v_sum0_hi);
                    PROCESS_GROUP(1, k_w_start1, k_start1, v_sum1_lo, v_sum1_hi);
                    PROCESS_GROUP(2, k_w_start2, k_start2, v_sum2_lo, v_sum2_hi);
                    PROCESS_GROUP(3, k_w_start3, k_start3, v_sum3_lo, v_sum3_hi);
                }
#undef PROCESS_GROUP

                // Reduce sums and Apply (W_dot_In - Z * Sum_In) * S

                __m512 v_sum0 = _mm512_add_ps(v_sum0_lo, v_sum0_hi);
                float w_dot_in0 = _mm512_reduce_add_ps(v_sum0);
                global_acc += (w_dot_in0 - z0_val * i_sum0) * s0;

                __m512 v_sum1 = _mm512_add_ps(v_sum1_lo, v_sum1_hi);
                float w_dot_in1 = _mm512_reduce_add_ps(v_sum1);
                global_acc += (w_dot_in1 - z1_val * i_sum1) * s1;

                __m512 v_sum2 = _mm512_add_ps(v_sum2_lo, v_sum2_hi);
                float w_dot_in2 = _mm512_reduce_add_ps(v_sum2);
                global_acc += (w_dot_in2 - z2_val * i_sum2) * s2;

                __m512 v_sum3 = _mm512_add_ps(v_sum3_lo, v_sum3_hi);
                float w_dot_in3 = _mm512_reduce_add_ps(v_sum3);
                global_acc += (w_dot_in3 - z3_val * i_sum3) * s3;
            }

            // Cleanup leftover groups
            for (; g < num_groups; ++g) {
                float s = static_cast<float>(row_s[g]);
                float z_val = (float)row_z[g];
                float i_sum = group_input_sums[g];

                int64_t k_start = g * group_size;
                int64_t k_w_start = k_start / 2;

                __m512 v_sum = _mm512_setzero_ps();

                for (int i = 0; i < 4; ++i) {
                    int w_offset = k_w_start + i * 16;
                    __m128i v_w_packed = _mm_loadu_si128((const __m128i *)(row_w + w_offset));
                    __m128i v_w_lo = _mm_and_si128(v_w_packed, _mm_set1_epi8(0x0F));
                    __m128i v_w_hi = _mm_and_si128(_mm_srli_epi16(v_w_packed, 4), _mm_set1_epi8(0x0F));
                    __m128i v_w_u8_lo = _mm_unpacklo_epi8(v_w_lo, v_w_hi);
                    __m128i v_w_u8_hi = _mm_unpackhi_epi8(v_w_lo, v_w_hi);

                    __m512i v_w_i32_lo = _mm512_cvtepu8_epi32(v_w_u8_lo);
                    __m512i v_w_i32_hi = _mm512_cvtepu8_epi32(v_w_u8_hi);

                    // No subtraction
                    __m512 v_w_f_lo = _mm512_cvtepi32_ps(v_w_i32_lo);
                    __m512 v_w_f_hi = _mm512_cvtepi32_ps(v_w_i32_hi);

                    int64_t k_base = k_start + i * 32;
                    __m512i v_in_bf16 = _mm512_loadu_si512((const void *)&in_ptr[k_base]);
                    __m512 v_in_f_lo, v_in_f_hi;
                    {
                        __m512i v_lo_bf16 = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(v_in_bf16));
                        __m512i v_lo_u32 = _mm512_slli_epi32(v_lo_bf16, 16);
                        v_in_f_lo = _mm512_castsi512_ps(v_lo_u32);
                        __m256i v_hi_256 = _mm512_extracti64x4_epi64(v_in_bf16, 1);
                        __m512i v_hi_bf16 = _mm512_cvtepu16_epi32(v_hi_256);
                        __m512i v_hi_u32 = _mm512_slli_epi32(v_hi_bf16, 16);
                        v_in_f_hi = _mm512_castsi512_ps(v_hi_u32);
                    }
                    v_sum = _mm512_fmadd_ps(v_w_f_lo, v_in_f_lo, v_sum);
                    v_sum = _mm512_fmadd_ps(v_w_f_hi, v_in_f_hi, v_sum);
                }
                float w_dot_in = _mm512_reduce_add_ps(v_sum);
                global_acc += (w_dot_in - z_val * i_sum) * s;
            }

#else
            // Fallback Scalar Loop 
            // Reuse input_sums
             for (int64_t g = 0; g < num_groups; ++g) {
                float s = static_cast<float>(row_s[g]);
                int8_t z = row_z[g]; 
                float i_sum = group_input_sums[g];

                int64_t k_start = g * group_size;
                int64_t k_end = k_start + group_size;
                int64_t k_div_2_start = k_start / 2;
                int64_t k_div_2_end = k_end / 2;

                float w_dot_in = 0.0f;

                for (int64_t k2 = k_div_2_start; k2 < k_div_2_end; ++k2) {
                    uint8_t packed = row_w[k2];
                    int32_t w0 = (int32_t)(packed & 0x0F);
                    int32_t w1 = (int32_t)((packed >> 4) & 0x0F);

                    int64_t k = k2 * 2;
                    w_dot_in += (float)w0 * static_cast<float>(in_ptr[k]);
                    w_dot_in += (float)w1 * static_cast<float>(in_ptr[k + 1]);
                }
                global_acc += (w_dot_in - (float)z * i_sum) * s;
            }
#endif
            out_ptr[n] = static_cast<at::BFloat16>(global_acc);
        }
    });
}
