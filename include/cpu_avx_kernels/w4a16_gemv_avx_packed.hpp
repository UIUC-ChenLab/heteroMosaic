#ifndef W4A16_GEMV_AVX_PACKED_HPP
#define W4A16_GEMV_AVX_PACKED_HPP

#include <ATen/ATen.h>
#include <hip/hip_runtime.h>

void w4a16_gemv_cpu_fused_packed(at::Tensor &output, const at::Tensor &input, const at::Tensor &packed_params, int64_t in_features,
                                 int64_t out_features, hipEvent_t hip_event = nullptr, int num_threads = 1, int64_t start_col = 0,
                                 int64_t end_col = -1);

#endif // W4A16_GEMV_AVX_PACKED_HPP
