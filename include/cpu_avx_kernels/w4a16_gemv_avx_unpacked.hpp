#ifndef W4A16_GEMV_AVX_UNPACKED_HPP
#define W4A16_GEMV_AVX_UNPACKED_HPP

#include <ATen/ATen.h>
#include <hip/hip_runtime.h>

void w4a16_gemv_cpu_fused_unpacked(at::Tensor &output, const at::Tensor &input, const at::Tensor &qweights, const at::Tensor &scales,
                                   const at::Tensor &zeros, int64_t in_features, int64_t out_features, int64_t group_size,
                                   hipEvent_t hip_event = nullptr, int num_threads = 1);

#endif // W4A16_GEMV_AVX_UNPACKED_HPP
