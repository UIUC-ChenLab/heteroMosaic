#pragma once

#include <hip/hip_runtime.h>
#include <torch/torch.h>

namespace hipkernels {

void w4a16_gemm_unpacked_fused(torch::Tensor &output, const torch::Tensor &input, const torch::Tensor &qweights,
                               const torch::Tensor &scales, const torch::Tensor &zeros, int64_t in_features, int64_t out_features,
                               int64_t group_size, hipStream_t stream = 0);

} // namespace hipkernels
