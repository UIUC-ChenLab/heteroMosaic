#pragma once
#include <torch/torch.h>

namespace hipkernels {

void launch_embedding_forward(const torch::Tensor &weight, const torch::Tensor &input_ids, torch::Tensor &output, hipStream_t stream = 0);

} // namespace hipkernels
