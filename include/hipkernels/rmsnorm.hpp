#pragma once
#include <torch/torch.h>

void launch_rmsnorm(torch::Tensor &output, const torch::Tensor &input, const torch::Tensor &weight, float epsilon,
                    bool gemma_style = false);
