#pragma once
#include <torch/torch.h>

void launch_rope(torch::Tensor &q, torch::Tensor &k, int start_pos, float theta);
