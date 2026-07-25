#pragma once

#include <hip/hip_runtime.h>
#include <torch/torch.h>

namespace hipkernels {

// Launch a 1D softmax on the current HIP stream.
// Returns true when launched successfully, false if input constraints are not met.
bool launch_softmax_1d(const torch::Tensor &input, torch::Tensor &output, hipStream_t stream = 0);

} // namespace hipkernels
