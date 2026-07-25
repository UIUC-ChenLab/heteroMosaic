#pragma once
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace hipkernels {

void launch_lm_head_forward(float *out_ptr, const hip_bfloat16 *input_ptr, const hip_bfloat16 *weight_ptr, int batch_size, int hidden_size,
                            int vocab_size, hipStream_t stream);

void launch_lm_head_forward(hip_bfloat16 *out_ptr, const hip_bfloat16 *input_ptr, const hip_bfloat16 *weight_ptr, int batch_size,
                            int hidden_size, int vocab_size, hipStream_t stream);

} // namespace hipkernels
