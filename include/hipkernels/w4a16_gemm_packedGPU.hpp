#ifndef W4A16_GEMM_PACKEDGPU_HPP
#define W4A16_GEMM_PACKEDGPU_HPP

#include <torch/torch.h>

namespace hipkernels {

void w4a16_gemm_packedGPU(
    torch::Tensor &output,
    const torch::Tensor &input,
    const torch::Tensor &packed_params,
    int M,
    int K,
    int N
);

} // namespace hipkernels

#endif // W4A16_GEMM_PACKEDGPU_HPP
