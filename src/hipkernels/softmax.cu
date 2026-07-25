#include "hipkernels/softmax.hpp"

#include <c10/hip/HIPStream.h>
#include <cfloat>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace hipkernels {

template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_float<half>(half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_float<__hip_bfloat16>(__hip_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ float from_float<float>(float v) { return v; }
template <> __device__ __forceinline__ half from_float<half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __hip_bfloat16 from_float<__hip_bfloat16>(float v) { return __float2bfloat16(v); }

template <typename T>
__global__ void softmax_1d_kernel(const T *__restrict__ input, T *__restrict__ output, int64_t n) {
    extern __shared__ float shared[];
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    float local_max = -FLT_MAX;
    for (int64_t i = tid; i < n; i += stride) {
        local_max = fmaxf(local_max, to_float<T>(input[i]));
    }

    shared[tid] = local_max;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    const float max_val = shared[0];

    float local_sum = 0.0f;
    for (int64_t i = tid; i < n; i += stride) {
        local_sum += __expf(to_float<T>(input[i]) - max_val);
    }

    shared[tid] = local_sum;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    const float inv_sum = (shared[0] > 0.0f) ? (1.0f / shared[0]) : 0.0f;

    for (int64_t i = tid; i < n; i += stride) {
        float p = __expf(to_float<T>(input[i]) - max_val) * inv_sum;
        output[i] = from_float<T>(p);
    }
}

bool launch_softmax_1d(const torch::Tensor &input, torch::Tensor &output, hipStream_t stream) {
    if (!input.is_cuda() || !output.is_cuda()) {
        return false;
    }
    if (!input.is_contiguous() || !output.is_contiguous()) {
        return false;
    }
    if (input.dim() != 1 || output.dim() != 1) {
        return false;
    }
    if (input.numel() != output.numel() || input.dtype() != output.dtype()) {
        return false;
    }

    const int64_t n = input.numel();
    if (n <= 0) {
        return true;
    }

    const int block = 256;
    const int grid = 1;
    const size_t shmem = static_cast<size_t>(block) * sizeof(float);
    if (stream == 0) {
        stream = c10::hip::getCurrentHIPStream().stream();
    }

    if (input.dtype() == torch::kFloat32) {
        softmax_1d_kernel<float><<<grid, block, shmem, stream>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
    } else if (input.dtype() == torch::kFloat16) {
        softmax_1d_kernel<half><<<grid, block, shmem, stream>>>(reinterpret_cast<const half *>(input.data_ptr<at::Half>()),
                                                                reinterpret_cast<half *>(output.data_ptr<at::Half>()), n);
    } else if (input.dtype() == torch::kBFloat16) {
        softmax_1d_kernel<__hip_bfloat16><<<grid, block, shmem, stream>>>(
            reinterpret_cast<const __hip_bfloat16 *>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<__hip_bfloat16 *>(output.data_ptr<at::BFloat16>()), n);
    } else {
        return false;
    }

    hipError_t err = hipGetLastError();
    return (err == hipSuccess);
}

} // namespace hipkernels
