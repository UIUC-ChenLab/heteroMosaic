#include "hipkernels/embedding.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

namespace hipkernels {

template<typename T>
__global__ void embedding_forward_kernel(const T* __restrict__ weight, const int64_t* __restrict__ input_ids, T* __restrict__ output,
                                         int hidden_size, int num_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread ID handling token * hidden_dim
    
    // One thread per element? Or one thread per token?
    // Let's do one thread per embedded element to maximize parallelism.
    // Total threads needed = num_tokens * hidden_size.
    
    int total_elements = num_tokens * hidden_size;
    
    if (idx < total_elements) {
        int token_idx = idx / hidden_size;
        int dim_idx = idx % hidden_size;
        
        int64_t token_id = input_ids[token_idx];
        
        // Simple lookup: weight[token_id, dim_idx]
        // Flattened: weight[token_id * hidden_size + dim_idx]
        
        output[idx] = weight[token_id * hidden_size + dim_idx];
    }
}

void launch_embedding_forward(const torch::Tensor &weight, const torch::Tensor &input_ids, torch::Tensor &output, hipStream_t stream) {
    int num_tokens = input_ids.numel();
    int hidden_size = weight.size(1);
    
    int total_elements = num_tokens * hidden_size;
    int block = 256;
    int grid = (total_elements + block - 1) / block;

    if (weight.dtype() == torch::kBFloat16) {
        embedding_forward_kernel<hip_bfloat16><<<grid, block, 0, stream>>>(
            (const hip_bfloat16*)weight.data_ptr(), 
            (const int64_t*)input_ids.data_ptr(), 
            (hip_bfloat16*)output.data_ptr(), 
            hidden_size, num_tokens);
    } else {
        // Fallback for float
         embedding_forward_kernel<float><<<grid, block, 0, stream>>>(
            (const float*)weight.data_ptr(), 
            (const int64_t*)input_ids.data_ptr(), 
            (float*)output.data_ptr(), 
            hidden_size, num_tokens);
    }
     hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP Embedding error: ") + hipGetErrorString(err));
    }
}

} // namespace hipkernels
