#include "hipkernels/lm_head.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

namespace hipkernels {

// Helper Functions
static __device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor(x, offset);
    }
    return x;
}

// Optimized Kernel: 4 Rows per Block + 128-bit Vectorization (float4)
// Loads 16 bytes (8 BF16s) per thread instruction.
// Optimized Kernel: 4 Rows per Block + 128-bit Vectorization (float4)
// Loads 16 bytes (8 BF16s) per thread instruction.
template <int block_size, int rows_per_block, typename OutType>
static __global__ void mul_mat_vec_bf16_generic_opt(
    const hip_bfloat16 * __restrict__ x, // Weights [Vocab, Hidden]
    const hip_bfloat16 * __restrict__ y, // Input [Hidden]
    OutType * __restrict__ dst,          // Output [Vocab]
    const int ncols,                     // Hidden size
    const int nrows                      // Vocab size
) {
    const int start_row = blockIdx.x * rows_per_block;
    const int tid = threadIdx.x;

    // Vectorize to float4 (16 bytes = 8 x BF16)
    // ncols must be divisible by 8. (4096 % 8 == 0)
    int ncols_vec = ncols / 8;

    const float4* y_vec = (const float4*)y;

    #pragma unroll
    for (int r = 0; r < rows_per_block; ++r) {
        int row = start_row + r;
        if (row >= nrows) break;

        const float4* w_vec = (const float4*)(x + row * ncols);
        
        float sum = 0.0f;

        for (int col = tid; col < ncols_vec; col += block_size) {
             float4 w_val = w_vec[col]; // 16 bytes
             float4 in_val = y_vec[col]; 
             
             // Unpack
             hip_bfloat16 w_elem[8];
             hip_bfloat16 in_elem[8];
             
             *(float4*)w_elem = w_val;
             *(float4*)in_elem = in_val;
             
             // UNROLL 8
             sum += static_cast<float>(w_elem[0]) * static_cast<float>(in_elem[0]);
             sum += static_cast<float>(w_elem[1]) * static_cast<float>(in_elem[1]);
             sum += static_cast<float>(w_elem[2]) * static_cast<float>(in_elem[2]);
             sum += static_cast<float>(w_elem[3]) * static_cast<float>(in_elem[3]);
             sum += static_cast<float>(w_elem[4]) * static_cast<float>(in_elem[4]);
             sum += static_cast<float>(w_elem[5]) * static_cast<float>(in_elem[5]);
             sum += static_cast<float>(w_elem[6]) * static_cast<float>(in_elem[6]);
             sum += static_cast<float>(w_elem[7]) * static_cast<float>(in_elem[7]);
        }

        sum = warp_reduce_sum(sum);

        __shared__ float shared_sum[64]; 
        
        int warp_id = tid / warpSize;
        int lane_id = tid % warpSize;
        
        if (lane_id == 0) shared_sum[warp_id] = sum;
        __syncthreads();
        
        if (tid < warpSize) {
             int num_warps = blockDim.x / warpSize;
             float val = (tid < num_warps) ? shared_sum[tid] : 0.0f;
             val = warp_reduce_sum(val);
             if (tid == 0) dst[row] = static_cast<OutType>(val);
        }
        __syncthreads();
    }
}

// Launcher Float Output
void launch_lm_head_forward(
    float* out_ptr,
    const hip_bfloat16* input_ptr,
    const hip_bfloat16* weight_ptr,
    int batch_size,
    int hidden_size,
    int vocab_size,
    hipStream_t stream
) {
    const int block_size = 256;
    const int rows_per_block = 4;
    
    dim3 block(block_size);
    dim3 grid((vocab_size + rows_per_block - 1) / rows_per_block);
    
    mul_mat_vec_bf16_generic_opt<block_size, rows_per_block, float><<<grid, block, 0, stream>>>(
        weight_ptr,
        input_ptr,
        out_ptr,
        hidden_size,
        vocab_size
    );
}

// Launcher BF16 Output
void launch_lm_head_forward(
    hip_bfloat16* out_ptr,
    const hip_bfloat16* input_ptr,
    const hip_bfloat16* weight_ptr,
    int batch_size,
    int hidden_size,
    int vocab_size,
    hipStream_t stream
) {
    const int block_size = 256;
    const int rows_per_block = 4;
    
    dim3 block(block_size);
    dim3 grid((vocab_size + rows_per_block - 1) / rows_per_block);
    
    mul_mat_vec_bf16_generic_opt<block_size, rows_per_block, hip_bfloat16><<<grid, block, 0, stream>>>(
        weight_ptr,
        input_ptr,
        out_ptr,
        hidden_size,
        vocab_size
    );
}

} // namespace hipkernels
