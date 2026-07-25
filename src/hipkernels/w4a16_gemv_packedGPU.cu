#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <torch/torch.h>

using bfloat16_t = hip_bfloat16;

namespace hipkernels {

__device__ __forceinline__ __half2 bf16_to_half2(bfloat16_t b0, bfloat16_t b1) {
    float f0 = static_cast<float>(b0);
    float f1 = static_cast<float>(b1);
    return __float22half2_rn(make_float2(f0, f1));
}

__device__ __forceinline__ __half2 float_to_half2(float f0, float f1) {
    return __float22half2_rn(make_float2(f0, f1));
}

__global__ void __launch_bounds__(256)
gemv_packedGPU_kernel_opt(
    bfloat16_t* __restrict__ output,
    const bfloat16_t* __restrict__ input, 
    const uint8_t* __restrict__ packed_params,
    int K,
    int N,
    int num_blocks_k
) {
    // Stride 20 bytes (Stride-20 Layout)
    size_t row_stride = (size_t)num_blocks_k * 20;
    
    // 512 half2 (1024 bfloat16) shared memory
    __shared__ __half2 s_input[512]; 

    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int tid = threadIdx.x;
    
    // Rows processed per block calculation
    int rows_per_block_call = 48; // 8 warps * 6 rows
    int total_grid_rows = gridDim.x * rows_per_block_call;
    
    // --- PERSISTENT GRID STRIDE LOOP ---
    // Iterate over N (output rows) in chunks
    // Block-256 processes 48 rows at a time.
    // Total grid processes (80 blocks * 48 rows.
    for (int block_row_start = blockIdx.x * rows_per_block_call; block_row_start < N; block_row_start += total_grid_rows) {
        
        int row_start = block_row_start + warpId * 6;

        // Accumulators for 6 rows
        __half2 acc0_0 = __float2half2_rn(0.0f); __half2 acc1_0 = __float2half2_rn(0.0f);
        __half2 acc2_0 = __float2half2_rn(0.0f); __half2 acc3_0 = __float2half2_rn(0.0f);
        __half2 acc0_1 = __float2half2_rn(0.0f); __half2 acc1_1 = __float2half2_rn(0.0f);
        __half2 acc2_1 = __float2half2_rn(0.0f); __half2 acc3_1 = __float2half2_rn(0.0f);
        __half2 acc0_2 = __float2half2_rn(0.0f); __half2 acc1_2 = __float2half2_rn(0.0f);
        __half2 acc2_2 = __float2half2_rn(0.0f); __half2 acc3_2 = __float2half2_rn(0.0f);
        __half2 acc0_3 = __float2half2_rn(0.0f); __half2 acc1_3 = __float2half2_rn(0.0f);
        __half2 acc2_3 = __float2half2_rn(0.0f); __half2 acc3_3 = __float2half2_rn(0.0f);
        __half2 acc0_4 = __float2half2_rn(0.0f); __half2 acc1_4 = __float2half2_rn(0.0f);
        __half2 acc2_4 = __float2half2_rn(0.0f); __half2 acc3_4 = __float2half2_rn(0.0f);
        __half2 acc0_5 = __float2half2_rn(0.0f); __half2 acc1_5 = __float2half2_rn(0.0f);
        __half2 acc2_5 = __float2half2_rn(0.0f); __half2 acc3_5 = __float2half2_rn(0.0f);

        bfloat16_t next_vals[4];
        
        // Reset K Loop State
        int k_blk_start = 0;
        
        // Prologue Load (Prefetch first chunk of input)
        {
            int input_offset = k_blk_start * 32;
            int t_idx_0 = tid * 4;
            
            for(int i=0;i<4;++i) next_vals[i]=static_cast<bfloat16_t>(0.0f);

            int input_idx = input_offset + t_idx_0;
            if (input_idx < K) {
                 if (input_idx + 3 < K) {
                     const int2* v_ptr = reinterpret_cast<const int2*>(input + input_idx);
                     int2 v = *v_ptr;
                     bfloat16_t* ptr = reinterpret_cast<bfloat16_t*>(&v);
                     next_vals[0] = ptr[0]; next_vals[1] = ptr[1]; next_vals[2] = ptr[2]; next_vals[3] = ptr[3];
                 } else {
                     for(int i=0; i<4; ++i) if(input_idx+i < K) next_vals[i] = input[input_idx+i];
                 }
            }
        }

        // Inner K Loop
        for (; k_blk_start < num_blocks_k; k_blk_start += 32) {
            
            // 1. Move Prefetched input to Shared Memory
            s_input[tid*2]     = bf16_to_half2(next_vals[0], next_vals[1]);
            s_input[tid*2+1]   = bf16_to_half2(next_vals[2], next_vals[3]);
            
            __syncthreads();
            
            // 2. Prefetch NEXT input (for k+32)
            int next_k = k_blk_start + 32;
            if (next_k < num_blocks_k) {
                int input_offset = next_k * 32;
                int t_idx_0 = tid * 4;
                
                // Re-use next_vals
                int input_idx = input_offset + t_idx_0;
                if (input_idx < K) {
                     if (input_idx + 3 < K) {
                         const int2* v_ptr = reinterpret_cast<const int2*>(input + input_idx);
                         int2 v = *v_ptr;
                         bfloat16_t* ptr = reinterpret_cast<bfloat16_t*>(&v);
                         next_vals[0] = ptr[0]; next_vals[1] = ptr[1]; next_vals[2] = ptr[2]; next_vals[3] = ptr[3];
                     } else {
                         for(int i=0; i<4; ++i) if(input_idx+i < K) next_vals[i] = input[input_idx+i];
                         if (input_idx+4 > K && input_idx < K) { for(int i=K-input_idx; i<4; ++i) next_vals[i]=static_cast<bfloat16_t>(0.0f); }
                     }
                } else {
                     for(int i=0;i<4;++i) next_vals[i]=static_cast<bfloat16_t>(0.0f);
                }
            }
            
            // 3. Compute 6 Rows
            int my_blk = k_blk_start + laneId;
            
            if (my_blk < num_blocks_k) {
                
                #define PROCESS_WORD_ACC(w_u32, i_off, acc0, acc1, acc2, acc3) \
                { \
                     uint32_t val = w_u32; \
                     int2 b_in; __half2* h_in; uint32_t w_low_byte; float wf0, wf1; __half2 w_pair; \
                     int smem_idx = laneId * 16; \
                     b_in = *reinterpret_cast<const int2*>(&s_input[smem_idx + i_off]); \
                     h_in = reinterpret_cast<__half2*>(&b_in); \
                     w_low_byte = val & 0xFF; \
                     wf0 = (float)(w_low_byte & 0xF); wf1 = (float)((w_low_byte >> 4) & 0xF); \
                     w_pair = float_to_half2(wf0, wf1); w_pair = __hmul2(w_pair, scale2); \
                     acc0 = __hfma2(w_pair, h_in[0], acc0); acc0 = __hfma2(neg_zs, h_in[0], acc0); \
                     val >>= 8; \
                     w_low_byte = val & 0xFF; \
                     wf0 = (float)(w_low_byte & 0xF); wf1 = (float)((w_low_byte >> 4) & 0xF); \
                     w_pair = float_to_half2(wf0, wf1); w_pair = __hmul2(w_pair, scale2); \
                     acc1 = __hfma2(w_pair, h_in[1], acc1); acc1 = __hfma2(neg_zs, h_in[1], acc1); \
                     val >>= 8; \
                     b_in = *reinterpret_cast<const int2*>(&s_input[smem_idx + i_off + 2]); \
                     w_low_byte = val & 0xFF; \
                     wf0 = (float)(w_low_byte & 0xF); wf1 = (float)((w_low_byte >> 4) & 0xF); \
                     w_pair = float_to_half2(wf0, wf1); w_pair = __hmul2(w_pair, scale2); \
                     acc2 = __hfma2(w_pair, h_in[0], acc2); acc2 = __hfma2(neg_zs, h_in[0], acc2); \
                     val >>= 8; \
                     w_low_byte = val & 0xFF; \
                     wf0 = (float)(w_low_byte & 0xF); wf1 = (float)((w_low_byte >> 4) & 0xF); \
                     w_pair = float_to_half2(wf0, wf1); w_pair = __hmul2(w_pair, scale2); \
                     acc3 = __hfma2(w_pair, h_in[1], acc3); acc3 = __hfma2(neg_zs, h_in[1], acc3); \
                     val >>= 8; \
                }
                
                #define PROCESS_ROW_COMPUTE(VAL1, VAL2, VAL3, VAL4, VAL0, A0, A1, A2, A3) \
                { \
                        bfloat16_t* p_bf16 = reinterpret_cast<bfloat16_t*>(&VAL0); \
                        float scale_f = static_cast<float>(p_bf16[0]); \
                        float zero_f = static_cast<float>(p_bf16[1]); \
                        __half2 scale2 = float_to_half2(scale_f, scale_f); \
                        __half2 zero2 = float_to_half2(zero_f, zero_f); \
                        __half2 neg_zs = __hneg2(__hmul2(zero2, scale2)); \
                        PROCESS_WORD_ACC(VAL1, 0, A0, A1, A2, A3); \
                        PROCESS_WORD_ACC(VAL2, 4, A0, A1, A2, A3); \
                        PROCESS_WORD_ACC(VAL3, 8, A0, A1, A2, A3); \
                        PROCESS_WORD_ACC(VAL4, 12, A0, A1, A2, A3); \
                }
                
                uint32_t r0_v[5], r1_v[5], r2_v[5], r3_v[5], r4_v[5], r5_v[5];
                
                // Load 20 bytes (Stride-20 Layout)
                #define LOAD_ROW(ROW_IDX, REGS) \
                { \
                    int my_row = row_start + ROW_IDX; \
                    if (my_row < N) { \
                        size_t offset = (size_t)my_row * row_stride + (size_t)my_blk * 20; \
                        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(packed_params + offset); \
                        REGS[0] = ptr[0]; REGS[1] = ptr[1]; REGS[2] = ptr[2]; REGS[3] = ptr[3]; REGS[4] = ptr[4]; \
                    } else { \
                         REGS[0]=0; REGS[1]=0; REGS[2]=0; REGS[3]=0; REGS[4]=0; \
                    } \
                }
                
                LOAD_ROW(0, r0_v);
                LOAD_ROW(1, r1_v);
                LOAD_ROW(2, r2_v);
                LOAD_ROW(3, r3_v);
                LOAD_ROW(4, r4_v);
                LOAD_ROW(5, r5_v);
                
                if (row_start + 0 < N) PROCESS_ROW_COMPUTE(r0_v[1], r0_v[2], r0_v[3], r0_v[4], r0_v[0], acc0_0, acc1_0, acc2_0, acc3_0);
                if (row_start + 1 < N) PROCESS_ROW_COMPUTE(r1_v[1], r1_v[2], r1_v[3], r1_v[4], r1_v[0], acc0_1, acc1_1, acc2_1, acc3_1);
                if (row_start + 2 < N) PROCESS_ROW_COMPUTE(r2_v[1], r2_v[2], r2_v[3], r2_v[4], r2_v[0], acc0_2, acc1_2, acc2_2, acc3_2);
                if (row_start + 3 < N) PROCESS_ROW_COMPUTE(r3_v[1], r3_v[2], r3_v[3], r3_v[4], r3_v[0], acc0_3, acc1_3, acc2_3, acc3_3);
                if (row_start + 4 < N) PROCESS_ROW_COMPUTE(r4_v[1], r4_v[2], r4_v[3], r4_v[4], r4_v[0], acc0_4, acc1_4, acc2_4, acc3_4);
                if (row_start + 5 < N) PROCESS_ROW_COMPUTE(r5_v[1], r5_v[2], r5_v[3], r5_v[4], r5_v[0], acc0_5, acc1_5, acc2_5, acc3_5);
            }
            __syncthreads();
        }

        // Reduce & Store
        #define REDUCE_AND_STORE(ROW_OFFSET, A0, A1, A2, A3) \
        { \
            int my_row = row_start + ROW_OFFSET; \
            if (my_row < N) { \
                __half2 s = __hadd2(__hadd2(A0, A1), __hadd2(A2, A3)); \
                float val = __half2float(s.x) + __half2float(s.y); \
                for (int offset = 16; offset > 0; offset /= 2) val += __shfl_down(val, offset); \
                if (laneId == 0) output[my_row] = static_cast<bfloat16_t>(val); \
            } \
        }

        REDUCE_AND_STORE(0, acc0_0, acc1_0, acc2_0, acc3_0);
        REDUCE_AND_STORE(1, acc0_1, acc1_1, acc2_1, acc3_1);
        REDUCE_AND_STORE(2, acc0_2, acc1_2, acc2_2, acc3_2);
        REDUCE_AND_STORE(3, acc0_3, acc1_3, acc2_3, acc3_3);
        REDUCE_AND_STORE(4, acc0_4, acc1_4, acc2_4, acc3_4);
        REDUCE_AND_STORE(5, acc0_5, acc1_5, acc2_5, acc3_5);
    }
}

void w4a16_gemv_packedGPU(torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& packed_params, int M, int K, int N) {
    if (M != 1) return;
    
    int num_blocks_k = (K + 31) / 32;
    int threadsPerBlock = 256; 
    
    // PERSISTENT CONFIG FOR STRIX HALO (40 CUs)
    // 40 CUs * 2 waves = 80 blocks.
    int blocks = 80; 
    
    auto input_bf16 = input.to(torch::kBFloat16);
    
    gemv_packedGPU_kernel_opt<<<blocks, threadsPerBlock>>>(
        (bfloat16_t*)output.data_ptr(),
        (const bfloat16_t*)input_bf16.data_ptr(),
        packed_params.data_ptr<uint8_t>(),
        K, N, num_blocks_k
    );
     
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "HIP Error in w4a16_gemv_packedGPU: " << hipGetErrorString(err) << std::endl;
    }
}
} // namespace hipkernels
