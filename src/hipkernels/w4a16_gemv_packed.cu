#include "hipkernels/w4a16_gemm_packed.hpp"
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using bfloat16_t = hip_bfloat16;

#define TILE_K 128
#define TILE_N 64
#define PACKED_TILE_SIZE 4352
#define WEIGHT_BYTES 4096
#define SCALE_BYTES 128
#define ZERO_BYTES 128

namespace hipkernels {

__device__ __forceinline__ float bf16_to_float_dev(bfloat16_t val) { return static_cast<float>(val); }
__device__ __forceinline__ bfloat16_t float_to_bf16_dev(float val) { return static_cast<bfloat16_t>(val); }

// Optimization 7.0: Warp-per-SC + Half2 Vectorization + ILP 4x
// - Uses __half2 for SIMD execution
// - Uses 4 separate accumulators to hide FMA latency
// - Manual unrolling
__global__ void __launch_bounds__(256) w4a16_gemv_kernel_packed(bfloat16_t *__restrict__ output, const bfloat16_t *__restrict__ input,
                                         const uint8_t *__restrict__ packed_params,
                                         const int K,           // Input features (must be multiple of 128)
                                         const int num_tiles_k, // K / 128
                                         const int num_tiles_n, // Total N / 64
                                         const int start_tile_n // Start Block Offset
) {
    const int bx = blockIdx.x; 
    const int tid = threadIdx.x;

    // Mapping
    const int warp_id = tid / 32;       // 0..7
    const int lane_id = tid % 32;       // 0..31
    const int row_in_SSR = lane_id >> 2;   // 0..7
    const int pair_in_SC = lane_id & 3;    // 0..3
    
    const int out_idx0 = warp_id * 8 + pair_in_SC * 2;
    const int out_idx1 = out_idx0 + 1;

    // ZigZag Addressing 
    const int abs_bx = bx + start_tile_n;
    int col_group = abs_bx % 8;
    int col_offset = abs_bx / 8;
    int tbg = 0;
    for (int g = 0; g < col_group; ++g) {
        tbg += ((num_tiles_n + 7 - g) / 8) * num_tiles_k;
    }
    int base_z = tbg + col_offset * num_tiles_k;

    // Shared memory for Input Tile (Half)
    __shared__ __half smem_input[128]; 

    // Accumulators (Half2) - Use 4 to hide latency
    __half2 acc0 = __float2half2_rn(0.0f);
    __half2 acc1 = __float2half2_rn(0.0f);
    __half2 acc2 = __float2half2_rn(0.0f);
    __half2 acc3 = __float2half2_rn(0.0f);

    // Loop over K tiles
    for (int k_tile = 0; k_tile < num_tiles_k; ++k_tile) {

        // Load Input + Convert to Half
        if (tid < 128) {
            float val_f = bf16_to_float_dev(input[k_tile * 128 + tid]);
            smem_input[tid] = __float2half(val_f);
        }
        __syncthreads(); 

        // Weight Base Pointer
        size_t tile_offset = (size_t)(base_z + k_tile) * PACKED_TILE_SIZE;
        const uint8_t *w_base = packed_params + tile_offset;

        // Load Scale/Zero (Convert to Half)
        const bfloat16_t *scale_ptr = (const bfloat16_t *)(w_base + WEIGHT_BYTES);
        const uint8_t *zero_ptr = w_base + WEIGHT_BYTES + SCALE_BYTES;

        __half scale0 = __float2half(bf16_to_float_dev(scale_ptr[out_idx0]));
        __half scale1 = __float2half(bf16_to_float_dev(scale_ptr[out_idx1]));
        __half2 scale2_v = __halves2half2(scale0, scale1);

        int zero_idx0 = (out_idx0 / 8) * 16 + (out_idx0 % 8);
        int zero_idx1 = (out_idx1 / 8) * 16 + (out_idx1 % 8); 
        __half zero0 = __float2half((float)(int8_t)zero_ptr[zero_idx0]);
        __half zero1 = __float2half((float)(int8_t)zero_ptr[zero_idx1]);
        __half2 zero2_v = __halves2half2(zero0, zero1);

        int w_offset_base = (warp_id << 5) + lane_id; 

        // Unroll loop manually by 4
        #pragma unroll 4
        for (int sr = 0; sr < 16; sr += 4) {
             // Iter 0
             {
                int byte_off = (sr << 8) + w_offset_base;
                uint8_t packed = w_base[byte_off];
                float q0_f = (float)(packed & 0x0F);
                float q1_f = (float)((packed >> 4) & 0x0F);
                __half2 w2 = __hmul2(__hsub2(__floats2half2_rn(q0_f, q1_f), zero2_v), scale2_v);
                
                int row = sr * 8 + row_in_SSR;
                __half2 in2 = __half2half2(smem_input[row]);
                acc0 = __hfma2(w2, in2, acc0);
             }
             // Iter 1
             {
                int byte_off = ((sr+1) << 8) + w_offset_base;
                uint8_t packed = w_base[byte_off];
                float q0_f = (float)(packed & 0x0F);
                float q1_f = (float)((packed >> 4) & 0x0F);
                __half2 w2 = __hmul2(__hsub2(__floats2half2_rn(q0_f, q1_f), zero2_v), scale2_v);
                
                int row = (sr+1) * 8 + row_in_SSR;
                __half2 in2 = __half2half2(smem_input[row]);
                acc1 = __hfma2(w2, in2, acc1);
             }
             // Iter 2
             {
                int byte_off = ((sr+2) << 8) + w_offset_base;
                uint8_t packed = w_base[byte_off];
                float q0_f = (float)(packed & 0x0F);
                float q1_f = (float)((packed >> 4) & 0x0F);
                __half2 w2 = __hmul2(__hsub2(__floats2half2_rn(q0_f, q1_f), zero2_v), scale2_v);
                
                int row = (sr+2) * 8 + row_in_SSR;
                __half2 in2 = __half2half2(smem_input[row]);
                acc2 = __hfma2(w2, in2, acc2);
             }
             // Iter 3
             {
                int byte_off = ((sr+3) << 8) + w_offset_base;
                uint8_t packed = w_base[byte_off];
                float q0_f = (float)(packed & 0x0F);
                float q1_f = (float)((packed >> 4) & 0x0F);
                __half2 w2 = __hmul2(__hsub2(__floats2half2_rn(q0_f, q1_f), zero2_v), scale2_v);
                
                int row = (sr+3) * 8 + row_in_SSR;
                __half2 in2 = __half2half2(smem_input[row]);
                acc3 = __hfma2(w2, in2, acc3);
             }
        }
        
        __syncthreads(); 
    }

    // Sum Accumulators
    acc0 = __hadd2(acc0, acc1);
    acc2 = __hadd2(acc2, acc3);
    acc0 = __hadd2(acc0, acc2);

    // Convert accumulator to float for reduction
    float2 acc2_f = __half22float2(acc0);

    // Warp Shuffle Reduction (Split float2 into 2 floats)
    float total_acc0 = acc2_f.x;
    float total_acc1 = acc2_f.y;
    
    // Reduce 0
    total_acc0 += __shfl_down(total_acc0, 4);  
    total_acc0 += __shfl_down(total_acc0, 8);  
    total_acc0 += __shfl_down(total_acc0, 16); 

    // Reduce 1
    total_acc1 += __shfl_down(total_acc1, 4);
    total_acc1 += __shfl_down(total_acc1, 8);
    total_acc1 += __shfl_down(total_acc1, 16);

    if (lane_id < 4) {
        int global_n0 = bx * 64 + out_idx0;
        int global_n1 = bx * 64 + out_idx1;
        output[global_n0] = float_to_bf16_dev(total_acc0);
        output[global_n1] = float_to_bf16_dev(total_acc1);
    }
}

void w4a16_gemv_fused_packed(torch::Tensor &output, const torch::Tensor &input, const torch::Tensor &packed_params, int64_t in_features,
                             int64_t out_features, int64_t total_out_features, int64_t start_col, hipStream_t stream) {
    const int K = in_features;
    const int N = out_features; // Work N

    int64_t total_N = (total_out_features == -1) ? out_features : total_out_features;
    int64_t start_N = start_col;

    // Grid: (N / 64) blocks
    dim3 grid((N + 63) / 64);
    dim3 block(256);

    const int num_tiles_k = (K + 127) / 128;     // TILE_K = 128
    const int num_tiles_n = (total_N + 63) / 64; // Total N / 64
    const int start_tile_n = start_N / 64;       // Start Block

    w4a16_gemv_kernel_packed<<<grid, block, 0, stream>>>((bfloat16_t *)output.data_ptr(), (const bfloat16_t *)input.data_ptr(),
                                                    packed_params.data_ptr<uint8_t>(), K, num_tiles_k, num_tiles_n, start_tile_n);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP GEMV kernel error: ") + hipGetErrorString(err));
    }
}

} // namespace hipkernels
