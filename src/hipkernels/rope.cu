#include <torch/torch.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <iostream>

using bfloat16 = hip_bfloat16;

__global__ void rope_kernel_bf16(
    bfloat16* q,           
    bfloat16* k,           
    int head_dim,
    int num_q_heads,
    int num_k_heads,
    int seq_len,
    int bsz,
    int start_pos,
    float theta,
    long long q_str_b, long long q_str_s, long long q_str_h, long long q_str_d,
    long long k_str_b, long long k_str_s, long long k_str_h, long long k_str_d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int dim_pairs = head_dim / 2;
    long long total_elements = (long long)bsz * seq_len * num_q_heads * dim_pairs;
    
    if (idx >= total_elements) return;
    
    int d_idx = idx % dim_pairs;
    long long rem = idx / dim_pairs;
    int h = rem % num_q_heads;
    rem = rem / num_q_heads;
    int s = rem % seq_len;
    int b = rem / seq_len;

    // Calc Q offset base
    long long q_base_offset = b * q_str_b + s * q_str_s + h * q_str_h;
    
    // Rotational math
    int pos = start_pos + s;
    float freq = 1.0f / powf(theta, (2.0f * d_idx) / head_dim);
    float val = pos * freq;
    float fcos = cosf(val);
    float fsin = sinf(val);
    
    // Load Q pair (half-rotate: pair element i with i + head_dim/2)
    bfloat16* q_ptr0 = q + q_base_offset + (long long)d_idx * q_str_d;
    bfloat16* q_ptr1 = q + q_base_offset + (long long)(d_idx + dim_pairs) * q_str_d;
    
    float q0 = static_cast<float>(*q_ptr0);
    float q1 = static_cast<float>(*q_ptr1);
    
    float q0_out = q0 * fcos - q1 * fsin;
    float q1_out = q0 * fsin + q1 * fcos;
    
    *q_ptr0 = static_cast<bfloat16>(q0_out);
    *q_ptr1 = static_cast<bfloat16>(q1_out);
    
    // Apply to K
    int ratio = num_q_heads / num_k_heads;
    if (h % ratio == 0) {
        int h_k = h / ratio;
        long long k_base_offset = b * k_str_b + s * k_str_s + h_k * k_str_h;
        
        bfloat16* k_ptr0 = k + k_base_offset + (long long)d_idx * k_str_d;
        bfloat16* k_ptr1 = k + k_base_offset + (long long)(d_idx + dim_pairs) * k_str_d;
        
        float k0 = static_cast<float>(*k_ptr0);
        float k1 = static_cast<float>(*k_ptr1);
        
        float k0_out = k0 * fcos - k1 * fsin;
        float k1_out = k0 * fsin + k1 * fcos;
        
        *k_ptr0 = static_cast<bfloat16>(k0_out);
        *k_ptr1 = static_cast<bfloat16>(k1_out);
    }
}

void launch_rope(torch::Tensor& q, torch::Tensor& k, int start_pos, float theta) {
    if (q.scalar_type() != torch::kBFloat16) {
        std::cerr << "Warning: launch_rope only supports BFloat16 currently. Skipping RoPE." << std::endl;
        return;
    }

    int bsz = q.size(0);
    int seq_len = q.size(1);
    int num_q_heads = q.size(2);
    int head_dim = q.size(3);
    int num_k_heads = k.size(2);
    
    long long total_elements = (long long)bsz * seq_len * num_q_heads * (head_dim / 2);
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    auto q_strides = q.strides();
    auto k_strides = k.strides();
    
    rope_kernel_bf16<<<grid_size, block_size, 0, c10::hip::getCurrentHIPStream().stream()>>>(
        (bfloat16*)q.data_ptr<at::BFloat16>(),
        (bfloat16*)k.data_ptr<at::BFloat16>(),
        head_dim,
        num_q_heads,
        num_k_heads,
        seq_len,
        bsz,
        start_pos,
        theta,
        q_strides[0], q_strides[1], q_strides[2], q_strides[3],
        k_strides[0], k_strides[1], k_strides[2], k_strides[3]
    );
     // hipDeviceSynchronize(); // Handled by Torch's CUDA caching allocator usually, or explicit sync in user code
}
