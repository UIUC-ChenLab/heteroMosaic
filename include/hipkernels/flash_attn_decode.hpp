#pragma once
#include <hip/hip_runtime.h>

extern "C" void launch_flash_attn_decode_hip(const void *Q, const void *K, const void *V, const void *mask, void *dst, int batch_size,
                                             int n_heads_Q, int n_heads_KV, int head_dim, int seq_len_kv, float scale, int stride_Q_seq,
                                             int stride_Q_head, int stride_Q_batch, int stride_K_seq, int stride_K_head, int stride_K_batch,
                                             int stride_V_seq, int stride_V_head, int stride_V_batch, int stride_mask_seq, bool is_bf16,
                                             hipStream_t stream);

extern "C" void launch_flash_attn_decode_hip_hd256(const void *Q, const void *K, const void *V, const void *mask, void *dst, int batch_size,
                                                   int n_heads_Q, int n_heads_KV, int head_dim, int seq_len_kv, float scale,
                                                   int stride_Q_seq, int stride_Q_head, int stride_Q_batch, int stride_K_seq,
                                                   int stride_K_head, int stride_K_batch, int stride_V_seq, int stride_V_head,
                                                   int stride_V_batch, int stride_mask_seq, bool is_bf16, hipStream_t stream);

extern "C" void launch_flash_attn_prefill_hip(const void *Q, const void *K, const void *V, void *dst, int batch_size, int n_heads_Q,
                                              int n_heads_KV, int head_dim, int seq_len_q, int seq_len_kv, float scale, int stride_Q_seq,
                                              int stride_Q_head, int stride_Q_batch, int stride_K_seq, int stride_K_head, int stride_K_batch,
                                              int stride_V_seq, int stride_V_head, int stride_V_batch, int stride_O_seq, int stride_O_head,
                                              int stride_O_batch, int q_start, bool is_bf16, hipStream_t stream);
