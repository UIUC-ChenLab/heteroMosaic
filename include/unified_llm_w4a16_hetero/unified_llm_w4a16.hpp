#pragma once

#include <atomic>
#include <c10/hip/HIPStream.h>
#include <cstdint>
#include <future>
#include <hip/hip_runtime.h>
#include <string>
#include <torch/torch.h>
#include <utility>
#include <vector>

// Attention mechanism selection:
// 0 = Manual matmul, 1 = PyTorch SDPA, 2 = Custom HIP kernel
#define LLAMA_USE_SCALED_ATTENTION 2
#define GEMMA_USE_SCALED_ATTENTION 2
#define QWEN_USE_SCALED_ATTENTION 2
#define PHI_USE_SCALED_ATTENTION 2

// LM_HEAD_ONLY_LAST: If set, only run final_norm + lm_head on last token during prefill; set 0 for full-sequence logits.
#define LM_HEAD_ONLY_LAST 1
#define USE_HIP_EMBEDDING 1
#define USE_HIP_LM_HEAD 1

// Single source of truth for hetero trace capture.
// Set to 1 to enable trace collection, or 0 to compile it out.
#define GET_TRACES 1

// Architecture type enum
enum class ArchitectureType { LLAMA3, GEMMA, QWEN15, QWEN25SMALL, PHI3 };

// Scratch space for pre-allocated buffers
struct GemmaScratchSpace {
    torch::Tensor x_buffer;
    torch::Tensor qkv_buffer;
    torch::Tensor gate_buffer;
    torch::Tensor up_buffer;
    torch::Tensor output_buffer;
    torch::Tensor hidden_states_buffer;
    torch::Tensor queries_buffer;
    torch::Tensor keys_buffer;
    torch::Tensor values_buffer;
    torch::Tensor attn_output_heads_buffer;
    torch::Tensor attn_output_buffer;
    torch::Tensor attn_output_proj_buffer;
    torch::Tensor norm_buffer;
};

// Per-worker scratch space for LLAMA3 chunked pipeline.
struct LlamaPipelineScratchSpace {
    torch::Tensor qkv_buffer;
    torch::Tensor gate_buffer;
    torch::Tensor up_buffer;
    torch::Tensor output_buffer;
    torch::Tensor queries_buffer;
    torch::Tensor keys_buffer;
    torch::Tensor values_buffer;
    torch::Tensor attn_output_buffer;
    torch::Tensor attn_output_proj_buffer;
    torch::Tensor norm_buffer;
};

// Quantized Linear Layer for w4a16 (4-bit weights, 16-bit activations)
// Weights are stored as 4-bit packed in uint8, with scales for dequantization
class QuantizedLinearImpl : public torch::nn::Module {
  public:
    QuantizedLinearImpl(int64_t in_features, int64_t out_features, bool bias = false, int64_t max_seq_len = 2048,
                        std::string layer_type = "");

    // Dequantize 4-bit weights to bf16 (Standard/Unpacked)
    torch::Tensor dequantize_weights();

    // Dequantize from packed parameters (ZigZag/Fused format)
    torch::Tensor dequantize_weights_packed();
    // Static helper for dequantization (used by fallback paths)
    static torch::Tensor dequantize_packed(const torch::Tensor &packed_params, int64_t in_features, int64_t out_features);

    // Forward pass: dequantize weights, then perform linear operation
    // Takes an optional output buffer for in-place operation
    // Returns a future if async execution is possible and no bias addition is needed immediately
    std::future<int> forward(torch::Tensor output_buffer, torch::Tensor input, std::string layer_type, int chunk_id = -1);

    // Set quantized weights (for loading from state dict)
    void set_quantized_weights(torch::Tensor qweight, torch::Tensor scale, torch::Tensor zero_point, torch::Tensor g_idx = torch::Tensor());
    // Directly set pre-packed or preprocessed weights
    void set_packed_params(torch::Tensor packed);
    void set_packed_params_split(torch::Tensor packed0, torch::Tensor packed1);
    void set_unpacked_params(torch::Tensor qweight_packed, torch::Tensor scale, torch::Tensor zero_point);
    bool is_k_split() const { return is_k_split_ > 0; }
    int64_t split_k() const { return static_cast<int64_t>(is_k_split_); }

    // Explicitly import weights to XDNA (for benchmark/testing)
    void import_weights_to_xdna();

    // Accessors
    int64_t in_features() const { return in_features_; }
    int64_t out_features() const { return out_features_; }
    torch::Tensor get_packed_params() const { return packed_params_; }
    // torch::Tensor get_packed_params_cpu() const { return packed_params_cpu_; }
    torch::Tensor get_quantized_weights() const { return quantized_weight_; }
    torch::Tensor get_scales() const { return scale_; }
    torch::Tensor get_zeros() const { return zero_point_; }

  private:
    int64_t in_features_;
    int64_t out_features_;
    int64_t padded_out_features_;
    int64_t padded_in_features_;     // Padded output dimension (if padPackedWeights is enabled)
    int64_t pad_k_alignment_ = 0;    // Per-layer K padding alignment when packed-weight padding is enabled
    int64_t pad_n_alignment_ = 0;    // Per-layer N padding alignment when packed-weight padding is enabled
    int64_t max_seq_len_;            // Max sequence length for buffer pre-allocation
    torch::Tensor packed_params_;    // Unified buffer for weights, scales, and zero points on GPU
    torch::Tensor packed_params_0_;  // First half of split weights (gpu_split)
    torch::Tensor packed_params_1_;  // Second half of split weights (gpu_split)
    torch::Tensor quantized_weight_; // Separate buffer for weights (if use_packed_weights=false)
    torch::Tensor scale_;            // Separate buffer for scales (if use_packed_weights=false)
    torch::Tensor zero_point_;       // Separate buffer for zero points (if use_packed_weights=false)
    torch::Tensor bias_;             // Optional bias [out_features]
    torch::Tensor g_idx_;            // Optional group index [in_features]

    // Optimization: Cache split decision
    int is_k_split_ = 0;
};
TORCH_MODULE(QuantizedLinear);

// Custom LM Head Layer (BF16)
// optimized for high-bandwidth generation using custom HIP kernels
class LmHeadLinearImpl : public torch::nn::Module {
  public:
    LmHeadLinearImpl(int64_t in_features, int64_t out_features, int64_t max_batch_size = 1, int64_t max_seq_len = 8192);
    torch::Tensor forward(torch::Tensor input);
    torch::Tensor weight;

  private:
    torch::Tensor logits_buffer;
};
TORCH_MODULE(LmHeadLinear);

// RMSNorm implementation
class RMSNormImpl : public torch::nn::Module {
  public:
    RMSNormImpl(int64_t dim, float eps, bool gemma_style = false);

    torch::Tensor forward(torch::Tensor x);
    void forward_out(torch::Tensor output, torch::Tensor x);

    // Setter for weight loading
    void set_weight(torch::Tensor weight);

  private:
    float eps_;
    bool gemma_style_;
    torch::Tensor weight_;
};
TORCH_MODULE(RMSNorm);

// Unified LLM Model Implementation for w4a16 quantization
// Custom HIP Embedding Layer
// optimized to skip unnecessary tensor copies
class HipEmbeddingImpl : public torch::nn::Module {
  public:
    HipEmbeddingImpl(int64_t num_embeddings, int64_t embedding_dim, int64_t max_batch_size = 1, int64_t max_seq_len = 8192);
    torch::Tensor forward(torch::Tensor input);
    torch::Tensor weight;

  private:
    torch::Tensor output_buffer;
};
TORCH_MODULE(HipEmbedding);

class UnifiedLLMW4A16Impl : public torch::nn::Module {
  public:
    UnifiedLLMW4A16Impl(ArchitectureType arch_type, int64_t vocab_size, int64_t hidden_size, int64_t intermediate_size,
                        int64_t num_hidden_layers, int64_t num_attention_heads, int64_t num_key_value_heads, int64_t head_dim,
                        float rms_norm_eps, float rope_theta, int64_t max_seq_len = 8192, int64_t max_batch_size = 1,
                        int64_t groupsize = 128, torch::Device device = torch::kCPU, std::string config_path = "",
                        float partial_rotary_factor = 1.0f, int64_t original_max_position_embeddings = 0,
                        std::vector<float> rope_short_factors = {}, std::vector<float> rope_long_factors = {},
                        int64_t model_max_position_embeddings = 0);
    ~UnifiedLLMW4A16Impl() override;

    // Forward pass: takes token IDs and returns logits
    torch::Tensor forward(torch::Tensor input_ids, int64_t start_pos = 0);

    // Generate tokens: takes prompt, generates max_new_tokens, returns all token IDs
    torch::Tensor generate(torch::Tensor input_ids, int64_t max_new_tokens, float temperature = 1.0f, float top_p = 0.9f,
                           int64_t top_k = 50, int64_t eos_token_id = -1);

    // Load quantized weights from safetensors file
    void load_quantized_weights_from_safetensors(const std::string &filename);
    // Load non-quantized weights only (embeddings, norms, lm_head) from safetensors
    void load_non_quantized_weights_from_safetensors(const std::string &filename);
    // Load quantized weights from preprocessed bin directory
    void load_quantized_weights_from_bins(const std::string &weights_dir);

    // Move model to device
    // Move model to device
    UnifiedLLMW4A16Impl &to(torch::Device device);

    // Initialize NPU configuration and resources
    int initialize_npu();

    // Import weights to NPU (must be called after loading weights)
    void import_weights();

    // Initialize all weights with dummy values (random) for testing without loading files
    void initialize_dummy_weights(int seed = 42);

    // Get scratch space with pre-allocated buffers
    GemmaScratchSpace get_scratch_space();

    // NPU Helper functions
    // We declare them as friends or static/global if they are not members
    // But they are global in npuSetup.cpp.
    // So we should declare them outside the class or as static members if we moved them.
    // Since they are global in npuSetup.cpp, we declare them here as global functions.
  private:
    ArchitectureType arch_type_;
    int64_t vocab_size_;
    int64_t hidden_size_;
    int64_t intermediate_size_;
    int64_t num_hidden_layers_;
    int64_t num_attention_heads_;
    int64_t num_key_value_heads_;
    int64_t head_dim_;
    float rms_norm_eps_;
    float rope_theta_;
    int64_t max_seq_len_;
    int64_t max_batch_size_;
    int64_t groupsize_;
    int64_t GQA_head_ratio_;
    std::string config_path_;
    bool warmup_;
    int64_t gemma_sliding_window_size_ = 1;
    int64_t gemma_cache_filled_ = 0;
    bool gemma_use_qkv_fused_ = false;
    bool qwen_use_qkv_fused_ = false;
    float phi_partial_rotary_factor_ = 1.0f;
    int64_t phi_original_max_position_embeddings_ = 0;
    int64_t phi_model_max_position_embeddings_ = 0;
    int64_t phi_rotary_dim_ = 0;
    float phi_attention_scaling_ = 1.0f;
    std::vector<float> phi_rope_short_factors_;
    std::vector<float> phi_rope_long_factors_;

    // Common components
    HipEmbedding token_embedding{nullptr};
    // Fused attention projection (q|k|v) for architectures that support it.
    std::vector<QuantizedLinear> qkv_layers;
    std::vector<QuantizedLinear> q_layers;
    std::vector<QuantizedLinear> k_layers;
    std::vector<QuantizedLinear> v_layers;
    std::vector<QuantizedLinear> o_layers;

    // MLP layers
    std::vector<QuantizedLinear> gate_layers;
    std::vector<QuantizedLinear> up_layers;
    std::vector<QuantizedLinear> down_layers;

    // LLaMA3 specific MLP layers (unified approach might not need separate vectors if logic handles it,
    // but UnifiedLLM.cpp used separate vectors for LLaMA3.
    // However, looking at UnifiedLLM.cpp, LLaMA3 uses gate/up/down just like others,
    // but the variable names in UnifiedLLM.cpp were llama_gate_layers etc. vs gate_layers for Gemma/Qwen.
    // I can reuse gate_layers/up_layers/down_layers for all architectures if I map them correctly during init and
    // weight loading.) Let's reuse the existing vectors to keep it simple, but we need to know which is which.

    // Normalization layers
    std::vector<RMSNorm> input_norms;
    std::vector<RMSNorm> post_attn_norms;
    RMSNorm final_norm{nullptr};

    // Output layer (can be quantized or regular)
    LmHeadLinear lm_head{nullptr};

    // KV caches
    std::vector<torch::Tensor> caches_k;
    std::vector<torch::Tensor> caches_v;

    // Scratch buffers
    torch::Tensor x_buffer;
    torch::Tensor gate_buffer;
    torch::Tensor up_buffer;
    torch::Tensor output_buffer;
    torch::Tensor hidden_states_buffer;
    torch::Tensor qkv_buffer;
    torch::Tensor queries_buffer;
    torch::Tensor keys_buffer;
    torch::Tensor values_buffer;
    torch::Tensor attn_output_heads_buffer;
    torch::Tensor attn_output_buffer;
    torch::Tensor attn_output_proj_buffer;
    torch::Tensor norm_buffer;
    std::vector<LlamaPipelineScratchSpace> llama_pipeline_scratch_slots;
    std::vector<c10::cuda::CUDAStream> llama_chunk_slot_streams_;
    std::vector<std::vector<hipEvent_t>> llama_chunk_kv_ready_events_;
    bool llama_chunk_async_runtime_ready_ = false;
    c10::DeviceIndex llama_chunk_async_device_index_ = static_cast<c10::DeviceIndex>(-1);
    void init_llama_chunk_async_runtime(bool force_rebuild = false);
    void release_llama_chunk_async_runtime();

    // RoPE embedding
    torch::Tensor compute_rope_freqs(int64_t seq_len, int64_t start_pos);
    std::pair<torch::Tensor, torch::Tensor> apply_rotary_emb(const torch::Tensor &xq, const torch::Tensor &xk,
                                                             const torch::Tensor &freqs_cis);

    // Gemma-specific RoPE
    std::pair<torch::Tensor, torch::Tensor> compute_gemma_rope(int64_t batch_size, int64_t seq_len, int64_t start_pos);
    std::pair<torch::Tensor, torch::Tensor> apply_gemma_rotary_emb(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &cos,
                                                                   const torch::Tensor &sin);
    std::pair<torch::Tensor, torch::Tensor> compute_phi3_rope(int64_t batch_size, int64_t seq_len, int64_t start_pos);
    std::pair<torch::Tensor, torch::Tensor> apply_phi3_rotary_emb(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &cos,
                                                                  const torch::Tensor &sin);

    // Architecture-specific forward methods
    torch::Tensor forward_llama3_chunked(torch::Tensor x, int64_t start_pos);
    torch::Tensor forward_llama3(torch::Tensor x, int64_t start_pos, bool return_logits = true, bool last_token_only = false);
    torch::Tensor forward_llama3_with_scratch(torch::Tensor x, int64_t start_pos, bool return_logits, bool last_token_only,
                                              torch::Tensor &queries_buffer_base, torch::Tensor &keys_buffer_base,
                                              torch::Tensor &values_buffer_base, torch::Tensor &attn_output_buffer_base,
                                              torch::Tensor &attn_output_proj_buffer_base, torch::Tensor &gate_buffer_base,
                                              torch::Tensor &up_buffer_base, torch::Tensor &output_buffer_base,
                                              torch::Tensor &norm_buffer_base,
                                              std::vector<std::vector<hipEvent_t>> *kv_ready_events = nullptr,
                                              std::vector<std::atomic<int>> *chunk_layer_progress = nullptr, int64_t chunk_id = -1,
                                              int64_t slot_id = -1);
    torch::Tensor forward_gemma_chunked(torch::Tensor x, int64_t start_pos);
    torch::Tensor forward_gemma(torch::Tensor x, int64_t start_pos);
    torch::Tensor forward_gemma_with_scratch(torch::Tensor x, int64_t start_pos, torch::Tensor &queries_buffer_base,
                                             torch::Tensor &keys_buffer_base, torch::Tensor &values_buffer_base,
                                             torch::Tensor &qkv_buffer_base,
                                             torch::Tensor &attn_output_buffer_base, torch::Tensor &attn_output_proj_buffer_base,
                                             torch::Tensor &gate_buffer_base, torch::Tensor &up_buffer_base,
                                             torch::Tensor &output_buffer_base, torch::Tensor &norm_buffer_base,
                                             std::vector<std::vector<hipEvent_t>> *kv_ready_events = nullptr,
                                             std::vector<std::atomic<int>> *chunk_layer_progress = nullptr, int64_t chunk_id = -1,
                                             int64_t slot_id = -1);
    torch::Tensor forward_qwen25_chunked(torch::Tensor x, int64_t start_pos);
    torch::Tensor forward_qwen25(torch::Tensor x, int64_t start_pos);
    torch::Tensor forward_qwen25_with_scratch(torch::Tensor x, int64_t start_pos, torch::Tensor &queries_buffer_base,
                                              torch::Tensor &keys_buffer_base, torch::Tensor &values_buffer_base,
                                              torch::Tensor &attn_output_buffer_base, torch::Tensor &attn_output_proj_buffer_base,
                                              torch::Tensor &gate_buffer_base, torch::Tensor &up_buffer_base,
                                              torch::Tensor &output_buffer_base, torch::Tensor &norm_buffer_base,
                                              std::vector<std::vector<hipEvent_t>> *kv_ready_events = nullptr,
                                              std::vector<std::atomic<int>> *chunk_layer_progress = nullptr, int64_t chunk_id = -1,
                                              int64_t slot_id = -1, bool return_logits = true,
                                              bool cache_only_last_layer_kv = false);
    torch::Tensor forward_qwen25small_chunked(torch::Tensor x, int64_t start_pos);
    torch::Tensor forward_qwen25small(torch::Tensor x, int64_t start_pos);
    torch::Tensor forward_qwen25small_with_scratch(torch::Tensor x, int64_t start_pos, torch::Tensor &queries_buffer_base,
                                              torch::Tensor &keys_buffer_base, torch::Tensor &values_buffer_base,
                                              torch::Tensor &qkv_buffer_base,
                                              torch::Tensor &attn_output_buffer_base, torch::Tensor &attn_output_proj_buffer_base,
                                              torch::Tensor &gate_buffer_base, torch::Tensor &up_buffer_base,
                                              torch::Tensor &output_buffer_base, torch::Tensor &norm_buffer_base,
                                              std::vector<std::vector<hipEvent_t>> *kv_ready_events = nullptr,
                                              std::vector<std::atomic<int>> *chunk_layer_progress = nullptr, int64_t chunk_id = -1,
                                              int64_t slot_id = -1, bool return_logits = true,
                                              bool cache_only_last_layer_kv = false);
    torch::Tensor forward_qwen25_impl(torch::Tensor x, int64_t start_pos, torch::Tensor &queries_buffer_base,
                                              torch::Tensor &keys_buffer_base, torch::Tensor &values_buffer_base,
                                              torch::Tensor *qkv_buffer_base, torch::Tensor &attn_output_buffer_base,
                                              torch::Tensor &attn_output_proj_buffer_base, torch::Tensor &gate_buffer_base,
                                              torch::Tensor &up_buffer_base, torch::Tensor &output_buffer_base,
                                              torch::Tensor &norm_buffer_base, bool use_fused_qkv,
                                              std::vector<std::vector<hipEvent_t>> *kv_ready_events = nullptr,
                                              std::vector<std::atomic<int>> *chunk_layer_progress = nullptr, int64_t chunk_id = -1,
                                              int64_t slot_id = -1, bool return_logits = true,
                                              bool cache_only_last_layer_kv = false);
    torch::Tensor forward_qwen25_chunked_impl(torch::Tensor x, int64_t start_pos, bool use_fused_qkv);
    torch::Tensor forward_phi3_chunked(torch::Tensor x, int64_t start_pos);
    torch::Tensor forward_phi3(torch::Tensor x, int64_t start_pos);
    torch::Tensor forward_phi3_with_scratch(torch::Tensor x, int64_t start_pos, torch::Tensor &queries_buffer_base,
                                            torch::Tensor &keys_buffer_base, torch::Tensor &values_buffer_base,
                                            torch::Tensor &attn_output_buffer_base, torch::Tensor &attn_output_proj_buffer_base,
                                            torch::Tensor &gate_buffer_base, torch::Tensor &up_buffer_base,
                                            torch::Tensor &output_buffer_base, torch::Tensor &norm_buffer_base,
                                            std::vector<std::vector<hipEvent_t>> *kv_ready_events = nullptr,
                                            std::vector<std::atomic<int>> *chunk_layer_progress = nullptr, int64_t chunk_id = -1,
                                            int64_t slot_id = -1);

    // Activation functions
    torch::Tensor silu(const torch::Tensor &x);
    torch::Tensor gelu(const torch::Tensor &x);
    torch::Tensor swiglu(const torch::Tensor &gate, const torch::Tensor &up);
};

// Global NPU functions
uint32_t import_dma_buf_to_xdna(void *hip_managed_ptr, size_t size, int dataTypeinBytes);
std::pair<int, int> get_npu_context(int M, int K, int N);
int npuMatmul_zero(int hwctx_numb, int instctx_numb, void *output_pointer, void *input_pointer, void *weight_pointer,
                   uint32_t output_xdna_handle, uint32_t input_xdna_handle, uint32_t weight_xdna_handle, void *hip_event);

// Reference Implementation Functions
torch::Tensor reference_dequantize_weights(torch::Tensor quantized_weight, torch::Tensor scale, torch::Tensor zero_point,
                                           int64_t in_features, int64_t out_features, torch::Tensor g_idx = torch::Tensor());

torch::Tensor reference_gemm(torch::Tensor input, torch::Tensor quantized_weight, torch::Tensor scale, torch::Tensor zero_point,
                             int64_t in_features, int64_t out_features, torch::Tensor g_idx = torch::Tensor());

TORCH_MODULE(UnifiedLLMW4A16);
