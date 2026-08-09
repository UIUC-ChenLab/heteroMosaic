#include "unified_llm_w4a16_hetero/unified_llm_w4a16.hpp"
#include "hipkernels/w4a16_gemm_packed.hpp"
#include "hipkernels/w4a16_gemv_unpacked.hpp"
#include "unified_llm_w4a16_hetero/helper.hpp"
#include "unified_llm_w4a16_hetero/hetero_compute.hpp"
#include "unified_llm_w4a16_hetero/hipblasltSetup.hpp"
#include "unified_llm_w4a16_hetero/npuSetup.hpp"
#include <ATen/ops/_scaled_dot_product_flash_attention.h>
#include <algorithm>
#include <atomic>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <torch/torch.h>

#include "cpu_avx_kernels/w4a16_gemv_avx_unpacked.hpp" // Added for CPU sanity check
#include "hipkernels/embedding.hpp"
#include "hipkernels/flash_attn_decode.hpp"
#include "hipkernels/lm_head.hpp"
#include "hipkernels/rmsnorm.hpp"
#include "hipkernels/rope.hpp"
#include "hipkernels/w4a16_gemm_packed.hpp"
#include "hipkernels/w4a16_gemm_unpacked.hpp" // Added for Unpacked GEMM fallback
#include "hipkernels/w4a16_gemv_unpacked.hpp"
#include <unistd.h>
#include <vector>

// #define FA2_HIP_KERNEL 1 // Removed as per refactor instructions

// Pipeline mode runs multiple chunk workers concurrently. QuantizedLinear modules
// use per-slot scratch and layer events for KV synchronization.

namespace {
using TraceClock = std::chrono::steady_clock;

int64_t round_up_to_alignment(int64_t dim, int64_t alignment) {
    if (alignment <= 0) {
        return dim;
    }
    return ((dim + alignment - 1) / alignment) * alignment;
}

int64_t round_up_packed_dim(int64_t dim) {
    if (pad_packed_weights <= 0) {
        return dim;
    }
    return round_up_to_alignment(dim, pad_packed_weights);
}

int64_t get_layer_padded_k_dim(const std::string &layer_type, int64_t dim) {
    if (pad_packed_weights <= 0) {
        return dim;
    }
    auto pad_spec = get_packed_weight_pad_alignment(layer_type);
    return round_up_to_alignment(dim, pad_spec[0]);
}

int64_t get_layer_padded_n_dim(const std::string &layer_type, int64_t dim) {
    if (pad_packed_weights <= 0) {
        return dim;
    }
    auto pad_spec = get_packed_weight_pad_alignment(layer_type);
    return round_up_to_alignment(dim, pad_spec[1]);
}

struct LlamaChunkSpan {
    int64_t start;
    int64_t len;
};

struct StageTraceRow {
    std::string run_id;
    std::string model;
    std::string arch;
    int64_t chunk_id = -1;
    int64_t slot_id = -1;
    int64_t layer_id = -1;
    std::string stage;
    int64_t host_ready_ts_us = 0;
    int64_t start_ts_us = 0;
    int64_t end_ts_us = 0;
    int64_t start_pos = 0;
    int64_t seq_len = 0;
};

struct TraceCollectorState {
    bool active = false;
    std::string run_id;
    std::string model;
    std::string arch;
    std::string output_path;
    TraceClock::time_point t0;
    std::vector<StageTraceRow> rows;
    std::mutex mutex;
};

struct StageTraceToken {
    bool trace_enabled = false;
    std::string run_id;
    std::string model;
    std::string arch;
    int64_t chunk_id = -1;
    int64_t slot_id = -1;
    int64_t layer_id = -1;
    std::string stage;
    int64_t host_ready_ts_us = 0;
    int64_t start_ts_us = 0;
    int64_t start_pos = 0;
    int64_t seq_len = 0;
};

TraceCollectorState &get_trace_collector_state() {
    static TraceCollectorState state;
    return state;
}

std::string arch_type_to_string(ArchitectureType arch_type) {
    switch (arch_type) {
    case ArchitectureType::LLAMA3:
        return "llama3";
    case ArchitectureType::GEMMA:
        return "gemma";
    case ArchitectureType::QWEN15:
        return "qwen15";
    case ArchitectureType::QWEN25SMALL:
        return "qwen25small";
    case ArchitectureType::PHI3:
        return "phi3";
    }
    return "unknown";
}

std::string build_trace_run_id(const std::string &tag) {
    const auto now = std::chrono::system_clock::now();
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    if (tag.empty()) {
        return std::string("trace_") + std::to_string(micros);
    }
    return tag + "_" + std::to_string(micros);
}

int64_t find_stage_bubble_delay_us(int64_t chunk_id, int64_t layer_id, const std::string &stage) {
    for (const auto &spec : stage_bubbles) {
        if (spec.delay_us <= 0) {
            continue;
        }
        if (spec.chunk_id != chunk_id || spec.layer_id != layer_id) {
            continue;
        }
        if (spec.stage != stage) {
            continue;
        }
        return spec.delay_us;
    }
    return 0;
}

bool begin_prefill_trace_capture(ArchitectureType arch_type) {
#if GET_TRACES
    if (trace_output_path.empty()) {
        return false;
    }

    auto &state = get_trace_collector_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.active = true;
    state.run_id = build_trace_run_id(trace_run_tag);
    state.arch = arch_type_to_string(arch_type);
    state.model = state.arch;
    state.output_path = trace_output_path;
    state.t0 = TraceClock::now();
    state.rows.clear();
    return true;
#else
    (void)arch_type;
    return false;
#endif
}

void finish_prefill_trace_capture() {
#if GET_TRACES
    auto &state = get_trace_collector_state();
    std::vector<StageTraceRow> rows;
    std::string output_path;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.active) {
            return;
        }
        state.active = false;
        rows = state.rows;
        output_path = state.output_path;
        state.rows.clear();
    }

    if (output_path.empty()) {
        return;
    }

    std::filesystem::path out_path(output_path);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }
    std::ofstream out(output_path, std::ios::app);
    if (!out.is_open()) {
        return;
    }
    for (const auto &row : rows) {
        out << "{\"run_id\":\"" << row.run_id << "\"," << "\"model\":\"" << row.model << "\"," << "\"arch\":\"" << row.arch << "\","
            << "\"chunk_id\":" << row.chunk_id << "," << "\"slot_id\":" << row.slot_id << "," << "\"layer_id\":" << row.layer_id << ","
            << "\"stage\":\"" << row.stage << "\"," << "\"host_ready_ts_us\":" << row.host_ready_ts_us << ","
            << "\"start_ts_us\":" << row.start_ts_us << "," << "\"end_ts_us\":" << row.end_ts_us << "," << "\"start_pos\":" << row.start_pos
            << "," << "\"seq_len\":" << row.seq_len << "}\n";
    }
#endif
}

struct PrefillTraceGuard {
    bool active = false;
    explicit PrefillTraceGuard(ArchitectureType arch_type) : active(begin_prefill_trace_capture(arch_type)) {}
    ~PrefillTraceGuard() {
        if (active) {
            finish_prefill_trace_capture();
        }
    }
};

StageTraceToken begin_stage_trace(int64_t chunk_id, int64_t slot_id, int64_t layer_id, const std::string &stage, int64_t start_pos,
                                  int64_t seq_len) {
    StageTraceToken token;
    token.chunk_id = chunk_id;
    token.slot_id = slot_id;
    token.layer_id = layer_id;
    token.stage = stage;
    token.start_pos = start_pos;
    token.seq_len = seq_len;

#if GET_TRACES
    auto &state = get_trace_collector_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.active) {
            token.trace_enabled = true;
            token.run_id = state.run_id;
            token.model = state.model;
            token.arch = state.arch;
            token.host_ready_ts_us = std::chrono::duration_cast<std::chrono::microseconds>(TraceClock::now() - state.t0).count();
        }
    }
#endif

    const int64_t delay_us = find_stage_bubble_delay_us(chunk_id, layer_id, stage);
    if (delay_us > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
    }
    return token;
}

void mark_stage_trace_started(StageTraceToken &token) {
#if GET_TRACES
    if (!token.trace_enabled) {
        return;
    }
    auto &state = get_trace_collector_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.active) {
        token.trace_enabled = false;
        return;
    }
    if (trace_sync_stages) {
        torch::cuda::synchronize();
    }
    token.start_ts_us = std::chrono::duration_cast<std::chrono::microseconds>(TraceClock::now() - state.t0).count();
#else
    (void)token;
#endif
}

void end_stage_trace(StageTraceToken token) {
#if GET_TRACES
    if (!token.trace_enabled) {
        return;
    }

    auto &state = get_trace_collector_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.active) {
        return;
    }
    if (trace_sync_stages) {
        torch::cuda::synchronize();
    }
    StageTraceRow row;
    row.run_id = token.run_id;
    row.model = token.model;
    row.arch = token.arch;
    row.chunk_id = token.chunk_id;
    row.slot_id = token.slot_id;
    row.layer_id = token.layer_id;
    row.stage = token.stage;
    row.host_ready_ts_us = token.host_ready_ts_us;
    row.start_ts_us = token.start_ts_us;
    row.end_ts_us = std::chrono::duration_cast<std::chrono::microseconds>(TraceClock::now() - state.t0).count();
    row.start_pos = token.start_pos;
    row.seq_len = token.seq_len;
    state.rows.push_back(std::move(row));
#else
    (void)token;
#endif
}

std::vector<int64_t> get_active_chunk_schedule() {
    if (!chunking_token_schedule.empty()) {
        return chunking_token_schedule;
    }
    if (chunking_tokens > 0) {
        return {chunking_tokens};
    }
    return {};
}

std::vector<LlamaChunkSpan> build_llama_chunk_plan(int64_t seq_len) {
    std::vector<LlamaChunkSpan> plan;
    if (seq_len <= 0) {
        return plan;
    }

    const std::vector<int64_t> schedule = get_active_chunk_schedule();
    if (schedule.empty()) {
        return plan;
    }

    if (schedule.size() == 1) {
        const int64_t chunk_size = schedule.front();
        if (chunk_size <= 0) {
            return plan;
        }
        for (int64_t start = 0; start < seq_len; start += chunk_size) {
            plan.push_back({start, std::min<int64_t>(chunk_size, seq_len - start)});
        }
        return plan;
    }

    int64_t start = 0;
    for (int64_t chunk_size : schedule) {
        if (start >= seq_len) {
            break;
        }
        if (chunk_size <= 0) {
            continue;
        }
        const int64_t len = std::min<int64_t>(chunk_size, seq_len - start);
        plan.push_back({start, len});
        start += len;
    }

    if (start < seq_len) {
        plan.push_back({start, seq_len - start});
    }

    return plan;
}

int64_t max_llama_chunk_count_for_seq_len(int64_t seq_len) { return static_cast<int64_t>(build_llama_chunk_plan(seq_len).size()); }

int64_t resolve_llama_chunk_id_from_start(int64_t start_pos) {
    if (start_pos < 0) {
        return -1;
    }

    const std::vector<int64_t> schedule = get_active_chunk_schedule();
    if (schedule.empty()) {
        return -1;
    }

    if (schedule.size() == 1) {
        const int64_t chunk_size = schedule.front();
        if (chunk_size <= 0) {
            return -1;
        }
        return start_pos / chunk_size;
    }

    int64_t boundary = 0;
    int64_t chunk_id = 0;
    for (int64_t chunk_size : schedule) {
        if (chunk_size <= 0) {
            continue;
        }
        if (start_pos < boundary + chunk_size) {
            return chunk_id;
        }
        boundary += chunk_size;
        ++chunk_id;
    }

    // All positions beyond the staged schedule map to the final remainder chunk.
    return chunk_id;
}

torch::Tensor maybe_narrow_lm_head_input(torch::Tensor x, int64_t seq_len, bool fallback_last_token_only = false) {
#if LM_HEAD_ONLY_LAST
    if (seq_len > 1) {
        // Prefill only consumes the final token logits, so avoid running
        // final_norm + lm_head across the entire prompt sequence.
        // Decode keeps seq_len == 1, so this does not change generation shape.
        x = x.narrow(1, seq_len - 1, 1);
    }
#else
    if (fallback_last_token_only && seq_len > 1) {
        // Preserve the older LLAMA-specific behavior when the global shortcut is
        // disabled: only explicit last-token callers narrow the LM-head input.
        x = x.narrow(1, seq_len - 1, 1);
    }
#endif
    return x;
}
} // namespace

torch::Tensor repeat_kv(const torch::Tensor &x, int64_t n_rep) {
    if (n_rep == 1) {
        return x;
    }
    auto sizes = x.sizes();
    int64_t batch = sizes[0];
    int64_t num_kv_heads = sizes[1];
    int64_t seq_len = sizes[2];
    int64_t head_dim = sizes[3];

    // Assuming [batch, num_kv_heads, seq_len, head_dim] layout
    // We want to repeat the heads (dim 1)
    auto expanded = x.unsqueeze(2).expand({batch, num_kv_heads, n_rep, seq_len, head_dim});
    return expanded.reshape({batch, num_kv_heads * n_rep, seq_len, head_dim});
}

int64_t positive_mod(int64_t value, int64_t mod) {
    if (mod <= 0) {
        return 0;
    }
    int64_t r = value % mod;
    return (r < 0) ? (r + mod) : r;
}

void write_kv_ring(torch::Tensor cache_slice, const torch::Tensor &kv, int64_t start_pos, int64_t window_size) {
    if (window_size <= 0 || kv.numel() == 0) {
        return;
    }

    int64_t seq_len = kv.size(2);
    int64_t tokens_to_write = std::min<int64_t>(seq_len, window_size);
    int64_t src_start = seq_len - tokens_to_write;
    int64_t write_head = positive_mod(start_pos + src_start, window_size);

    auto ring_cache = cache_slice.narrow(2, 0, window_size);
    int64_t first_chunk = std::min<int64_t>(tokens_to_write, window_size - write_head);
    ring_cache.narrow(2, write_head, first_chunk).copy_(kv.narrow(2, src_start, first_chunk));

    int64_t remaining = tokens_to_write - first_chunk;
    if (remaining > 0) {
        ring_cache.narrow(2, 0, remaining).copy_(kv.narrow(2, src_start + first_chunk, remaining));
    }
}

torch::Tensor read_kv_window(const torch::Tensor &cache_slice, int64_t kv_len, int64_t oldest_pos, int64_t window_size) {
    auto ring_cache = cache_slice.narrow(2, 0, window_size);
    if (kv_len <= 0) {
        return ring_cache.narrow(2, 0, 0);
    }

    int64_t oldest_idx = positive_mod(oldest_pos, window_size);
    if (oldest_idx + kv_len <= window_size) {
        return ring_cache.narrow(2, oldest_idx, kv_len);
    }

    int64_t first_chunk = window_size - oldest_idx;
    int64_t second_chunk = kv_len - first_chunk;
    return torch::cat({ring_cache.narrow(2, oldest_idx, first_chunk), ring_cache.narrow(2, 0, second_chunk)}, 2);
}

// Tile sizes
constexpr int LARGE_TILE_SIZE_ROW = 128;
constexpr int LARGE_TILE_SIZE_COL = 64;
constexpr int SMALL_TILE_SIZE = 8;

// Helper to unpack int4 to int8
int8_t unpack_nibble(uint8_t packed, bool high) { return high ? (int8_t)((packed >> 4) & 0x0F) : (int8_t)(packed & 0x0F); }

// Helper to quantize float to int4 (simple round/clip)

// QuantizedLinear declaration (move to header if not already there, but here we modify signature in source)
// We need to modify header first. Let's do header.

// RMSNormImpl Implementation
RMSNormImpl::RMSNormImpl(int64_t dim, float eps, bool gemma_style) : eps_(eps), gemma_style_(gemma_style) {
    // Gemma initializes with zeros to be 1-centered (since it does 1.0 + weight)
    // LLaMA initializes with ones
    auto init_value = gemma_style ? torch::zeros({dim}) : torch::ones({dim});
    weight_ = register_parameter("weight", init_value);
}

torch::Tensor RMSNormImpl::forward(torch::Tensor x) {
    if (x.device().is_cuda()) {
        auto output = torch::empty_like(x);
        launch_rmsnorm(output, x, weight_, eps_, gemma_style_);
        return output;
    }

    auto rms = torch::sqrt(x.pow(2).mean(-1, true) + eps_);
    auto normalized = x / rms;

    if (gemma_style_) {
        // Gemma: output = normalized * (1.0 + weight)
        return normalized * (1.0 + weight_.to(x.dtype()));
    } else {
        // LLaMA/Qwen: output = normalized * weight
        return normalized * weight_.to(x.dtype());
    }
}

void RMSNormImpl::set_weight(torch::Tensor weight) {
    weight_.set_requires_grad(false);
    weight_.copy_(weight.to(weight_.dtype()).to(weight_.device()));
}

void RMSNormImpl::forward_out(torch::Tensor output, torch::Tensor x) {
    if (x.device().is_cuda()) {
        launch_rmsnorm(output, x, weight_, eps_, gemma_style_);
        return;
    }

    // Compute variance and normalize
    auto var = torch::mean(torch::square(x), -1, true);
    auto normed = x * torch::rsqrt(var + eps_);

    // Apply weight
    if (gemma_style_) {
        normed = normed * (1.0 + weight_.to(x.dtype()));
    } else {
        normed = normed * weight_.to(x.dtype());
    }

    // Copy result to output buffer
    output.copy_(normed);
}

// Helper to get tile indices
std::tuple<torch::Tensor, int64_t, int64_t> get_tile_indices(int64_t rows, int64_t cols) {
    int64_t num_tiles_row = (rows + LARGE_TILE_SIZE_ROW - 1) / LARGE_TILE_SIZE_ROW;
    int64_t num_tiles_col = (cols + LARGE_TILE_SIZE_COL - 1) / LARGE_TILE_SIZE_COL;

    std::vector<int64_t> tile_indices;
    tile_indices.reserve(num_tiles_row * num_tiles_col);

    // The specific ordering: col % 8, then col, then row
    for (int col_mod = 0; col_mod < 8; col_mod++) {
        for (int c = col_mod; c < num_tiles_col; c += 8) {
            for (int r = 0; r < num_tiles_row; r++) {
                tile_indices.push_back(r * num_tiles_col + c);
            }
        }
    }

    return std::make_tuple(torch::tensor(tile_indices, torch::kLong), num_tiles_row, num_tiles_col);
}

// HipEmbeddingImpl Implementation
// HipEmbeddingImpl Implementation
HipEmbeddingImpl::HipEmbeddingImpl(int64_t num_embeddings, int64_t embedding_dim, int64_t max_batch_size, int64_t max_seq_len) {
    weight = register_parameter("weight", torch::empty({num_embeddings, embedding_dim}, torch::kBFloat16));
    // Pre-allocate buffer [max_bsz, max_seq, hidden] (BF16)
    output_buffer = register_buffer(
        "output_buffer", torch::empty({max_batch_size, max_seq_len, embedding_dim}, torch::TensorOptions().dtype(torch::kBFloat16)));
}

torch::Tensor HipEmbeddingImpl::forward(torch::Tensor input) {
#if USE_HIP_EMBEDDING
    int64_t bsz = input.size(0);
    int64_t seq_len = input.size(1);
    int64_t total_tokens = bsz * seq_len;

    // Only use Optimized HIP Path for generation (single token)
    if (total_tokens == 1) {
        // Slice pre-allocated buffer
        auto output = output_buffer.slice(0, 0, bsz).slice(1, 0, seq_len);

        hipkernels::launch_embedding_forward(weight, input, output, c10::hip::getCurrentHIPStream().stream());
        return output;
    }
#endif
    // Fallback path (LibTorch Embedding)
    return torch::nn::functional::embedding(input, weight);
}

// LmHeadLinearImpl Implementation
LmHeadLinearImpl::LmHeadLinearImpl(int64_t in_features, int64_t out_features, int64_t max_batch_size, int64_t max_seq_len) {
    // Initialize weight as a parameter [Out, In] with BF16
    weight = register_parameter("weight", torch::empty({out_features, in_features}, torch::kBFloat16));
    // Pre-allocate logits buffer [max_bsz, max_seq, vocab] using BF16
    // This will match the model dtype (BF16) when .to() is called.
    logits_buffer = register_buffer(
        "logits_buffer", torch::empty({max_batch_size, max_seq_len, out_features}, torch::TensorOptions().dtype(torch::kBFloat16)));
}

torch::Tensor LmHeadLinearImpl::forward(torch::Tensor input) {
#if USE_HIP_LM_HEAD
    int64_t bsz = input.size(0);
    int64_t seq_len = input.size(1);
    int64_t total_tokens = bsz * seq_len;

    // Only use Optimized HIP Path for generation (single token)
    if (total_tokens == 1) {
        int64_t vocab_size = weight.size(0);
        int64_t hidden_size = weight.size(1);

        // Slice pre-allocated buffer
        auto logits = logits_buffer.slice(0, 0, bsz).slice(1, 0, seq_len);

        hipkernels::launch_lm_head_forward(
            (hip_bfloat16 *)logits.data_ptr<at::BFloat16>(), (const hip_bfloat16 *)input.data_ptr<at::BFloat16>(),
            (const hip_bfloat16 *)weight.data_ptr<at::BFloat16>(), 1, hidden_size, vocab_size, c10::hip::getCurrentHIPStream().stream());
        // Cast to Float32 for sampling
        return logits.to(torch::kFloat32);
    }
#endif
    // Fallback path (LibTorch Linear) for prefill or if HIP disabled
    return torch::nn::functional::linear(input, weight).to(torch::kFloat32);
}

torch::Tensor QuantizedLinearImpl::dequantize_weights_packed() {
    return QuantizedLinearImpl::dequantize_packed(packed_params_, in_features_, out_features_);
}

torch::Tensor QuantizedLinearImpl::dequantize_weights() {
    torch::NoGradGuard no_grad;
    auto device = quantized_weight_.device();

    const int64_t out_features = out_features_;
    const int64_t in_features = in_features_;
    const int64_t packed_cols = quantized_weight_.size(1);

    // 1) Unpack 4-bit weights without creating temporary stacks.
    // Optimization: Use _out variants to avoid temporary tensor allocations
    auto unpack_pairs = torch::empty({out_features, packed_cols, 2}, torch::dtype(torch::kUInt8).device(device));

    auto low_bits = unpack_pairs.select(-1, 0);
    auto high_bits = unpack_pairs.select(-1, 1);

    // Write directly to the output slices to avoid intermediate allocations
    torch::bitwise_and_out(low_bits, quantized_weight_, 0x0F);
    torch::bitwise_right_shift_out(high_bits, quantized_weight_, 4);

    auto unpacked_uint8 = unpack_pairs.view({out_features, packed_cols * 2});
    if (unpacked_uint8.size(1) != in_features) {
        unpacked_uint8 = unpacked_uint8.slice(1, 0, in_features);
    }
    auto unpacked = unpacked_uint8.to(torch::kInt8);

    // 2) Apply scale / zero via broadcasting-friendly views.
    if (scale_.dim() == 2) {
        auto apply_group = [&](const torch::Tensor &scales, const torch::Tensor &zeros) {
            int64_t n_groups = scales.size(1);
            TORCH_CHECK(in_features % n_groups == 0, "Group size mismatch");
            int64_t group_size = in_features / n_groups;

            auto w_view = unpacked.view({out_features, n_groups, group_size});
            auto s_view = scales.view({out_features, n_groups, 1});
            auto z_view = zeros.view({out_features, n_groups, 1}); // Int8

            // w_view is Int8, z_view is Int8.
            // Perform subtraction in Int8, then convert to BFloat16 and apply scale.
            auto w_sub = w_view.sub(z_view);
            return w_sub.to(torch::kBFloat16).mul_(s_view).view({out_features, in_features});
        };

        if (g_idx_.defined()) {
            // std::cout << "Applying group index" << std::endl;
            auto g_idx_long = g_idx_.to(torch::kLong);
            auto s_reordered = scale_.index_select(1, g_idx_long);
            auto z_reordered = zero_point_.index_select(1, g_idx_long);
            return apply_group(s_reordered, z_reordered);
        }

        return apply_group(scale_, zero_point_);
    } else {
        auto s_view = scale_.view({out_features, 1});
        auto z_view = zero_point_.view({out_features, 1}); // Int8
        // Perform subtraction in Int8, then convert to BFloat16 and apply scale
        return unpacked.sub(z_view).to(torch::kBFloat16).mul_(s_view);
    }
}

torch::Tensor QuantizedLinearImpl::dequantize_packed(const torch::Tensor &packed_params, int64_t in_features, int64_t out_features) {
    torch::NoGradGuard no_grad;
    auto device = packed_params.device();

    int64_t K = in_features;
    int64_t N = out_features;

    int64_t num_tiles_row = (K + LARGE_TILE_SIZE_ROW - 1) / LARGE_TILE_SIZE_ROW;
    int64_t num_tiles_col = (N + LARGE_TILE_SIZE_COL - 1) / LARGE_TILE_SIZE_COL;
    int64_t num_large_tiles = num_tiles_row * num_tiles_col;

    int64_t K_padded = num_tiles_row * LARGE_TILE_SIZE_ROW;
    int64_t N_padded = num_tiles_col * LARGE_TILE_SIZE_COL;

    // Check packed size
    if (packed_params.numel() != num_large_tiles * 4352) {
        // Warning: Size mismatch possibly due to padding or legacy packing
    }

    auto packed_reshaped = packed_params.view({num_large_tiles, 4352});

    // 1. Unpack Weights to Tiles (ZigZag Order)
    auto w_bytes = packed_reshaped.index({torch::indexing::Slice(), torch::indexing::Slice(0, 4096)});
    auto w_small = w_bytes.view({-1, 16, 8, 8, 4});

    auto w_even = w_small & 0x0F;
    auto w_odd = torch::bitwise_right_shift(w_small, 4) & 0x0F;

    auto w_unpacked_blk = torch::empty({num_large_tiles, 16, 8, 8, 8}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    w_unpacked_blk.index_put_({"...", torch::indexing::Slice(0, torch::indexing::None, 2)}, w_even);
    w_unpacked_blk.index_put_({"...", torch::indexing::Slice(1, torch::indexing::None, 2)}, w_odd);

    // Restore [Tile, 128, 64]
    // Permute {SR, IR, SC, IC} -> {SR, SC, IR, IC} to satisfy 128x64
    // Dims: 0:Tile, 1:SR, 2:SC, 3:IR, 4:IC -> 0, 1, 3, 2, 4
    auto w_restored = w_unpacked_blk.permute({0, 1, 3, 2, 4}).contiguous();
    auto w_large = w_restored.view({num_large_tiles, 128, 64});

    // 2. Unpack scales and zeros (ZigZag Order)
    auto s_bytes = packed_reshaped.index({torch::indexing::Slice(), torch::indexing::Slice(4096, 4096 + 128)});
    auto s_int16 = s_bytes.view(torch::kInt16);
    auto s_bf16 = s_int16.view(torch::kBFloat16); // [TotalTiles, 64]

    // Zeros
    auto z_bytes = packed_reshaped.index({torch::indexing::Slice(), torch::indexing::Slice(4096 + 128, torch::indexing::None)});
    // Consumed interleaved zeros (8 bytes real, 8 bytes duplicate, etc.)
    auto z_reshaped = z_bytes.contiguous().view({num_large_tiles, 8, 2, 8});      // [T, 8, 2, 8]
    auto z_u8 = z_reshaped.select(2, 0).contiguous().view({num_large_tiles, 64}); // [T, 64]

    // 3. Dequantize Computation (Per Tile)
    // w: [T, 128, 64]
    // s, z: [T, 64] -> unsqueeze(1) -> [T, 1, 64]
    auto s_exp = s_bf16.unsqueeze(1);
    auto z_exp = z_u8.unsqueeze(1).to(torch::kBFloat16);
    // Dequantize: (w - z) * s
    // `test_layout` uses `(float(w) - float(z)) * scale`.
    // Or if `z` is `uint8`.

    auto w_fp = (w_large.to(torch::kBFloat16) - z_exp).mul_(s_exp);
    // w_fp is [TotalTiles, 128, 64] (ZigZag)

    // 4. Reorder Tiles (ZigZag -> RowMajor)
    auto [tile_indices, _r, _c] = get_tile_indices(K, N);
    tile_indices = tile_indices.to(device).to(torch::kLong);
    auto inverse_indices = torch::argsort(tile_indices);

    auto w_ordered = w_fp.index_select(0, inverse_indices); // [TotalTiles, 128, 64] (Row Major)

    // 5. Form Grid and Transpose
    // [RowTiles, ColTiles, 128, 64]
    auto w_grid = w_ordered.view({num_tiles_row, num_tiles_col, 128, 64});
    // [RowTiles, 128, ColTiles, 64] -> [K_padded, N_padded]
    w_grid = w_grid.permute({0, 2, 1, 3}).contiguous();
    auto w_out_in_padded = w_grid.view({K_padded, N_padded});

    // Crop to valid area
    auto w_out_in = w_out_in_padded.slice(0, 0, K).slice(1, 0, N);

    // Return [Out, In]
    return w_out_in.t();
}

// QuantizedLinearImpl Implementation
QuantizedLinearImpl::QuantizedLinearImpl(int64_t in_features, int64_t out_features, bool bias, int64_t max_seq_len, std::string layer_type)
    : in_features_(in_features), out_features_(out_features), padded_out_features_(out_features), padded_in_features_(in_features),
      max_seq_len_(max_seq_len) {
    // Unified packed buffer
    // Number of large tiles = (N / 64) * (K / 128)
    // Size per large tile = 4608 bytes

    if (use_packed_weights) {
        auto pad_spec = get_packed_weight_pad_alignment(layer_type);
        pad_k_alignment_ = pad_spec[0];
        pad_n_alignment_ = pad_spec[1];
        if (pad_packed_weights > 0) {
            padded_in_features_ = round_up_to_alignment(in_features_, pad_k_alignment_);
            padded_out_features_ = round_up_to_alignment(out_features_, pad_n_alignment_);
        }

        // Determine if K-split is needed based on config lookup
        // Check config map for ANY layer with these dimensions that requires K-split
        // Pass empty string as layer_type to search all layers
        if (hw_target == "hetero" || hw_target == "gpu_split") {
            int split_type = get_split_type(in_features, out_features, layer_type);
            if (split_type > 0) {
                is_k_split_ = split_type;
            }
        }

        if (is_k_split_ > 0) {
            if (debug_verbosity >= 1)
                std::cout << "QuantizedLinearImpl: Enabling K-Split for " << in_features << "x" << out_features << " (NPU K=" << is_k_split_
                          << ")" << std::endl;
            // Split K dimension
            int64_t K = in_features;
            int64_t K0 = is_k_split_;
            int64_t K1 = K - K0;
            int64_t N = padded_out_features_;
            int64_t K0_internal = (pad_packed_weights > 0) ? round_up_to_alignment(K0, pad_k_alignment_) : K0;
            int64_t K1_internal = (pad_packed_weights > 0) ? round_up_to_alignment(K1, pad_k_alignment_) : K1;

            // Calc buffer size for first half
            int64_t num_tiles_row_0 = (K0_internal + LARGE_TILE_SIZE_ROW - 1) / LARGE_TILE_SIZE_ROW;
            int64_t num_tiles_col = (N + LARGE_TILE_SIZE_COL - 1) / LARGE_TILE_SIZE_COL; // Shared N
            int64_t total_tiles_0 = num_tiles_col * num_tiles_row_0;
            int64_t buffer_size_0 = total_tiles_0 * 4352;

            packed_params_0_ = register_parameter("packed_params_0", torch::zeros({buffer_size_0}, torch::kUInt8), /*requires_grad=*/false);

            // Calc buffer size for second half
            int64_t num_tiles_row_1 = (K1_internal + LARGE_TILE_SIZE_ROW - 1) / LARGE_TILE_SIZE_ROW;
            int64_t total_tiles_1 = num_tiles_col * num_tiles_row_1;
            int64_t buffer_size_1 = total_tiles_1 * 4352;

            packed_params_1_ = register_parameter("packed_params_1", torch::zeros({buffer_size_1}, torch::kUInt8), /*requires_grad=*/false);

        } else {
            int64_t K = padded_in_features_;
            int64_t N = padded_out_features_;
            int64_t num_tiles_row = (K + LARGE_TILE_SIZE_ROW - 1) / LARGE_TILE_SIZE_ROW;
            int64_t num_tiles_col = (N + LARGE_TILE_SIZE_COL - 1) / LARGE_TILE_SIZE_COL;
            int64_t total_tiles = num_tiles_col * num_tiles_row;
            int64_t buffer_size = total_tiles * 4352;

            packed_params_ = register_parameter("packed_params", torch::zeros({buffer_size}, torch::kUInt8), /*requires_grad=*/false);
        }
    } else {
        // Separate buffers for manual path
        int64_t packed_size = (in_features + 1) / 2;
        quantized_weight_ = register_buffer("quantized_weight", torch::zeros({out_features, packed_size}, torch::kUInt8));
        scale_ = register_buffer("scale", torch::ones({out_features}, torch::kBFloat16));
        zero_point_ = register_buffer("zero_point", torch::zeros({out_features}, torch::kInt8));
    }

    if (bias) {
        bias_ = register_parameter("bias", torch::zeros({out_features}, torch::kBFloat16));
    }
}

std::future<int> QuantizedLinearImpl::forward(torch::Tensor output_buffer, torch::Tensor input, std::string layer_type, int chunk_id) {

    // Dequantize weights to bf16
    std::future<int> fut;

    // Handle Padding Output Buffer
    torch::Tensor result_buffer = output_buffer;
    int64_t N_internal = out_features_;
    bool is_padded = false;

    if (use_packed_weights && padded_out_features_ > out_features_) {
        // Check if output_buffer is large enough to hold padded result
        if (output_buffer.size(-1) >= padded_out_features_) {
            if (debug_verbosity >= 2) {
                std::cout << "Using Preallocated Buffers" << std::endl;
            }
            result_buffer = output_buffer.slice(-1, 0, padded_out_features_);
            N_internal = padded_out_features_;
            is_padded = false; // Direct write, no copy back needed
        } else {
            std::cerr << "Error: output_buffer is not large enough to hold padded results. " << "Expected " << padded_out_features_
                      << ", got " << output_buffer.size(-1) << std::endl;
            std::exit(1);
        }
    }

    if (debug_verbosity >= 2) {
        std::cout << "Forward " << layer_type << " (Target: " << hw_target << ")" << std::endl;
    }

    // Calculate M and decode flag
    int64_t input_width = input.size(-1);
    int64_t M = input_width > 0 ? (input.numel() / input_width) : 0;
    bool is_decode = (M == 1);

    if (hw_target == "cpu") {
        if (packed_params_.defined() && packed_params_.numel() > 0) {
            throw std::runtime_error("error when packed for cpu");
        } else {
            // Unpacked Weights Path
            if (is_decode) {
                // GEMV: Use CPU Fused Unpacked
                if (debug_verbosity >= 2)
                    std::cout << "CPU Unpacked GEMV" << std::endl;

                int64_t num_groups = scale_.size(1);
                int64_t group_size = in_features_ / num_groups;

                // Sync with GPU to ensure input is ready (XDNA shared memory)
                // This is critical because CPU kernel is host-side and won't wait for GPU stream
                // Use DeviceSynchronize for simplicity (blocks host until all GPU work done)
                HIP_CHECK(hipDeviceSynchronize());

                // Call global CPU kernel via thread pool for async execution
                fut = g_thread_pool.submit_task([=]() mutable {
                    w4a16_gemv_cpu_fused_unpacked(output_buffer, input, quantized_weight_, scale_, zero_point_, in_features_, out_features_,
                                                  group_size, nullptr, 1);
                    return 0;
                });
            } else {
                // GEMM: Use Unpacked GPU Kernel (Fallback)
                if (debug_verbosity >= 1)
                    std::cout << "CPU Unpacked GEMM: Fallback to GPU only (No CPU GEMM yet)" << std::endl;

                // Calculate group size from scales
                int64_t num_groups = scale_.size(1);
                int64_t group_size = in_features_ / num_groups;

                // View input/output as 2D
                auto input_2d = input.view({-1, in_features_});
                auto output_2d = output_buffer.view({-1, out_features_});

                hipkernels::w4a16_gemm_unpacked_fused(output_2d, input_2d, quantized_weight_, scale_, zero_point_, in_features_,
                                                      out_features_, group_size);
                // Return empty future
                fut = std::future<int>();
            }
        }
    } else if (hw_target == "hetero") {
        if (debug_verbosity >= 2) {
            std::cout << "Hetero Path (" << (is_decode ? "Optimized GEMV" : "Optimized Fused Kernel") << ")" << std::endl;
        }

        if ((packed_params_.defined() && packed_params_.numel() > 0) || (packed_params_0_.defined() && packed_params_0_.numel() > 0)) {
            // Packed Weights Path
            // Prevent GEMM-derived K-split from forcing GEMV K-split when GEMV split mode is disabled.
            int runtime_k_split = is_k_split_;
            const bool has_split_packed_params =
                packed_params_0_.defined() && packed_params_0_.numel() > 0 && packed_params_1_.defined() && packed_params_1_.numel() > 0;
            if (is_decode && !gemv_driven_split_K) {
                runtime_k_split = 0;
                if (debug_verbosity >= 2 && is_k_split_ > 0) {
                    std::cout << "Decode GEMV: gemv_driven_split_K=false, overriding K-split to GPU path." << std::endl;
                }
            }

            const bool serialize_chunked_split_k = (chunk_id >= 0 && async_chunking && has_split_packed_params);
            if (runtime_k_split > 0) {
                // K-Split Path
                if (debug_verbosity >= 2)
                    std::cout << "Hetero Path (K-Split)" << std::endl;
                const bool force_splitk_gpu = (std::getenv("HETERO_SPLITK_FORCE_GPU") != nullptr);
                if (debug_verbosity >= 2 && serialize_chunked_split_k) {
                    std::cout << "Using runtime chunked K-split scratch for chunk_id=" << chunk_id << std::endl;
                }
                fut = is_decode ? hetero_matmul_out_gemv_packed_K(result_buffer, input, packed_params_0_, packed_params_1_, in_features_,
                                                                  N_internal, runtime_k_split, layer_type, chunk_id, force_splitk_gpu)
                                : hetero_matmul_out_gemm_packed_K(result_buffer, input, packed_params_0_, packed_params_1_, in_features_,
                                                                  N_internal, runtime_k_split, layer_type, chunk_id, force_splitk_gpu);
            } else if (is_decode && has_split_packed_params) {
                // Split-K layers do not populate packed_params_. When decode disables NPU-driven split-K,
                // keep using the split buffers but force the GPU-only split implementation.
                fut = hetero_matmul_out_gemv_packed_K(result_buffer, input, packed_params_0_, packed_params_1_, in_features_, N_internal,
                                                      is_k_split_, layer_type, chunk_id, true);
            } else {
                // M-Split (Standard) Path
                fut = is_decode ? hetero_matmul_out_gemv_packed_M(result_buffer, input, packed_params_, in_features_, N_internal,
                                                                  layer_type, chunk_id)
                                : hetero_matmul_out_gemm_packed_M(result_buffer, input, packed_params_, in_features_, N_internal,
                                                                  layer_type, chunk_id);
            }
        } else {
            // Unpacked Weights Path
            if (is_decode) {
                // GEMV: Use Hetero Unpacked implementation
                fut = hetero_matmul_out_gemv_unpacked_M(output_buffer, input, quantized_weight_, scale_, zero_point_, in_features_,
                                                        out_features_, layer_type, chunk_id);

            } else {
                // GEMM: Use Unpacked GPU Kernel (Fallback, no NPU/CPU splitting implemented for GEMM yet)
                if (debug_verbosity >= 2)
                    std::cout << "Hetero Unpacked GEMM: Fallback to GPU only" << std::endl;

                // Calculate group size from scales
                int64_t num_groups = scale_.size(1);
                int64_t group_size = in_features_ / num_groups;

                // View input/output as 2D
                auto input_2d = input.view({-1, in_features_});
                auto output_2d = output_buffer.view({-1, out_features_});

                hipkernels::w4a16_gemm_unpacked_fused(output_2d, input_2d, quantized_weight_, scale_, zero_point_, in_features_,
                                                      out_features_, group_size);
                // Return empty future
                fut = std::future<int>();
            }
        }
    } else if (hw_target == "npu" || hw_target == "npu-sim") {
        if (debug_verbosity >= 2) {
            std::cout << (hw_target == "npu-sim" ? "NPU Sim Path" : "NPU Path") << " (" << (is_decode ? "Strict GEMV" : "Strict") << ")"
                      << std::endl;
        }
        int64_t K_internal = padded_in_features_;

        fut = is_decode
                  ? npu_top_matmul_out_gemv_packed(result_buffer, input, packed_params_, K_internal, N_internal, layer_type, chunk_id)
                  : npu_top_matmul_out_gemm_packed(result_buffer, input, packed_params_, K_internal, N_internal, layer_type, chunk_id);
    } else if (hw_target == "gpu_split") {
        // Deprecated gpu_split path - effectively merged into hetero K-split
        // But if explicitly requested, map to similar logic or warn
        std::cerr << "Warning: hw_target=gpu_split is deprecated. Use hetero with appropriate config." << std::endl;
        fut = std::future<int>(); // No-op catch

    } else {
        if (debug_verbosity >= 2) {
            std::cout << "GPU Path (Optimized Fused Kernel)" << std::endl;
        }
        // Use optimized fused kernel (vectorized + register blocked)
        int64_t K_internal = padded_in_features_;

        // Check for padding logic mismatch
        if (input.size(-1) < K_internal) {
            if (debug_verbosity >= 2)
                std::cout << "Warning: Input unpadded (" << input.size(-1) << ") but K is padded (" << K_internal << ")" << std::endl;
        }

        auto input_2d = input.view({-1, input.size(-1)}); // View as is
        auto output_2d = result_buffer.view({-1, N_internal});

        if (debug_verbosity >= 2) {
            std::cout << "DEBUG: Calling " << (is_decode ? "GEMV" : "GEMM") << " Input: " << input.sizes()
                      << " Output: " << result_buffer.sizes() << " Weight: [" << K_internal << ", " << N_internal << "]" << std::endl;
        }

        if (use_packed_weights) {
            hipStream_t current_stream = c10::hip::getCurrentHIPStream().stream();
            if (is_decode) {
                hipkernels::w4a16_gemv_fused_packed(output_2d, input_2d, packed_params_, K_internal, N_internal, -1, 0, current_stream);
            } else {
                hipkernels::w4a16_gemm_fused_packed(output_2d, input_2d, packed_params_, K_internal, N_internal, current_stream);
            }
        } else {
            if (debug_verbosity >= 2)
                std::cout << "Hetero Unpacked GPU only" << std::endl;

            // Use optimized unpacked HIP kernel for manual weights path
            // Determine group size for scales
            int64_t group_size = 128; // Default
            if (scale_.dim() == 2) {
                // scale_ is [Out, Groups] -> group_size = In / Groups
                int64_t n_groups = scale_.size(1);
                if (n_groups > 0)
                    group_size = in_features_ / n_groups;
            } else if (scale_.size(0) == out_features_) {
                // Per channel -> group_size = In (single group)
                group_size = in_features_;
            }

            if (is_decode) {
                hipkernels::w4a16_gemv_unpacked_fused(output_2d, input_2d, quantized_weight_, scale_, zero_point_, in_features_,
                                                      out_features_, group_size, c10::hip::getCurrentHIPStream().stream());
            } else {
                hipkernels::w4a16_gemm_unpacked_fused(output_2d, input_2d, quantized_weight_, scale_, zero_point_, in_features_,
                                                      out_features_, group_size, c10::hip::getCurrentHIPStream().stream());
            }
            // auto w_dq = dequantize_weights();
            // torch::matmul_out(output_2d, input_2d, w_dq.t());
        }
    }

    if (bias_.defined()) {
        // If bias is defined, we must wait for GEMM to finish before adding bias
        // This makes it synchronous for this layer
        if (fut.valid()) {
            fut.wait();
        }

        // Handle copy back if padded and NOT aliased
        if (is_padded && !result_buffer.is_alias_of(output_buffer)) {
            output_buffer.copy_(result_buffer.slice(-1, 0, out_features_));
        }

        if (output_buffer.size(-1) == bias_.size(-1)) {
            output_buffer.add_(bias_);
        } else {
            // Case: optimizing padding, output_buffer is padded, bias is not
            output_buffer.slice(-1, 0, bias_.size(-1)).add_(bias_);
        }
        return std::future<int>(); // Return invalid/empty future as we are done
    }

    // If no bias, but padded, we MUST wait and copy back
    if (is_padded) {
        if (fut.valid())
            fut.wait();
        output_buffer.copy_(result_buffer.slice(-1, 0, out_features_));
        return std::future<int>();
    }

    return fut;
}

void QuantizedLinearImpl::set_quantized_weights(torch::Tensor qweight, torch::Tensor scale, torch::Tensor zero_point, torch::Tensor g_idx) {

    auto device = qweight.device();

    // 1. Prepare unpacked int8 weights [out_features, in_features]
    // (Normalized to [In, Out] Column Major)

    torch::Tensor unpacked_qweight; // Declare unpacked_qweight here

    if (qweight.size(0) == in_features_ && qweight.size(1) == out_features_) {
        // [In, Out] -> Use directly
        // Prioritize this check so that square matrices (In==Out) are treated as [In, Out]
        if (debug_verbosity >= 2) {
            std::cout << "Using qweight INT8 COLUMN major [In, Out]" << std::endl;
        }
        unpacked_qweight = qweight.contiguous().to(torch::kInt8);
    } else if (qweight.size(0) == out_features_ && qweight.size(1) == in_features_) {
        // [Out, In] -> Transpose to [In, Out] (NPU Expects [In, Out])
        if (debug_verbosity >= 2) {
            std::cout << "Using qweight INT8 ROW major [Out, In] -> Transposing to [In, Out]" << std::endl;
        }
        unpacked_qweight = qweight.to(torch::kInt8).t().contiguous();
    } else if (qweight.size(0) == out_features_ && qweight.size(1) == (in_features_ + 1) / 2) {

        // [out, in/2] packed -> unpack
        int64_t out_features = qweight.size(0);
        int64_t in_features = qweight.size(1) * 2;

        auto q_view = qweight.view({out_features, in_features / 2, 1});
        auto w_low = torch::bitwise_and(q_view, 0x0F).to(torch::kInt8);
        auto w_high = torch::bitwise_right_shift(q_view, 4).to(torch::kInt8);

        std::vector<torch::Tensor> cat_tensors;
        cat_tensors.push_back(w_low);
        cat_tensors.push_back(w_high);
        // Result is [Out, In] -> Transpose to [In, Out]
        unpacked_qweight = torch::cat(cat_tensors, 2).view({out_features, in_features}).t().contiguous();
    } else {
        std::cerr << "Error: qweight shape " << qweight.sizes() << " not supported in set_quantized_weights" << std::endl;
        return;
    }

    // 0. Manual / Unpacked Mode Handling
    if (!use_packed_weights) {
        if (debug_verbosity >= 2) {
            std::cout << "Setting weights (Manual Mode) - repacking to [Out, In/2]" << std::endl;
        }

        // 1. Pack weights: [In, Out] -> [Out, In] -> [Out, In/2]
        // unpacked_qweight is [In, Out]
        auto w_out_in = unpacked_qweight.t().contiguous(); // [Out, In]

        // Pack into uint8 (2 val/byte)
        // [Out, In/2, 2]
        auto w_view = w_out_in.view({out_features_, in_features_ / 2, 2});
        auto w_low = w_view.select(-1, 0).to(torch::kUInt8);
        auto w_high = w_view.select(-1, 1).to(torch::kUInt8);

        // Standard packing: Low nibble comes first in input stream (Column 0), High nibble (Column 1)
        auto packed_w = torch::bitwise_or(torch::bitwise_and(w_low, 0x0F), torch::bitwise_left_shift(torch::bitwise_and(w_high, 0x0F), 4));

        quantized_weight_ = packed_w;

        // 2. Handle Scales and Zeros
        // Check current shapes
        if (scale.size(0) == out_features_ && scale.dim() == 1) {
            // [Out] -> OK (Per Channel)
            scale_ = scale.to(device).to(torch::kBFloat16);
            zero_point_ = zero_point.to(device).to(torch::kInt8);
        } else {
            // Assuming Grouped
            int64_t num_scales = scale.numel();
            int64_t n_groups = num_scales / out_features_;

            if (scale.size(1) == out_features_) {
                // [Groups, Out] -> Transpose to [Out, Groups]
                scale_ = scale.t().contiguous().to(device).to(torch::kBFloat16);
                zero_point_ = zero_point.t().contiguous().to(device).to(torch::kInt8);
            } else {
                // Already [Out, Groups] or Flat
                scale_ = scale.reshape({out_features_, n_groups}).to(device).to(torch::kBFloat16);
                zero_point_ = zero_point.reshape({out_features_, n_groups}).to(device).to(torch::kInt8);
            }
        }

        if (g_idx.defined()) {
            g_idx_ = register_buffer("g_idx", g_idx.to(device));
        }
        return;
    }

    scale = scale.to(torch::kBFloat16);

    // Explicitly unpack Int32 qzeros (AWQ/GPTQ format) if needed
    if (zero_point.dtype() == torch::kInt32) {
        // [Groups, N/8] -> Unpack 8x 4-bit values to [Groups, N]
        auto zeros_i32 = zero_point;
        std::vector<torch::Tensor> unpacked_vec;
        for (int i = 0; i < 8; ++i) {
            unpacked_vec.push_back(torch::bitwise_and(torch::bitwise_right_shift(zeros_i32, 4 * i), 0x0F).to(torch::kInt8));
        }
        // Stack along last dimension and flattening
        auto zeros_stacked = torch::stack(unpacked_vec, -1); // [Groups, N/8, 8]
        zero_point = zeros_stacked.flatten(1, 2);            // [Groups, N]
    }

    zero_point = zero_point.to(torch::kInt8); // Ensure zeros are Int8 for packing

    // Local variables for potentially padded tensors
    torch::Tensor qweight_final = unpacked_qweight;
    torch::Tensor scale_final = scale;
    torch::Tensor zero_final = zero_point;

    // Helper Lambda for packing
    auto pack_weights_lambda = [&](torch::Tensor &dest_packed, torch::Tensor qw, torch::Tensor s, torch::Tensor z, int64_t K_in,
                                   int64_t N_in) {
        int64_t K_npu = K_in;
        int64_t N_npu = N_in;

        torch::Tensor qw_p = qw;
        torch::Tensor s_p = s;
        torch::Tensor z_p = z;

        if (pad_packed_weights) {
            int64_t pad_align_N = pad_n_alignment_;
            int64_t pad_align_K = pad_k_alignment_;

            // Pad N (Output Features)
            if (pad_align_N > 0 && N_in % pad_align_N != 0) {
                int64_t padded_N = round_up_to_alignment(N_in, pad_align_N);
                int64_t pad_val = padded_N - N_in;
                N_npu = padded_N;
                // Note: We don't update class member padded_out_features_ here as this might be a split

                if (debug_verbosity >= 2) {
                    std::cout << "Padding packed weights N: " << N_in << " -> " << padded_N << " (pad=" << pad_val << ")" << std::endl;
                }

                qw_p = torch::nn::functional::pad(qw_p, torch::nn::functional::PadFuncOptions({0, pad_val}));

                if (s.size(0) == N_in) {
                    if (s.dim() == 1) { // [N]
                        s_p = torch::nn::functional::pad(s_p, torch::nn::functional::PadFuncOptions({0, pad_val}));
                        z_p = torch::nn::functional::pad(z_p, torch::nn::functional::PadFuncOptions({0, pad_val}));
                    } else { // [N, G]
                        s_p = torch::nn::functional::pad(s_p, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_val}));
                        z_p = torch::nn::functional::pad(z_p, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_val}));
                    }
                } else if (s.size(1) == N_in) { // [G, N]
                    s_p = torch::nn::functional::pad(s_p, torch::nn::functional::PadFuncOptions({0, pad_val, 0, 0}));
                    z_p = torch::nn::functional::pad(z_p, torch::nn::functional::PadFuncOptions({0, pad_val, 0, 0}));
                }
            }

            // Pad K (Input Features)
            if (pad_align_K > 0 && K_in % pad_align_K != 0) {
                int64_t padded_K = round_up_to_alignment(K_in, pad_align_K);
                int64_t pad_val_k = padded_K - K_in;
                K_npu = padded_K;

                if (debug_verbosity >= 2) {
                    std::cout << "Padding packed weights K: " << K_in << " -> " << padded_K << " (pad=" << pad_val_k << ")" << std::endl;
                }

                qw_p = torch::nn::functional::pad(qw_p, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_val_k}));

                // Handle Scale Padding for K (Groups)
                bool is_grouped = (s.numel() > s.size(0) && s.numel() > s.size(1));
                if (s.dim() == 1 && s.size(0) == N_in)
                    is_grouped = false;

                if (is_grouped) {
                    int64_t num_groups = -1;
                    int64_t group_dim = -1;
                    if (s_p.size(0) == N_npu && s_p.dim() == 2) {
                        num_groups = s_p.size(1);
                        group_dim = 1;
                    } else if (s_p.size(1) == N_npu && s_p.dim() == 2) {
                        num_groups = s_p.size(0);
                        group_dim = 0;
                    }

                    if (num_groups > 0) {
                        int64_t group_size = 128;
                        int64_t expected_groups = (padded_K + group_size - 1) / group_size;
                        int64_t pad_groups = expected_groups - num_groups;
                        if (pad_groups > 0) {
                            if (group_dim == 1) { // [N, G]
                                s_p = torch::nn::functional::pad(s_p, torch::nn::functional::PadFuncOptions({0, pad_groups, 0, 0}));
                                z_p = torch::nn::functional::pad(z_p, torch::nn::functional::PadFuncOptions({0, pad_groups, 0, 0}));
                            } else { // [G, N]
                                s_p = torch::nn::functional::pad(s_p, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_groups}));
                                z_p = torch::nn::functional::pad(z_p, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_groups}));
                            }
                        }
                    }
                }
            }
        }

        // Preparation
        auto scale_flat = s_p.contiguous().view({-1});
        auto zero_flat = z_p.contiguous().view({-1});
        int64_t num_scales = scale_flat.size(0);
        int64_t total_weights = N_npu * K_npu;
        int64_t group_size = (num_scales > 0) ? (total_weights / num_scales) : 128; // Estimate

        int64_t num_tiles_row = (K_npu + LARGE_TILE_SIZE_ROW - 1) / LARGE_TILE_SIZE_ROW;
        int64_t num_tiles_col = (N_npu + LARGE_TILE_SIZE_COL - 1) / LARGE_TILE_SIZE_COL;
        int64_t K_padded = num_tiles_row * LARGE_TILE_SIZE_ROW;
        int64_t N_padded = num_tiles_col * LARGE_TILE_SIZE_COL;

        auto w_padded = torch::zeros({K_padded, N_padded}, torch::TensorOptions().dtype(torch::kInt8).device(device));
        w_padded.slice(0, 0, K_npu).slice(1, 0, N_npu).copy_(qw_p);

        // Extract Tiles
        auto w_tiles = w_padded.view({num_tiles_row, 128, num_tiles_col, 64});
        w_tiles = w_tiles.permute({0, 2, 1, 3}).contiguous().view({-1, 128, 64});

        // Reorder
        auto [tile_indices, _r, _c] = get_tile_indices(K_npu, N_npu);
        tile_indices = tile_indices.to(device).to(torch::kLong);
        auto w_ordered = w_tiles.index_select(0, tile_indices);

        // Pack 4-bit
        int64_t total_tiles = w_ordered.size(0);
        auto w_reshaped = w_ordered.view({total_tiles, 16, 8, 8, 8});
        auto w_permuted = w_reshaped.permute({0, 1, 3, 2, 4}).contiguous();
        auto w_even = w_permuted.index({"...", torch::indexing::Slice(0, torch::indexing::None, 2)});
        auto w_odd = w_permuted.index({"...", torch::indexing::Slice(1, torch::indexing::None, 2)});
        auto w_packed_blk =
            torch::bitwise_or(torch::bitwise_and(w_even, 0x0F), torch::bitwise_left_shift(torch::bitwise_and(w_odd, 0x0F), 4))
                .to(torch::kUInt8);
        auto packed_weights_flat = w_packed_blk.view({-1, 4096});

        // Pack Scales/Zeros
        auto tile_rows = torch::div(tile_indices, num_tiles_col, "floor");
        auto tile_cols = torch::remainder(tile_indices, num_tiles_col);
        auto col_offsets = torch::arange(64, device).unsqueeze(0);
        auto global_col_indices = tile_cols.unsqueeze(1) * 64 + col_offsets;
        auto global_rows_start = tile_rows.unsqueeze(1) * 128;
        auto group_idx = torch::div(global_rows_start, group_size, "floor");
        auto scale_indices = group_idx * N_npu + global_col_indices;
        scale_indices = torch::clamp(scale_indices, 0, num_scales - 1).to(torch::kLong);

        auto gathered_scales = scale_flat.index_select(0, scale_indices.view({-1})).view({-1, 64});
        auto gathered_zeros = zero_flat.index_select(0, scale_indices.view({-1})).view({-1, 64});

        auto scales_uint8 = gathered_scales.view(torch::kUInt8).view({-1, 128});
        auto zeros_dup = gathered_zeros.view(torch::kUInt8).view({-1, 8, 8}).repeat_interleave(2, 1).view({-1, 128});

        std::vector<torch::Tensor> tensors_to_cat;
        tensors_to_cat.push_back(packed_weights_flat);
        tensors_to_cat.push_back(scales_uint8);
        tensors_to_cat.push_back(zeros_dup);
        auto packed_final = torch::cat(tensors_to_cat, 1);

        dest_packed.resize_({packed_final.numel()});
        dest_packed.copy_(packed_final.view({-1}));
    };

    if (is_k_split_ > 0) {
        std::cout << "Packing weights for K-Split mode... (Split K=" << is_k_split_ << ")" << std::endl;
        // Split tensors along K
        int64_t K = in_features_;
        int64_t K0 = (is_k_split_ > 0) ? is_k_split_ : (K / 2);
        int64_t K1 = K - K0;

        // Split qweight [K, N]
        auto q0 = qweight_final.slice(0, 0, K0);
        auto q1 = qweight_final.slice(0, K0, K);

        // Split scales/zeros
        // Assuming [Groups, N] or [N, Groups]. If per channel, share.
        torch::Tensor s0 = scale_final, s1 = scale_final;
        torch::Tensor z0 = zero_final, z1 = zero_final;

        bool is_grouped = (scale.numel() > scale.size(0) && scale.numel() > scale.size(1));
        if (scale.dim() == 1 && scale.size(0) == out_features_)
            is_grouped = false;

        if (is_grouped) {
            // We need to split groups.
            // Calculate groups in K0 partition
            // Total Groups G corresponds to Total K
            // G0 = G * (K0 / K)

            int64_t num_groups_total = 0;
            int dim_groups = -1;

            if (scale_final.dim() == 2) {
                if (scale_final.size(1) == out_features_) { // [G, N]
                    num_groups_total = scale_final.size(0);
                    dim_groups = 0;
                } else { // [N, G]
                    num_groups_total = scale_final.size(1);
                    dim_groups = 1;
                }
            }

            if (dim_groups != -1) {
                // Calculate G0 based on K0 proportion (assuming groups are evenly distributed along K)
                // group_size = K / G
                int64_t group_size = K / num_groups_total;
                // G0 = K0 / group_size
                int64_t G0 = K0 / group_size;

                if (dim_groups == 0) { // [G, N]
                    s0 = scale_final.slice(0, 0, G0);
                    s1 = scale_final.slice(0, G0, num_groups_total);
                    z0 = zero_final.slice(0, 0, G0);
                    z1 = zero_final.slice(0, G0, num_groups_total);
                } else { // [N, G]
                    s0 = scale_final.slice(1, 0, G0);
                    s1 = scale_final.slice(1, G0, num_groups_total);
                    z0 = zero_final.slice(1, 0, G0);
                    z1 = zero_final.slice(1, G0, num_groups_total);
                }
            }
        }

        // Pack Half 0
        pack_weights_lambda(packed_params_0_, q0, s0, z0, K0, out_features_);

        // Pack Half 1
        pack_weights_lambda(packed_params_1_, q1, s1, z1, K1, out_features_);

        // Use K_npu = total padded K for reference if needed, but here we split.

    } else {
        // Standard single packing
        int64_t K_npu = in_features_;

        // Ensure member variables for padded dimensions are updated for the forward pass
        padded_in_features_ = in_features_;
        padded_out_features_ = out_features_;

        if (pad_packed_weights) {
            padded_out_features_ = round_up_to_alignment(out_features_, pad_n_alignment_);
            padded_in_features_ = round_up_to_alignment(in_features_, pad_k_alignment_);
        }

        pack_weights_lambda(packed_params_, qweight_final, scale_final, zero_final, in_features_, out_features_);
    }

    // Cache CPU weights to avoid runtime D2H copy
    // packed_params_cpu_ = packed_params_.to(torch::kCPU);

    if (g_idx.defined()) {
        g_idx_ = register_buffer("g_idx", g_idx.to(device));
    }
}

void QuantizedLinearImpl::set_packed_params(torch::Tensor packed) {
    if (!use_packed_weights) {
        std::cerr << "set_packed_params called but use_packed_weights=false" << std::endl;
        return;
    }
    if (is_k_split_ > 0) {
        std::cerr << "set_packed_params called for K-split layer; use set_packed_params_split instead" << std::endl;
        return;
    }
    if (!packed_params_.defined()) {
        std::cerr << "set_packed_params: packed_params_ not defined" << std::endl;
        return;
    }

    auto device = packed_params_.device();
    auto src = packed.to(torch::kUInt8).contiguous().view({-1});

    if (src.numel() != packed_params_.numel()) {
        std::cerr << "set_packed_params: size mismatch. Expected " << packed_params_.numel() << " bytes, got " << src.numel() << std::endl;
        return;
    }

    packed_params_.resize_({src.numel()});
    packed_params_.copy_(src.to(device));
}

void QuantizedLinearImpl::set_packed_params_split(torch::Tensor packed0, torch::Tensor packed1) {
    if (!use_packed_weights) {
        std::cerr << "set_packed_params_split called but use_packed_weights=false" << std::endl;
        return;
    }
    if (is_k_split_ <= 0) {
        std::cerr << "set_packed_params_split called but layer is not K-split" << std::endl;
        return;
    }
    if (!packed_params_0_.defined() || !packed_params_1_.defined()) {
        std::cerr << "set_packed_params_split: packed_params_0/1 not defined" << std::endl;
        return;
    }

    auto device = packed_params_0_.device();
    auto src0 = packed0.to(torch::kUInt8).contiguous().view({-1});
    auto src1 = packed1.to(torch::kUInt8).contiguous().view({-1});

    if (src0.numel() != packed_params_0_.numel()) {
        std::cerr << "set_packed_params_split: size mismatch for split0. Expected " << packed_params_0_.numel() << " bytes, got "
                  << src0.numel() << std::endl;
        return;
    }
    if (src1.numel() != packed_params_1_.numel()) {
        std::cerr << "set_packed_params_split: size mismatch for split1. Expected " << packed_params_1_.numel() << " bytes, got "
                  << src1.numel() << std::endl;
        return;
    }

    packed_params_0_.resize_({src0.numel()});
    packed_params_1_.resize_({src1.numel()});
    packed_params_0_.copy_(src0.to(device));
    packed_params_1_.copy_(src1.to(device));
}

void QuantizedLinearImpl::set_unpacked_params(torch::Tensor qweight_packed, torch::Tensor scale, torch::Tensor zero_point) {
    if (use_packed_weights) {
        std::cerr << "set_unpacked_params called but use_packed_weights=true" << std::endl;
        return;
    }
    auto device = quantized_weight_.device();

    quantized_weight_ = qweight_packed.to(torch::kUInt8).contiguous().to(device);
    scale_ = scale.to(torch::kBFloat16).contiguous().to(device);
    zero_point_ = zero_point.to(torch::kInt8).contiguous().to(device);
}

void QuantizedLinearImpl::import_weights_to_xdna() {
    // Import buffers to XDNA to allow CPU access to GPU memory
    // This is crucial for the CPU path in hetero mode
    if (packed_params_.defined() && packed_params_.numel() > 0) {
        import_dma_buf_to_xdna(packed_params_.data_ptr(), packed_params_.numel(), 1); // uint8
    } else {
        if (quantized_weight_.defined()) {
            import_dma_buf_to_xdna(quantized_weight_.data_ptr(), quantized_weight_.numel(), 1); // uint8
        }
        if (scale_.defined()) {
            import_dma_buf_to_xdna(scale_.data_ptr(), scale_.numel(), 2); // bf16
        }
        if (zero_point_.defined()) {
            import_dma_buf_to_xdna(zero_point_.data_ptr(), zero_point_.numel(), 1); // int8
        }
    }

    if (g_idx_.defined()) {
        int dtype_bytes = g_idx_.element_size();
        import_dma_buf_to_xdna(g_idx_.data_ptr(), g_idx_.numel(), dtype_bytes);
    }

    if (bias_.defined()) {
        import_dma_buf_to_xdna(bias_.data_ptr(), bias_.numel(), bias_.element_size());
    }
}

// Prefetch removed
// void QuantizedLinearImpl::prefetch_cpu_weights() {}
UnifiedLLMW4A16Impl::UnifiedLLMW4A16Impl(ArchitectureType arch_type, int64_t vocab_size, int64_t hidden_size, int64_t intermediate_size,
                                         int64_t num_hidden_layers, int64_t num_attention_heads, int64_t num_key_value_heads,
                                         int64_t head_dim, float rms_norm_eps, float rope_theta, int64_t max_seq_len,
                                         int64_t max_batch_size, int64_t groupsize, torch::Device device, std::string config_path,
                                         float partial_rotary_factor, int64_t original_max_position_embeddings,
                                         std::vector<float> rope_short_factors, std::vector<float> rope_long_factors,
                                         int64_t model_max_position_embeddings)
    : arch_type_(arch_type), vocab_size_(vocab_size), hidden_size_(hidden_size), intermediate_size_(intermediate_size),
      num_hidden_layers_(num_hidden_layers), num_attention_heads_(num_attention_heads), num_key_value_heads_(num_key_value_heads),
      head_dim_(head_dim), rms_norm_eps_(rms_norm_eps), rope_theta_(rope_theta), max_seq_len_(max_seq_len), max_batch_size_(max_batch_size),
      groupsize_(groupsize), GQA_head_ratio_(num_attention_heads / num_key_value_heads), config_path_(config_path),
      phi_partial_rotary_factor_(partial_rotary_factor), phi_original_max_position_embeddings_(original_max_position_embeddings),
      phi_model_max_position_embeddings_(model_max_position_embeddings), phi_rope_short_factors_(std::move(rope_short_factors)),
      phi_rope_long_factors_(std::move(rope_long_factors)) {

    std::cout << "Initializing UnifiedLLMW4A16Impl_hetero" << std::endl;

    // Read NPU config early to set debug verbosity
    std::cout << "Using NPU config: " << config_path_ << std::endl;
    // Initialize NPU (reads config internally)
    if (this->initialize_npu() != 0) {
        throw std::runtime_error("Failed to initialize NPU context");
    }

    // Use global warmup setting from npuSetup
    warmup_ = warmup_enabled;
    std::cout << "Warmup: " << (warmup_ ? "Enabled" : "Disabled") << std::endl;

    // Gemma uses a fixed retained KV window for sliding-window attention.
    gemma_sliding_window_size_ = std::max<int64_t>(1, std::min<int64_t>(4096, max_seq_len_));
    gemma_cache_filled_ = 0;
    if (phi_original_max_position_embeddings_ <= 0) {
        phi_original_max_position_embeddings_ = max_seq_len_;
    }
    if (phi_model_max_position_embeddings_ <= 0) {
        phi_model_max_position_embeddings_ = std::max<int64_t>(max_seq_len_, phi_original_max_position_embeddings_);
    }
    phi_rotary_dim_ = static_cast<int64_t>(std::floor(static_cast<double>(head_dim_) * static_cast<double>(phi_partial_rotary_factor_)));
    phi_rotary_dim_ = std::max<int64_t>(0, std::min<int64_t>(head_dim_, phi_rotary_dim_));
    if ((phi_rotary_dim_ % 2) != 0) {
        phi_rotary_dim_ -= 1;
    }
    if (arch_type_ == ArchitectureType::PHI3 && phi_model_max_position_embeddings_ > phi_original_max_position_embeddings_ &&
        phi_original_max_position_embeddings_ > 1) {
        const double factor =
            static_cast<double>(phi_model_max_position_embeddings_) / static_cast<double>(phi_original_max_position_embeddings_);
        phi_attention_scaling_ =
            factor <= 1.0 ? 1.0f : static_cast<float>(std::sqrt(1.0 + std::log(factor) / std::log(phi_original_max_position_embeddings_)));
    }

    // Token embedding - initialize on device
    token_embedding = register_module("token_embedding", HipEmbedding(vocab_size_, hidden_size_, max_batch_size_, max_seq_len_));
    token_embedding->to(device);
    token_embedding->to(torch::kBFloat16);

    bool use_qkv_bias = (arch_type_ == ArchitectureType::QWEN15 || arch_type_ == ArchitectureType::QWEN25SMALL);
    bool gemma_style_norm = (arch_type_ == ArchitectureType::GEMMA);

    // Initialize quantized layers for each transformer block
    // Initialize quantized layers for each transformer block
    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        const int64_t q_proj_size = num_attention_heads_ * head_dim_;
        const int64_t kv_proj_size = num_key_value_heads_ * head_dim_;

        if (arch_type_ == ArchitectureType::GEMMA || arch_type_ == ArchitectureType::QWEN25SMALL) {
            qkv_layers.push_back(register_module("qkv_" + std::to_string(i), QuantizedLinear(hidden_size_, q_proj_size + (2 * kv_proj_size),
                                                                                             use_qkv_bias, max_seq_len_, "qkv")));
        }

        // Attention layers (quantized)
        q_layers.push_back(register_module(
            "q_" + std::to_string(i), QuantizedLinear(hidden_size_, num_attention_heads_ * head_dim_, use_qkv_bias, max_seq_len_, "q")));
        k_layers.push_back(register_module(
            "k_" + std::to_string(i), QuantizedLinear(hidden_size_, num_key_value_heads_ * head_dim_, use_qkv_bias, max_seq_len_, "k")));
        v_layers.push_back(register_module(
            "v_" + std::to_string(i), QuantizedLinear(hidden_size_, num_key_value_heads_ * head_dim_, use_qkv_bias, max_seq_len_, "v")));
        o_layers.push_back(register_module("o_" + std::to_string(i),
                                           QuantizedLinear(num_attention_heads_ * head_dim_, hidden_size_, false, max_seq_len_, "o")));

        // MLP layers (quantized)
        gate_layers.push_back(
            register_module("gate_" + std::to_string(i), QuantizedLinear(hidden_size_, intermediate_size_, false, max_seq_len_, "gate")));
        up_layers.push_back(
            register_module("up_" + std::to_string(i), QuantizedLinear(hidden_size_, intermediate_size_, false, max_seq_len_, "up")));
        down_layers.push_back(
            register_module("down_" + std::to_string(i), QuantizedLinear(intermediate_size_, hidden_size_, false, max_seq_len_, "down")));

        // Normalization layers (not quantized)
        input_norms.push_back(register_module("input_norm_" + std::to_string(i), RMSNorm(hidden_size_, rms_norm_eps_, gemma_style_norm)));
        post_attn_norms.push_back(
            register_module("post_attn_norm_" + std::to_string(i), RMSNorm(hidden_size_, rms_norm_eps_, gemma_style_norm)));

        // KV caches - initialize on device with bf16
        caches_k.push_back(
            register_buffer("cache_k_" + std::to_string(i), torch::zeros({max_batch_size_, num_key_value_heads_, max_seq_len_, head_dim_},
                                                                         torch::TensorOptions().device(device).dtype(torch::kBFloat16))));
        caches_v.push_back(
            register_buffer("cache_v_" + std::to_string(i), torch::zeros({max_batch_size_, num_key_value_heads_, max_seq_len_, head_dim_},
                                                                         torch::TensorOptions().device(device).dtype(torch::kBFloat16))));
    }

    // Final norm and output head
    final_norm = register_module("final_norm", RMSNorm(hidden_size_, rms_norm_eps_, gemma_style_norm));
    final_norm->to(device);
    final_norm->to(torch::kBFloat16);

    lm_head = register_module("lm_head", LmHeadLinear(hidden_size_, vocab_size_, max_batch_size_, max_seq_len_));

    // Move all layers to device
    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        if (!qkv_layers.empty()) {
            qkv_layers[i]->to(device);
        }
        q_layers[i]->to(device);
        k_layers[i]->to(device);
        v_layers[i]->to(device);
        o_layers[i]->to(device);
        gate_layers[i]->to(device);
        up_layers[i]->to(device);
        down_layers[i]->to(device);
        input_norms[i]->to(device);
        input_norms[i]->to(torch::kBFloat16);
        post_attn_norms[i]->to(device);
        post_attn_norms[i]->to(torch::kBFloat16);
    }
    lm_head->to(device);
    lm_head->to(torch::kBFloat16);

    // Register default scratch buffers.
    // These are always needed because forward_llama3() (non-chunked prefill + decode)
    // uses this base set directly.

    const int64_t q_proj_size = num_attention_heads_ * head_dim_;
    const int64_t kv_proj_size = num_key_value_heads_ * head_dim_;
    const int64_t qkv_proj_size = q_proj_size + (2 * kv_proj_size);

    const int64_t padded_hidden_norm = std::max(
        {hidden_size_,
         get_layer_padded_k_dim("qkv", hidden_size_),
         get_layer_padded_k_dim("q", hidden_size_),
         get_layer_padded_k_dim("k", hidden_size_),
         get_layer_padded_k_dim("v", hidden_size_),
         get_layer_padded_k_dim("gate", hidden_size_),
         get_layer_padded_k_dim("up", hidden_size_)});
    const int64_t padded_hidden_out =
        std::max({hidden_size_, get_layer_padded_n_dim("o", hidden_size_), get_layer_padded_n_dim("down", hidden_size_)});
    const int64_t padded_attn_input = std::max(hidden_size_, get_layer_padded_k_dim("o", hidden_size_));
    const int64_t padded_intermediate = std::max(
        {intermediate_size_,
         get_layer_padded_n_dim("gate", intermediate_size_),
         get_layer_padded_n_dim("up", intermediate_size_),
         get_layer_padded_k_dim("down", intermediate_size_)});
    const int64_t padded_q_heads = std::max<int64_t>(q_proj_size, get_layer_padded_n_dim("q", q_proj_size));
    const int64_t padded_k_heads = std::max<int64_t>(kv_proj_size, get_layer_padded_n_dim("k", kv_proj_size));
    const int64_t padded_v_heads = std::max<int64_t>(kv_proj_size, get_layer_padded_n_dim("v", kv_proj_size));
    const int64_t padded_kv_heads = std::max(padded_k_heads, padded_v_heads);
    const int64_t padded_qkv_heads = std::max<int64_t>(qkv_proj_size, get_layer_padded_n_dim("qkv", qkv_proj_size));

    x_buffer = register_buffer("x_buffer", torch::zeros({max_batch_size_, max_seq_len_, hidden_size_}, torch::kBFloat16));
    gate_buffer = register_buffer("gate_buffer", torch::zeros({max_batch_size_, max_seq_len_, padded_intermediate}, torch::kBFloat16));
    up_buffer = register_buffer("up_buffer", torch::zeros({max_batch_size_, max_seq_len_, padded_intermediate}, torch::kBFloat16));
    output_buffer = register_buffer("output_buffer", torch::zeros({max_batch_size_, max_seq_len_, padded_hidden_out}, torch::kBFloat16));

    hidden_states_buffer =
        register_buffer("hidden_states_buffer", torch::zeros({max_batch_size_, max_seq_len_, hidden_size_}, torch::kBFloat16));
    qkv_buffer = register_buffer("qkv_buffer", torch::zeros({max_batch_size_, max_seq_len_, padded_qkv_heads}, torch::kBFloat16));
    queries_buffer = register_buffer("queries_buffer", torch::zeros({max_batch_size_, max_seq_len_, padded_q_heads}, torch::kBFloat16));
    keys_buffer = register_buffer("keys_buffer", torch::zeros({max_batch_size_, max_seq_len_, padded_kv_heads}, torch::kBFloat16));
    values_buffer = register_buffer("values_buffer", torch::zeros({max_batch_size_, max_seq_len_, padded_kv_heads}, torch::kBFloat16));
    // Decode output buffer uses q_len=1 to keep contiguous [B, H, 1, D] layout.
    attn_output_heads_buffer =
        register_buffer("attn_output_heads_buffer", torch::zeros({max_batch_size_, num_attention_heads_, 1, head_dim_}, torch::kBFloat16));
    attn_output_buffer =
        register_buffer("attn_output_buffer", torch::zeros({max_batch_size_, max_seq_len_, padded_attn_input}, torch::kBFloat16));
    attn_output_proj_buffer =
        register_buffer("attn_output_proj_buffer", torch::zeros({max_batch_size_, max_seq_len_, padded_hidden_out}, torch::kBFloat16));

    norm_buffer = register_buffer("norm_buffer", torch::zeros({max_batch_size_, max_seq_len_, padded_hidden_norm}, torch::kBFloat16));

    // Chunked async prefill allocates additional per-slot scratch sets so inflight
    // chunk workers never contend on temporary Q/K/V/MLP buffers.
    // Note: this is additive; base buffers above remain for non-chunked/decode paths.
    if (async_chunking) {
        llama_pipeline_scratch_slots.reserve(static_cast<size_t>(chunking_inflight));
        for (int64_t slot = 0; slot < chunking_inflight; ++slot) {
            const std::string slot_prefix = "pipeline_slot_" + std::to_string(slot) + "_";
            // One full scratch set per inflight pipeline slot.
            LlamaPipelineScratchSpace slot_buffers;
            slot_buffers.gate_buffer = register_buffer(
                slot_prefix + "gate_buffer", torch::zeros({max_batch_size_, max_seq_len_, padded_intermediate}, torch::kBFloat16));
            slot_buffers.up_buffer = register_buffer(slot_prefix + "up_buffer",
                                                     torch::zeros({max_batch_size_, max_seq_len_, padded_intermediate}, torch::kBFloat16));
            slot_buffers.output_buffer = register_buffer(slot_prefix + "output_buffer",
                                                         torch::zeros({max_batch_size_, max_seq_len_, padded_hidden_out}, torch::kBFloat16));
            slot_buffers.qkv_buffer = register_buffer(slot_prefix + "qkv_buffer",
                                                      torch::zeros({max_batch_size_, max_seq_len_, padded_qkv_heads}, torch::kBFloat16));
            slot_buffers.queries_buffer = register_buffer(slot_prefix + "queries_buffer",
                                                          torch::zeros({max_batch_size_, max_seq_len_, padded_q_heads}, torch::kBFloat16));
            slot_buffers.keys_buffer = register_buffer(slot_prefix + "keys_buffer",
                                                       torch::zeros({max_batch_size_, max_seq_len_, padded_kv_heads}, torch::kBFloat16));
            slot_buffers.values_buffer = register_buffer(slot_prefix + "values_buffer",
                                                         torch::zeros({max_batch_size_, max_seq_len_, padded_kv_heads}, torch::kBFloat16));
            slot_buffers.attn_output_buffer = register_buffer(
                slot_prefix + "attn_output_buffer", torch::zeros({max_batch_size_, max_seq_len_, padded_attn_input}, torch::kBFloat16));
            slot_buffers.attn_output_proj_buffer = register_buffer(
                slot_prefix + "attn_output_proj_buffer", torch::zeros({max_batch_size_, max_seq_len_, padded_hidden_out}, torch::kBFloat16));
            slot_buffers.norm_buffer = register_buffer(slot_prefix + "norm_buffer",
                                                       torch::zeros({max_batch_size_, max_seq_len_, padded_hidden_norm}, torch::kBFloat16));
            llama_pipeline_scratch_slots.push_back(std::move(slot_buffers));
        }
        if (debug_verbosity >= 1) {
            std::cout << "Allocated chunk pipeline scratch slots: " << llama_pipeline_scratch_slots.size() << std::endl;
        }
    }

    // Move entire module (including all buffers and parameters) to device
    this->to(device);

    if (debug_verbosity >= 1) {
        std::cout << "UnifiedLLMW4A16 initialized on device: " << device << " with w4a16 quantization" << std::endl;
    }

    // Initialize NPU call removed from constructor, to be called from Python
    // this->initialize_npu();
}

UnifiedLLMW4A16Impl::~UnifiedLLMW4A16Impl() { release_llama_chunk_async_runtime(); }

void UnifiedLLMW4A16Impl::release_llama_chunk_async_runtime() {
    for (auto &layer_events : llama_chunk_kv_ready_events_) {
        for (auto &evt : layer_events) {
            if (evt != nullptr) {
                hipError_t destroy_err = hipEventDestroy(evt);
                (void)destroy_err;
                evt = nullptr;
            }
        }
    }
    llama_chunk_kv_ready_events_.clear();
    llama_chunk_slot_streams_.clear();
    llama_chunk_async_runtime_ready_ = false;
    llama_chunk_async_device_index_ = static_cast<c10::DeviceIndex>(-1);
}

void UnifiedLLMW4A16Impl::init_llama_chunk_async_runtime(bool force_rebuild) {
    const int64_t scratch_slots = static_cast<int64_t>(llama_pipeline_scratch_slots.size());
    const bool can_use_async_runtime = (async_chunking && scratch_slots > 1);
    if (!can_use_async_runtime) {
        release_llama_chunk_async_runtime();
        return;
    }

    const int64_t max_pipeline_slots = std::min<int64_t>(chunking_inflight, scratch_slots);
    const int64_t max_chunks = max_llama_chunk_count_for_seq_len(max_seq_len_);
    if (max_pipeline_slots <= 1 || max_chunks <= 1) {
        release_llama_chunk_async_runtime();
        return;
    }

    c10::DeviceIndex device_index = static_cast<c10::DeviceIndex>(-1);
    if (!caches_k.empty()) {
        device_index = static_cast<c10::DeviceIndex>(caches_k[0].device().index());
    }

    const bool shape_match = llama_chunk_async_runtime_ready_ && llama_chunk_async_device_index_ == device_index &&
                             static_cast<int64_t>(llama_chunk_slot_streams_.size()) == max_pipeline_slots &&
                             static_cast<int64_t>(llama_chunk_kv_ready_events_.size()) == max_chunks;
    if (!force_rebuild && shape_match) {
        return;
    }

    release_llama_chunk_async_runtime();

    llama_chunk_slot_streams_.reserve(static_cast<size_t>(max_pipeline_slots));
    for (int64_t slot = 0; slot < max_pipeline_slots; ++slot) {
        llama_chunk_slot_streams_.push_back(c10::hip::getStreamFromPool(false, device_index));
    }

    llama_chunk_kv_ready_events_.assign(static_cast<size_t>(max_chunks),
                                        std::vector<hipEvent_t>(static_cast<size_t>(num_hidden_layers_), nullptr));
    for (int64_t chunk_id = 0; chunk_id < max_chunks; ++chunk_id) {
        for (int64_t layer = 0; layer < num_hidden_layers_; ++layer) {
            hipEvent_t evt = nullptr;
            hipError_t err = hipEventCreateWithFlags(&evt, hipEventDisableTiming);
            if (err != hipSuccess) {
                release_llama_chunk_async_runtime();
                throw std::runtime_error("hipEventCreateWithFlags failed while initializing async chunk runtime");
            }
            llama_chunk_kv_ready_events_[static_cast<size_t>(chunk_id)][static_cast<size_t>(layer)] = evt;
        }
    }

    llama_chunk_async_runtime_ready_ = true;
    llama_chunk_async_device_index_ = device_index;
    if (debug_verbosity >= 1) {
        std::cout << "Initialized async chunk runtime: slots=" << max_pipeline_slots << " max_chunks=" << max_chunks << std::endl;
    }
}

UnifiedLLMW4A16Impl &UnifiedLLMW4A16Impl::to(torch::Device device) {
    torch::nn::Module::to(device);

    for (auto &cache : caches_k) {
        cache = cache.to(device);
    }
    for (auto &cache : caches_v) {
        cache = cache.to(device);
    }

    // Explicitly move scratch buffers
    x_buffer = x_buffer.to(device);
    gate_buffer = gate_buffer.to(device);
    up_buffer = up_buffer.to(device);
    output_buffer = output_buffer.to(device);
    hidden_states_buffer = hidden_states_buffer.to(device);
    qkv_buffer = qkv_buffer.to(device);
    queries_buffer = queries_buffer.to(device);
    keys_buffer = keys_buffer.to(device);
    values_buffer = values_buffer.to(device);
    attn_output_heads_buffer = attn_output_heads_buffer.to(device);
    attn_output_buffer = attn_output_buffer.to(device);
    attn_output_proj_buffer = attn_output_proj_buffer.to(device);
    norm_buffer = norm_buffer.to(device);
    for (auto &slot : llama_pipeline_scratch_slots) {
        slot.qkv_buffer = slot.qkv_buffer.to(device);
        slot.gate_buffer = slot.gate_buffer.to(device);
        slot.up_buffer = slot.up_buffer.to(device);
        slot.output_buffer = slot.output_buffer.to(device);
        slot.queries_buffer = slot.queries_buffer.to(device);
        slot.keys_buffer = slot.keys_buffer.to(device);
        slot.values_buffer = slot.values_buffer.to(device);
        slot.attn_output_buffer = slot.attn_output_buffer.to(device);
        slot.attn_output_proj_buffer = slot.attn_output_proj_buffer.to(device);
        slot.norm_buffer = slot.norm_buffer.to(device);
    }

    // Pre-create async chunk runtime resources on the target device so forward() can reuse them.
    init_llama_chunk_async_runtime(true);

    if (debug_verbosity >= 1) {
        std::cout << "UnifiedLLMW4A16 moved to device: " << device << std::endl;
    }
    return *this;
}

torch::Tensor UnifiedLLMW4A16Impl::compute_rope_freqs(int64_t seq_len, int64_t start_pos) {
    auto arange = torch::arange(0, head_dim_, 2, torch::kFloat32).slice(0, 0, head_dim_ / 2);
    auto freqs = 1.0 / torch::pow(rope_theta_, arange / head_dim_);

    if (rope_scaling_enabled && rope_scaling_type == "llama3" && rope_scaling_factor > 0.0f &&
        rope_scaling_original_max_position_embeddings > 0.0f && rope_scaling_high_freq_factor != rope_scaling_low_freq_factor) {
        const float kPi = 3.14159265358979323846f;
        const float low_wavelen = rope_scaling_original_max_position_embeddings / rope_scaling_low_freq_factor;
        const float high_wavelen = rope_scaling_original_max_position_embeddings / rope_scaling_high_freq_factor;

        auto wavelen = (2.0f * kPi) / freqs;
        auto smooth = (rope_scaling_original_max_position_embeddings / wavelen - rope_scaling_low_freq_factor) /
                      (rope_scaling_high_freq_factor - rope_scaling_low_freq_factor);

        auto ones = torch::ones_like(freqs);
        auto factor = torch::full_like(freqs, rope_scaling_factor);
        auto mid = 1.0f / ((1.0f - smooth) / rope_scaling_factor + smooth);

        auto rope_factors = torch::where(wavelen < high_wavelen, ones, torch::where(wavelen > low_wavelen, factor, mid));
        freqs = freqs / rope_factors;
    }

    auto t = torch::arange(start_pos, start_pos + seq_len, torch::kFloat32);
    auto freqs_matrix = torch::outer(t, freqs);
    return torch::polar(torch::ones_like(freqs_matrix), freqs_matrix);
}

std::pair<torch::Tensor, torch::Tensor> UnifiedLLMW4A16Impl::apply_rotary_emb(const torch::Tensor &xq, const torch::Tensor &xk,
                                                                              const torch::Tensor &freqs_cis) {
    auto cos = torch::real(freqs_cis);
    auto sin = torch::imag(freqs_cis);

    // Concatenate to get full head_dim
    std::vector<torch::Tensor> cos_chunks = {cos, cos};
    std::vector<torch::Tensor> sin_chunks = {sin, sin};
    cos = torch::cat(cos_chunks, -1);
    sin = torch::cat(sin_chunks, -1);

    // Reshape for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(2).to(xq.device()).to(xq.dtype());
    sin = sin.unsqueeze(0).unsqueeze(2).to(xq.device()).to(xq.dtype());

    // rotate_half helper
    auto rotate_half = [](const torch::Tensor &x) -> torch::Tensor {
        int64_t head_dim = x.size(-1);
        auto x1 = x.slice(-1, 0, head_dim / 2);
        auto x2 = x.slice(-1, head_dim / 2);
        return torch::cat({-x2, x1}, -1);
    };

    auto xq_out = (xq * cos) + (rotate_half(xq) * sin);
    auto xk_out = (xk * cos) + (rotate_half(xk) * sin);

    return std::make_pair(xq_out, xk_out);
}

// Gemma-specific RoPE
std::pair<torch::Tensor, torch::Tensor> UnifiedLLMW4A16Impl::compute_gemma_rope(int64_t batch_size, int64_t seq_len, int64_t start_pos) {
    // Compute inv_freq: [head_dim/2]
    auto arange = torch::arange(0, head_dim_, 2, torch::kInt64).to(torch::kFloat32);
    auto inv_freq = 1.0 / torch::pow(rope_theta_, arange / static_cast<float>(head_dim_));

    // Create position_ids: [batch_size, seq_len]
    auto position_ids_base = torch::arange(start_pos, start_pos + seq_len, torch::kInt64);
    auto position_ids = position_ids_base.unsqueeze(0).repeat({batch_size, 1});

    // inv_freq_expanded: [batch_size, head_dim/2, 1]
    auto inv_freq_expanded = inv_freq.unsqueeze(0).unsqueeze(-1).to(torch::kFloat32).repeat({batch_size, 1, 1});

    // position_ids_expanded: [batch_size, 1, seq_len]
    auto position_ids_expanded = position_ids.unsqueeze(1).to(torch::kFloat32);

    // freqs: [batch_size, seq_len, head_dim/2] (after transpose)
    auto freqs = torch::matmul(inv_freq_expanded.to(torch::kFloat32), position_ids_expanded.to(torch::kFloat32)).transpose(1, 2);

    // emb: [batch_size, seq_len, head_dim] (concatenate freqs with itself)
    auto emb = torch::cat({freqs, freqs}, -1);

    // cos and sin: [batch_size, seq_len, head_dim]
    float attention_scaling = 1.0f; // Gemma uses 1.0 for attention_scaling
    auto cos = emb.cos() * attention_scaling;
    auto sin = emb.sin() * attention_scaling;

    if (cos.dim() == 2) {
        cos = cos.unsqueeze(0);
    }
    if (sin.dim() == 2) {
        sin = sin.unsqueeze(0);
    }

    return {cos, sin};
}

std::pair<torch::Tensor, torch::Tensor> UnifiedLLMW4A16Impl::apply_gemma_rotary_emb(const torch::Tensor &q, const torch::Tensor &k,
                                                                                    const torch::Tensor &cos, const torch::Tensor &sin) {
    auto cos_fixed = cos;
    auto sin_fixed = sin;
    if (cos.dim() == 2) {
        int64_t bsz = q.size(0);
        cos_fixed = cos.unsqueeze(0).expand({bsz, -1, -1});
    }
    if (sin.dim() == 2) {
        int64_t bsz = q.size(0);
        sin_fixed = sin.unsqueeze(0).expand({bsz, -1, -1});
    }

    auto cos_expanded = cos_fixed.unsqueeze(1); // [bsz, 1, seq_len, head_dim]
    auto sin_expanded = sin_fixed.unsqueeze(1); // [bsz, 1, seq_len, head_dim]

    auto rotate_half = [](const torch::Tensor &x) {
        int64_t head_dim = x.size(-1);
        auto x1 = x.slice(-1, 0, head_dim / 2);
        auto x2 = x.slice(-1, head_dim / 2);
        return torch::cat({-x2, x1}, -1);
    };

    auto q_embed = (q * cos_expanded) + (rotate_half(q) * sin_expanded);
    auto k_embed = (k * cos_expanded) + (rotate_half(k) * sin_expanded);

    return {q_embed, k_embed};
}

std::pair<torch::Tensor, torch::Tensor> UnifiedLLMW4A16Impl::compute_phi3_rope(int64_t batch_size, int64_t seq_len, int64_t start_pos) {
    if (phi_rotary_dim_ <= 0) {
        return {torch::Tensor(), torch::Tensor()};
    }

    const bool use_longrope = (start_pos + seq_len) > phi_original_max_position_embeddings_;
    const auto &rope_factors = use_longrope ? phi_rope_long_factors_ : phi_rope_short_factors_;
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor ext_factors;
    if (static_cast<int64_t>(rope_factors.size()) == (phi_rotary_dim_ / 2)) {
        ext_factors = torch::tensor(rope_factors, options);
    } else {
        ext_factors = torch::ones({phi_rotary_dim_ / 2}, options);
    }

    auto inv_freq_shape = torch::arange(0, phi_rotary_dim_, 2, torch::kFloat32) / static_cast<float>(phi_rotary_dim_);
    auto inv_freq = 1.0f / (ext_factors * torch::pow(rope_theta_, inv_freq_shape));

    auto position_ids_base = torch::arange(start_pos, start_pos + seq_len, torch::kInt64);
    auto position_ids = position_ids_base.unsqueeze(0).repeat({batch_size, 1});
    auto inv_freq_expanded = inv_freq.unsqueeze(0).unsqueeze(-1).repeat({batch_size, 1, 1});
    auto position_ids_expanded = position_ids.unsqueeze(1).to(torch::kFloat32);
    auto freqs = torch::matmul(inv_freq_expanded.to(torch::kFloat32), position_ids_expanded).transpose(1, 2);
    auto emb = torch::cat({freqs, freqs}, -1);
    auto cos = emb.cos() * phi_attention_scaling_;
    auto sin = emb.sin() * phi_attention_scaling_;

    return {cos, sin};
}

std::pair<torch::Tensor, torch::Tensor> UnifiedLLMW4A16Impl::apply_phi3_rotary_emb(const torch::Tensor &q, const torch::Tensor &k,
                                                                                   const torch::Tensor &cos, const torch::Tensor &sin) {
    if (phi_rotary_dim_ <= 0 || !cos.defined() || !sin.defined()) {
        return {q, k};
    }

    auto cos_fixed = cos;
    auto sin_fixed = sin;
    if (cos_fixed.dim() == 2) {
        cos_fixed = cos_fixed.unsqueeze(0).expand({q.size(0), -1, -1});
    }
    if (sin_fixed.dim() == 2) {
        sin_fixed = sin_fixed.unsqueeze(0).expand({q.size(0), -1, -1});
    }

    auto cos_expanded = cos_fixed.unsqueeze(2).to(q.device()).to(q.dtype());
    auto sin_expanded = sin_fixed.unsqueeze(2).to(q.device()).to(q.dtype());

    auto rotate_half = [](const torch::Tensor &x) {
        int64_t rotary_dim = x.size(-1);
        auto x1 = x.slice(-1, 0, rotary_dim / 2);
        auto x2 = x.slice(-1, rotary_dim / 2);
        return torch::cat({-x2, x1}, -1);
    };

    auto q_rot = q.slice(-1, 0, phi_rotary_dim_);
    auto q_pass = q.slice(-1, phi_rotary_dim_, head_dim_);
    auto k_rot = k.slice(-1, 0, phi_rotary_dim_);
    auto k_pass = k.slice(-1, phi_rotary_dim_, head_dim_);

    auto q_embed = torch::cat({(q_rot * cos_expanded) + (rotate_half(q_rot) * sin_expanded), q_pass}, -1);
    auto k_embed = torch::cat({(k_rot * cos_expanded) + (rotate_half(k_rot) * sin_expanded), k_pass}, -1);
    return {q_embed, k_embed};
}

torch::Tensor UnifiedLLMW4A16Impl::silu(const torch::Tensor &x) { return torch::silu(x); }
torch::Tensor UnifiedLLMW4A16Impl::gelu(const torch::Tensor &x) { return torch::gelu(x); }
torch::Tensor UnifiedLLMW4A16Impl::swiglu(const torch::Tensor &gate, const torch::Tensor &up) { return silu(gate) * up; }

torch::Tensor UnifiedLLMW4A16Impl::forward(torch::Tensor x, int64_t start_pos) {
    switch (arch_type_) {
    case ArchitectureType::LLAMA3: {
        int64_t seq_len = x.size(1);
        const auto chunk_plan = build_llama_chunk_plan(seq_len);
        if (start_pos == 0 && chunk_plan.size() > 1) {
            return forward_llama3_chunked(x, start_pos);
        }
        return forward_llama3(x, start_pos);
    }
    case ArchitectureType::GEMMA: {
        int64_t seq_len = x.size(1);
        const auto chunk_plan = build_llama_chunk_plan(seq_len);
        if (start_pos == 0 && chunk_plan.size() > 1) {
            return forward_gemma_chunked(x, start_pos);
        }
        return forward_gemma(x, start_pos);
    }
    case ArchitectureType::QWEN15: {
        int64_t seq_len = x.size(1);
        const auto chunk_plan = build_llama_chunk_plan(seq_len);
        if (start_pos == 0 && chunk_plan.size() > 1) {
            return forward_qwen25_chunked(x, start_pos);
        }
        return forward_qwen25(x, start_pos);
    }
    case ArchitectureType::QWEN25SMALL: {
        int64_t seq_len = x.size(1);
        const auto chunk_plan = build_llama_chunk_plan(seq_len);
        if (start_pos == 0 && chunk_plan.size() > 1) {
            return forward_qwen25small_chunked(x, start_pos);
        }
        return forward_qwen25small(x, start_pos);
    }
    case ArchitectureType::PHI3: {
        int64_t seq_len = x.size(1);
        const auto chunk_plan = build_llama_chunk_plan(seq_len);
        if (start_pos == 0 && chunk_plan.size() > 1) {
            return forward_phi3_chunked(x, start_pos);
        }
        return forward_phi3(x, start_pos);
    }
    default:
        throw std::runtime_error("Unknown architecture type");
    }
}

torch::Tensor UnifiedLLMW4A16Impl::forward_llama3_chunked(torch::Tensor x, int64_t start_pos) {
    int64_t seq_len = x.size(1);
    const auto chunk_plan = build_llama_chunk_plan(seq_len);
    if (debug_verbosity >= 1) {
        std::cout << "Chunking prefill (cache-only intermediate chunks): ";
        if (chunking_token_schedule.size() > 1) {
            std::cout << "[";
            for (size_t i = 0; i < chunking_token_schedule.size(); ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << chunking_token_schedule[i];
            }
            std::cout << "]";
        } else {
            std::cout << chunking_tokens;
        }
        std::cout << std::endl;
    }
    // Chunking is prefill-only; decode and disabled modes use the normal path.
    if (start_pos != 0 || chunk_plan.size() <= 1) {
        return forward_llama3(x, start_pos);
    }

    // Compute chunk count and effective number of reusable scratch slots.
    const int64_t num_chunks = static_cast<int64_t>(chunk_plan.size());
    const int64_t scratch_slots = static_cast<int64_t>(llama_pipeline_scratch_slots.size());
    const int64_t max_pipeline_slots = std::min<int64_t>(std::min<int64_t>(chunking_inflight, scratch_slots), num_chunks);
    const bool use_pipeline = (max_pipeline_slots > 1);
    const bool run_async_pipeline = (async_chunking && use_pipeline);

    // Serial chunk execution path.
    if (!run_async_pipeline) {
        if (debug_verbosity >= 1) {
            std::cout << "Chunking pipeline disabled, serial chunk execution (async_chunking=false or slots unavailable)" << std::endl;
        }
        torch::Tensor output;
        for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
            // Process chunk [s, s+len) with absolute start position s.
            const auto &span = chunk_plan[static_cast<size_t>(chunk_id)];
            const int64_t s = span.start;
            const int64_t len = span.len;
            const bool is_last_chunk = (chunk_id + 1) == num_chunks;
            auto chunk = x.narrow(1, s, len);
            if (use_pipeline) {
                // Reuse one scratch slot per chunk modulo slot count.
                auto &slot_buffers = llama_pipeline_scratch_slots[static_cast<size_t>(chunk_id % max_pipeline_slots)];
                output = forward_llama3_with_scratch(chunk, s, is_last_chunk, is_last_chunk, slot_buffers.queries_buffer,
                                                     slot_buffers.keys_buffer, slot_buffers.values_buffer, slot_buffers.attn_output_buffer,
                                                     slot_buffers.attn_output_proj_buffer, slot_buffers.gate_buffer, slot_buffers.up_buffer,
                                                     slot_buffers.output_buffer, slot_buffers.norm_buffer, nullptr, nullptr, chunk_id,
                                                     chunk_id % max_pipeline_slots);
            } else {
                output = forward_llama3(chunk, s, is_last_chunk, is_last_chunk);
            }
        }
        return output;
    }

    if (debug_verbosity >= 1) {
        std::cout << "Chunking pipeline enabled with " << max_pipeline_slots << " inflight workers" << std::endl;
    }

    // Reuse async resources pre-allocated at model init / to(device).
    if (!llama_chunk_async_runtime_ready_) {
        init_llama_chunk_async_runtime(false);
    }
    if (!llama_chunk_async_runtime_ready_) {
        throw std::runtime_error("Async chunk runtime is unavailable despite async path selection");
    }
    if (num_chunks > static_cast<int64_t>(llama_chunk_kv_ready_events_.size())) {
        throw std::runtime_error("Chunk count exceeds preallocated async KV event capacity");
    }
    if (max_pipeline_slots > static_cast<int64_t>(llama_chunk_slot_streams_.size())) {
        throw std::runtime_error("Required inflight slots exceed preallocated async stream capacity");
    }

    auto &slot_streams = llama_chunk_slot_streams_;
    auto &kv_ready_events = llama_chunk_kv_ready_events_;
    std::vector<std::atomic<int>> chunk_layer_progress(static_cast<size_t>(num_chunks));
    for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
        chunk_layer_progress[static_cast<size_t>(chunk_id)].store(-1, std::memory_order_relaxed);
    }

    std::vector<std::future<torch::Tensor>> slot_futures(static_cast<size_t>(max_pipeline_slots));
    std::vector<int64_t> slot_chunk_ids(static_cast<size_t>(max_pipeline_slots), -1);
    std::vector<torch::Tensor> chunk_outputs(static_cast<size_t>(num_chunks));

    // Schedule chunks round-robin across slots and collect completed slot work.
    for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
        const int64_t slot_id = chunk_id % max_pipeline_slots;
        if (slot_futures[static_cast<size_t>(slot_id)].valid()) {
            const int64_t done_chunk = slot_chunk_ids[static_cast<size_t>(slot_id)];
            chunk_outputs[static_cast<size_t>(done_chunk)] = slot_futures[static_cast<size_t>(slot_id)].get();
        }

        const auto &span = chunk_plan[static_cast<size_t>(chunk_id)];
        const int64_t s = span.start;
        const int64_t len = span.len;
        const bool is_last_chunk = (chunk_id + 1) == num_chunks;
        auto chunk = x.narrow(1, s, len);

        slot_chunk_ids[static_cast<size_t>(slot_id)] = chunk_id;
        slot_futures[static_cast<size_t>(slot_id)] =
            std::async(std::launch::async,
                       [this, chunk, s, is_last_chunk, slot_id, chunk_id, &slot_streams, &kv_ready_events, &chunk_layer_progress]() {
                           torch::NoGradGuard no_grad_guard;
                           c10::cuda::CUDAStreamGuard stream_guard(slot_streams[static_cast<size_t>(slot_id)]);
                           auto &slot_buffers = llama_pipeline_scratch_slots[static_cast<size_t>(slot_id)];
                           return this->forward_llama3_with_scratch(
                               chunk, s, is_last_chunk, is_last_chunk, slot_buffers.queries_buffer, slot_buffers.keys_buffer,
                               slot_buffers.values_buffer, slot_buffers.attn_output_buffer, slot_buffers.attn_output_proj_buffer,
                               slot_buffers.gate_buffer, slot_buffers.up_buffer, slot_buffers.output_buffer, slot_buffers.norm_buffer,
                               &kv_ready_events, &chunk_layer_progress, chunk_id, slot_id);
                       });
    }

    for (int64_t slot_id = 0; slot_id < max_pipeline_slots; ++slot_id) {
        if (slot_futures[static_cast<size_t>(slot_id)].valid()) {
            const int64_t done_chunk = slot_chunk_ids[static_cast<size_t>(slot_id)];
            chunk_outputs[static_cast<size_t>(done_chunk)] = slot_futures[static_cast<size_t>(slot_id)].get();
        }
    }

    // Ensure all worker streams have completed before returning to caller stream.
    // Without this, decode may read stale logits/KV state from unfinished async streams.
    for (int64_t slot_id = 0; slot_id < max_pipeline_slots; ++slot_id) {
        HIP_CHECK(hipStreamSynchronize(slot_streams[static_cast<size_t>(slot_id)].stream()));
    }

    return chunk_outputs.back();
}

torch::Tensor UnifiedLLMW4A16Impl::forward_llama3(torch::Tensor x, int64_t start_pos, bool return_logits, bool last_token_only) {
    // When chunked kernel tables are active, prefill calls that do not enter the
    // chunk-loop path (e.g., seq_len == chunk_size) still need a real chunk_id.
    // Decode (seq_len == 1) remains unchunked and uses chunk_id=-1.
    int64_t effective_chunk_id = -1;
    if (start_pos >= 0 && x.dim() >= 2 && x.size(1) > 1) {
        effective_chunk_id = resolve_llama_chunk_id_from_start(start_pos);
    }
    return forward_llama3_with_scratch(x, start_pos, return_logits, last_token_only, queries_buffer, keys_buffer, values_buffer,
                                       attn_output_buffer, attn_output_proj_buffer, gate_buffer, up_buffer, output_buffer, norm_buffer,
                                       nullptr, nullptr, effective_chunk_id, -1);
}

torch::Tensor UnifiedLLMW4A16Impl::forward_llama3_with_scratch(
    torch::Tensor x, int64_t start_pos, bool return_logits, bool last_token_only, torch::Tensor &queries_buffer_base,
    torch::Tensor &keys_buffer_base, torch::Tensor &values_buffer_base, torch::Tensor &attn_output_buffer_base,
    torch::Tensor &attn_output_proj_buffer_base, torch::Tensor &gate_buffer_base, torch::Tensor &up_buffer_base,
    torch::Tensor &output_buffer_base, torch::Tensor &norm_buffer_base, std::vector<std::vector<hipEvent_t>> *kv_ready_events,
    std::vector<std::atomic<int>> *chunk_layer_progress, int64_t chunk_id, int64_t slot_id) {
    int64_t bsz = x.size(0);
    int64_t seq_len = x.size(1);

    // Embedding
    x = token_embedding->forward(x);

    // Create causal mask
    torch::Tensor mask;
#if LLAMA_USE_SCALED_ATTENTION != 2
    if (seq_len > 1) {
        if (seq_len > 1) {
            mask = torch::full({seq_len, seq_len}, -std::numeric_limits<float>::infinity(),
                               torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
            mask = torch::triu(mask, 1);
            mask =
                torch::hstack({torch::zeros({seq_len, start_pos}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device())), mask});
            mask = mask.to(x.dtype());
        }
    }
#endif

    // Forward through layers
    // Slice buffers for current batch size and sequence length
    auto q_buf = queries_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto k_buf = keys_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto v_buf = values_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto attn_proj_buf = attn_output_proj_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);

    auto gate_buf = gate_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto up_buf = up_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto out_buf = output_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto normed = norm_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);

    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        auto g1_trace = begin_stage_trace(chunk_id, slot_id, i, "G1", start_pos, seq_len);
        mark_stage_trace_started(g1_trace);
        // Pre-attention norm (write to norm_buffer)
        // Use sliced buffer to avoid allocation
        // Pre-attention norm (write to norm_buffer)
        // Use sliced buffer to avoid allocation
        input_norms[i]->forward_out(normed, x);

        // Async calls for Q, K, V
        torch::Tensor q, k, v;
        auto f_q = q_layers[i]->forward(q_buf, normed, "q", static_cast<int>(chunk_id));
        auto f_k = k_layers[i]->forward(k_buf, normed, "k", static_cast<int>(chunk_id));
        auto f_v = v_layers[i]->forward(v_buf, normed, "v", static_cast<int>(chunk_id));
        if (f_q.valid())
            f_q.wait();
        if (f_k.valid())
            f_k.wait();
        if (f_v.valid())
            f_v.wait();

        // Slice padded buffers to valid dimensions before usage
        q = q_buf.slice(-1, 0, num_attention_heads_ * head_dim_);
        k = k_buf.slice(-1, 0, num_key_value_heads_ * head_dim_);

        // Reshape
        q = q.view({bsz, seq_len, num_attention_heads_, head_dim_});
        k = k.view({bsz, seq_len, num_key_value_heads_, head_dim_});

        // Apply RoPE (Llama3 scaling if configured)
        if (rope_scaling_enabled && rope_scaling_type == "llama3") {
            auto freqs_cis = compute_rope_freqs(seq_len, start_pos);
            auto rope_result = apply_rotary_emb(q, k, freqs_cis);
            q = rope_result.first;
            k = rope_result.second;
        } else {
            launch_rope(q, k, start_pos, rope_theta_);
        }

        v = v_buf.slice(-1, 0, num_key_value_heads_ * head_dim_);
        v = v.view({bsz, seq_len, num_key_value_heads_, head_dim_});

        // Reshape K/V to [B, H, S, D] for storage
        auto k_transposed = k.transpose(1, 2);
        auto v_transposed = v.transpose(1, 2);

        caches_k[i].narrow(0, 0, bsz).narrow(2, start_pos, seq_len).copy_(k_transposed);
        caches_v[i].narrow(0, 0, bsz).narrow(2, start_pos, seq_len).copy_(v_transposed);

        // Async chunk pipeline ordering for chunked prefill:
        // 1) Wait on previous chunk's same-layer KV-ready event (transitive prefix guarantee),
        // 2) Publish current chunk's KV-ready event immediately after cache writes.
        const bool async_chunk_kv_sync = (kv_ready_events != nullptr && chunk_layer_progress != nullptr && chunk_id >= 0);
        if (async_chunk_kv_sync) {
            if (chunk_id >= static_cast<int64_t>(kv_ready_events->size()) ||
                i >= static_cast<int64_t>((*kv_ready_events)[static_cast<size_t>(chunk_id)].size()) ||
                chunk_id >= static_cast<int64_t>(chunk_layer_progress->size())) {
                throw std::runtime_error("Chunked pipeline KV event index out of range");
            }

            auto stream = c10::hip::getCurrentHIPStream().stream();

            if (chunk_id > 0) {
                if (i >= static_cast<int64_t>((*kv_ready_events)[static_cast<size_t>(chunk_id - 1)].size()) ||
                    (chunk_id - 1) >= static_cast<int64_t>(chunk_layer_progress->size())) {
                    throw std::runtime_error("Chunked pipeline KV event index out of range");
                }
                while ((*chunk_layer_progress)[static_cast<size_t>(chunk_id - 1)].load(std::memory_order_acquire) < i) {
                    std::this_thread::yield();
                }
                hipEvent_t dependency_event = (*kv_ready_events)[static_cast<size_t>(chunk_id - 1)][static_cast<size_t>(i)];
                HIP_CHECK(hipStreamWaitEvent(stream, dependency_event, 0));
            }

            hipEvent_t ready_event = (*kv_ready_events)[static_cast<size_t>(chunk_id)][static_cast<size_t>(i)];
            HIP_CHECK(hipEventRecord(ready_event, stream));
            (*chunk_layer_progress)[static_cast<size_t>(chunk_id)].store(static_cast<int>(i), std::memory_order_release);
        }
        end_stage_trace(std::move(g1_trace));

        // Retrieve full cache in [B, H, S, D]
        auto a_trace = begin_stage_trace(chunk_id, slot_id, i, "A", start_pos, seq_len);
        mark_stage_trace_started(a_trace);
        k = caches_k[i].narrow(0, 0, bsz).narrow(2, 0, start_pos + seq_len);
        v = caches_v[i].narrow(0, 0, bsz).narrow(2, 0, start_pos + seq_len);

        // Expand KV heads for GQA (8 KV heads -> 32 Q heads)
        // For SDPA: Skip repeat during decoding (start_pos > 0) to use native GQA.
        // For Prompt (start_pos == 0) or non-SDPA: Apply repeat (workaround for potential GQA+Causal issues).
#if LLAMA_USE_SCALED_ATTENTION >= 1
        if (start_pos == 0) {
            k = repeat_kv(k, GQA_head_ratio_);
            v = repeat_kv(v, GQA_head_ratio_);
        }
#else
        k = repeat_kv(k, GQA_head_ratio_);
        v = repeat_kv(v, GQA_head_ratio_);
#endif

        // Transpose for attention
        q = q.transpose(1, 2);
        // k, v are already [B, H, S, D]

        // std::cout << "Attention Shapes:\n"
        //           << "Q: " << q.sizes() << "\n"
        //           << "K: " << k.sizes() << "\n"
        //           << "V: " << v.sizes() << "\n";
        // torch::cuda::synchronize();
        // auto start_attn = std::chrono::high_resolution_clock::now();

        torch::Tensor attn_output;
#if LLAMA_USE_SCALED_ATTENTION == 1
        // Mode 1: PyTorch SDPA
        if (start_pos == 0 && seq_len > 1) {
            // First prompt pass: use native causal optimization
            attn_output = torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true, std::nullopt, false);
        } else {
            // Decoding or subsequent chunks: use explicit mask if defined
            c10::optional<torch::Tensor> opt_mask;
            if (mask.defined() && mask.numel() > 0) {
                // Expand mask from [seq_len, kv_seq_len] to [1, 1, seq_len, kv_seq_len]
                opt_mask = mask.unsqueeze(0).unsqueeze(0).to(q.dtype());
            }
            attn_output = torch::scaled_dot_product_attention(q, k, v, opt_mask, 0.0, false, std::nullopt, true);
        }

#elif LLAMA_USE_SCALED_ATTENTION == 2
        // Mode 2: SDPA for first prompt pass, sdpf for chunked prefill and custom hip for decode.
        if (seq_len > 1) {
            if (start_pos == 0) {
                attn_output = torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true, std::nullopt, false);
            } else {
                // Extract only the forward attention output, ignoring backward pass data (like LSE)
                attn_output = std::get<0>(at::_scaled_dot_product_flash_attention(q, k, v, 0.0, true, false, c10::nullopt));
            }
        } else {
            // Decoding phase: use Custom HIP Kernel
            int batch_size = q.size(0);
            int n_heads_Q = q.size(1);
            int n_heads_KV = k.size(1);
            int head_dim = q.size(3);
            int seq_len_kv = k.size(2);
            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

            // Output tensor: slice from preallocated heads buffer when q_len == 1
            if (q.size(2) == 1) {
                attn_output = attn_output_heads_buffer.narrow(0, 0, batch_size);
            } else {
                attn_output = torch::empty_like(q);
            }

            int element_size = q.element_size(); // Should be 2 for BF16

            launch_flash_attn_decode_hip(q.data_ptr(), k.data_ptr(), v.data_ptr(),
                                         (mask.defined() && mask.numel() > 0) ? mask.data_ptr() : nullptr, // Basic mask support check
                                         attn_output.data_ptr(), batch_size, n_heads_Q, n_heads_KV, head_dim, seq_len_kv, scale,
                                         q.stride(2) * element_size, q.stride(1) * element_size, q.stride(0) * element_size,
                                         k.stride(2) * element_size, k.stride(1) * element_size, k.stride(0) * element_size,
                                         v.stride(2) * element_size, v.stride(1) * element_size, v.stride(0) * element_size,
                                         0, // stride_mask_seq (not fully supported yet in this call site, assuming basic usage)
                                         q.dtype() == torch::kBFloat16, c10::hip::getCurrentHIPStream().stream());
        }
#else
        // Mode 0: Manual Matmul
        auto att = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<float>(head_dim_));

        if (mask.defined() && mask.numel() > 0) {
            att = att + mask.to(q.dtype());
        }

        auto attn_weights = torch::softmax(att.to(torch::kFloat32), -1).to(q.dtype());
        attn_output = torch::matmul(attn_weights, v);
#endif
        end_stage_trace(std::move(a_trace));

        // torch::cuda::synchronize();
        // auto end_attn = std::chrono::high_resolution_clock::now();
        // std::cout << "Attention Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end_attn - start_attn).count() / 1000.0
        //           << " ms\n";

        // Reshape and output projection
        auto g2_trace = begin_stage_trace(chunk_id, slot_id, i, "G2", start_pos, seq_len);
        mark_stage_trace_started(g2_trace);
        attn_output = attn_output.transpose(1, 2).contiguous().view({bsz, seq_len, hidden_size_});

        const int64_t o_input_size = get_layer_padded_k_dim("o", hidden_size_);
        const int64_t gate_input_size = get_layer_padded_k_dim("gate", hidden_size_);
        const int64_t up_input_size = get_layer_padded_k_dim("up", hidden_size_);
        const int64_t norm_input_size = std::max(gate_input_size, up_input_size);
        const int64_t down_input_size = get_layer_padded_k_dim("down", intermediate_size_);

        // Copy to attn_output_buffer to ensure contiguous memory and registered handle
        auto attn_out_buf = attn_output_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
        attn_out_buf.slice(-1, 0, hidden_size_).copy_(attn_output);

        auto attn_out_buf_proj = attn_output_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);

        // Input to o_proj: Slice to Padded K
        auto f_o = o_layers[i]->forward(attn_proj_buf, attn_out_buf_proj.slice(-1, 0, o_input_size), "o", static_cast<int>(chunk_id));
        if (f_o.valid())
            f_o.wait();

        // Residual connection: x = x + attn_output
        // x points to x_buf. We add attn_proj_buf to it in-place.
        x.add_(attn_proj_buf.slice(-1, 0, hidden_size_));

        // MLP (GeGLU)
        // Post-attention
        // RMSNorm (write to norm_buffer)
        // Slice x and normed to logical size to ensure correct Mean calculation
        post_attn_norms[i]->forward_out(normed.slice(-1, 0, hidden_size_), x.slice(-1, 0, hidden_size_));

        // QKV Projections
        // Input to q/k/v_proj: Slice normed to Padded K (1536 -> 2048) and use Padded Buffer
        auto normed_sliced_gate = normed.slice(-1, 0, gate_input_size);
        auto normed_sliced_up = normed.slice(-1, 0, up_input_size);

        // MLP projections with buffers
        auto f_gate = gate_layers[i]->forward(gate_buf, normed_sliced_gate, "gate", static_cast<int>(chunk_id));
        auto f_up = up_layers[i]->forward(up_buf, normed_sliced_up, "up", static_cast<int>(chunk_id));
        if (f_gate.valid())
            f_gate.wait();
        if (f_up.valid())
            f_up.wait();

        // In-place GELU and multiplication
        torch::silu_(gate_buf); // Llama3 uses SiLU (Swish), not GELU

        gate_buf.mul_(up_buf);

        // Down projection with buffer: Slice to Padded K
        auto f_down = down_layers[i]->forward(out_buf, gate_buf.slice(-1, 0, down_input_size), "down", static_cast<int>(chunk_id));
        if (f_down.valid())
            f_down.wait();

        // Residual connection: x = x + mlp_output
        // x points to x_buf. We add out_buf to it in-place.
        x.add_(out_buf.slice(-1, 0, hidden_size_));
        end_stage_trace(std::move(g2_trace));
    }

    if (!return_logits) {
        return torch::Tensor();
    }

    x = maybe_narrow_lm_head_input(x, x.size(1), last_token_only);

    // Final norm and output
    x = final_norm->forward(x);

    // Optimized LM Head (Ported from Llama.cpp mmvf)
    x = lm_head->forward(x);

    return x;
}

torch::Tensor UnifiedLLMW4A16Impl::forward_gemma_chunked(torch::Tensor x, int64_t start_pos) {
    int64_t seq_len = x.size(1);
    const auto chunk_plan = build_llama_chunk_plan(seq_len);
    if (debug_verbosity >= 1) {
        std::cout << "Gemma chunking prefill: ";
        if (chunking_token_schedule.size() > 1) {
            std::cout << "[";
            for (size_t i = 0; i < chunking_token_schedule.size(); ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << chunking_token_schedule[i];
            }
            std::cout << "]";
        } else {
            std::cout << chunking_tokens;
        }
        std::cout << std::endl;
    }
    if (start_pos != 0 || chunk_plan.size() <= 1) {
        return forward_gemma(x, start_pos);
    }

    const int64_t num_chunks = static_cast<int64_t>(chunk_plan.size());
    const int64_t scratch_slots = static_cast<int64_t>(llama_pipeline_scratch_slots.size());
    const int64_t max_pipeline_slots = std::min<int64_t>(std::min<int64_t>(chunking_inflight, scratch_slots), num_chunks);
    const bool use_pipeline = (max_pipeline_slots > 1);
    const bool run_async_pipeline = (async_chunking && use_pipeline);

    if (!run_async_pipeline) {
        if (debug_verbosity >= 1) {
            std::cout << "Gemma chunking pipeline disabled, serial chunk execution" << std::endl;
        }
        torch::Tensor output;
        for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
            const auto &span = chunk_plan[static_cast<size_t>(chunk_id)];
            const int64_t s = span.start;
            const int64_t len = span.len;
            auto chunk = x.narrow(1, s, len);
            if (use_pipeline) {
                auto &slot_buffers = llama_pipeline_scratch_slots[static_cast<size_t>(chunk_id % max_pipeline_slots)];
                output = forward_gemma_with_scratch(
                    chunk, s, slot_buffers.queries_buffer, slot_buffers.keys_buffer, slot_buffers.values_buffer, slot_buffers.qkv_buffer,
                    slot_buffers.attn_output_buffer, slot_buffers.attn_output_proj_buffer, slot_buffers.gate_buffer, slot_buffers.up_buffer,
                    slot_buffers.output_buffer, slot_buffers.norm_buffer, nullptr, nullptr, chunk_id, chunk_id % max_pipeline_slots);
            } else {
                output = forward_gemma_with_scratch(chunk, s, queries_buffer, keys_buffer, values_buffer, qkv_buffer, attn_output_buffer,
                                                    attn_output_proj_buffer, gate_buffer, up_buffer, output_buffer, norm_buffer, nullptr,
                                                    nullptr, chunk_id, -1);
            }
        }
        return output;
    }

    if (debug_verbosity >= 1) {
        std::cout << "Gemma chunking pipeline enabled with " << max_pipeline_slots << " inflight workers" << std::endl;
    }
    if (!llama_chunk_async_runtime_ready_) {
        init_llama_chunk_async_runtime(false);
    }
    if (!llama_chunk_async_runtime_ready_) {
        throw std::runtime_error("Async chunk runtime is unavailable despite async path selection");
    }
    if (num_chunks > static_cast<int64_t>(llama_chunk_kv_ready_events_.size())) {
        throw std::runtime_error("Chunk count exceeds preallocated async KV event capacity");
    }
    if (max_pipeline_slots > static_cast<int64_t>(llama_chunk_slot_streams_.size())) {
        throw std::runtime_error("Required inflight slots exceed preallocated async stream capacity");
    }

    auto &slot_streams = llama_chunk_slot_streams_;
    auto &kv_ready_events = llama_chunk_kv_ready_events_;
    std::vector<std::atomic<int>> chunk_layer_progress(static_cast<size_t>(num_chunks));
    for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
        chunk_layer_progress[static_cast<size_t>(chunk_id)].store(-1, std::memory_order_relaxed);
    }

    std::vector<std::future<torch::Tensor>> slot_futures(static_cast<size_t>(max_pipeline_slots));
    std::vector<int64_t> slot_chunk_ids(static_cast<size_t>(max_pipeline_slots), -1);
    std::vector<torch::Tensor> chunk_outputs(static_cast<size_t>(num_chunks));

    for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
        const int64_t slot_id = chunk_id % max_pipeline_slots;
        if (slot_futures[static_cast<size_t>(slot_id)].valid()) {
            const int64_t done_chunk = slot_chunk_ids[static_cast<size_t>(slot_id)];
            chunk_outputs[static_cast<size_t>(done_chunk)] = slot_futures[static_cast<size_t>(slot_id)].get();
        }

        const auto &span = chunk_plan[static_cast<size_t>(chunk_id)];
        const int64_t s = span.start;
        const int64_t len = span.len;
        auto chunk = x.narrow(1, s, len);

        slot_chunk_ids[static_cast<size_t>(slot_id)] = chunk_id;
        slot_futures[static_cast<size_t>(slot_id)] =
            std::async(std::launch::async, [this, chunk, s, slot_id, chunk_id, &slot_streams, &kv_ready_events, &chunk_layer_progress]() {
                torch::NoGradGuard no_grad_guard;
                c10::cuda::CUDAStreamGuard stream_guard(slot_streams[static_cast<size_t>(slot_id)]);
                auto &slot_buffers = llama_pipeline_scratch_slots[static_cast<size_t>(slot_id)];
                return this->forward_gemma_with_scratch(
                    chunk, s, slot_buffers.queries_buffer, slot_buffers.keys_buffer, slot_buffers.values_buffer, slot_buffers.qkv_buffer,
                    slot_buffers.attn_output_buffer, slot_buffers.attn_output_proj_buffer, slot_buffers.gate_buffer, slot_buffers.up_buffer,
                    slot_buffers.output_buffer, slot_buffers.norm_buffer, &kv_ready_events, &chunk_layer_progress, chunk_id, slot_id);
            });
    }

    for (int64_t slot_id = 0; slot_id < max_pipeline_slots; ++slot_id) {
        if (slot_futures[static_cast<size_t>(slot_id)].valid()) {
            const int64_t done_chunk = slot_chunk_ids[static_cast<size_t>(slot_id)];
            chunk_outputs[static_cast<size_t>(done_chunk)] = slot_futures[static_cast<size_t>(slot_id)].get();
        }
    }

    for (int64_t slot_id = 0; slot_id < max_pipeline_slots; ++slot_id) {
        HIP_CHECK(hipStreamSynchronize(slot_streams[static_cast<size_t>(slot_id)].stream()));
    }

    return chunk_outputs.back();
}

torch::Tensor UnifiedLLMW4A16Impl::forward_gemma(torch::Tensor x, int64_t start_pos) {
    int64_t effective_chunk_id = -1;
    if (start_pos >= 0 && x.dim() >= 2 && x.size(1) > 1) {
        effective_chunk_id = resolve_llama_chunk_id_from_start(start_pos);
    }
    return forward_gemma_with_scratch(x, start_pos, queries_buffer, keys_buffer, values_buffer, qkv_buffer, attn_output_buffer,
                                      attn_output_proj_buffer, gate_buffer, up_buffer, output_buffer, norm_buffer, nullptr, nullptr,
                                      effective_chunk_id, -1);
}

torch::Tensor UnifiedLLMW4A16Impl::forward_gemma_with_scratch(
    torch::Tensor x, int64_t start_pos, torch::Tensor &queries_buffer_base, torch::Tensor &keys_buffer_base,
    torch::Tensor &values_buffer_base, torch::Tensor &qkv_buffer_base, torch::Tensor &attn_output_buffer_base,
    torch::Tensor &attn_output_proj_buffer_base, torch::Tensor &gate_buffer_base, torch::Tensor &up_buffer_base,
    torch::Tensor &output_buffer_base, torch::Tensor &norm_buffer_base, std::vector<std::vector<hipEvent_t>> *kv_ready_events,
    std::vector<std::atomic<int>> *chunk_layer_progress, int64_t chunk_id, int64_t slot_id) {
    int64_t bsz = x.size(0);
    int64_t seq_len = x.size(1);
    int64_t kv_seq_len = start_pos + seq_len;
    if (start_pos == 0) {
        gemma_cache_filled_ = 0;
    }
    gemma_cache_filled_ = kv_seq_len;

    x = token_embedding(x) * std::sqrt(static_cast<float>(hidden_size_));

    torch::Tensor cos;
    torch::Tensor sin;
    auto rope_cos_sin = compute_gemma_rope(bsz, seq_len, start_pos);
    cos = rope_cos_sin.first.to(x.device()).to(x.dtype());
    sin = rope_cos_sin.second.to(x.device()).to(x.dtype());

    torch::Tensor mask;
    if (seq_len > 1 && kv_seq_len > 0) {
        auto pos_opts = torch::TensorOptions().dtype(torch::kInt64).device(x.device());
        auto q_pos = torch::arange(start_pos, start_pos + seq_len, pos_opts).unsqueeze(1);
        auto k_pos = torch::arange(0, kv_seq_len, pos_opts).unsqueeze(0);
        mask = torch::zeros({seq_len, kv_seq_len}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
        mask.masked_fill_(k_pos > q_pos, -std::numeric_limits<float>::infinity());
        mask = mask.to(x.dtype());
    }

    auto q_buf = queries_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto k_buf = keys_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto v_buf = values_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto qkv_buf = qkv_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto attn_out_buf = attn_output_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto attn_proj_buf = attn_output_proj_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);

    auto gate_buf = gate_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto up_buf = up_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto out_buf = output_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto normed = norm_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);

    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        auto g1_trace = begin_stage_trace(chunk_id, slot_id, i, "G1", start_pos, seq_len);
        mark_stage_trace_started(g1_trace);
        input_norms[i]->forward_out(normed, x);

        const int64_t q_proj_size = num_attention_heads_ * head_dim_;
        const int64_t kv_proj_size = num_key_value_heads_ * head_dim_;
        const int64_t qkv_proj_size = q_proj_size + (2 * kv_proj_size);
        const int64_t qkv_out_size = get_layer_padded_n_dim("qkv", qkv_proj_size);

        TORCH_CHECK(gemma_use_qkv_fused_ && !qkv_layers.empty(),
                    "Gemma forward requires fused qkv path. Ensure fused qkv bins are loaded.");
        TORCH_CHECK(qkv_buf.size(-1) >= qkv_out_size, "Gemma fused qkv requires dedicated qkv scratch width >= padded qkv projection size");

        auto qkv_out_buf = qkv_buf.slice(-1, 0, qkv_out_size);
        auto f_qkv = qkv_layers[i]->forward(qkv_out_buf, normed, "qkv", static_cast<int>(chunk_id));
        if (f_qkv.valid()) {
            f_qkv.wait();
        }

        auto qkv = qkv_out_buf.slice(-1, 0, qkv_proj_size);
        torch::Tensor q = qkv.slice(-1, 0, q_proj_size);
        torch::Tensor k = qkv.slice(-1, q_proj_size, q_proj_size + kv_proj_size);
        torch::Tensor v = qkv.slice(-1, q_proj_size + kv_proj_size, qkv_proj_size);

        q = q.view({bsz, seq_len, num_attention_heads_, head_dim_});
        k = k.view({bsz, seq_len, num_key_value_heads_, head_dim_});
        v = v.view({bsz, seq_len, num_key_value_heads_, head_dim_});

        q = q.transpose(1, 2);
        k = k.transpose(1, 2);
        v = v.transpose(1, 2);

        auto rope_result = apply_gemma_rotary_emb(q, k, cos, sin);
        q = rope_result.first;
        k = rope_result.second;

        caches_k[i].narrow(0, 0, bsz).narrow(2, start_pos, seq_len).copy_(k);
        caches_v[i].narrow(0, 0, bsz).narrow(2, start_pos, seq_len).copy_(v);

        const bool async_chunk_kv_sync = (kv_ready_events != nullptr && chunk_layer_progress != nullptr && chunk_id >= 0);
        if (async_chunk_kv_sync) {
            if (chunk_id >= static_cast<int64_t>(kv_ready_events->size()) ||
                i >= static_cast<int64_t>((*kv_ready_events)[static_cast<size_t>(chunk_id)].size()) ||
                chunk_id >= static_cast<int64_t>(chunk_layer_progress->size())) {
                throw std::runtime_error("Gemma async chunk runtime metadata index out of bounds");
            }
            hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
            if (chunk_id > 0) {
                if (i >= static_cast<int64_t>((*kv_ready_events)[static_cast<size_t>(chunk_id - 1)].size()) ||
                    (chunk_id - 1) >= static_cast<int64_t>(chunk_layer_progress->size())) {
                    throw std::runtime_error("Gemma async chunk dependency index out of bounds");
                }
                while ((*chunk_layer_progress)[static_cast<size_t>(chunk_id - 1)].load(std::memory_order_acquire) < i) {
                    std::this_thread::yield();
                }
                hipEvent_t dependency_event = (*kv_ready_events)[static_cast<size_t>(chunk_id - 1)][static_cast<size_t>(i)];
                HIP_CHECK(hipStreamWaitEvent(stream, dependency_event, 0));
            }
            hipEvent_t ready_event = (*kv_ready_events)[static_cast<size_t>(chunk_id)][static_cast<size_t>(i)];
            HIP_CHECK(hipEventRecord(ready_event, stream));
            (*chunk_layer_progress)[static_cast<size_t>(chunk_id)].store(static_cast<int>(i), std::memory_order_release);
        }
        end_stage_trace(std::move(g1_trace));

        auto a_trace = begin_stage_trace(chunk_id, slot_id, i, "A", start_pos, seq_len);
        mark_stage_trace_started(a_trace);
        k = caches_k[i].narrow(0, 0, bsz).narrow(2, 0, kv_seq_len);
        v = caches_v[i].narrow(0, 0, bsz).narrow(2, 0, kv_seq_len);

        torch::Tensor attn_output;
        const bool enable_gqa = (GQA_head_ratio_ > 1);
#if GEMMA_USE_SCALED_ATTENTION == 1
        if (start_pos == 0 && seq_len > 1 && kv_seq_len == seq_len) {
            attn_output = torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true, std::nullopt, enable_gqa);
        } else {
            c10::optional<torch::Tensor> opt_mask;
            if (mask.defined() && mask.numel() > 0) {
                opt_mask = mask.unsqueeze(0).unsqueeze(0).to(q.dtype());
            }
            attn_output = torch::scaled_dot_product_attention(q, k, v, opt_mask, 0.0, false, std::nullopt, enable_gqa);
        }
#elif GEMMA_USE_SCALED_ATTENTION == 2
        if (seq_len > 1) {
            c10::optional<torch::Tensor> opt_mask;
            if (mask.defined() && mask.numel() > 0) {
                opt_mask = mask.unsqueeze(0).unsqueeze(0).to(q.dtype());
            }
            if (start_pos == 0 && kv_seq_len == seq_len) {
                attn_output = torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true, std::nullopt, enable_gqa);
            } else {
                attn_output = torch::scaled_dot_product_attention(q, k, v, opt_mask, 0.0, false, std::nullopt, enable_gqa);
            }
        } else {
            if (head_dim_ == 256) {
                int batch_size = q.size(0);
                int n_heads_Q = q.size(1);
                int n_heads_KV = k.size(1);
                int head_dim = q.size(3);
                int seq_len_kv = k.size(2);
                float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

                if (q.size(2) == 1) {
                    attn_output = attn_output_heads_buffer.narrow(0, 0, batch_size);
                } else {
                    attn_output = torch::empty_like(q);
                }

                int element_size = q.element_size();
                launch_flash_attn_decode_hip_hd256(
                    q.data_ptr(), k.data_ptr(), v.data_ptr(), (mask.defined() && mask.numel() > 0) ? mask.data_ptr() : nullptr,
                    attn_output.data_ptr(), batch_size, n_heads_Q, n_heads_KV, head_dim, seq_len_kv, scale, q.stride(2) * element_size,
                    q.stride(1) * element_size, q.stride(0) * element_size, k.stride(2) * element_size, k.stride(1) * element_size,
                    k.stride(0) * element_size, v.stride(2) * element_size, v.stride(1) * element_size, v.stride(0) * element_size, 0,
                    q.dtype() == torch::kBFloat16, c10::hip::getCurrentHIPStream().stream());
            } else {
                attn_output = torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, false, std::nullopt, enable_gqa);
            }
        }
#else
        if (enable_gqa) {
            k = repeat_kv(k, GQA_head_ratio_);
            v = repeat_kv(v, GQA_head_ratio_);
        }
        auto att = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<float>(head_dim_));
        if (mask.defined() && mask.numel() > 0) {
            att = att + mask.to(q.dtype());
        }
        auto attn_weights = torch::softmax(att.to(torch::kFloat32), -1).to(q.dtype());
        attn_output = torch::matmul(attn_weights, v);
#endif
        end_stage_trace(std::move(a_trace));

        auto g2_trace = begin_stage_trace(chunk_id, slot_id, i, "G2", start_pos, seq_len);
        mark_stage_trace_started(g2_trace);
        attn_output = attn_output.transpose(1, 2).contiguous().view({bsz, seq_len, hidden_size_});
        const int64_t o_input_size = get_layer_padded_k_dim("o", hidden_size_);
        const int64_t gate_input_size = get_layer_padded_k_dim("gate", hidden_size_);
        const int64_t up_input_size = get_layer_padded_k_dim("up", hidden_size_);
        const int64_t down_input_size = get_layer_padded_k_dim("down", intermediate_size_);

        attn_out_buf.slice(-1, 0, hidden_size_).copy_(attn_output);
        auto f_o = o_layers[i]->forward(attn_proj_buf, attn_out_buf.slice(-1, 0, o_input_size), "o", static_cast<int>(chunk_id));
        if (f_o.valid())
            f_o.wait();

        x.add_(attn_proj_buf.slice(-1, 0, hidden_size_));

        post_attn_norms[i]->forward_out(normed.slice(-1, 0, hidden_size_), x.slice(-1, 0, hidden_size_));
        auto normed_sliced_gate = normed.slice(-1, 0, gate_input_size);
        auto normed_sliced_up = normed.slice(-1, 0, up_input_size);

        auto f_gate = gate_layers[i]->forward(gate_buf, normed_sliced_gate, "gate", static_cast<int>(chunk_id));
        if (f_gate.valid())
            f_gate.wait();
        auto f_up = up_layers[i]->forward(up_buf, normed_sliced_up, "up", static_cast<int>(chunk_id));
        if (f_up.valid())
            f_up.wait();

        torch::gelu_(gate_buf);
        gate_buf.mul_(up_buf);

        auto f_down = down_layers[i]->forward(out_buf, gate_buf.slice(-1, 0, down_input_size), "down", static_cast<int>(chunk_id));
        if (f_down.valid())
            f_down.wait();

        x.add_(out_buf.slice(-1, 0, hidden_size_));
        end_stage_trace(std::move(g2_trace));
    }

    x = maybe_narrow_lm_head_input(x, seq_len);

    x = final_norm->forward(x);
    x = lm_head->forward(x);
    return x;
}

torch::Tensor UnifiedLLMW4A16Impl::forward_qwen25_chunked_impl(torch::Tensor x, int64_t start_pos, bool use_fused_qkv) {
    const char *qwen_label = use_fused_qkv ? "Qwen25small" : "Qwen";
    int64_t seq_len = x.size(1);
    const auto chunk_plan = build_llama_chunk_plan(seq_len);
    if (debug_verbosity >= 1) {
        std::cout << qwen_label << " chunking prefill: ";
        if (chunking_token_schedule.size() > 1) {
            std::cout << "[";
            for (size_t i = 0; i < chunking_token_schedule.size(); ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << chunking_token_schedule[i];
            }
            std::cout << "]";
        } else {
            std::cout << chunking_tokens;
        }
        std::cout << std::endl;
    }
    if (start_pos != 0 || chunk_plan.size() <= 1) {
        return use_fused_qkv ? forward_qwen25small(x, start_pos) : forward_qwen25(x, start_pos);
    }

    const int64_t num_chunks = static_cast<int64_t>(chunk_plan.size());
    const int64_t scratch_slots = static_cast<int64_t>(llama_pipeline_scratch_slots.size());
    const int64_t max_pipeline_slots = std::min<int64_t>(std::min<int64_t>(chunking_inflight, scratch_slots), num_chunks);
    const bool use_pipeline = (max_pipeline_slots > 1);
    const bool run_async_pipeline = (async_chunking && use_pipeline);

    if (!run_async_pipeline) {
        if (debug_verbosity >= 1) {
            std::cout << qwen_label << " chunking pipeline disabled, serial chunk execution" << std::endl;
        }
        torch::Tensor output;
        for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
            const auto &span = chunk_plan[static_cast<size_t>(chunk_id)];
            const int64_t s = span.start;
            const int64_t len = span.len;
            const bool is_last_chunk = (chunk_id + 1) == num_chunks;
            auto chunk = x.narrow(1, s, len);
            if (use_pipeline) {
                auto &slot_buffers = llama_pipeline_scratch_slots[static_cast<size_t>(chunk_id % max_pipeline_slots)];
                output = use_fused_qkv
                             ? forward_qwen25small_with_scratch(
                                   chunk, s, slot_buffers.queries_buffer, slot_buffers.keys_buffer, slot_buffers.values_buffer,
                                   slot_buffers.qkv_buffer, slot_buffers.attn_output_buffer, slot_buffers.attn_output_proj_buffer,
                                   slot_buffers.gate_buffer, slot_buffers.up_buffer, slot_buffers.output_buffer, slot_buffers.norm_buffer,
                                   nullptr, nullptr, chunk_id, chunk_id % max_pipeline_slots, is_last_chunk, !is_last_chunk)
                             : forward_qwen25_with_scratch(
                                   chunk, s, slot_buffers.queries_buffer, slot_buffers.keys_buffer, slot_buffers.values_buffer,
                                   slot_buffers.attn_output_buffer, slot_buffers.attn_output_proj_buffer, slot_buffers.gate_buffer,
                                   slot_buffers.up_buffer, slot_buffers.output_buffer, slot_buffers.norm_buffer, nullptr, nullptr, chunk_id,
                                   chunk_id % max_pipeline_slots, is_last_chunk, !is_last_chunk);
            } else {
                output = use_fused_qkv
                             ? forward_qwen25small_with_scratch(chunk, s, queries_buffer, keys_buffer, values_buffer, qkv_buffer,
                                                                attn_output_buffer, attn_output_proj_buffer, gate_buffer, up_buffer,
                                                                output_buffer, norm_buffer, nullptr, nullptr, chunk_id, -1, is_last_chunk,
                                                                !is_last_chunk)
                             : forward_qwen25_with_scratch(chunk, s, queries_buffer, keys_buffer, values_buffer, attn_output_buffer,
                                                           attn_output_proj_buffer, gate_buffer, up_buffer, output_buffer, norm_buffer, nullptr,
                                                           nullptr, chunk_id, -1, is_last_chunk, !is_last_chunk);
            }
        }
        return output;
    }

    if (debug_verbosity >= 1) {
        std::cout << qwen_label << " chunking pipeline enabled with " << max_pipeline_slots << " inflight workers" << std::endl;
    }
    if (!llama_chunk_async_runtime_ready_) {
        init_llama_chunk_async_runtime(false);
    }
    if (!llama_chunk_async_runtime_ready_) {
        throw std::runtime_error("Async chunk runtime is unavailable despite async path selection");
    }
    if (num_chunks > static_cast<int64_t>(llama_chunk_kv_ready_events_.size())) {
        throw std::runtime_error("Chunk count exceeds preallocated async KV event capacity");
    }
    if (max_pipeline_slots > static_cast<int64_t>(llama_chunk_slot_streams_.size())) {
        throw std::runtime_error("Required inflight slots exceed preallocated async stream capacity");
    }

    auto &slot_streams = llama_chunk_slot_streams_;
    auto &kv_ready_events = llama_chunk_kv_ready_events_;
    std::vector<std::atomic<int>> chunk_layer_progress(static_cast<size_t>(num_chunks));
    for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
        chunk_layer_progress[static_cast<size_t>(chunk_id)].store(-1, std::memory_order_relaxed);
    }

    std::vector<std::future<torch::Tensor>> slot_futures(static_cast<size_t>(max_pipeline_slots));
    std::vector<int64_t> slot_chunk_ids(static_cast<size_t>(max_pipeline_slots), -1);
    std::vector<torch::Tensor> chunk_outputs(static_cast<size_t>(num_chunks));
    std::mutex ready_mutex;
    std::condition_variable ready_cv;
    std::deque<int64_t> ready_slots;

    auto launch_chunk_on_slot = [&](int64_t slot_id, int64_t chunk_id) {
        const auto &span = chunk_plan[static_cast<size_t>(chunk_id)];
        const int64_t s = span.start;
        const int64_t len = span.len;
        const bool is_last_chunk = (chunk_id + 1) == num_chunks;
        auto chunk = x.narrow(1, s, len);

        if (debug_verbosity >= 2) {
            std::cout << "Qwen chunk dispatch: chunk " << chunk_id << " -> slot " << slot_id << std::endl;
        }

        slot_chunk_ids[static_cast<size_t>(slot_id)] = chunk_id;
        slot_futures[static_cast<size_t>(slot_id)] =
            std::async(std::launch::async,
                       [this, chunk, s, slot_id, chunk_id, is_last_chunk, use_fused_qkv, &slot_streams, &kv_ready_events,
                        &chunk_layer_progress, &ready_mutex, &ready_cv, &ready_slots]() {
                           torch::NoGradGuard no_grad_guard;
                           c10::cuda::CUDAStreamGuard stream_guard(slot_streams[static_cast<size_t>(slot_id)]);
                           auto &slot_buffers = llama_pipeline_scratch_slots[static_cast<size_t>(slot_id)];
                           auto result = use_fused_qkv
                                             ? this->forward_qwen25small_with_scratch(
                                                   chunk, s, slot_buffers.queries_buffer, slot_buffers.keys_buffer, slot_buffers.values_buffer,
                                                   slot_buffers.qkv_buffer, slot_buffers.attn_output_buffer, slot_buffers.attn_output_proj_buffer,
                                                   slot_buffers.gate_buffer, slot_buffers.up_buffer, slot_buffers.output_buffer,
                                                   slot_buffers.norm_buffer, &kv_ready_events, &chunk_layer_progress, chunk_id, slot_id,
                                                   is_last_chunk, !is_last_chunk)
                                             : this->forward_qwen25_with_scratch(
                                                   chunk, s, slot_buffers.queries_buffer, slot_buffers.keys_buffer, slot_buffers.values_buffer,
                                                   slot_buffers.attn_output_buffer, slot_buffers.attn_output_proj_buffer,
                                                   slot_buffers.gate_buffer, slot_buffers.up_buffer, slot_buffers.output_buffer,
                                                   slot_buffers.norm_buffer, &kv_ready_events, &chunk_layer_progress, chunk_id, slot_id,
                                                   is_last_chunk, !is_last_chunk);
                           {
                               std::lock_guard<std::mutex> lock(ready_mutex);
                               ready_slots.push_back(slot_id);
                           }
                           ready_cv.notify_one();
                           return result;
                       });
    };

    int64_t next_chunk_id = 0;
    for (; next_chunk_id < num_chunks && next_chunk_id < max_pipeline_slots; ++next_chunk_id) {
        launch_chunk_on_slot(next_chunk_id, next_chunk_id);
    }

    while (next_chunk_id < num_chunks) {
        int64_t ready_slot = -1;
        {
            std::unique_lock<std::mutex> lock(ready_mutex);
            ready_cv.wait(lock, [&ready_slots]() { return !ready_slots.empty(); });
            ready_slot = ready_slots.front();
            ready_slots.pop_front();
        }

        const int64_t done_chunk = slot_chunk_ids[static_cast<size_t>(ready_slot)];
        if (done_chunk >= 0) {
            chunk_outputs[static_cast<size_t>(done_chunk)] = slot_futures[static_cast<size_t>(ready_slot)].get();
        }
        launch_chunk_on_slot(ready_slot, next_chunk_id);
        ++next_chunk_id;
    }

    for (int64_t slot_id = 0; slot_id < max_pipeline_slots; ++slot_id) {
        if (slot_futures[static_cast<size_t>(slot_id)].valid()) {
            const int64_t done_chunk = slot_chunk_ids[static_cast<size_t>(slot_id)];
            chunk_outputs[static_cast<size_t>(done_chunk)] = slot_futures[static_cast<size_t>(slot_id)].get();
        }
    }

    for (int64_t slot_id = 0; slot_id < max_pipeline_slots; ++slot_id) {
        HIP_CHECK(hipStreamSynchronize(slot_streams[static_cast<size_t>(slot_id)].stream()));
    }

    return chunk_outputs.back();
}

torch::Tensor UnifiedLLMW4A16Impl::forward_qwen25_chunked(torch::Tensor x, int64_t start_pos) {
    return forward_qwen25_chunked_impl(std::move(x), start_pos, false);
}

torch::Tensor UnifiedLLMW4A16Impl::forward_qwen25small_chunked(torch::Tensor x, int64_t start_pos) {
    return forward_qwen25_chunked_impl(std::move(x), start_pos, true);
}

torch::Tensor UnifiedLLMW4A16Impl::forward_qwen25(torch::Tensor x, int64_t start_pos) {
    int64_t effective_chunk_id = -1;
    if (start_pos >= 0 && x.dim() >= 2 && x.size(1) > 1) {
        effective_chunk_id = resolve_llama_chunk_id_from_start(start_pos);
    }
    return forward_qwen25_with_scratch(x, start_pos, queries_buffer, keys_buffer, values_buffer, attn_output_buffer,
                                       attn_output_proj_buffer, gate_buffer, up_buffer, output_buffer, norm_buffer, nullptr, nullptr,
                                       effective_chunk_id, -1, true, false);
}

torch::Tensor UnifiedLLMW4A16Impl::forward_qwen25small(torch::Tensor x, int64_t start_pos) {
    int64_t effective_chunk_id = -1;
    if (start_pos >= 0 && x.dim() >= 2 && x.size(1) > 1) {
        effective_chunk_id = resolve_llama_chunk_id_from_start(start_pos);
    }
    return forward_qwen25small_with_scratch(x, start_pos, queries_buffer, keys_buffer, values_buffer, qkv_buffer, attn_output_buffer,
                                            attn_output_proj_buffer, gate_buffer, up_buffer, output_buffer, norm_buffer, nullptr,
                                            nullptr, effective_chunk_id, -1, true, false);
}

torch::Tensor UnifiedLLMW4A16Impl::forward_qwen25_with_scratch(
    torch::Tensor x, int64_t start_pos, torch::Tensor &queries_buffer_base, torch::Tensor &keys_buffer_base,
    torch::Tensor &values_buffer_base, torch::Tensor &attn_output_buffer_base, torch::Tensor &attn_output_proj_buffer_base,
    torch::Tensor &gate_buffer_base, torch::Tensor &up_buffer_base, torch::Tensor &output_buffer_base, torch::Tensor &norm_buffer_base,
    std::vector<std::vector<hipEvent_t>> *kv_ready_events, std::vector<std::atomic<int>> *chunk_layer_progress, int64_t chunk_id,
    int64_t slot_id, bool return_logits, bool cache_only_last_layer_kv) {
    return forward_qwen25_impl(x, start_pos, queries_buffer_base, keys_buffer_base, values_buffer_base, nullptr, attn_output_buffer_base,
                               attn_output_proj_buffer_base, gate_buffer_base, up_buffer_base, output_buffer_base, norm_buffer_base, false,
                               kv_ready_events, chunk_layer_progress, chunk_id, slot_id, return_logits, cache_only_last_layer_kv);
}

torch::Tensor UnifiedLLMW4A16Impl::forward_qwen25small_with_scratch(
    torch::Tensor x, int64_t start_pos, torch::Tensor &queries_buffer_base, torch::Tensor &keys_buffer_base,
    torch::Tensor &values_buffer_base, torch::Tensor &qkv_buffer_base, torch::Tensor &attn_output_buffer_base,
    torch::Tensor &attn_output_proj_buffer_base, torch::Tensor &gate_buffer_base, torch::Tensor &up_buffer_base,
    torch::Tensor &output_buffer_base, torch::Tensor &norm_buffer_base, std::vector<std::vector<hipEvent_t>> *kv_ready_events,
    std::vector<std::atomic<int>> *chunk_layer_progress, int64_t chunk_id, int64_t slot_id, bool return_logits,
    bool cache_only_last_layer_kv) {
    return forward_qwen25_impl(x, start_pos, queries_buffer_base, keys_buffer_base, values_buffer_base, &qkv_buffer_base,
                               attn_output_buffer_base, attn_output_proj_buffer_base, gate_buffer_base, up_buffer_base,
                               output_buffer_base, norm_buffer_base, true, kv_ready_events, chunk_layer_progress, chunk_id, slot_id,
                               return_logits, cache_only_last_layer_kv);
}

torch::Tensor UnifiedLLMW4A16Impl::forward_qwen25_impl(
    torch::Tensor x, int64_t start_pos, torch::Tensor &queries_buffer_base, torch::Tensor &keys_buffer_base,
    torch::Tensor &values_buffer_base, torch::Tensor *qkv_buffer_base, torch::Tensor &attn_output_buffer_base,
    torch::Tensor &attn_output_proj_buffer_base, torch::Tensor &gate_buffer_base, torch::Tensor &up_buffer_base,
    torch::Tensor &output_buffer_base, torch::Tensor &norm_buffer_base, bool use_fused_qkv,
    std::vector<std::vector<hipEvent_t>> *kv_ready_events, std::vector<std::atomic<int>> *chunk_layer_progress, int64_t chunk_id,
    int64_t slot_id, bool return_logits, bool cache_only_last_layer_kv) {
    int64_t bsz = x.size(0);
    int64_t seq_len = x.size(1);

    x = token_embedding(x);
    auto freqs_cis = compute_rope_freqs(seq_len, start_pos);

    torch::Tensor mask;
#if QWEN_USE_SCALED_ATTENTION != 2
    if (seq_len > 1) {
        mask = torch::full({seq_len, seq_len}, -std::numeric_limits<float>::infinity(),
                           torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
        mask = torch::triu(mask, 1);
        mask = torch::hstack({torch::zeros({seq_len, start_pos}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device())), mask});
        mask = mask.to(x.dtype());
    }
#endif

    auto q_buf = queries_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto k_buf = keys_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto v_buf = values_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    torch::Tensor qkv_buf;
    if (qkv_buffer_base != nullptr) {
        qkv_buf = qkv_buffer_base->narrow(0, 0, bsz).narrow(1, 0, seq_len);
    }
    auto attn_proj_buf = attn_output_proj_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);

    auto gate_buf = gate_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto up_buf = up_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto out_buf = output_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto normed = norm_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);

    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        const int64_t q_proj_size = num_attention_heads_ * head_dim_;
        const int64_t kv_proj_size = num_key_value_heads_ * head_dim_;
        const int64_t qkv_proj_size = q_proj_size + (2 * kv_proj_size);
        const auto pad_k_legacy = [](int64_t dim) { return (dim % 2048 == 0) ? dim : ((dim / 2048 + 1) * 2048); };

        const int64_t q_input_size = use_fused_qkv ? get_layer_padded_k_dim("q", hidden_size_) : pad_k_legacy(hidden_size_);
        const int64_t k_input_size = use_fused_qkv ? get_layer_padded_k_dim("k", hidden_size_) : pad_k_legacy(hidden_size_);
        const int64_t v_input_size = use_fused_qkv ? get_layer_padded_k_dim("v", hidden_size_) : pad_k_legacy(hidden_size_);
        const int64_t qkv_input_size = use_fused_qkv ? get_layer_padded_k_dim("qkv", hidden_size_) : pad_k_legacy(hidden_size_);
        const int64_t hidden_input_size =
            use_fused_qkv ? std::max({qkv_input_size, q_input_size, k_input_size, v_input_size}) : pad_k_legacy(hidden_size_);
        const int64_t o_input_size = use_fused_qkv ? get_layer_padded_k_dim("o", hidden_size_) : pad_k_legacy(hidden_size_);
        const int64_t gate_input_size = use_fused_qkv ? get_layer_padded_k_dim("gate", hidden_size_) : pad_k_legacy(hidden_size_);
        const int64_t up_input_size = use_fused_qkv ? get_layer_padded_k_dim("up", hidden_size_) : pad_k_legacy(hidden_size_);
        const int64_t mlp_input_size = use_fused_qkv ? std::max(gate_input_size, up_input_size) : pad_k_legacy(hidden_size_);
        const int64_t down_input_size = use_fused_qkv ? get_layer_padded_k_dim("down", intermediate_size_) : pad_k_legacy(intermediate_size_);
        const int64_t qkv_out_size = get_layer_padded_n_dim("qkv", qkv_proj_size);

        auto g1_trace = begin_stage_trace(chunk_id, slot_id, i, "G1", start_pos, seq_len);
        mark_stage_trace_started(g1_trace);
        input_norms[i]->forward_out(normed.slice(-1, 0, hidden_size_), x);
        normed.slice(-1, hidden_size_, hidden_input_size).zero_();

        torch::Tensor q;
        torch::Tensor k;
        torch::Tensor v;

        if (use_fused_qkv && qwen_use_qkv_fused_ && !qkv_layers.empty()) {
            TORCH_CHECK(qkv_buffer_base != nullptr, "Qwen25small fused qkv path requires qkv scratch buffer");
            TORCH_CHECK(qkv_buf.size(-1) >= qkv_out_size, "Qwen fused qkv requires dedicated qkv scratch width >= padded qkv projection size");

            auto normed_sliced_qkv = normed.slice(-1, 0, qkv_input_size);
            auto qkv_out_buf = qkv_buf.slice(-1, 0, qkv_out_size);
            auto f_qkv = qkv_layers[i]->forward(qkv_out_buf, normed_sliced_qkv, "qkv", static_cast<int>(chunk_id));
            if (f_qkv.valid()) {
                f_qkv.wait();
            }

            auto qkv = qkv_out_buf.slice(-1, 0, qkv_proj_size);
            q = qkv.slice(-1, 0, q_proj_size);
            k = qkv.slice(-1, q_proj_size, q_proj_size + kv_proj_size);
            v = qkv.slice(-1, q_proj_size + kv_proj_size, qkv_proj_size);
        } else {
            auto normed_sliced_q = normed.slice(-1, 0, q_input_size);
            auto normed_sliced_k = normed.slice(-1, 0, k_input_size);
            auto normed_sliced_v = normed.slice(-1, 0, v_input_size);

            auto f_q = q_layers[i]->forward(q_buf, normed_sliced_q, "q", static_cast<int>(chunk_id));
            auto f_k = k_layers[i]->forward(k_buf, normed_sliced_k, "k", static_cast<int>(chunk_id));
            auto f_v = v_layers[i]->forward(v_buf, normed_sliced_v, "v", static_cast<int>(chunk_id));

            if (f_q.valid())
                f_q.wait();
            if (f_k.valid())
                f_k.wait();
            if (f_v.valid())
                f_v.wait();

            q = q_buf.slice(-1, 0, q_proj_size);
            k = k_buf.slice(-1, 0, kv_proj_size);
            v = v_buf.slice(-1, 0, kv_proj_size);
        }

        q = q.view({bsz, seq_len, num_attention_heads_, head_dim_});
        k = k.view({bsz, seq_len, num_key_value_heads_, head_dim_});
        v = v.view({bsz, seq_len, num_key_value_heads_, head_dim_});

        auto rope_result = apply_rotary_emb(q, k, freqs_cis);
        q = rope_result.first;
        k = rope_result.second;

        auto k_transposed = k.transpose(1, 2);
        auto v_transposed = v.transpose(1, 2);

        caches_k[i].narrow(0, 0, bsz).narrow(2, start_pos, seq_len).copy_(k_transposed);
        caches_v[i].narrow(0, 0, bsz).narrow(2, start_pos, seq_len).copy_(v_transposed);

        const bool async_chunk_kv_sync = (kv_ready_events != nullptr && chunk_layer_progress != nullptr && chunk_id >= 0);
        if (async_chunk_kv_sync) {
            if (chunk_id >= static_cast<int64_t>(kv_ready_events->size()) ||
                i >= static_cast<int64_t>((*kv_ready_events)[static_cast<size_t>(chunk_id)].size()) ||
                chunk_id >= static_cast<int64_t>(chunk_layer_progress->size())) {
                throw std::runtime_error("Qwen async chunk runtime metadata index out of bounds");
            }
            hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
            if (chunk_id > 0) {
                if (i >= static_cast<int64_t>((*kv_ready_events)[static_cast<size_t>(chunk_id - 1)].size()) ||
                    (chunk_id - 1) >= static_cast<int64_t>(chunk_layer_progress->size())) {
                    throw std::runtime_error("Qwen async chunk dependency index out of bounds");
                }
                while ((*chunk_layer_progress)[static_cast<size_t>(chunk_id - 1)].load(std::memory_order_acquire) < i) {
                    std::this_thread::yield();
                }
                hipEvent_t dependency_event = (*kv_ready_events)[static_cast<size_t>(chunk_id - 1)][static_cast<size_t>(i)];
                HIP_CHECK(hipStreamWaitEvent(stream, dependency_event, 0));
            }
            hipEvent_t ready_event = (*kv_ready_events)[static_cast<size_t>(chunk_id)][static_cast<size_t>(i)];
            HIP_CHECK(hipEventRecord(ready_event, stream));
            (*chunk_layer_progress)[static_cast<size_t>(chunk_id)].store(static_cast<int>(i), std::memory_order_release);
        }
        end_stage_trace(std::move(g1_trace));

        if (cache_only_last_layer_kv && i == (num_hidden_layers_ - 1)) {
            return torch::Tensor();
        }

        auto a_trace = begin_stage_trace(chunk_id, slot_id, i, "A", start_pos, seq_len);
        mark_stage_trace_started(a_trace);
        k = caches_k[i].narrow(0, 0, bsz).narrow(2, 0, start_pos + seq_len);
        v = caches_v[i].narrow(0, 0, bsz).narrow(2, 0, start_pos + seq_len);

#if QWEN_USE_SCALED_ATTENTION >= 1
        if (start_pos == 0) {
            k = repeat_kv(k, GQA_head_ratio_);
            v = repeat_kv(v, GQA_head_ratio_);
        }
#else
        k = repeat_kv(k, GQA_head_ratio_);
        v = repeat_kv(v, GQA_head_ratio_);
#endif

        q = q.transpose(1, 2);

        torch::Tensor attn_output;
#if QWEN_USE_SCALED_ATTENTION == 1
        if (start_pos == 0 && seq_len > 1) {
            attn_output = torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true, std::nullopt, false);
        } else {
            c10::optional<torch::Tensor> opt_mask;
            if (mask.defined() && mask.numel() > 0) {
                opt_mask = mask.unsqueeze(0).unsqueeze(0).to(q.dtype());
            }
            attn_output = torch::scaled_dot_product_attention(q, k, v, opt_mask, 0.0, false, std::nullopt, true);
        }
#elif QWEN_USE_SCALED_ATTENTION == 2
        if (seq_len > 1) {
            if (start_pos == 0) {
                attn_output = torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true, std::nullopt, false);
            } else {
                attn_output = std::get<0>(at::_scaled_dot_product_flash_attention(q, k, v, 0.0, true, false, c10::nullopt));
            }
        } else {
            if (head_dim_ == 128 || head_dim_ == 256) {
                int batch_size = q.size(0);
                int n_heads_Q = q.size(1);
                int n_heads_KV = k.size(1);
                int head_dim = q.size(3);
                int seq_len_kv = k.size(2);
                float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

                if (q.size(2) == 1) {
                    attn_output = attn_output_heads_buffer.narrow(0, 0, batch_size);
                } else {
                    attn_output = torch::empty_like(q);
                }

                int element_size = q.element_size();
                if (head_dim_ == 128) {
                    launch_flash_attn_decode_hip(
                        q.data_ptr(), k.data_ptr(), v.data_ptr(), (mask.defined() && mask.numel() > 0) ? mask.data_ptr() : nullptr,
                        attn_output.data_ptr(), batch_size, n_heads_Q, n_heads_KV, head_dim, seq_len_kv, scale, q.stride(2) * element_size,
                        q.stride(1) * element_size, q.stride(0) * element_size, k.stride(2) * element_size, k.stride(1) * element_size,
                        k.stride(0) * element_size, v.stride(2) * element_size, v.stride(1) * element_size, v.stride(0) * element_size, 0,
                        q.dtype() == torch::kBFloat16, c10::hip::getCurrentHIPStream().stream());
                } else {
                    launch_flash_attn_decode_hip_hd256(
                        q.data_ptr(), k.data_ptr(), v.data_ptr(), (mask.defined() && mask.numel() > 0) ? mask.data_ptr() : nullptr,
                        attn_output.data_ptr(), batch_size, n_heads_Q, n_heads_KV, head_dim, seq_len_kv, scale, q.stride(2) * element_size,
                        q.stride(1) * element_size, q.stride(0) * element_size, k.stride(2) * element_size, k.stride(1) * element_size,
                        k.stride(0) * element_size, v.stride(2) * element_size, v.stride(1) * element_size, v.stride(0) * element_size, 0,
                        q.dtype() == torch::kBFloat16, c10::hip::getCurrentHIPStream().stream());
                }
            } else {
                attn_output = torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, false, std::nullopt, true);
            }
        }
#else
        auto att = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<float>(head_dim_));
        if (mask.defined() && mask.numel() > 0) {
            att = att + mask.to(q.dtype());
        }
        auto attn_weights = torch::softmax(att.to(torch::kFloat32), -1).to(q.dtype());
        attn_output = torch::matmul(attn_weights, v);
#endif
        end_stage_trace(std::move(a_trace));

        auto g2_trace = begin_stage_trace(chunk_id, slot_id, i, "G2", start_pos, seq_len);
        mark_stage_trace_started(g2_trace);
        attn_output = attn_output.transpose(1, 2).contiguous().view({bsz, seq_len, hidden_size_});

        auto attn_out_buf = attn_output_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
        attn_out_buf.slice(-1, 0, hidden_size_).copy_(attn_output);
        attn_out_buf.slice(-1, hidden_size_, o_input_size).zero_();

        auto f_o = o_layers[i]->forward(attn_proj_buf, attn_out_buf.slice(-1, 0, o_input_size), "o", static_cast<int>(chunk_id));
        if (f_o.valid())
            f_o.wait();

        x.add_(attn_proj_buf.slice(-1, 0, hidden_size_));

        post_attn_norms[i]->forward_out(normed.slice(-1, 0, hidden_size_), x.slice(-1, 0, hidden_size_));
        normed.slice(-1, hidden_size_, mlp_input_size).zero_();

        auto normed_sliced_gate = normed.slice(-1, 0, gate_input_size);
        auto normed_sliced_up = normed.slice(-1, 0, up_input_size);

        auto f_gate = gate_layers[i]->forward(gate_buf, normed_sliced_gate, "gate", static_cast<int>(chunk_id));
        auto f_up = up_layers[i]->forward(up_buf, normed_sliced_up, "up", static_cast<int>(chunk_id));

        if (f_gate.valid())
            f_gate.wait();
        if (f_up.valid())
            f_up.wait();

        torch::silu_(gate_buf);
        gate_buf.mul_(up_buf);

        auto f_down = down_layers[i]->forward(out_buf, gate_buf.slice(-1, 0, down_input_size), "down", static_cast<int>(chunk_id));
        if (f_down.valid())
            f_down.wait();

        x.add_(out_buf.slice(-1, 0, hidden_size_));
        end_stage_trace(std::move(g2_trace));
    }

    if (!return_logits) {
        return torch::Tensor();
    }

    x = maybe_narrow_lm_head_input(x, seq_len);
    x = final_norm->forward(x);
    x = lm_head->forward(x);
    return x;
}

torch::Tensor UnifiedLLMW4A16Impl::forward_phi3_chunked(torch::Tensor x, int64_t start_pos) {
    int64_t seq_len = x.size(1);
    const auto chunk_plan = build_llama_chunk_plan(seq_len);
    if (debug_verbosity >= 1) {
        std::cout << "Phi chunking prefill: ";
        if (chunking_token_schedule.size() > 1) {
            std::cout << "[";
            for (size_t i = 0; i < chunking_token_schedule.size(); ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << chunking_token_schedule[i];
            }
            std::cout << "]";
        } else {
            std::cout << chunking_tokens;
        }
        std::cout << std::endl;
    }
    if (start_pos != 0 || chunk_plan.size() <= 1) {
        return forward_phi3(x, start_pos);
    }

    const int64_t num_chunks = static_cast<int64_t>(chunk_plan.size());
    const int64_t scratch_slots = static_cast<int64_t>(llama_pipeline_scratch_slots.size());
    const int64_t max_pipeline_slots = std::min<int64_t>(std::min<int64_t>(chunking_inflight, scratch_slots), num_chunks);
    const bool use_pipeline = (max_pipeline_slots > 1);
    const bool run_async_pipeline = (async_chunking && use_pipeline);

    if (!run_async_pipeline) {
        if (debug_verbosity >= 1) {
            std::cout << "Phi chunking pipeline disabled, serial chunk execution" << std::endl;
        }
        torch::Tensor output;
        for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
            const auto &span = chunk_plan[static_cast<size_t>(chunk_id)];
            const int64_t s = span.start;
            const int64_t len = span.len;
            auto chunk = x.narrow(1, s, len);
            if (use_pipeline) {
                auto &slot_buffers = llama_pipeline_scratch_slots[static_cast<size_t>(chunk_id % max_pipeline_slots)];
                output = forward_phi3_with_scratch(
                    chunk, s, slot_buffers.queries_buffer, slot_buffers.keys_buffer, slot_buffers.values_buffer,
                    slot_buffers.attn_output_buffer, slot_buffers.attn_output_proj_buffer, slot_buffers.gate_buffer, slot_buffers.up_buffer,
                    slot_buffers.output_buffer, slot_buffers.norm_buffer, nullptr, nullptr, chunk_id, chunk_id % max_pipeline_slots);
            } else {
                output = forward_phi3_with_scratch(chunk, s, queries_buffer, keys_buffer, values_buffer, attn_output_buffer,
                                                   attn_output_proj_buffer, gate_buffer, up_buffer, output_buffer, norm_buffer, nullptr,
                                                   nullptr, chunk_id, -1);
            }
        }
        return output;
    }

    if (debug_verbosity >= 1) {
        std::cout << "Phi chunking pipeline enabled with " << max_pipeline_slots << " inflight workers" << std::endl;
    }
    if (!llama_chunk_async_runtime_ready_) {
        init_llama_chunk_async_runtime(false);
    }
    if (!llama_chunk_async_runtime_ready_) {
        throw std::runtime_error("Async chunk runtime is unavailable despite async path selection");
    }
    if (num_chunks > static_cast<int64_t>(llama_chunk_kv_ready_events_.size())) {
        throw std::runtime_error("Chunk count exceeds preallocated async KV event capacity");
    }
    if (max_pipeline_slots > static_cast<int64_t>(llama_chunk_slot_streams_.size())) {
        throw std::runtime_error("Required inflight slots exceed preallocated async stream capacity");
    }

    auto &slot_streams = llama_chunk_slot_streams_;
    auto &kv_ready_events = llama_chunk_kv_ready_events_;
    std::vector<std::atomic<int>> chunk_layer_progress(static_cast<size_t>(num_chunks));
    for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
        chunk_layer_progress[static_cast<size_t>(chunk_id)].store(-1, std::memory_order_relaxed);
    }

    std::vector<std::future<torch::Tensor>> slot_futures(static_cast<size_t>(max_pipeline_slots));
    std::vector<int64_t> slot_chunk_ids(static_cast<size_t>(max_pipeline_slots), -1);
    std::vector<torch::Tensor> chunk_outputs(static_cast<size_t>(num_chunks));

    for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
        const int64_t slot_id = chunk_id % max_pipeline_slots;
        if (slot_futures[static_cast<size_t>(slot_id)].valid()) {
            const int64_t done_chunk = slot_chunk_ids[static_cast<size_t>(slot_id)];
            chunk_outputs[static_cast<size_t>(done_chunk)] = slot_futures[static_cast<size_t>(slot_id)].get();
        }

        const auto &span = chunk_plan[static_cast<size_t>(chunk_id)];
        const int64_t s = span.start;
        const int64_t len = span.len;
        auto chunk = x.narrow(1, s, len);

        slot_chunk_ids[static_cast<size_t>(slot_id)] = chunk_id;
        slot_futures[static_cast<size_t>(slot_id)] =
            std::async(std::launch::async, [this, chunk, s, slot_id, chunk_id, &slot_streams, &kv_ready_events, &chunk_layer_progress]() {
                torch::NoGradGuard no_grad_guard;
                c10::cuda::CUDAStreamGuard stream_guard(slot_streams[static_cast<size_t>(slot_id)]);
                auto &slot_buffers = llama_pipeline_scratch_slots[static_cast<size_t>(slot_id)];
                return this->forward_phi3_with_scratch(
                    chunk, s, slot_buffers.queries_buffer, slot_buffers.keys_buffer, slot_buffers.values_buffer,
                    slot_buffers.attn_output_buffer, slot_buffers.attn_output_proj_buffer, slot_buffers.gate_buffer, slot_buffers.up_buffer,
                    slot_buffers.output_buffer, slot_buffers.norm_buffer, &kv_ready_events, &chunk_layer_progress, chunk_id, slot_id);
            });
    }

    for (int64_t slot_id = 0; slot_id < max_pipeline_slots; ++slot_id) {
        if (slot_futures[static_cast<size_t>(slot_id)].valid()) {
            const int64_t done_chunk = slot_chunk_ids[static_cast<size_t>(slot_id)];
            chunk_outputs[static_cast<size_t>(done_chunk)] = slot_futures[static_cast<size_t>(slot_id)].get();
        }
    }

    for (int64_t slot_id = 0; slot_id < max_pipeline_slots; ++slot_id) {
        HIP_CHECK(hipStreamSynchronize(slot_streams[static_cast<size_t>(slot_id)].stream()));
    }

    return chunk_outputs.back();
}

torch::Tensor UnifiedLLMW4A16Impl::forward_phi3(torch::Tensor x, int64_t start_pos) {
    int64_t effective_chunk_id = -1;
    if (start_pos >= 0 && x.dim() >= 2 && x.size(1) > 1) {
        effective_chunk_id = resolve_llama_chunk_id_from_start(start_pos);
    }
    return forward_phi3_with_scratch(x, start_pos, queries_buffer, keys_buffer, values_buffer, attn_output_buffer, attn_output_proj_buffer,
                                     gate_buffer, up_buffer, output_buffer, norm_buffer, nullptr, nullptr, effective_chunk_id, -1);
}

torch::Tensor UnifiedLLMW4A16Impl::forward_phi3_with_scratch(
    torch::Tensor x, int64_t start_pos, torch::Tensor &queries_buffer_base, torch::Tensor &keys_buffer_base,
    torch::Tensor &values_buffer_base, torch::Tensor &attn_output_buffer_base, torch::Tensor &attn_output_proj_buffer_base,
    torch::Tensor &gate_buffer_base, torch::Tensor &up_buffer_base, torch::Tensor &output_buffer_base, torch::Tensor &norm_buffer_base,
    std::vector<std::vector<hipEvent_t>> *kv_ready_events, std::vector<std::atomic<int>> *chunk_layer_progress, int64_t chunk_id,
    int64_t slot_id) {
    int64_t bsz = x.size(0);
    int64_t seq_len = x.size(1);

    x = token_embedding(x);
    auto phi_rope = compute_phi3_rope(bsz, seq_len, start_pos);
    auto cos = phi_rope.first;
    auto sin = phi_rope.second;

    torch::Tensor mask;
#if PHI_USE_SCALED_ATTENTION != 2
    if (seq_len > 1) {
        mask = torch::full({seq_len, seq_len}, -std::numeric_limits<float>::infinity(),
                           torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
        mask = torch::triu(mask, 1);
        mask = torch::hstack({torch::zeros({seq_len, start_pos}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device())), mask});
        mask = mask.to(x.dtype());
    }
#endif

    auto q_buf = queries_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto k_buf = keys_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto v_buf = values_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto attn_proj_buf = attn_output_proj_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);

    auto gate_buf = gate_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto up_buf = up_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto out_buf = output_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto normed = norm_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);

    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        const int64_t q_input_size = get_layer_padded_k_dim("q", hidden_size_);
        const int64_t k_input_size = get_layer_padded_k_dim("k", hidden_size_);
        const int64_t v_input_size = get_layer_padded_k_dim("v", hidden_size_);
        const int64_t hidden_input_size = std::max({q_input_size, k_input_size, v_input_size});
        const int64_t o_input_size = get_layer_padded_k_dim("o", hidden_size_);
        const int64_t gate_input_size = get_layer_padded_k_dim("gate", hidden_size_);
        const int64_t up_input_size = get_layer_padded_k_dim("up", hidden_size_);
        const int64_t mlp_input_size = std::max(gate_input_size, up_input_size);
        const int64_t down_input_size = get_layer_padded_k_dim("down", intermediate_size_);

        auto g1_trace = begin_stage_trace(chunk_id, slot_id, i, "G1", start_pos, seq_len);
        mark_stage_trace_started(g1_trace);
        input_norms[i]->forward_out(normed.slice(-1, 0, hidden_size_), x);
        normed.slice(-1, hidden_size_, hidden_input_size).zero_();

        auto normed_sliced_q = normed.slice(-1, 0, q_input_size);
        auto normed_sliced_k = normed.slice(-1, 0, k_input_size);
        auto normed_sliced_v = normed.slice(-1, 0, v_input_size);

        auto f_q = q_layers[i]->forward(q_buf, normed_sliced_q, "q", static_cast<int>(chunk_id));
        auto f_k = k_layers[i]->forward(k_buf, normed_sliced_k, "k", static_cast<int>(chunk_id));
        auto f_v = v_layers[i]->forward(v_buf, normed_sliced_v, "v", static_cast<int>(chunk_id));

        if (f_q.valid())
            f_q.wait();
        if (f_k.valid())
            f_k.wait();
        if (f_v.valid())
            f_v.wait();

        auto q = q_buf.slice(-1, 0, num_attention_heads_ * head_dim_);
        auto k = k_buf.slice(-1, 0, num_key_value_heads_ * head_dim_);
        auto v = v_buf.slice(-1, 0, num_key_value_heads_ * head_dim_);

        q = q.view({bsz, seq_len, num_attention_heads_, head_dim_});
        k = k.view({bsz, seq_len, num_key_value_heads_, head_dim_});
        v = v.view({bsz, seq_len, num_key_value_heads_, head_dim_});

        auto rope_result = apply_phi3_rotary_emb(q, k, cos, sin);
        q = rope_result.first;
        k = rope_result.second;

        auto k_transposed = k.transpose(1, 2);
        auto v_transposed = v.transpose(1, 2);

        caches_k[i].narrow(0, 0, bsz).narrow(2, start_pos, seq_len).copy_(k_transposed);
        caches_v[i].narrow(0, 0, bsz).narrow(2, start_pos, seq_len).copy_(v_transposed);

        const bool async_chunk_kv_sync = (kv_ready_events != nullptr && chunk_layer_progress != nullptr && chunk_id >= 0);
        if (async_chunk_kv_sync) {
            if (chunk_id >= static_cast<int64_t>(kv_ready_events->size()) ||
                i >= static_cast<int64_t>((*kv_ready_events)[static_cast<size_t>(chunk_id)].size()) ||
                chunk_id >= static_cast<int64_t>(chunk_layer_progress->size())) {
                throw std::runtime_error("Phi async chunk runtime metadata index out of bounds");
            }
            hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
            if (chunk_id > 0) {
                if (i >= static_cast<int64_t>((*kv_ready_events)[static_cast<size_t>(chunk_id - 1)].size()) ||
                    (chunk_id - 1) >= static_cast<int64_t>(chunk_layer_progress->size())) {
                    throw std::runtime_error("Phi async chunk dependency index out of bounds");
                }
                while ((*chunk_layer_progress)[static_cast<size_t>(chunk_id - 1)].load(std::memory_order_acquire) < i) {
                    std::this_thread::yield();
                }
                hipEvent_t dependency_event = (*kv_ready_events)[static_cast<size_t>(chunk_id - 1)][static_cast<size_t>(i)];
                HIP_CHECK(hipStreamWaitEvent(stream, dependency_event, 0));
            }
            hipEvent_t ready_event = (*kv_ready_events)[static_cast<size_t>(chunk_id)][static_cast<size_t>(i)];
            HIP_CHECK(hipEventRecord(ready_event, stream));
            (*chunk_layer_progress)[static_cast<size_t>(chunk_id)].store(static_cast<int>(i), std::memory_order_release);
        }
        end_stage_trace(std::move(g1_trace));

        auto a_trace = begin_stage_trace(chunk_id, slot_id, i, "A", start_pos, seq_len);
        mark_stage_trace_started(a_trace);
        k = caches_k[i].narrow(0, 0, bsz).narrow(2, 0, start_pos + seq_len);
        v = caches_v[i].narrow(0, 0, bsz).narrow(2, 0, start_pos + seq_len);

#if PHI_USE_SCALED_ATTENTION >= 1
        if (start_pos == 0) {
            k = repeat_kv(k, GQA_head_ratio_);
            v = repeat_kv(v, GQA_head_ratio_);
        }
#else
        k = repeat_kv(k, GQA_head_ratio_);
        v = repeat_kv(v, GQA_head_ratio_);
#endif

        q = q.transpose(1, 2);

        torch::Tensor attn_output;
#if PHI_USE_SCALED_ATTENTION == 1
        if (start_pos == 0 && seq_len > 1) {
            attn_output = torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true, std::nullopt, false);
        } else {
            c10::optional<torch::Tensor> opt_mask;
            if (mask.defined() && mask.numel() > 0) {
                opt_mask = mask.unsqueeze(0).unsqueeze(0).to(q.dtype());
            }
            attn_output = torch::scaled_dot_product_attention(q, k, v, opt_mask, 0.0, false, std::nullopt, true);
        }
#elif PHI_USE_SCALED_ATTENTION == 2
        if (seq_len > 1) {
            if (start_pos == 0) {
                attn_output = torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true, std::nullopt, false);
            } else {
                attn_output = std::get<0>(at::_scaled_dot_product_flash_attention(q, k, v, 0.0, true, false, c10::nullopt));
            }
        } else {
            // Phi decode quality is more reliable on the generic SDPA GQA path than the
            // custom single-token kernel, especially after long prefills.
            attn_output = torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, false, std::nullopt, true);
        }
#else
        auto att = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<float>(head_dim_));
        if (mask.defined() && mask.numel() > 0) {
            att = att + mask.to(q.dtype());
        }
        auto attn_weights = torch::softmax(att.to(torch::kFloat32), -1).to(q.dtype());
        attn_output = torch::matmul(attn_weights, v);
#endif
        end_stage_trace(std::move(a_trace));

        auto g2_trace = begin_stage_trace(chunk_id, slot_id, i, "G2", start_pos, seq_len);
        mark_stage_trace_started(g2_trace);
        attn_output = attn_output.transpose(1, 2).contiguous().view({bsz, seq_len, hidden_size_});

        auto attn_out_buf = attn_output_buffer_base.narrow(0, 0, bsz).narrow(1, 0, seq_len);
        attn_out_buf.slice(-1, 0, hidden_size_).copy_(attn_output);
        attn_out_buf.slice(-1, hidden_size_, o_input_size).zero_();

        auto f_o = o_layers[i]->forward(attn_proj_buf, attn_out_buf.slice(-1, 0, o_input_size), "o", static_cast<int>(chunk_id));
        if (f_o.valid())
            f_o.wait();

        x.add_(attn_proj_buf.slice(-1, 0, hidden_size_));

        post_attn_norms[i]->forward_out(normed.slice(-1, 0, hidden_size_), x.slice(-1, 0, hidden_size_));
        normed.slice(-1, hidden_size_, mlp_input_size).zero_();

        auto normed_sliced_gate = normed.slice(-1, 0, gate_input_size);
        auto normed_sliced_up = normed.slice(-1, 0, up_input_size);

        auto f_gate = gate_layers[i]->forward(gate_buf, normed_sliced_gate, "gate", static_cast<int>(chunk_id));
        auto f_up = up_layers[i]->forward(up_buf, normed_sliced_up, "up", static_cast<int>(chunk_id));

        if (f_gate.valid())
            f_gate.wait();
        if (f_up.valid())
            f_up.wait();

        torch::silu_(gate_buf);
        gate_buf.mul_(up_buf);

        auto f_down = down_layers[i]->forward(out_buf, gate_buf.slice(-1, 0, down_input_size), "down", static_cast<int>(chunk_id));
        if (f_down.valid())
            f_down.wait();

        x.add_(out_buf.slice(-1, 0, hidden_size_));
        end_stage_trace(std::move(g2_trace));
    }

    x = maybe_narrow_lm_head_input(x, seq_len);
    x = final_norm->forward(x);
    x = lm_head->forward(x);
    return x;
}

// Helper function to sample from logits

torch::Tensor UnifiedLLMW4A16Impl::generate(torch::Tensor input_ids, int64_t max_new_tokens, float temperature, float top_p, int64_t top_k,
                                            int64_t eos_token_id) {
    std::cout << "Libtorch input_ids shape: " << input_ids.sizes() << std::endl;

    int64_t batch_size = input_ids.size(0);
    int64_t prompt_len = input_ids.size(1);
    int64_t start_pos = 0;
    torch::Tensor output;
    torch::Tensor next_token;
    torch::Tensor last_token;

    // Ensure model is in eval mode (no dropout, etc.)
    this->eval();
    torch::NoGradGuard no_grad;

    // Warmup cycle
    if (warmup_) {
        std::cout << "Running warmup..." << std::endl;
        int warm_up = 1;
        // Prefill warmup
        for (int i = 0; i < warm_up; i++) {
            forward(input_ids, start_pos);
        }

        // M=1 Warmup (Single token generation after prefill)
        std::cout << "Running M=1 Warmup (Hetero Path)..." << std::endl;

        auto dummy_token = torch::zeros({1, 1}, torch::TensorOptions().dtype(torch::kInt64).device(input_ids.device()));
        int64_t warmup_start_pos = input_ids.size(1); // Position after prefill
        for (int i = 0; i < 1; i++) {
            forward(dummy_token, warmup_start_pos + i);
        }
    } else {
        if (debug_verbosity >= 1)
            std::cout << "Skipping warmup." << std::endl;
    }

    // Prefill phase: process initial prompt
    if (debug_verbosity >= 2) {
        std::cout << "Prefill phase" << std::endl;
    }
    torch::cuda::synchronize();
    PrefillTraceGuard prefill_trace_guard(arch_type_);
    auto start_prefill = std::chrono::high_resolution_clock::now();

    output = forward(input_ids, start_pos);

    torch::cuda::synchronize();
    auto end_prefill = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_prefill = end_prefill - start_prefill;
    // std::cout << "Prefill time: " << elapsed_prefill.count() << " seconds" << std::endl;

    // Get next token: implementation when temperature is 0
    last_token = output.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});
    if (temperature < 0.01f) {
        // Greedy decoding: argmax
        next_token = torch::argmax(last_token, -1, true);
    } else {
        // Sampling
        int64_t next_token_id = sample_token(last_token.squeeze(0), temperature, top_p, top_k);
        next_token = torch::tensor({{next_token_id}}, torch::TensorOptions().dtype(torch::kInt64).device(input_ids.device()));
    }

    // Check for EOS token
    int64_t next_token_id = next_token.item<int64_t>();
    if (eos_token_id >= 0 && next_token_id == eos_token_id) {
        return input_ids;
    }

    torch::Tensor input_tensor = torch::cat({input_ids, next_token}, 1);
    int64_t token_len = input_tensor.size(1);
    start_pos = token_len - 1;

    // Generate remaining tokens
    torch::cuda::synchronize();
    auto start_gen = std::chrono::high_resolution_clock::now();
    int64_t actual_generated = 0;

    // Generation loop
    std::cout << "Generation phase" << std::endl;
    while (token_len < max_new_tokens + prompt_len) {
        output = forward(next_token, start_pos);
        last_token = output.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});

        if (temperature < 0.01f) {
            next_token = torch::argmax(last_token, -1, true);
        } else {
            int64_t next_token_id = sample_token(last_token.squeeze(0), temperature, top_p, top_k);
            next_token = torch::tensor({{next_token_id}}, torch::TensorOptions().dtype(torch::kInt64).device(input_ids.device()));
        }

        next_token_id = next_token.item<int64_t>();
        if (eos_token_id >= 0 && next_token_id == eos_token_id) {
            break;
        }

        input_tensor = torch::cat({input_tensor, next_token}, 1);
        actual_generated++;
        token_len++;
        start_pos++;
    }

    torch::cuda::synchronize();
    auto end_gen = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> generation_time = end_gen - start_gen;

    // print to terminal
    std::cout << "Prefill time: " << elapsed_prefill.count() << " seconds" << std::endl;
    std::cout << "Total Generation Time: " << generation_time.count() << " seconds" << std::endl;
    if (actual_generated > 0) {
        double time_per_token = generation_time.count() / actual_generated;
        std::cout << "Average Time per Token: " << time_per_token << " seconds" << std::endl;
    }

    return input_tensor;
}

// Load a tensor from safetensors file in helper.cpp

int UnifiedLLMW4A16Impl::initialize_npu() {
    // Read config first to set debug_verbosity (already called in constructor, but safe to call again)
    read_npu_config(config_path_);

    if (debug_verbosity >= 1)
        std::cout << "Initializing NPU..." << std::endl;

    if (hw_target == "gpu") {
        if (debug_verbosity >= 1) {
            std::cout << "GPU-only mode: skipping XDNA/NPU init" << std::endl;
        }
        init_gemm_resources();
        return 0;
    }

    // Hardcoded driver path
    const char *drv_path = "/dev/accel/accel0";

    // Open XDNA driver (using global xdna_drv_fd defined in npuSetup.cpp)
    if (initialize_xdna_driver(drv_path) != 0) {
        return -1;
    }

    if (debug_verbosity >= 1)
        std::cout << "Verbosity: " << debug_verbosity << std::endl;

    // Import all weights to XDNA using this class instance
    if (debug_verbosity >= 1)
        std::cout << "Loading NPU kernels..." << std::endl;
    init_npu(); // Initialize NPU context arrays with max capacity

    if (hw_target == "npu" || hw_target == "npu-sim") {
        if (debug_verbosity >= 1) {
            std::cout << (hw_target == "npu-sim" ? "NPU Sim Kernels" : "NPU Only Kernels") << std::endl;
        }
        load_npu_only_kernels(xdna_drv_fd, config_path_);
    } else if (hw_target == "hetero") {
        if (debug_verbosity >= 1) {
            std::cout << "Hetero Kernels" << std::endl;
        }
        load_npu_kernels(xdna_drv_fd, config_path_);
    } else {
        std::cout << "Skipping loading Kernels" << std::endl;
    }

    // Initialize GEMM resources (HIP events)
    init_gemm_resources();

    return 0;
}

void UnifiedLLMW4A16Impl::import_weights() {
    if (hw_target == "gpu") {
        if (debug_verbosity >= 1) {
            std::cout << "GPU-only mode: skipping XDNA import" << std::endl;
        }
        return;
    }
    if (debug_verbosity >= 1)
        std::cout << "Importing weights to XDNA..." << std::endl;
    // Import all weights and buffers to XDNA
    import_all_weights_to_xdna(*this);
    if (debug_verbosity >= 1)
        std::cout << "Weights imported." << std::endl;
}
