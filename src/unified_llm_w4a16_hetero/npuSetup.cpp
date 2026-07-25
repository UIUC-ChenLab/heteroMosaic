#include "unified_llm_w4a16_hetero/npuSetup.hpp"
#include "unified_llm_w4a16_hetero/hetero_compute.hpp"
#include "unified_llm_w4a16_hetero/hipblasltSetup.hpp"
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <libdrm/drm.h>
#include <map>
#include <set>
#include <stdint.h>
#include <string_view>
#include <sys/ioctl.h>
#include <torch/torch.h>
#include <unistd.h>

// Include actual amdxdna_accel.h header
#ifdef __KERNEL__
#include <drm/drm.h>
#else
#include <libdrm/drm.h>
#endif
#include "amdxdna_accel.h"

// Define IOMMU_STRIDE
#define IOMMU_STRIDE 1024

// Define myBfloat type (bfloat16 as uint16_t)
typedef uint16_t myBfloat;

// XDNA driver file descriptor
// Initialize to -1 (invalid)
int xdna_drv_fd = -1;

// Global map to cache imported handles: ptr -> handle
std::map<void *, uint32_t> ptr_to_handle_map;

// Global hardware target setting from kernels.json
std::string hw_target = "gpu";

// Global debug verbosity level from kernels.json
int debug_verbosity = 1;

// Global dummy weights flag
bool dummy_weights_enabled = false;

// Global warmup flag
bool warmup_enabled = true;

// Global minimal PDI flag
bool minimal_pdi = false;
bool split_M_only = false;

// Global NPU dim
int npu_dim = -1;
int64_t chunking_tokens = 0;
std::vector<int64_t> chunking_token_schedule;
int64_t chunking_inflight = 1;
bool chunking_scheduled = true;
bool async_chunking = false;

// Global Packed Weights flags
bool use_packed_weights = true;
int64_t pad_packed_weights = 0;
bool cpu_decode = false;
bool gemv_driven_split_K = false;
std::string gemv_npu_col;
std::string trace_output_path;
std::string trace_run_tag;
bool trace_sync_stages = false;
std::vector<StageBubbleSpec> stage_bubbles;

// Global RoPE scaling settings (e.g., Llama3)
bool rope_scaling_enabled = false;
std::string rope_scaling_type;
float rope_scaling_factor = 1.0f;
float rope_scaling_low_freq_factor = 1.0f;
float rope_scaling_high_freq_factor = 1.0f;
float rope_scaling_original_max_position_embeddings = 0.0f;

// Global NPU contexts
std::vector<hwctxt> hwctxt_array;
std::vector<instctxt> instctxt_array;
std::map<NPUKey, NPUValue> config_map;
std::map<std::string, PackedWeightPadSpec> packed_weight_pad_map;

#include "third_party/nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>

namespace json = nlohmann;

namespace {
constexpr int64_t kLegacyPadPackedWeightsAlignment = 2048;

int64_t get_optional_int64(const json::json &obj, std::initializer_list<const char *> keys, int64_t default_value) {
    for (const char *key : keys) {
        if (!obj.contains(key)) {
            continue;
        }
        const auto &value = obj.at(key);
        if (value.is_number_integer()) {
            return value.get<int64_t>();
        }
        if (value.is_number_unsigned()) {
            return static_cast<int64_t>(value.get<uint64_t>());
        }
    }
    return default_value;
}

bool get_optional_bool(const json::json &obj, std::initializer_list<const char *> keys, bool default_value) {
    for (const char *key : keys) {
        if (!obj.contains(key)) {
            continue;
        }
        const auto &value = obj.at(key);
        if (value.is_boolean()) {
            return value.get<bool>();
        }
        if (value.is_number_integer()) {
            return value.get<int64_t>() != 0;
        }
        if (value.is_number_unsigned()) {
            return value.get<uint64_t>() != 0;
        }
    }
    return default_value;
}

double get_optional_double(const json::json &obj, std::initializer_list<const char *> keys, double default_value) {
    for (const char *key : keys) {
        if (!obj.contains(key)) {
            continue;
        }
        const auto &value = obj.at(key);
        if (value.is_number_float()) {
            return value.get<double>();
        }
        if (value.is_number_integer()) {
            return static_cast<double>(value.get<int64_t>());
        }
        if (value.is_number_unsigned()) {
            return static_cast<double>(value.get<uint64_t>());
        }
    }
    return default_value;
}

int64_t get_optional_pad_packed_weights_alignment(const json::json &obj, std::initializer_list<const char *> keys, int64_t default_value) {
    for (const char *key : keys) {
        if (!obj.contains(key)) {
            continue;
        }
        const auto &value = obj.at(key);
        if (value.is_boolean()) {
            return value.get<bool>() ? kLegacyPadPackedWeightsAlignment : 0;
        }
        if (value.is_number_integer()) {
            const int64_t numeric_value = value.get<int64_t>();
            return numeric_value > 0 ? numeric_value : 0;
        }
        if (value.is_number_unsigned()) {
            const int64_t numeric_value = static_cast<int64_t>(value.get<uint64_t>());
            return numeric_value > 0 ? numeric_value : 0;
        }
    }
    return default_value;
}

int64_t normalize_pad_alignment_value(const json::json &value) {
    if (value.is_boolean()) {
        return value.get<bool>() ? kLegacyPadPackedWeightsAlignment : 0;
    }
    if (value.is_number_integer()) {
        const int64_t numeric_value = value.get<int64_t>();
        return numeric_value > 0 ? numeric_value : 0;
    }
    if (value.is_number_unsigned()) {
        const int64_t numeric_value = static_cast<int64_t>(value.get<uint64_t>());
        return numeric_value > 0 ? numeric_value : 0;
    }
    return 0;
}

int64_t round_up_to_alignment(int64_t dim, int64_t alignment) {
    if (alignment <= 0) {
        return dim;
    }
    return ((dim + alignment - 1) / alignment) * alignment;
}

bool parse_packed_weight_pad_spec(const json::json &value, PackedWeightPadSpec &pad_spec) {
    if (!value.is_array() || value.size() < 2) {
        return false;
    }
    pad_spec = {normalize_pad_alignment_value(value[0]), normalize_pad_alignment_value(value[1])};
    return true;
}

bool parse_kernel_dims_entry(const json::json &value, int &K, int &N, int &gops, PackedWeightPadSpec *pad_spec_out = nullptr) {
    K = 0;
    N = 0;
    gops = 0;
    if (pad_spec_out != nullptr) {
        *pad_spec_out = {0, 0};
    }

    const json::json *dims = nullptr;
    if (value.is_array()) {
        dims = &value;
    } else if (value.is_object()) {
        if (!value.contains("dims") || !value["dims"].is_array()) {
            return false;
        }
        dims = &value["dims"];
        if (pad_spec_out != nullptr && value.contains("pad")) {
            parse_packed_weight_pad_spec(value["pad"], *pad_spec_out);
        }
        if (value.contains("gops")) {
            gops = static_cast<int>(normalize_pad_alignment_value(value["gops"]));
        }
    } else {
        return false;
    }

    if (dims->size() < 2) {
        return false;
    }

    K = (*dims)[0].get<int64_t>();
    N = (*dims)[1].get<int64_t>();
    if (dims->size() > 2) {
        gops = (*dims)[2].get<int64_t>();
    }

    return true;
}

std::string canonical_pad_group_name(const std::string &group_name) {
    constexpr std::string_view gen_suffix = "-gen";
    if (group_name.size() > gen_suffix.size() &&
        group_name.compare(group_name.size() - gen_suffix.size(), gen_suffix.size(), gen_suffix) == 0) {
        return group_name.substr(0, group_name.size() - gen_suffix.size());
    }
    return group_name;
}

void store_group_pad_spec(const std::string &group_name, const PackedWeightPadSpec &pad_spec) {
    if (pad_spec[0] <= 0 && pad_spec[1] <= 0) {
        return;
    }

    if (group_name.size() >= 4 && group_name.compare(group_name.size() - 4, 4, "-gen") == 0) {
        return;
    }

    const std::string canonical = canonical_pad_group_name(group_name);
    auto assign = [&](const std::string &layer_name) { packed_weight_pad_map[layer_name] = pad_spec; };

    if (canonical == "qkv") {
        assign("qkv");
    } else if (canonical == "o") {
        assign("o");
    } else if (canonical == "qo") {
        assign("q");
        assign("o");
    } else if (canonical == "kv") {
        assign("k");
        assign("v");
    } else if (canonical == "upgate") {
        assign("gate");
        assign("up");
    } else if (canonical == "down") {
        assign("down");
    }
}

std::vector<int64_t> get_optional_chunk_schedule(const json::json &obj, std::initializer_list<const char *> keys) {
    for (const char *key : keys) {
        if (!obj.contains(key)) {
            continue;
        }
        const auto &value = obj.at(key);
        if (value.is_number_integer()) {
            const int64_t v = value.get<int64_t>();
            if (v > 0) {
                return {v};
            }
            return {};
        }
        if (value.is_number_unsigned()) {
            const int64_t v = static_cast<int64_t>(value.get<uint64_t>());
            if (v > 0) {
                return {v};
            }
            return {};
        }
        if (value.is_array()) {
            std::vector<int64_t> out;
            out.reserve(value.size());
            for (const auto &item : value) {
                int64_t parsed = 0;
                if (item.is_number_integer()) {
                    parsed = item.get<int64_t>();
                } else if (item.is_number_unsigned()) {
                    parsed = static_cast<int64_t>(item.get<uint64_t>());
                } else {
                    continue;
                }
                if (parsed > 0) {
                    out.push_back(parsed);
                }
            }
            return out;
        }
    }
    return {};
}

std::vector<int64_t> get_preferred_chunk_schedule(const json::json &obj, bool prefer_schedule,
                                                  std::initializer_list<const char *> schedule_keys,
                                                  std::initializer_list<const char *> single_keys) {
    if (prefer_schedule) {
        std::vector<int64_t> schedule = get_optional_chunk_schedule(obj, schedule_keys);
        if (!schedule.empty()) {
            return schedule;
        }
        // Backward compatibility: if dedicated schedule key is missing, allow array-valued single keys.
        return get_optional_chunk_schedule(obj, single_keys);
    }

    std::vector<int64_t> single = get_optional_chunk_schedule(obj, single_keys);
    if (!single.empty()) {
        return {single.front()};
    }
    // Fallback for configs that only provide schedule keys while scheduling is disabled.
    std::vector<int64_t> schedule = get_optional_chunk_schedule(obj, schedule_keys);
    if (!schedule.empty()) {
        return {schedule.front()};
    }
    return {};
}

bool has_any_key(const json::json &obj, std::initializer_list<const char *> keys) {
    for (const char *key : keys) {
        if (obj.contains(key)) {
            return true;
        }
    }
    return false;
}

std::string get_optional_string(const json::json &obj, std::initializer_list<const char *> keys, const std::string &default_value) {
    for (const char *key : keys) {
        if (!obj.contains(key)) {
            continue;
        }
        const auto &value = obj.at(key);
        if (value.is_string()) {
            return value.get<std::string>();
        }
    }
    return default_value;
}

std::string format_chunk_schedule(const std::vector<int64_t> &schedule) {
    if (schedule.empty()) {
        return "[]";
    }
    std::string out = "[";
    for (size_t i = 0; i < schedule.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += std::to_string(schedule[i]);
    }
    out += "]";
    return out;
}

bool is_grouped_chunked_kernel_config(const json::json &value) {
    return value.is_array() && !value.empty() && value.front().is_object() && value.front().contains("kernels");
}

std::vector<StageBubbleSpec> parse_stage_bubbles(const json::json &value) {
    std::vector<StageBubbleSpec> out;
    if (!value.is_array()) {
        return out;
    }

    for (const auto &entry : value) {
        if (!entry.is_object()) {
            continue;
        }
        const int64_t delay_us = get_optional_int64(entry, {"delay_us"}, 0);
        const std::string stage = get_optional_string(entry, {"stage"}, "");
        if (delay_us <= 0 || stage.empty()) {
            continue;
        }
        StageBubbleSpec spec;
        spec.chunk_id = get_optional_int64(entry, {"chunk_id"}, -1);
        spec.layer_id = get_optional_int64(entry, {"layer_id"}, -1);
        spec.stage = stage;
        spec.delay_us = delay_us;
        out.push_back(spec);
    }

    std::sort(out.begin(), out.end(), [](const StageBubbleSpec &a, const StageBubbleSpec &b) {
        if (a.chunk_id != b.chunk_id) {
            return a.chunk_id < b.chunk_id;
        }
        if (a.layer_id != b.layer_id) {
            return a.layer_id < b.layer_id;
        }
        if (a.stage != b.stage) {
            return a.stage < b.stage;
        }
        return a.delay_us < b.delay_us;
    });
    return out;
}

struct ChunkedLayerPolicyState {
    bool chunk0_split_seen = false;
    int chunk0_split_npuK = 0;
    std::set<std::pair<int, int>> nonzero_chunk_dims;
};

std::string format_chunked_layer_key(const std::string &layer, int forK, int forN) {
    return (layer.empty() ? std::string("<any>") : layer) + " (" + std::to_string(forK) + "x" + std::to_string(forN) + ")";
}

void validate_chunked_gemm_split_policy(const json::json &kernels_gemm) {
    std::map<std::string, ChunkedLayerPolicyState> policy_by_layer;

    for (const auto &entry : kernels_gemm) {
        if (!entry.is_object()) {
            continue;
        }

        const bool use = get_optional_bool(entry, {"use"}, true);
        if (!use) {
            continue;
        }

        const int chunk_id = static_cast<int>(get_optional_int64(entry, {"chunk_id"}, -1));
        if (chunk_id < 0) {
            continue;
        }

        const std::string layer = entry.contains("layer") ? entry.at("layer").get<std::string>() : "";
        const int forM = entry.at("forM").get<int64_t>();
        const int forK = entry.at("forK").get<int64_t>();
        const int forN = entry.at("forN").get<int64_t>();
        const int npuK = entry.at("npuK").get<int64_t>();
        const bool is_k_split = (npuK > 0 && npuK < forK);

        auto &policy = policy_by_layer[format_chunked_layer_key(layer, forK, forN)];

        if (chunk_id == 0) {
            if (!is_k_split) {
                continue;
            }
            if (!policy.chunk0_split_seen) {
                policy.chunk0_split_seen = true;
                policy.chunk0_split_npuK = npuK;
                continue;
            }
            if (policy.chunk0_split_npuK != npuK) {
                throw std::runtime_error("Invalid chunked GEMM config for layer=" + format_chunked_layer_key(layer, forK, forN) +
                                         ": chunk_id=0 must use a single split-K npuK value, but found both " +
                                         std::to_string(policy.chunk0_split_npuK) + " and " + std::to_string(npuK) + ".");
            }
            continue;
        }

        policy.nonzero_chunk_dims.insert({forM, npuK});
        if (policy.nonzero_chunk_dims.size() > 1) {
            std::string dims_summary;
            for (const auto &dims : policy.nonzero_chunk_dims) {
                if (!dims_summary.empty()) {
                    dims_summary += ", ";
                }
                dims_summary += "(forM=" + std::to_string(dims.first) + ", npuK=" + std::to_string(dims.second) + ")";
            }
            throw std::runtime_error("Invalid chunked GEMM config for layer=" + format_chunked_layer_key(layer, forK, forN) +
                                     ": chunk_id>0 entries must share one (forM, npuK) shape, but found " + dims_summary + ".");
        }
    }
}

const json::json *select_chunked_kernel_group(const json::json &groups, int64_t desired_prompt_len, int64_t desired_chunk_size,
                                              int64_t desired_inflight, bool prefer_schedule) {
    if (!is_grouped_chunked_kernel_config(groups)) {
        return nullptr;
    }

    const json::json *fallback_group = nullptr;
    const json::json *prompt_fallback_group = nullptr;
    const json::json *prompt_mode_mismatch_fallback = nullptr;
    const json::json *exact_mode_mismatch_candidate = nullptr;
    bool has_prompt_len_group = false;
    const std::initializer_list<const char *> schedule_keys = {"hetero_chunk_size_schedule", "hetero_chunk_schedule",
                                                                "gpu_chunk_size_schedule", "gpu_chunk_schedule",
                                                                "gpu_chunk_shedule", "chunk_schedule"};
    for (const auto &group : groups) {
        if (!group.is_object() || !group.contains("kernels") || !group.at("kernels").is_array()) {
            continue;
        }
        if (fallback_group == nullptr) {
            fallback_group = &group;
        }

        const int64_t group_prompt_len = get_optional_int64(group, {"prompt_len"}, 0);
        if (group_prompt_len > 0) {
            has_prompt_len_group = true;
        }
        const std::vector<int64_t> group_schedule = get_preferred_chunk_schedule(
            group, prefer_schedule, schedule_keys,
            {"hetero_chunk_size", "gpu_chunk_size", "chunk_size"});
        const int64_t group_chunk_size = group_schedule.empty() ? 0 : group_schedule.front();
        const int64_t group_inflight =
            get_optional_int64(group, {"hetero_inflight", "gpu_chunking_inflight", "chunking_inflight", "inflight"}, 1);
        const bool group_has_schedule_key = has_any_key(group, schedule_keys);
        const bool group_matches_mode = prefer_schedule ? group_has_schedule_key : !group_has_schedule_key;

        if (desired_prompt_len > 0 && group_prompt_len > 0 && group_prompt_len != desired_prompt_len) {
            continue;
        }
        if (desired_prompt_len > 0 && group_prompt_len == desired_prompt_len) {
            if (group_matches_mode) {
                if (prompt_fallback_group == nullptr) {
                    prompt_fallback_group = &group;
                }
            } else if (prompt_mode_mismatch_fallback == nullptr) {
                prompt_mode_mismatch_fallback = &group;
            }
        }

        const bool chunk_size_matches = (desired_chunk_size <= 0 || group_chunk_size <= 0 || group_chunk_size == desired_chunk_size);
        const bool inflight_matches = (desired_inflight <= 0 || group_inflight <= 0 || group_inflight == desired_inflight);
        if (chunk_size_matches && inflight_matches) {
            if (group_matches_mode) {
                return &group;
            }
            if (exact_mode_mismatch_candidate == nullptr) {
                exact_mode_mismatch_candidate = &group;
            }
        }
    }

    if (prompt_fallback_group != nullptr) {
        return prompt_fallback_group;
    }
    if (exact_mode_mismatch_candidate != nullptr) {
        return exact_mode_mismatch_candidate;
    }
    if (prompt_mode_mismatch_fallback != nullptr) {
        return prompt_mode_mismatch_fallback;
    }
    if (desired_prompt_len > 0 && has_prompt_len_group) {
        // Scheduled chunking encodes explicit per-prompt schedules and must remain strict.
        if (prefer_schedule) {
            return nullptr;
        }
        // Non-scheduled chunking (single chunk_size) is prompt-agnostic enough to
        // use a grouped fallback rather than disabling chunking outright.
        if (fallback_group != nullptr) {
            return fallback_group;
        }
        return nullptr;
    }

    return fallback_group;
}
} // namespace

// Helper to remove C-style comments from JSON string
std::string removeComments(const std::string &input) {
    std::string output;
    output.reserve(input.size());
    bool in_string = false;
    bool in_comment_line = false;
    bool in_comment_block = false;

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        if (in_comment_line) {
            if (c == '\n') {
                in_comment_line = false;
                output += c;
            }
        } else if (in_comment_block) {
            if (c == '*' && i + 1 < input.size() && input[i + 1] == '/') {
                in_comment_block = false;
                i++; // Skip '/'
            }
        } else if (in_string) {
            if (c == '"' && (i == 0 || input[i - 1] != '\\')) {
                in_string = false;
            }
            output += c;
        } else {
            if (c == '"') {
                in_string = true;
                output += c;
            } else if (c == '/' && i + 1 < input.size() && input[i + 1] == '/') {
                in_comment_line = true;
                i++; // Skip second '/'
            } else if (c == '/' && i + 1 < input.size() && input[i + 1] == '*') {
                in_comment_block = true;
                i++; // Skip '*'
            } else {
                output += c;
            }
        }
    }
    return output;
}

// Find index of path in vector
template <typename T> int find_path_index(const std::vector<T> &vec, const std::string &path, const std::vector<std::string> &paths) {
    auto it = std::find(paths.begin(), paths.end(), path);
    if (it != paths.end()) {
        return std::distance(paths.begin(), it);
    }
    return -1;
}

int get_layer_id(const std::string &layer_type) {
    if (layer_type == "qkv")
        return 8;
    if (layer_type == "q")
        return 1;
    if (layer_type == "k")
        return 2;
    if (layer_type == "v")
        return 3;
    if (layer_type == "o")
        return 4;
    if (layer_type == "gate")
        return 5;
    if (layer_type == "up")
        return 6;
    if (layer_type == "down")
        return 7;
    return 0; // Default/Any
}

// Initialize NPU context arrays with maximum capacity
void init_npu() {
    if (debug_verbosity >= 1) {
        std::cout << "Initializing NPU context arrays with max sizes: HW=" << MAX_NPU_HW_CTX << ", Inst=" << MAX_NPU_INST_CTX << std::endl;
    }

    // Resize arrays to max capacity (actual elements, not just reserved space)
    hwctxt_array.resize(MAX_NPU_HW_CTX);
    instctxt_array.resize(MAX_NPU_INST_CTX);
}

void read_npu_config(const std::string &config_path) {
    // Use provided path or default
    std::string json_path = config_path;
    if (json_path.empty()) {
        json_path = "src/unified_llm_w4a16_hetero/kernels.json";
    }

    if (!std::filesystem::exists(json_path)) {
        // Fallback to absolute path
        if (config_path.empty()) {
            const char *env_root = std::getenv("HETEROMOSAIC_ROOT");
            std::string root = env_root ? env_root : "/home/greg/Desktop/heteroMosaic";
            json_path = root + "/src/unified_llm_w4a16_hetero/kernels.json";
        }
    }

    std::ifstream f(json_path);
    if (!f.is_open()) {
        std::cerr << "Failed to open kernels.json at " << json_path << std::endl;
        return;
    }

    std::stringstream buffer;
    buffer << f.rdbuf();
    std::string json_str = removeComments(buffer.str());

    try {
        json::json jv = json::json::parse(json_str);
        json::json &root = jv;

        // Reset rope scaling defaults for each config read
        rope_scaling_enabled = false;
        rope_scaling_type.clear();
        rope_scaling_factor = 1.0f;
        rope_scaling_low_freq_factor = 1.0f;
        rope_scaling_high_freq_factor = 1.0f;
        rope_scaling_original_max_position_embeddings = 0.0f;
        gemv_driven_split_K = false;
        gemv_npu_col.clear();
        trace_output_path.clear();
        trace_run_tag.clear();
        trace_sync_stages = false;
        stage_bubbles.clear();
        packed_weight_pad_map.clear();
        npu_dim = -1;
        chunking_tokens = 0;
        chunking_token_schedule.clear();
        chunking_inflight = 1;
        chunking_scheduled = true;
        async_chunking = false;
        const json::json *selected_chunk_group_for_runtime = nullptr;

        // Read heterogeneity setting and store in global hw_target
        if (root.contains("heterogeneity")) {
            hw_target = root["heterogeneity"].get<std::string>();
        }

        if (root.contains("debug_verbosity")) {
            debug_verbosity = root["debug_verbosity"].get<int64_t>();
            if (debug_verbosity >= 1)
                std::cout << "Debug verbosity level: " << debug_verbosity << std::endl;
        }
        set_npu_debug_verbosity(debug_verbosity);

        if (root.contains("usePackedWeights")) {
            use_packed_weights = root["usePackedWeights"].get<bool>();
            if (debug_verbosity >= 1) {
                std::cout << "Use Packed Weights: " << (use_packed_weights ? "true" : "false") << std::endl;
            }
        }

        if (root.contains("padPackedWeights")) {
            pad_packed_weights = get_optional_pad_packed_weights_alignment(root, {"padPackedWeights"}, 0);
            if (debug_verbosity >= 1) {
                if (pad_packed_weights > 0) {
                    std::cout << "Pad Packed Weights Alignment: " << pad_packed_weights << std::endl;
                } else {
                    std::cout << "Pad Packed Weights: false" << std::endl;
                }
            }
        }

        if (pad_packed_weights > 0 && root.contains("npuOnlydefault") && root["npuOnlydefault"].is_array()) {
            for (const auto &entry : root["npuOnlydefault"]) {
                if (!entry.is_object()) {
                    continue;
                }
                for (auto it = entry.begin(); it != entry.end(); ++it) {
                    PackedWeightPadSpec pad_spec = {0, 0};
                    int unused_k = 0;
                    int unused_n = 0;
                    int unused_gops = 0;
                    if (!parse_kernel_dims_entry(it.value(), unused_k, unused_n, unused_gops, &pad_spec)) {
                        continue;
                    }
                    store_group_pad_spec(it.key(), pad_spec);
                }
            }

            if (debug_verbosity >= 2 && !packed_weight_pad_map.empty()) {
                for (const auto &entry : packed_weight_pad_map) {
                    std::cout << "Packed pad spec for layer " << entry.first << ": K=" << entry.second[0] << " N=" << entry.second[1]
                              << std::endl;
                }
            }
        }

        if (root.contains("cpu_decode")) {
            cpu_decode = root["cpu_decode"].get<bool>();
            if (debug_verbosity >= 1) {
                std::cout << "CPU Decode: " << (cpu_decode ? "true" : "false") << std::endl;
            }
        }

        if (root.contains("dummy_weights")) {
            dummy_weights_enabled = root["dummy_weights"].get<bool>();
            if (debug_verbosity >= 1 && dummy_weights_enabled) {
                std::cout << "Dummy weights enabled: true" << std::endl;
            }
        }

        if (root.contains("warmup")) {
            warmup_enabled = root["warmup"].get<bool>();
            if (debug_verbosity >= 1) {
                std::cout << "Warmup enabled: " << (warmup_enabled ? "true" : "false") << std::endl;
            }
        }

        if (root.contains("minimal_pdi")) {
            minimal_pdi = root["minimal_pdi"].get<bool>();
            if (debug_verbosity >= 1) {
                std::cout << "Minimal PDI enabled: " << (minimal_pdi ? "true" : "false") << std::endl;
            }
        }

        split_M_only = root.value("split_M_only", false);
        if (debug_verbosity >= 1) {
            std::cout << "Split M only: " << (split_M_only ? "true" : "false") << std::endl;
        }

        int64_t configured_prompt_len = -1;
        if (root.contains("prompt_len")) {
            npu_dim = root["prompt_len"].get<int64_t>();
            configured_prompt_len = npu_dim;
            if (debug_verbosity >= 1) {
                std::cout << "Prompt length filter enabled: " << npu_dim << std::endl;
            }
        } else if (root.contains("npu_dim")) {
            // Backward-compatible fallback for older configs.
            npu_dim = root["npu_dim"].get<int64_t>();
            configured_prompt_len = npu_dim;
            if (debug_verbosity >= 1) {
                std::cout << "Prompt length filter enabled (legacy npu_dim): " << npu_dim << std::endl;
            }
        }

        bool chunking_enabled = false;
        if (root.contains("chunking")) {
            if (root["chunking"].is_boolean()) {
                chunking_enabled = root["chunking"].get<bool>();
            } else {
                int64_t parsed_chunking = root["chunking"].get<int64_t>();
                chunking_tokens = parsed_chunking > 0 ? parsed_chunking : 0;
                if (chunking_tokens > 0) {
                    chunking_token_schedule = {chunking_tokens};
                }
                chunking_enabled = chunking_tokens > 0;
            }
        }

        if (root.contains("chunking_inflight")) {
            int64_t parsed_chunking_inflight = root["chunking_inflight"].get<int64_t>();
            chunking_inflight = parsed_chunking_inflight > 0 ? parsed_chunking_inflight : 1;
        }
        chunking_scheduled = get_optional_bool(root, {"chunking_scheduled"}, true);

        if (chunking_enabled) {
            const json::json *chunking_cfg = nullptr;
            if (hw_target == "gpu") {
                if (root.contains("gpu_chunking") && root["gpu_chunking"].is_object()) {
                    chunking_cfg = &root["gpu_chunking"];
                } else if (root.contains("gpu_chunk_size") || root.contains("gpu_chunking_inflight")) {
                    chunking_cfg = &root;
                }
            }

            if (chunking_cfg != nullptr) {
                std::vector<int64_t> parsed_schedule = get_preferred_chunk_schedule(
                    *chunking_cfg, chunking_scheduled, {"gpu_chunk_size_schedule", "gpu_chunk_schedule", "gpu_chunk_shedule", "chunk_schedule"},
                    {"gpu_chunk_size", "chunk_size"});
                if (!parsed_schedule.empty()) {
                    chunking_token_schedule = std::move(parsed_schedule);
                    chunking_tokens = chunking_token_schedule.front();
                } else {
                    chunking_tokens =
                        std::max<int64_t>(0, get_optional_int64(*chunking_cfg, {"gpu_chunk_size", "chunk_size"}, chunking_tokens));
                    if (chunking_tokens > 0) {
                        chunking_token_schedule = {chunking_tokens};
                    }
                }
                chunking_inflight = std::max<int64_t>(
                    1, get_optional_int64(*chunking_cfg, {"gpu_chunking_inflight", "chunking_inflight", "inflight"}, chunking_inflight));
            }

            if (chunking_tokens <= 0 && root.contains("kernels_gemm_chunked")) {
                const json::json *selected_group =
                    select_chunked_kernel_group(root["kernels_gemm_chunked"], configured_prompt_len, -1, -1, chunking_scheduled);
                if (selected_group != nullptr) {
                    selected_chunk_group_for_runtime = selected_group;
                    std::vector<int64_t> selected_schedule = get_preferred_chunk_schedule(
                        *selected_group, chunking_scheduled,
                        {"hetero_chunk_size_schedule", "hetero_chunk_schedule", "gpu_chunk_size_schedule", "gpu_chunk_schedule",
                         "gpu_chunk_shedule", "chunk_schedule"},
                        {"hetero_chunk_size", "gpu_chunk_size", "chunk_size"});
                    if (!selected_schedule.empty()) {
                        const int64_t selected_prompt_len =
                            get_optional_int64(*selected_group, {"prompt_len"}, configured_prompt_len);
                        if (hw_target == "hetero" && selected_schedule.size() > 1) {
                            if (selected_prompt_len <= 0) {
                                throw std::runtime_error("hetero_chunk_size schedule requires a valid prompt_len for hetero mode");
                            }
                            const int64_t schedule_sum =
                                std::accumulate(selected_schedule.begin(), selected_schedule.end(), static_cast<int64_t>(0));
                            if (schedule_sum != selected_prompt_len) {
                                throw std::runtime_error("hetero_chunk_size schedule must sum to prompt_len in hetero mode (prompt_len=" +
                                                         std::to_string(selected_prompt_len) +
                                                         ", schedule_sum=" + std::to_string(schedule_sum) +
                                                         ", schedule=" + format_chunk_schedule(selected_schedule) + ")");
                            }
                        }
                        chunking_token_schedule = std::move(selected_schedule);
                        chunking_tokens = chunking_token_schedule.front();
                    } else {
                        chunking_tokens =
                            std::max<int64_t>(0, get_optional_int64(*selected_group, {"hetero_chunk_size", "gpu_chunk_size", "chunk_size"}, 0));
                    }
                    chunking_inflight = std::max<int64_t>(
                        1,
                        get_optional_int64(*selected_group, {"hetero_inflight", "gpu_chunking_inflight", "chunking_inflight", "inflight"},
                                           chunking_inflight));
                    if (chunking_tokens > 0 && chunking_token_schedule.empty()) {
                        chunking_token_schedule = {chunking_tokens};
                    }
                }
            }

            if (chunking_tokens > 0 && chunking_token_schedule.empty()) {
                chunking_token_schedule = {chunking_tokens};
            }

            // In chunked mode, M filtering should follow the active chunk size.
            if (chunking_tokens > 0) {
                npu_dim = static_cast<int>(chunking_tokens);
            }
        } else {
            chunking_tokens = 0;
            chunking_token_schedule.clear();
            chunking_inflight = 1;
        }

        if (selected_chunk_group_for_runtime == nullptr && root.contains("kernels_gemm_chunked")) {
            selected_chunk_group_for_runtime = select_chunked_kernel_group(
                root["kernels_gemm_chunked"],
                configured_prompt_len,
                chunking_tokens > 0 ? chunking_tokens : -1,
                chunking_inflight,
                chunking_scheduled);
        }

        if (debug_verbosity >= 1) {
            std::cout << "Chunking enabled: " << (chunking_enabled ? "true" : "false") << std::endl;
            if (chunking_enabled) {
                std::cout << "Chunking scheduled: " << (chunking_scheduled ? "true" : "false") << std::endl;
                std::cout << "Prefill chunk size: " << chunking_tokens << std::endl;
                if (chunking_token_schedule.size() > 1) {
                    std::cout << "Prefill chunk schedule: [";
                    for (size_t i = 0; i < chunking_token_schedule.size(); ++i) {
                        if (i > 0) {
                            std::cout << ", ";
                        }
                        std::cout << chunking_token_schedule[i];
                    }
                    std::cout << "]" << std::endl;
                }
                std::cout << "Chunking inflight workers: " << chunking_inflight << std::endl;
            }
        }

        // Async chunking is enabled only when chunking is active and >1 inflight worker is requested.
        async_chunking = (chunking_enabled && chunking_tokens > 0 && chunking_inflight > 1);
        if (debug_verbosity >= 1) {
            std::cout << "Async chunking: " << (async_chunking ? "true" : "false") << std::endl;
        }

        if (root.contains("gemv_driven_split_K")) {
            gemv_driven_split_K = root["gemv_driven_split_K"].get<bool>();
            if (debug_verbosity >= 1) {
                std::cout << "GEMV Driven Split K: " << (gemv_driven_split_K ? "true" : "false") << std::endl;
            }
        }

        if (root.contains("gemv_npu_col")) {
            gemv_npu_col = root["gemv_npu_col"].get<std::string>();
            if (debug_verbosity >= 1) {
                std::cout << "GEMV NPU Col Prefix: " << gemv_npu_col << std::endl;
            }
        }

        trace_output_path = get_optional_string(root, {"trace_output_path"}, "");
        trace_run_tag = get_optional_string(root, {"trace_run_tag"}, "");
        trace_sync_stages = get_optional_bool(root, {"trace_sync_stages"}, false);
        if (debug_verbosity >= 1 && !trace_output_path.empty()) {
            std::cout << "Trace output path: " << trace_output_path << std::endl;
            std::cout << "Trace sync stages: " << (trace_sync_stages ? "true" : "false") << std::endl;
        }

        if (root.contains("rope_scaling")) {
            const auto &rope_val = root["rope_scaling"];
            if (rope_val.is_object()) {
                const json::json &rope_obj = rope_val;
                auto to_float = [](const json::json &val, float default_value) {
                    if (val.is_number_float()) {
                        return static_cast<float>(val.get<double>());
                    }
                    if (val.is_number_integer()) {
                        return static_cast<float>(val.get<int64_t>());
                    }
                    if (val.is_number_unsigned()) {
                        return static_cast<float>(val.get<uint64_t>());
                    }
                    return default_value;
                };

                if (rope_obj.contains("type")) {
                    rope_scaling_type = rope_obj.at("type").get<std::string>();
                } else if (rope_obj.contains("rope_type")) {
                    rope_scaling_type = rope_obj.at("rope_type").get<std::string>();
                }

                if (rope_obj.contains("factor")) {
                    rope_scaling_factor = to_float(rope_obj.at("factor"), rope_scaling_factor);
                }
                if (rope_obj.contains("low_freq_factor")) {
                    rope_scaling_low_freq_factor = to_float(rope_obj.at("low_freq_factor"), rope_scaling_low_freq_factor);
                }
                if (rope_obj.contains("high_freq_factor")) {
                    rope_scaling_high_freq_factor = to_float(rope_obj.at("high_freq_factor"), rope_scaling_high_freq_factor);
                }
                if (rope_obj.contains("original_max_position_embeddings")) {
                    rope_scaling_original_max_position_embeddings =
                        to_float(rope_obj.at("original_max_position_embeddings"), rope_scaling_original_max_position_embeddings);
                }

                if (!rope_scaling_type.empty() && rope_scaling_type != "none") {
                    rope_scaling_enabled = true;
                }
            }

            if (debug_verbosity >= 1 && rope_scaling_enabled) {
                std::cout << "RoPE scaling enabled: type=" << rope_scaling_type << " factor=" << rope_scaling_factor
                          << " low_freq_factor=" << rope_scaling_low_freq_factor << " high_freq_factor=" << rope_scaling_high_freq_factor
                          << " orig_ctx=" << rope_scaling_original_max_position_embeddings << std::endl;
            }
        }

        if (selected_chunk_group_for_runtime != nullptr && selected_chunk_group_for_runtime->contains("stage_bubbles")) {
            stage_bubbles = parse_stage_bubbles(selected_chunk_group_for_runtime->at("stage_bubbles"));
        } else if (root.contains("stage_bubbles")) {
            stage_bubbles = parse_stage_bubbles(root.at("stage_bubbles"));
        }
        if (debug_verbosity >= 1 && !stage_bubbles.empty()) {
            std::cout << "Configured stage bubbles: " << stage_bubbles.size() << std::endl;
        }

        if (debug_verbosity >= 1 && root.contains("heterogeneity")) {
            std::cout << "Heterogeneity mode: " << hw_target << std::endl;
        }

    } catch (const std::exception &e) {
        std::cerr << "Error parsing kernels.json in read_npu_config: " << e.what() << std::endl;
        throw;
    }
}

// Helper to determine split type (optimized)
int get_split_type(int64_t K, int64_t N, std::string layer_type) {
    int layer_id = get_layer_id(layer_type);

    int M_key = npu_dim;
    if (gemv_driven_split_K) {
        M_key = 1;
    }

    auto find_positive_split_for_chunk = [&](int chunk_id) -> int {
        auto exact_it = config_map.find({M_key, (int)K, (int)N, layer_id, chunk_id});
        if (exact_it != config_map.end() && exact_it->second.split_type > 0) {
            return exact_it->second.split_type;
        }

        for (const auto &pair : config_map) {
            const auto &map_key = pair.first;
            if (map_key[1] == K && map_key[2] == N && map_key[3] == layer_id && map_key[4] == chunk_id && pair.second.split_type > 0) {
                return pair.second.split_type;
            }
        }
        return -1;
    };

    if (chunking_tokens > 0) {
        // Preferred source for split-K in chunked mode is chunk_id=0 when present.
        const int chunk0_split = find_positive_split_for_chunk(0);
        if (chunk0_split > 0) {
            return chunk0_split;
        }

        // Fallback: allow split-K to be discovered from any chunk config.
        // This is required when chunk_id=0 is intentionally GPU-only (npuM=0)
        // and split-K is activated on later chunks (chunk_id>0).
        int fallback_split = -1;
        int fallback_chunk_id = -1;
        for (const auto &pair : config_map) {
            const auto &map_key = pair.first;
            if (map_key[1] != K || map_key[2] != N || map_key[3] != layer_id || map_key[4] < 0 || pair.second.split_type <= 0) {
                continue;
            }

            if (fallback_split < 0 || map_key[4] < fallback_chunk_id) {
                fallback_split = pair.second.split_type;
                fallback_chunk_id = map_key[4];
            } else if (pair.second.split_type != fallback_split) {
                // Keep behavior deterministic when multiple chunk entries disagree.
                // Chunk policy validation should prevent this, but prefer the first
                // discovered split and emit a warning.
                if (debug_verbosity >= 1) {
                    std::cerr << "Warning: inconsistent split-K values for layer " << layer_type << " (" << K << "x" << N
                              << ") across chunked configs; using split-K=" << fallback_split << " from chunk_id="
                              << fallback_chunk_id << ", ignoring split-K=" << pair.second.split_type << " from chunk_id=" << map_key[4]
                              << std::endl;
                }
            }
        }
        if (fallback_split > 0) {
            return fallback_split;
        }

        // Chunking override: do not fall back to non-chunk split configs.
        // When chunked execution is enabled, split policy must be derived
        // only from chunk-specific entries.
        if (debug_verbosity >= 2) {
            std::cout << "Chunked split override: no split-K found in chunk configs for layer " << layer_type << " (" << K << "x" << N
                      << "), forcing non-split." << std::endl;
        }
        return -1;
    }

    NPUKey key = {M_key, (int)K, (int)N, layer_id, -1};

    if (config_map.count(key)) {
        return config_map[key].split_type;
    }

    // Fallback: search for matching K, N, Layer regardless of M
    for (const auto &pair : config_map) {
        const auto &map_key = pair.first;
        // Key format: {M, K, N, Layer, chunk_id}
        if (map_key[1] == K && map_key[2] == N && map_key[3] == layer_id && map_key[4] == -1) {
            return pair.second.split_type;
        }
    }

    return -1; // Default to M-split
}

void load_npu_kernels(int drv_fd, const std::string &config_path) {
    std::string json_path = config_path;
    if (json_path.empty()) {
        json_path = "src/unified_llm_w4a16_hetero/kernels.json";
    }

    if (!std::filesystem::exists(json_path)) {
        if (config_path.empty()) {
            const char *env_root = std::getenv("HETEROMOSAIC_ROOT");
            std::string root = env_root ? env_root : "/home/greg/Desktop/heteroMosaic";
            json_path = root + "/src/unified_llm_w4a16_hetero/kernels.json";
        }
    }

    // Root dir for xclbin/inst paths
    const char *env_root_ptr = std::getenv("HETEROMOSAIC_ROOT");
    std::string root_dir = env_root_ptr ? env_root_ptr : "/home/greg/Desktop/heteroMosaic";

    std::ifstream f(json_path);
    if (!f.is_open()) {
        std::cerr << "Failed to open kernels.json at " << json_path << std::endl;
        return;
    }

    std::stringstream buffer;
    buffer << f.rdbuf();
    std::string json_str = removeComments(buffer.str());

    // Track the current index for HW and Inst contexts
    int hw_ctx_count = 0;
    int inst_ctx_count = 0;

    try {
        json::json jv = json::json::parse(json_str);
        json::json &root = jv;

        // Read heterogeneity setting
        if (root.contains("heterogeneity")) {
            hw_target = root["heterogeneity"].get<std::string>();
            // if (debug_verbosity >= 1) std::cout << "Heterogeneity mode: " << hw_target << std::endl;
        }

        // Read debug verbosity level and store in global debug_verbosity
        // Already read in read_npu_config
        if (root.contains("debug_verbosity")) {
            debug_verbosity = root["debug_verbosity"].get<int64_t>();
            // if (debug_verbosity >= 1) std::cout << "Debug verbosity level: " << debug_verbosity << std::endl;
        }
        set_npu_debug_verbosity(debug_verbosity);

        // Parse GEMM kernels. When chunking is active, prefer chunk-aware GEMM configs.
        json::json kernels_gemm;
        int m_filter_dim = npu_dim;
        const int64_t configured_prompt_len = get_optional_int64(root, {"prompt_len", "npu_dim"}, -1);
        if (chunking_tokens > 0 && root.contains("kernels_gemm_chunked")) {
            const json::json &chunked_cfg = root["kernels_gemm_chunked"];
            if (is_grouped_chunked_kernel_config(chunked_cfg)) {
                const json::json *selected_group =
                    select_chunked_kernel_group(chunked_cfg, configured_prompt_len, chunking_tokens, chunking_inflight, chunking_scheduled);
                if (selected_group != nullptr) {
                    kernels_gemm = selected_group->at("kernels");
                    const std::vector<int64_t> group_chunk_schedule = get_preferred_chunk_schedule(
                        *selected_group, chunking_scheduled,
                        {"hetero_chunk_size_schedule", "hetero_chunk_schedule", "gpu_chunk_size_schedule", "gpu_chunk_schedule",
                         "gpu_chunk_shedule", "chunk_schedule"},
                        {"hetero_chunk_size", "gpu_chunk_size", "chunk_size"});
                    const int64_t group_chunk_size =
                        !group_chunk_schedule.empty() ? group_chunk_schedule.front()
                                                      : get_optional_int64(*selected_group, {"hetero_chunk_size", "gpu_chunk_size", "chunk_size"},
                                                                           chunking_tokens);
                    if (group_chunk_size > 0) {
                        m_filter_dim = static_cast<int>(group_chunk_size);
                    }
                    if (group_chunk_schedule.size() > 1) {
                        m_filter_dim = -1;
                    }
                    if (debug_verbosity >= 1) {
                        const int64_t group_inflight = get_optional_int64(
                            *selected_group, {"hetero_inflight", "gpu_chunking_inflight", "chunking_inflight", "inflight"},
                            chunking_inflight);
                        const int64_t group_prompt_len = get_optional_int64(*selected_group, {"prompt_len"}, -1);
                        std::cout << "Using grouped kernels_gemm_chunked (chunk_size=" << group_chunk_size
                                  << ", chunk_schedule=" << format_chunk_schedule(group_chunk_schedule)
                                  << ", inflight=" << group_inflight << ", prompt_len=" << group_prompt_len << ")" << std::endl;
                    }
                }
            } else {
                kernels_gemm = chunked_cfg;
                if (chunking_tokens > 0) {
                    m_filter_dim = static_cast<int>(chunking_tokens);
                }
                if (debug_verbosity >= 1) {
                    std::cout << "Using flat kernels_gemm_chunked (chunking_tokens=" << chunking_tokens << ")" << std::endl;
                }
            }

            if (kernels_gemm.is_null()) {
                std::cerr << "No valid kernels_gemm_chunked group found for prompt_len=" << configured_prompt_len
                          << " chunk_size=" << chunking_tokens << " inflight=" << chunking_inflight << std::endl;
                return;
            }
        } else if (root.contains("kernels_gemm")) {
            kernels_gemm = root["kernels_gemm"];
        } else if (root.contains("kernels")) {
            // Fallback for backward compatibility
            kernels_gemm = root["kernels"];
        }
        if (debug_verbosity >= 1 && m_filter_dim > 0) {
            std::cout << "Active GEMM M filter: " << m_filter_dim << (chunking_tokens > 0 ? " (from chunked config)" : " (from prompt_len)")
                      << std::endl;
        }

        if (chunking_tokens > 0 && !kernels_gemm.is_null()) {
            validate_chunked_gemm_split_policy(kernels_gemm);
        }

        // Cache for PDI reuse: key is {npuK, tile_size, col, dtype} -> hw_idx
        std::map<std::tuple<int, std::string, std::string, std::string>, int> pdi_cache;

        for (auto &k : kernels_gemm) {
            json::json &obj = k;

            // Check if this kernel should be loaded
            bool use = true; // default to true
            if (obj.contains("use")) {
                use = obj["use"].get<bool>();
            }

            // Skip if use is false
            if (!use) {
                if (debug_verbosity >= 1)
                    std::cout << "Skipping kernel (use=false)" << std::endl;
                continue;
            }

            // Read dims
            int npuM = obj["npuM"].get<int64_t>();
            int npuK = obj["npuK"].get<int64_t>();
            int npuN = obj["npuN"].get<int64_t>();

            int forM = obj["forM"].get<int64_t>();
            int forK = obj["forK"].get<int64_t>();
            int forN = obj["forN"].get<int64_t>();

            // Determine split type
            int split_type = -1; // default to M
            if (forK != npuK) {
                split_type = npuK; // K-split (Positive Integer) - Takes precedence
            } else if (forM != npuM && forN == npuN) {
                split_type = -1; // M-split
            } else if (forN != npuN) {
                split_type = -2; // N-split
            }

            // If GEMV-Driven Split K is enabled, skip loading GEMM configs that define a Split-K
            if (gemv_driven_split_K && split_type > 0) {
                if (debug_verbosity >= 2) {
                    std::cout << "Skipping GEMM Split-K config (gemv_driven_split_K=true): forK=" << forK << " npuK=" << npuK << std::endl;
                }
                continue;
            }
            // Read layer (optional, default "") - needed for config_map
            std::string layer = "";
            if (obj.contains("layer")) {
                layer = obj["layer"].get<std::string>();
            }
            int layer_id = get_layer_id(layer);

            int config = -1;
            if (obj.contains("config")) {
                config = obj["config"].get<int64_t>();
            }

            // Skip if npuM is 0 (GPU-only mode)
            if (npuM == 0) {
                if (debug_verbosity >= 1) {
                    std::cout << "Skipping kernel (npuM=0, GPU-only mode)" << std::endl;
                    std::cout << "  forM=" << forM << " forK=" << forK << " forN=" << forN << " layer=" << layer << std::endl;
                }
                continue;
            }

            // Determine if we should skip PDI loading (K-split with npu_dim filter)
            bool skip_pdi = false;
            // split_type == -1 is M-split
            if (m_filter_dim > 0 && (split_type != -1 || forM != m_filter_dim)) {
                if (split_type > 0) {
                    // Enforce strict M match for GEMM split kernels when npu_dim is active.
                    if (forM != m_filter_dim) {
                        if (debug_verbosity >= 1) {
                            std::cout << "Skipping K-split kernel (M filter mismatch): forM=" << forM << ", m_filter=" << m_filter_dim
                                      << ", split=" << split_type << std::endl;
                        }
                        continue;
                    }

                    bool has_k = false;
                    if (obj.contains("fw_path")) {
                        std::string fw = obj["fw_path"].get<std::string>();
                        if (fw.find("_K") != std::string::npos) {
                            has_k = true;
                        }
                    }

                    if (has_k) {
                        skip_pdi = false;
                    } else {
                        if (debug_verbosity >= 1) {
                            std::cout << "Skipping K-split kernel (missing _K): forM=" << forM << ", split=" << split_type << std::endl;
                        }
                        continue;
                    }
                } else {
                    if (debug_verbosity >= 1) {
                        std::cout << "Skipping kernel (M filter): forM=" << forM << ", m_filter=" << m_filter_dim
                                  << ", split=" << split_type << std::endl;
                    }
                    continue;
                }
            }

            int hw_idx = -1;
            int inst_idx = -1;

            if (!skip_pdi) {
                // Read num_tiles (optional, default 1)
                int num_tiles = 1;
                if (obj.contains("num_tiles")) {
                    num_tiles = obj["num_tiles"].get<int64_t>();
                }

                std::string xclbin, inst;
                bool reuse_pdi = false;
                int reused_hw_idx = -1;

                // Props for caching
                std::tuple<int, std::string, std::string, std::string> cache_key;
                bool can_cache = false;

                if (obj.contains("xclbin") && obj.contains("inst")) {
                    xclbin = obj["xclbin"].get<std::string>();
                    inst = obj["inst"].get<std::string>();
                } else if (obj.contains("fw_path") && obj.contains("tile_size") && obj.contains("col") && obj.contains("dtype")) {
                    std::string fw_path = obj["fw_path"].get<std::string>();
                    std::string tile_size = obj["tile_size"].get<std::string>();
                    std::string col = obj["col"].get<std::string>();
                    std::string dtype = obj["dtype"].get<std::string>();

                    // Check for PDI reuse
                    if (minimal_pdi) {
                        cache_key = std::make_tuple(npuK, tile_size, col, dtype);
                        can_cache = true;
                        if (pdi_cache.find(cache_key) != pdi_cache.end()) {
                            reuse_pdi = true;
                            reused_hw_idx = pdi_cache[cache_key];
                            if (debug_verbosity >= 1) {
                                std::cout << "Reusing PDI for " << layer << " (hw_idx=" << reused_hw_idx << ")" << std::endl;
                            }
                        }
                    }

                    // Ensure fw_path ends with /
                    if (fw_path.back() != '/') {
                        fw_path += "/";
                    }

                    std::string dims_str = std::to_string(npuM) + "x" + std::to_string(npuK) + "x" + std::to_string(npuN);
                    inst = fw_path + "insts_" + dims_str + "_" + tile_size + "_" + col + "_" + dtype + ".txt";

                    if (!reuse_pdi) {
                        xclbin = fw_path + "final_" + dims_str + "_" + tile_size + "_" + col + "_" + dtype + ".pdi";
                    }

                } else {
                    std::cerr << "Kernel definition missing path info (xclbin/inst OR fw_path/tile_size/col/dtype)" << std::endl;
                    continue;
                }

                std::string inst_path = root_dir + "/" + inst;

                // Load HW Context (PDI)
                if (reuse_pdi) {
                    hw_idx = reused_hw_idx;
                } else {
                    std::string xclbin_path = root_dir + "/" + xclbin;
                    hw_idx = hw_ctx_count;
                    if (debug_verbosity >= 1)
                        std::cout << "Loading PDI: " << xclbin_path << " with num_tiles=" << num_tiles << std::endl;
                    if (createHWctxt(drv_fd, hwctxt_array[hw_idx], xclbin_path.c_str(), num_tiles) != 0) {
                        std::cerr << "Failed to create HW context for " << xclbin_path << std::endl;
                        continue;
                    }
                    FlushCpuCache((const void *)hwctxt_array[hw_idx].pdi_vaddr, 0, hwctxt_array[hw_idx].pdi_size);
                    hw_ctx_count++;

                    if (minimal_pdi && can_cache) {
                        pdi_cache[cache_key] = hw_idx;
                    }
                }

                if (hw_ctx_count >= MAX_NPU_HW_CTX) {
                    std::cerr << "Warning: Reached maximum HW contexts (" << MAX_NPU_HW_CTX << ")" << std::endl;
                }

                // Load Instruction Context
                inst_idx = inst_ctx_count;
                if (debug_verbosity >= 1)
                    std::cout << "Loading Instructions: " << inst_path << std::endl;
                if (createInstctxt(drv_fd, instctxt_array[inst_idx], inst_path.c_str(), true) != 0) {
                    std::cerr << "Failed to create Inst context for " << inst_path << std::endl;
                    continue;
                }
                FlushCpuCache((const void *)instctxt_array[inst_idx].dpu_0_vaddr, 0,
                              instctxt_array[inst_idx].num_dpu_0_insts * sizeof(uint32_t));
                inst_ctx_count++;

                if (inst_ctx_count >= MAX_NPU_INST_CTX) {
                    std::cerr << "Warning: Reached maximum Inst contexts (" << MAX_NPU_INST_CTX << ")" << std::endl;
                }
            } // End of PDI loading block

            int chunk_id = -1;
            if (obj.contains("chunk_id")) {
                chunk_id = obj["chunk_id"].get<int64_t>();
            }

            // Update config map with all values
            NPUKey key = {forM, forK, forN, layer_id, chunk_id};
            config_map[key] = {hw_idx, inst_idx, npuM, npuK, npuN, 0, 1, config, split_type}; // cpuN=0, cpuThreads=1 for GEMM
            if (debug_verbosity >= 1) {
                std::cout << "Mapped kernel " << forM << "x" << forK << "x" << forN << " layer=" << layer << "(" << layer_id
                          << ") -> npuM=" << npuM << " npuK=" << npuK << " npuN=" << npuN << " config=" << config << " split=" << split_type
                          << " chunk_id=" << chunk_id << " to HW[" << hw_idx << "] Inst[" << inst_idx << "]" << std::endl;
            }
        } // End of for loop

        // Parse GEMV kernels only when minimal PDI is enabled.
        if (minimal_pdi && root.contains("kernels_gemv")) {
            json::json &kernels_gemv = root["kernels_gemv"];
            for (auto &k : kernels_gemv) {
                json::json &obj = k;
                bool use = true;
                if (obj.contains("use"))
                    use = obj["use"].get<bool>();
                if (!use)
                    continue;

                int npuM = obj["npuM"].get<int64_t>();
                int npuK = obj["npuK"].get<int64_t>();
                int npuN = obj["npuN"].get<int64_t>();

                // Keep npuN==0 entries: they should still be mapped for host-only (CPU/GPU) GEMV routing.

                int cpuN = 0;
                if (obj.contains("cpuN"))
                    cpuN = obj["cpuN"].get<int64_t>();
                int threads = 1;
                if (obj.contains("cpuThreads"))
                    threads = obj["cpuThreads"].get<int64_t>();

                int forM = obj["forM"].get<int64_t>();
                int forK = obj["forK"].get<int64_t>();
                int forN = obj["forN"].get<int64_t>();

                // Read layer (optional)
                std::string layer = "";
                if (obj.contains("layer")) {
                    layer = obj["layer"].get<std::string>();
                }
                int layer_id = get_layer_id(layer);

                int chunk_id = -1;
                if (obj.contains("chunk_id")) {
                    chunk_id = obj["chunk_id"].get<int64_t>();
                }

                // GEMV runtime policy:
                // - gemv_driven_split_K=true: honor GEMV split config and allow NPU path.
                // - gemv_driven_split_K=false: force GPU-only GEMV and do NOT load GEMV NPU kernels.
                if (!gemv_driven_split_K) {
                    if (debug_verbosity >= 1) {
                        std::cout << "GEMV: gemv_driven_split_K=false, forcing GPU-only path for layer=" << layer << " (" << forK << "x"
                                  << forN << ")" << std::endl;
                    }
                    npuK = 0;
                    npuN = 0;
                    cpuN = 0;
                    threads = 1;
                } else if (npuK < forK) {
                    if (debug_verbosity >= 1) {
                        std::cout << "GEMV: Enforcing GEMV-Driven K-split: npuK=" << npuK << " (forK=" << forK << ")" << std::endl;
                    }

                    bool has_k = false;
                    if (obj.contains("fw_path")) {
                        std::string fw = obj["fw_path"].get<std::string>();
                        if (fw.find("_K") != std::string::npos) {
                            has_k = true;
                        }
                    }

                    if (!has_k) {
                        if (debug_verbosity >= 2) {
                            std::cout << "Skipping K-split GEMV kernel (missing _K): forM=" << forM << ", npuK=" << npuK << std::endl;
                        }
                        continue;
                    }
                }

                // GEMV validation policy:
                // - npuK == 0: NPU disabled path (must also have npuN == 0).
                // - 0 < npuK < forK: true K-split (must have npuN == forN).
                // - npuK == forK: full-K path (npuN can follow normal M-split policy).
                // - npuK > forK: invalid.
                if (npuK == 0) {
                    if (npuN != 0) {
                        std::cerr << "Invalid GEMV config for layer=" << layer << " (" << forM << "x" << forK << "x" << forN
                                  << "): npuK=0 requires npuN=0." << std::endl;
                        std::exit(1);
                    }
                } else if (npuK < 0 || npuK > forK) {
                    std::cerr << "Invalid GEMV config for layer=" << layer << " (" << forM << "x" << forK << "x" << forN
                              << "): npuK=" << npuK << " must satisfy 0 <= npuK <= forK." << std::endl;
                    std::exit(1);
                } else if (npuK < forK) {
                    if (npuN != forN) {
                        std::cerr << "Invalid GEMV K-split config for layer=" << layer << " (" << forM << "x" << forK << "x" << forN
                                  << "): npuN=" << npuN << " must equal forN=" << forN << " when 0 < npuK < forK." << std::endl;
                        std::exit(1);
                    }
                }

                int num_tiles = 1;
                if (obj.contains("num_tiles"))
                    num_tiles = obj["num_tiles"].get<int64_t>();

                int config = -1;
                if (obj.contains("config"))
                    config = obj["config"].get<int64_t>();

                // PDI/Inst loading logic (simplified reuse of logic above - extracting common parts would be cleaner but copying for
                // now to keep diff small)
                std::string xclbin, inst;
                bool reuse_pdi = false;
                int reused_hw_idx = -1;

                std::tuple<int, std::string, std::string, std::string> cache_key;
                bool can_cache = false;

                // Only load NPU resources if npuN > 0 (otherwise strictly CPU/GPU)
                int hw_idx = -1;
                int inst_idx = -1;

                if (npuN > 0 && (npuN % 2048 == 0)) { // Requirement: positive and multiple of 2048
                    if (obj.contains("xclbin") && obj.contains("inst")) {
                        xclbin = obj["xclbin"].get<std::string>();
                        inst = obj["inst"].get<std::string>();
                    } else if (obj.contains("fw_path") && obj.contains("tile_size") && obj.contains("col") && obj.contains("dtype")) {
                        std::string fw_path = obj["fw_path"].get<std::string>();
                        std::string tile_size = obj["tile_size"].get<std::string>();
                        std::string col = obj["col"].get<std::string>();
                        // For GEMV, allow global override of filename col token (e.g., 4c vs 8c variants).
                        std::string effective_col = gemv_npu_col.empty() ? col : gemv_npu_col;
                        std::string dtype = obj["dtype"].get<std::string>();

                        auto apply_gemv_col_suffix = [&](std::string path_in) {
                            if (!path_in.empty() && path_in.back() == '/')
                                path_in.pop_back();
                            if (!gemv_npu_col.empty())
                                path_in += "_" + gemv_npu_col;
                            if (!path_in.empty() && path_in.back() != '/')
                                path_in += "/";
                            return path_in;
                        };

                        std::string effective_fw_path = apply_gemv_col_suffix(fw_path);

                        // Helper lambda to load a kernel
                        auto load_kernel_variant = [&](int target_M) {
                            // GEMV PDI cache key is keyed by npuK/tile/col/dtype.
                            std::tuple<int, std::string, std::string, std::string> variant_cache_key = {npuK, tile_size, effective_col,
                                                                                                        dtype};
                            bool reuse_pdi_variant = false;
                            int reused_hw_idx_variant = -1;

                            if (minimal_pdi) {
                                if (pdi_cache.count(variant_cache_key)) {
                                    reuse_pdi_variant = true;
                                    reused_hw_idx_variant = pdi_cache[variant_cache_key];
                                }
                            }

                            std::string dims_str = std::to_string(target_M) + "x" + std::to_string(npuK) + "x" + std::to_string(npuN);

                            // Prefer runtime dims directory (1x<npuK>x<npuN>) when present; fallback to configured fw_path.
                            std::vector<std::string> fw_candidates;
                            fw_candidates.push_back(effective_fw_path);

                            std::string fw_no_slash = fw_path;
                            if (!fw_no_slash.empty() && fw_no_slash.back() == '/')
                                fw_no_slash.pop_back();

                            size_t dims_pos = fw_no_slash.find("/1x");
                            if (dims_pos != std::string::npos) {
                                size_t dims_end = fw_no_slash.find('/', dims_pos + 1);
                                if (dims_end != std::string::npos) {
                                    std::string runtime_fw_no_col = fw_no_slash.substr(0, dims_pos + 1) + "1x" + std::to_string(npuK) +
                                                                    "x" + std::to_string(npuN) + fw_no_slash.substr(dims_end);
                                    std::string runtime_effective_fw_path = apply_gemv_col_suffix(runtime_fw_no_col);
                                    if (runtime_effective_fw_path != effective_fw_path) {
                                        fw_candidates.insert(fw_candidates.begin(), runtime_effective_fw_path);
                                    }
                                }
                            }

                            std::string variant_inst;
                            std::string variant_xclbin;
                            bool selected_variant_path = false;
                            for (const auto &fw_candidate : fw_candidates) {
                                std::string cand_inst =
                                    fw_candidate + "insts_" + dims_str + "_" + tile_size + "_" + effective_col + "_" + dtype + ".txt";
                                std::string cand_xclbin =
                                    fw_candidate + "final_" + dims_str + "_" + tile_size + "_" + effective_col + "_" + dtype + ".pdi";
                                std::string cand_inst_abs = root_dir + "/" + cand_inst;
                                std::string cand_xclbin_abs = root_dir + "/" + cand_xclbin;
                                if (std::filesystem::exists(cand_inst_abs) && std::filesystem::exists(cand_xclbin_abs)) {
                                    variant_inst = cand_inst;
                                    variant_xclbin = cand_xclbin;
                                    selected_variant_path = true;
                                    break;
                                }
                            }

                            if (!selected_variant_path) {
                                // Fall back to the first candidate and allow existing create* paths to emit precise errors.
                                variant_inst = fw_candidates.front() + "insts_" + dims_str + "_" + tile_size + "_" + effective_col + "_" +
                                               dtype + ".txt";
                                variant_xclbin = fw_candidates.front() + "final_" + dims_str + "_" + tile_size + "_" + effective_col + "_" +
                                                 dtype + ".pdi";
                                if (debug_verbosity >= 1) {
                                    std::cout << "GEMV variant path probe failed for dims=" << dims_str << ", tried:";
                                    for (const auto &fw_candidate : fw_candidates) {
                                        std::cout << " " << (root_dir + "/" + fw_candidate);
                                    }
                                    std::cout << std::endl;
                                }
                            }

                            // Load HW
                            int hw_idx_v = hw_ctx_count;
                            if (reuse_pdi_variant) {
                                hw_idx_v = reused_hw_idx_variant;
                                if (debug_verbosity >= 1) {
                                    std::cout << "Reusing GEMV PDI from HW[" << hw_idx_v << "] for layer=" << layer << " npuK=" << npuK
                                              << " npuN=" << npuN << std::endl;
                                }
                            } else {
                                std::string xclbin_path = root_dir + "/" + variant_xclbin;
                                if (debug_verbosity >= 1)
                                    std::cout << "Loading GEMV PDI: " << xclbin_path << " PDI_TILES: " << num_tiles << std::endl;
                                if (createHWctxt(drv_fd, hwctxt_array[hw_idx_v], xclbin_path.c_str(), num_tiles) != 0) {
                                    std::cerr << "Failed to create GEMV HW context for " << xclbin_path << std::endl;
                                    return -1;
                                }
                                FlushCpuCache((const void *)hwctxt_array[hw_idx_v].pdi_vaddr, 0, hwctxt_array[hw_idx_v].pdi_size);
                                hw_ctx_count++;

                                if (minimal_pdi)
                                    pdi_cache[variant_cache_key] = hw_idx_v;
                            }

                            // Load Inst
                            int inst_idx_v = inst_ctx_count;
                            std::string inst_path = root_dir + "/" + variant_inst;
                            if (debug_verbosity >= 1)
                                std::cout << "Loading GEMV Inst: " << inst_path << std::endl;
                            if (createInstctxt(drv_fd, instctxt_array[inst_idx_v], inst_path.c_str(), true) != 0) {
                                std::cerr << "Failed to create GEMV Inst context" << std::endl;
                                return -1;
                            }
                            FlushCpuCache((const void *)instctxt_array[inst_idx_v].dpu_0_vaddr, 0,
                                          instctxt_array[inst_idx_v].num_dpu_0_insts * sizeof(uint32_t));
                            inst_ctx_count++;

                            // Map: If primary (target_M == npuM), map to layer 'forM'. If fallback (target_M == 1), map to 1.
                            int key_M = (target_M == npuM) ? forM : 1;

                            int recorded_split_type = 0;
                            // Check if npuK implies a split relative to forK
                            if (npuK < forK) {
                                recorded_split_type = npuK;
                            }

                            config_map[{key_M, forK, forN, layer_id, chunk_id}] = {
                                hw_idx_v, inst_idx_v, target_M, npuK, npuN, cpuN, threads, config, recorded_split_type};

                            if (debug_verbosity >= 1) {
                                std::cout << "Mapped GEMV kernel " << key_M << "x" << forK << "x" << forN << " layer=" << layer << "("
                                          << layer_id << ") -> npuM=" << target_M << " npuK=" << npuK << " npuN=" << npuN
                                          << " config=" << config << " to HW[" << hw_idx_v << "] Inst[" << inst_idx_v
                                          << "] Split=" << recorded_split_type << " chunk_id=" << chunk_id << std::endl;
                            }
                            return 0;
                        };

                        // 1. Load Primary Kernel
                        load_kernel_variant(1);

                        // 2. Load Fallback (M=1) if primary was not 1
                        if (npuM != 1) {
                            std::cerr << "GEMV kernel dims is not npuM == 1" << std::endl;
                            std::exit(0);
                        }
                    } else {
                        std::cerr << "GEMV Kernel missing path info" << std::endl;
                        continue;
                    }
                } else {
                    // npuN == 0 or invalid (not multiple of 2048): Just map the config for CPU/GPU usage
                    // IMPORTANT: Force npuN to 0 in the map so hetero_compute doesn't try to launch with invalid hw_idx
                    if (debug_verbosity >= 1) {
                        std::cout << "Mapped GEMV kernel (Host Only) " << forM << "x" << forK << "x" << forN << " layer=" << layer << "("
                                  << layer_id << ") -> cpuN=" << cpuN << " (npuN forced to 0)" << std::endl;
                    }
                    config_map[{forM, forK, forN, layer_id, chunk_id}] = {-1, -1, npuM, npuK, 0, cpuN, threads, config, 0}; // split_type=0
                }
            }
        } else if (!minimal_pdi && root.contains("kernels_gemv") && debug_verbosity >= 1) {
            std::cout << "Skipping kernels_gemv because minimal_pdi=false" << std::endl;
        }

        // Resize arrays to actual size used
        hwctxt_array.resize(hw_ctx_count);
        instctxt_array.resize(inst_ctx_count);

        if (debug_verbosity >= 1)
            std::cout << "Loaded " << hw_ctx_count << " HW contexts and " << inst_ctx_count << " Inst contexts" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error parsing kernels.json: " << e.what() << std::endl;
    }
}

std::pair<int, int> get_npu_context(int M, int K, int N, const std::string &layer_type) {
    int layer_id = get_layer_id(layer_type);
    NPUKey key = {M, K, N, layer_id, -1};
    auto it = config_map.find(key);
    if (it != config_map.end()) {
        return {it->second.hw_idx, it->second.inst_idx};
    }
    return {-1, -1};
}

PackedWeightPadSpec get_packed_weight_pad_alignment(const std::string &layer_type) {
    if (pad_packed_weights <= 0) {
        return {0, 0};
    }

    auto it = packed_weight_pad_map.find(layer_type);
    if (it != packed_weight_pad_map.end()) {
        return it->second;
    }

    return {pad_packed_weights, pad_packed_weights};
}

// Initialize XDNA driver
int initialize_xdna_driver(const char *drv_path) {
    if (xdna_drv_fd >= 0) {
        // Already initialized
        return 0;
    }
    xdna_drv_fd = open(drv_path, O_RDWR);
    if (xdna_drv_fd < 0) {
        std::cerr << "Failed to open XDNA driver at " << drv_path << ": " << strerror(errno) << std::endl;
        return -1;
    }
    if (debug_verbosity >= 1) {
        std::cout << "XDNA driver opened successfully: " << drv_path << " (fd: " << xdna_drv_fd << ")" << std::endl;
    }

    // Allocate device heap
    if (debug_verbosity >= 1)
        std::cout << "Allocating device heap..." << std::endl;
    allocate_heap_and_error(xdna_drv_fd);

    return 0;
}

#include <mutex>
std::mutex npu_mutex;
std::mutex create_cmd_packet_mutex;

int npuMatmul_zero(int hwctx_numb, int instctx_numb, void *output_pointer, void *input_pointer, void *weight_pointer,
                   __u32 output_xdna_handle, __u32 input_xdna_handle, __u32 weight_xdna_handle, hipEvent_t hip_event) {

    struct amdxdna_drm_exec_cmd exec_cmd;
    // Pass all handles (Instruction, Input, Weight, Output) to ensure residency
    uint32_t bo_args[4] = {instctxt_array[instctx_numb].dpu_0_handle, input_xdna_handle, weight_xdna_handle, output_xdna_handle};

    if (debug_verbosity >= 3)
        std::cout << "Creating NPU command" << std::endl;

    int ret = 0;
    {
        std::lock_guard<std::mutex> create_cmd_lock(create_cmd_packet_mutex);
        ret = create_cmd_packet(xdna_drv_fd, hwctxt_array[hwctx_numb].pdi_handle, instctxt_array[instctx_numb].dpu_0_sram_vaddr,
                                instctxt_array[instctx_numb].dpu_0_handle, instctxt_array[instctx_numb].num_dpu_0_insts,
                                (__u64)input_pointer, (__u64)weight_pointer, (__u64)output_pointer, input_xdna_handle, weight_xdna_handle,
                                output_xdna_handle, hwctxt_array[hwctx_numb].hw_ctx, exec_cmd, bo_args, 4);
    }

    if (ret != 0) {
        perror("Failed to create command packet chain");
        return -1;
    }

    // Wait for input GPU buffers to be ready (Polling loop as requested)
    // The previous atomic check allowed a race; separate polling is safe here because:
    // 1. We wait for THIS thread's input data to be ready on GPU.
    // 2. THEN we lock the NPU to ensure exclusive access for the execution.
    if (hip_event != nullptr) {
        for (;;) {
            hipError_t query_status = hipEventQuery(hip_event);
            if (query_status == hipSuccess) {
                break;
            }
            if (query_status != hipErrorNotReady) {
                HIP_CHECK(query_status);
            }
            // Spin-wait / Poll until GPU is ready
            // (User requested avoiding heavy hipEventSynchronize)
        }
        if (debug_verbosity >= 3) {
            std::cout << "Hip Event Sync Success" << std::endl;
        }
    }

    // Lock the NPU for execution
    // EXPLANATION: std::lock_guard acquires 'npu_mutex'.
    // If another thread holds it, this thread will BLOCK (sleep/wait) until it is released.
    // It does NOT strictly spin-wait (burn CPU) unless the mutex implementation decides to spin briefly.
    std::lock_guard<std::mutex> lock(npu_mutex);

    if (debug_verbosity >= 2)
        std::cout << "Executing NPU command" << std::endl;

    ret = ioctl(xdna_drv_fd, DRM_IOCTL_AMDXDNA_EXEC_CMD, &exec_cmd);
    // Execute the command
    if (ret != 0) {
        perror("Failed to submit work");
        return -1;
    }

    // Wait for the command to complete
    struct amdxdna_drm_wait_cmd wait_cmd = {
        .ctx = hwctxt_array[hwctx_numb].hw_ctx.handle,
        .timeout = 500, // 50ms timeout
        .seq = exec_cmd.seq,
    };

    if (debug_verbosity >= 3)
        std::cout << "Waiting for NPU" << std::endl;

    ret = ioctl(xdna_drv_fd, DRM_IOCTL_AMDXDNA_WAIT_CMD, &wait_cmd);
    if (ret != 0) {
        perror("Failed to wait");
        return -1;
    }

    // lock_guard destructor is called HERE automatically, releasing npu_mutex.
    return 0;
}

// Import DMA-BUF to XDNA
uint32_t import_dma_buf_to_xdna(void *hip_managed_ptr, size_t size, int dataTypeinBytes) {
    // Check if XDNA driver is initialized
    if (xdna_drv_fd < 0) {
        std::cerr << "XDNA driver not initialized. Call initialize_xdna_driver() first." << std::endl;
        return 0;
    }

    // Check cache first
    if (ptr_to_handle_map.find(hip_managed_ptr) != ptr_to_handle_map.end()) {
        if (debug_verbosity >= 1)
            std::cout << "Using cached handle for ptr: " << hip_managed_ptr << " handle: " << ptr_to_handle_map[hip_managed_ptr]
                      << std::endl;
        return ptr_to_handle_map[hip_managed_ptr];
    }

    // std::cout << "Importing pointer to XDNA: " << hip_managed_ptr << std::endl;

    int dmabuf_fd;
    uint64_t offset;
    hsa_status_t status = hsa_amd_portable_export_dmabuf(hip_managed_ptr, size * dataTypeinBytes, &dmabuf_fd, &offset);
    if (status != HSA_STATUS_SUCCESS) {
        std::cerr << "Failed to export DMA-BUF. Status: " << status << std::endl;
        return 0;
    }

    // std::cout << "DMA-BUF export successful. FD: " << dmabuf_fd << ", Offset: " << offset << " "
    // << hip_managed_ptr << std::endl;

    if (dmabuf_fd < 0) {
        if (debug_verbosity >= 1)
            std::cout << "Invalid DMA-BUF FD: " << dmabuf_fd << std::endl;
        return 0;
    }

    drm_prime_handle prime_params;
    prime_params.handle = 0;
    prime_params.flags = 0;
    prime_params.fd = dmabuf_fd;

    if (ioctl(xdna_drv_fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &prime_params) < 0) {
        std::cerr << "Failed to import DMA-BUF: " << strerror(errno) << " (errno=" << errno << ")" << std::endl;
        std::cerr << "xdna_drv_fd=" << xdna_drv_fd << ", dmabuf_fd=" << dmabuf_fd << std::endl;
        close(dmabuf_fd); // Close the DMA-BUF FD on error
        return 0;
    }

    // std::cout << "Successfully imported DMA-BUF. Handle: " << prime_params.handle << std::endl;

    // Dummy operation to prevent compiler optimizations
    if (dataTypeinBytes == 4) {
        volatile float dummy_buffer = 0;
        for (int i = 0; i < size; i += IOMMU_STRIDE) {
            dummy_buffer += reinterpret_cast<float *>(hip_managed_ptr)[i];
        }
    } else if (dataTypeinBytes == 2) {
        volatile myBfloat dummy_buffer = 0;
        for (int i = 0; i < size; i += IOMMU_STRIDE) {
            dummy_buffer += reinterpret_cast<myBfloat *>(hip_managed_ptr)[i];
        }
    } else if (dataTypeinBytes == 1) {
        volatile uint8_t dummy_buffer = 0;
        for (int i = 0; i < size; i += IOMMU_STRIDE) {
            dummy_buffer += reinterpret_cast<uint8_t *>(hip_managed_ptr)[i];
        }
    } else {
        std::cerr << "Invalid data type size: " << dataTypeinBytes << std::endl;
        return 0;
    }

    // mmap the buffer in XDNA
    struct amdxdna_drm_get_bo_info get_bo_info = {.handle = prime_params.handle};
    int ret = ioctl(xdna_drv_fd, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &get_bo_info);
    if (ret != 0) {
        perror("Failed to get BO info: ");
        return -2;
    }

    // Cache the handle
    ptr_to_handle_map[hip_managed_ptr] = prime_params.handle;
    // std::cout << "Cached handle " << prime_params.handle << " for ptr: " << hip_managed_ptr << " (map size: " <<
    // ptr_to_handle_map.size() << ")" << std::endl;

    // Return the imported handle
    return prime_params.handle;
}

// Import all module params/buffers to XDNA
void import_all_weights_to_xdna(torch::nn::Module &module) {
    if (debug_verbosity >= 1)
        std::cout << "Starting to import all weights and parameters to XDNA..." << std::endl;

    size_t imported_params = 0;
    size_t failed_params = 0;
    size_t skipped_params = 0;
    size_t imported_buffers = 0;
    size_t failed_buffers = 0;
    size_t skipped_buffers = 0;

    // Iterate weights (including sub-modules)
    auto named_parameters = module.named_parameters(true);
    for (const auto &param : named_parameters) {
        const std::string &name = param.key();
        const torch::Tensor &tensor = param.value();

        // Skip if tensor is empty
        if (tensor.numel() == 0) {
            if (debug_verbosity >= 1)
                std::cout << "Skipping empty parameter: " << name << std::endl;
            skipped_params++;
            continue;
        }

        // Check if tensor is on HIP/CUDA device (required for DMA-BUF export)
        if (!tensor.device().is_cuda() && !tensor.device().is_hip()) {
            if (debug_verbosity >= 1) {
                std::cout << "Skipping parameter not on HIP/CUDA device: " << name << " (device: " << tensor.device() << ")" << std::endl;
            }
            skipped_params++;
            continue;
        }

        // Get tensor data pointer
        void *data_ptr = tensor.data_ptr();
        size_t numel = tensor.numel();

        // Determine data type size in bytes
        int dtype_bytes = 0;
        if (tensor.dtype() == torch::kFloat32) {
            dtype_bytes = 4;
        } else if (tensor.dtype() == torch::kBFloat16 || tensor.dtype() == torch::kFloat16) {
            dtype_bytes = 2;
        } else if (tensor.dtype() == torch::kUInt8 || tensor.dtype() == torch::kInt8) {
            dtype_bytes = 1;
        } else {
            std::cerr << "Unsupported dtype for parameter: " << name << " (dtype: " << tensor.dtype() << ")" << std::endl;
            skipped_params++;
            continue;
        }

        // std::cout << "Importing parameter: " << name << " (size: " << numel
        //           << ", dtype_bytes: " << dtype_bytes << ")" << std::endl;

        // Import to XDNA
        uint32_t handle = import_dma_buf_to_xdna(data_ptr, numel, dtype_bytes);
        if (handle == 0 || handle == static_cast<uint32_t>(-2)) {
            std::cerr << "Failed to import parameter to XDNA: " << name << std::endl;
            failed_params++;
        } else {
            imported_params++;
            if (debug_verbosity >= 1) {
                std::cout << "Successfully imported parameter to XDNA: " << name << " (handle: " << handle << ", numel: " << numel
                          << ", dtype_bytes: " << dtype_bytes << ")" << std::endl;
            }
        }
    }

    // Iterate buffers
    if (debug_verbosity >= 1)
        std::cout << "\nStarting to import all buffers to XDNA..." << std::endl;
    auto named_buffers = module.named_buffers(true);
    for (const auto &buf : named_buffers) {
        const std::string &name = buf.key();
        const torch::Tensor &tensor = buf.value();

        // KV caches remain GPU-resident and do not need XDNA handles.
        // Skipping them avoids spending DMA-BUF export budget on large,
        // per-layer cache tensors.
        const bool is_kv_cache =
            (name.rfind("cache_k_", 0) == 0) || (name.rfind("cache_v_", 0) == 0) ||
            (name.find(".cache_k_") != std::string::npos) || (name.find(".cache_v_") != std::string::npos);
        if (is_kv_cache) {
            if (debug_verbosity >= 2) {
                std::cout << "Skipping KV cache buffer import: " << name << std::endl;
            }
            skipped_buffers++;
            continue;
        }

        // Split-K staging buffers (input_buf_0 and slot variants) are optional for NPU:
        // split-K can use full-input handles instead. Skipping eager import avoids
        // DMA-BUF/handle exhaustion on deep models.
        if (name.find(".input_buf_0") != std::string::npos) {
            if (debug_verbosity >= 2) {
                std::cout << "Skipping optional split-K staging buffer import: " << name << std::endl;
            }
            skipped_buffers++;
            continue;
        }

        // Skip if tensor is empty
        if (tensor.numel() == 0) {
            if (debug_verbosity >= 1)
                std::cout << "Skipping empty buffer: " << name << std::endl;
            skipped_buffers++;
            continue;
        }

        // Check if tensor is on HIP/CUDA device (required for DMA-BUF export)
        if (!tensor.device().is_cuda() && !tensor.device().is_hip()) {
            if (debug_verbosity >= 1) {
                std::cout << "Skipping buffer not on HIP/CUDA device: " << name << " (device: " << tensor.device() << ")" << std::endl;
            }
            skipped_buffers++;
            continue;
        }

        // Get tensor data pointer
        void *data_ptr = tensor.data_ptr();
        size_t numel = tensor.numel();

        // Determine data type size in bytes
        int dtype_bytes = 0;
        if (tensor.dtype() == torch::kFloat32) {
            dtype_bytes = 4;
        } else if (tensor.dtype() == torch::kBFloat16 || tensor.dtype() == torch::kFloat16) {
            dtype_bytes = 2;
        } else if (tensor.dtype() == torch::kUInt8 || tensor.dtype() == torch::kInt8) {
            dtype_bytes = 1;
        } else {
            std::cerr << "Unsupported dtype for buffer: " << name << " (dtype: " << tensor.dtype() << ")" << std::endl;
            skipped_buffers++;
            continue;
        }

        // Import to XDNA
        uint32_t handle = import_dma_buf_to_xdna(data_ptr, numel, dtype_bytes);
        if (handle == 0 || handle == static_cast<uint32_t>(-2)) {
            std::cerr << "Failed to import buffer to XDNA: " << name << std::endl;
            failed_buffers++;
        } else {
            imported_buffers++;
            if (debug_verbosity >= 1) {
                std::cout << "Successfully imported buffer to XDNA: " << name << " (handle: " << handle << ", numel: " << numel
                          << ", dtype_bytes: " << dtype_bytes << ")" << std::endl;
            }
        }
    }

    if (debug_verbosity >= 1)
        std::cout << "Finished importing all weights and parameters to XDNA." << std::endl;
    if (debug_verbosity >= 1) {
        std::cout << "XDNA import summary: params imported=" << imported_params << ", failed=" << failed_params
                  << ", skipped=" << skipped_params << std::endl;
        std::cout << "XDNA import summary: buffers imported=" << imported_buffers << ", failed=" << failed_buffers
                  << ", skipped=" << skipped_buffers << std::endl;
    }
}

// Load NPU-only kernels (strict mode for "npu" target)
void load_npu_only_kernels(int drv_fd, const std::string &config_path) {
    std::string json_path = config_path;
    if (json_path.empty()) {
        json_path = "src/unified_llm_w4a16_hetero/kernels.json";
    }

    if (!std::filesystem::exists(json_path)) {
        if (config_path.empty()) {
            json_path = "/home/greg/Desktop/heteroMosaic/src/unified_llm_w4a16_hetero/kernels.json";
        }
    }

    // Root dir for paths
    std::string root_dir;
    const char *env_root = std::getenv("HETEROMOSAIC_ROOT");
    if (env_root) {
        root_dir = std::string(env_root);
    } else {
        std::cerr << "Warning: HETEROMOSAIC_ROOT not set. Using default." << std::endl;
        root_dir = "/home/greg/Desktop/heteroMosaic";
    }

    std::ifstream f(json_path);
    if (!f.is_open()) {
        std::cerr << "Failed to open kernels.json at " << json_path << std::endl;
        return;
    }

    std::stringstream buffer;
    buffer << f.rdbuf();
    std::string json_str = removeComments(buffer.str());

    // Track contexts (separate from standard lists, but using the same global arrays)
    int hw_ctx_count = 0;
    int inst_ctx_count = 0;

    // We assume init_npu() has been called to resize vectors

    try {
        json::json jv = json::json::parse(json_str);
        json::json &root = jv;

        if (root.contains("heterogeneity") && root["heterogeneity"].is_string()) {
            hw_target = root["heterogeneity"].get<std::string>();
        }

        // Get npuOnlydefault array
        if (!root.contains("npuOnlydefault")) {
            std::cerr << "Error: npuOnlydefault not found in config for NPU-only mode" << std::endl;
            return;
        }

        json::json &configs = root["npuOnlydefault"];

        // Cache for PDI reuse
        std::map<std::tuple<int, std::string, std::string, std::string>, int> pdi_cache;

        for (auto &entry_val : configs) {
            json::json &entry = entry_val;

            std::string fw_path = entry["fw_path"].get<std::string>();
            if (fw_path.back() != '/')
                fw_path += "/";

            std::string tile_size = entry["tile_size"].get<std::string>();
            std::string col = entry["col"].get<std::string>();
            std::string dtype = entry["dtype"].get<std::string>();

            const double emulated_tops = get_optional_double(entry, {"tops"}, -1.0);
            const bool emulate_tops = (hw_target == "npu-sim" && emulated_tops >= 0.0);
            const int emulated_gops =
                emulate_tops ? std::max(1, static_cast<int>(std::llround(emulated_tops * 1000.0))) : -1;
            if (emulate_tops && debug_verbosity >= 1) {
                std::cout << "NPU TOPS emulation enabled: " << emulated_tops << " TOPS (" << emulated_gops << " GOPS)" << std::endl;
            }

            int num_tiles = 1;
            if (entry.contains("num_tiles")) {
                num_tiles = entry["num_tiles"].get<int64_t>();
            }

            // Layers to process
            // Read max_ctx_len if available
            int max_ctx_len = -1;
            if (entry.contains("max_ctx_len")) {
                max_ctx_len = entry["max_ctx_len"].get<int64_t>();
            }

            std::vector<std::string> groups = {"qkv", "o", "qo", "kv", "upgate", "down"};

            for (const auto &grp : groups) {
                if (!entry.contains(grp))
                    continue;

                PackedWeightPadSpec pad_spec = {0, 0};
                int K = 0;
                int N = 0;
                int gops = 0;
                if (!parse_kernel_dims_entry(entry[grp], K, N, gops, &pad_spec)) {
                    std::cerr << "Error: invalid npuOnlydefault entry for group '" << grp << "'" << std::endl;
                    continue;
                }
                if (emulate_tops) {
                    gops = emulated_gops;
                }
                if (pad_packed_weights > 0) {
                    K = static_cast<int>(round_up_to_alignment(K, pad_spec[0]));
                    N = static_cast<int>(round_up_to_alignment(N, pad_spec[1]));
                }
                int M = npu_dim; // Use global npu_dim

                if (M <= 0) {
                    std::cerr << "Error: npu_dim not set!" << std::endl;
                    continue;
                }

                // Construct paths
                // Filename uses current M (npu_dim)
                std::string file_dims_str = std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N);

                // Folder uses max_ctx_len if set, otherwise M
                int folder_M = (max_ctx_len > 0) ? max_ctx_len : M;
                std::string folder_dims_str = std::to_string(folder_M) + "x" + std::to_string(K) + "x" + std::to_string(N);

                // Construct intermediate folder path
                const bool is_special_16k_npu_case = max_ctx_len == 16384;
                const bool use_down_k_suffix = (grp == "down") && !is_special_16k_npu_case && !split_M_only;
                std::string group_dtype_folder = dtype + (use_down_k_suffix ? "_K" : "_M");
                std::string bin_folder = fw_path + folder_dims_str + "/" + group_dtype_folder + "/";

                // Filenames
                std::string filename_suffix = "_" + file_dims_str + "_" + tile_size + "_" + col + "_" + dtype;
                std::string inst_file = bin_folder + "insts" + filename_suffix + ".txt";
                std::string xclbin_file = bin_folder + "final" + filename_suffix + ".pdi";

                bool reuse_pdi = false;
                int reused_hw_idx = -1;
                std::tuple<int, std::string, std::string, std::string> cache_key = {K, tile_size, col, group_dtype_folder};

                int hw_idx = hw_ctx_count;
                int inst_idx = inst_ctx_count;

                // Simulation Mode Check
                if (gops > 0) {
                    if (debug_verbosity >= 1) {
                        std::cout << "Simulation Mode (GOPS=" << gops << "): Skipping PDI/Inst load for " << grp << std::endl;
                    }
                    hw_idx = -1;
                    inst_idx = -1;
                } else {
                    // Normal Hardware Loading Path
                    if (minimal_pdi) {
                        if (pdi_cache.count(cache_key)) {
                            reuse_pdi = true;
                            reused_hw_idx = pdi_cache[cache_key];
                            if (debug_verbosity >= 1)
                                std::cout << "Reusing PDI for group " << grp << " (hw_idx=" << reused_hw_idx << ")" << std::endl;
                        }
                    }

                    if (reuse_pdi) {
                        hw_idx = reused_hw_idx;
                    } else {
                        std::string xclbin_path = root_dir + "/" + xclbin_file;
                        if (debug_verbosity >= 1)
                            std::cout << "Loading PDI: " << xclbin_path << " with num_tiles=" << num_tiles << std::endl;

                        if (createHWctxt(drv_fd, hwctxt_array[hw_idx], xclbin_path.c_str(), num_tiles) != 0) {
                            std::cerr << "Failed to create HW context for " << xclbin_path << std::endl;
                            continue;
                        }
                        FlushCpuCache((const void *)hwctxt_array[hw_idx].pdi_vaddr, 0, hwctxt_array[hw_idx].pdi_size);
                        hw_ctx_count++;

                        if (minimal_pdi) {
                            pdi_cache[cache_key] = hw_idx;
                        }
                    }

                    // Load Inst

                    std::string inst_path = root_dir + "/" + inst_file;
                    if (debug_verbosity >= 1)
                        std::cout << "Loading Instructions: " << inst_path << std::endl;

                    if (createInstctxt(drv_fd, instctxt_array[inst_idx], inst_path.c_str(), true) != 0) {
                        std::cerr << "Failed to create Inst context for " << inst_path << std::endl;
                        continue;
                    }
                    FlushCpuCache((const void *)instctxt_array[inst_idx].dpu_0_vaddr, 0,
                                  instctxt_array[inst_idx].num_dpu_0_insts * sizeof(uint32_t));
                    inst_ctx_count++;
                } // End of Normal Hardware Loading Path

                // Map layers
                if (grp == "qkv") {
                    NPUKey key_qkv = {M, K, N, 8, -1};
                    config_map[key_qkv] = {hw_idx, inst_idx, M, K, N, -1, 1, gops, 0};

                    if (debug_verbosity >= 1) {
                        std::cout << "Mapped kernel " << M << "x" << K << "x" << N << " group=" << grp << " -> npuM=" << M << " npuK=" << K
                                  << " npuN=" << N << " to HW[" << hw_idx << "] Inst[" << inst_idx << "]" << std::endl;
                    }

                } else if (grp == "o") {
                    NPUKey key_o = {M, K, N, 4, -1};
                    config_map[key_o] = {hw_idx, inst_idx, M, K, N, -1, 1, gops, 0};

                    if (debug_verbosity >= 1) {
                        std::cout << "Mapped kernel " << M << "x" << K << "x" << N << " group=" << grp << " -> npuM=" << M << " npuK=" << K
                                  << " npuN=" << N << " to HW[" << hw_idx << "] Inst[" << inst_idx << "]" << std::endl;
                    }

                } else if (grp == "qo") {
                    // q=1, o=4
                    NPUKey key_q = {M, K, N, 1, -1};
                    config_map[key_q] = {hw_idx, inst_idx, M, K, N, -1, 1, gops, 0};
                    NPUKey key_o = {M, K, N, 4, -1};
                    config_map[key_o] = {hw_idx, inst_idx, M, K, N, -1, 1, gops, 0};

                    if (debug_verbosity >= 1) {
                        std::cout << "Mapped kernel " << M << "x" << K << "x" << N << " group=" << grp << " -> npuM=" << M << " npuK=" << K
                                  << " npuN=" << N << " to HW[" << hw_idx << "] Inst[" << inst_idx << "]" << std::endl;
                    }

                } else if (grp == "kv") {
                    // k=2, v=3
                    NPUKey key_k = {M, K, N, 2, -1};
                    config_map[key_k] = {hw_idx, inst_idx, M, K, N, -1, 1, gops, 0};
                    NPUKey key_v = {M, K, N, 3, -1};
                    config_map[key_v] = {hw_idx, inst_idx, M, K, N, -1, 1, gops, 0};

                    if (debug_verbosity >= 1) {
                        std::cout << "Mapped kernel " << M << "x" << K << "x" << N << " group=" << grp << " -> npuM=" << M << " npuK=" << K
                                  << " npuN=" << N << " to HW[" << hw_idx << "] Inst[" << inst_idx << "]" << std::endl;
                    }

                } else if (grp == "upgate") {
                    // gate=5, up=6
                    NPUKey key_g = {M, K, N, 5, -1};
                    config_map[key_g] = {hw_idx, inst_idx, M, K, N, -1, 1, gops, 0};
                    NPUKey key_u = {M, K, N, 6, -1};
                    config_map[key_u] = {hw_idx, inst_idx, M, K, N, -1, 1, gops, 0};

                    if (debug_verbosity >= 1) {
                        std::cout << "Mapped kernel " << M << "x" << K << "x" << N << " group=" << grp << " -> npuM=" << M << " npuK=" << K
                                  << " npuN=" << N << " to HW[" << hw_idx << "] Inst[" << inst_idx << "]" << std::endl;
                    }

                } else if (grp == "down") {
                    // down=7
                    NPUKey key_d = {M, K, N, 7, -1};
                    config_map[key_d] = {hw_idx, inst_idx, M, K, N, -1, 1, gops};

                    if (debug_verbosity >= 1) {
                        std::cout << "Mapped kernel " << M << "x" << K << "x" << N << " group=" << grp << " -> npuM=" << M << " npuK=" << K
                                  << " npuN=" << N << " to HW[" << hw_idx << "] Inst[" << inst_idx << "]" << std::endl;
                    }
                }
            }

            // --- Special Handling for Generation (M=1) Kernels (Added) ---

            std::vector<std::string> gen_groups = {"qkv-gen", "o-gen", "qo-gen", "kv-gen", "upgate-gen", "down-gen"};
            for (const auto &grp : gen_groups) {
                // Only load explicit *-gen entries. If a config omits or comments out
                // a generation kernel, treat that as disabled instead of inferring it
                // from the corresponding prefill group.
                if (!entry.contains(grp))
                    continue;

                const json::json *group_value = &entry[grp];

                if (group_value == nullptr)
                    continue;

                PackedWeightPadSpec pad_spec = {0, 0};
                int K = 0;
                int N = 0;
                int gops = 0;
                if (!parse_kernel_dims_entry(*group_value, K, N, gops, &pad_spec)) {
                    std::cerr << "Error: invalid generation kernel entry for group '" << grp << "'" << std::endl;
                    continue;
                }
                if (emulate_tops) {
                    gops = emulated_gops;
                }

                if (pad_packed_weights > 0) {
                    K = static_cast<int>(round_up_to_alignment(K, pad_spec[0]));
                    N = static_cast<int>(round_up_to_alignment(N, pad_spec[1]));
                }
                int M = 1; // M=1 for generation

                std::vector<int> layers;
                if (grp == "qkv-gen")
                    layers = {8}; // fused QKV
                else if (grp == "o-gen")
                    layers = {4}; // O
                else if (grp == "qo-gen")
                    layers = {1, 4}; // Q, O
                else if (grp == "kv-gen")
                    layers = {2, 3}; // K, V
                else if (grp == "upgate-gen")
                    layers = {5, 6}; // Gate, Up
                else if (grp == "down-gen")
                    layers = {7}; // Down

                // Use paths derived from dims
                std::string dims_str = "1x" + std::to_string(K) + "x" + std::to_string(N);
                // Reuse entry-level fw_path/col with per-group dtype suffix.
                // Gen kernels use "128x64" tile size in filename, while config
                // has "64x128x64". We force tile_size for Gen kernels.
                std::string gen_tile_size = "128x64";

                const bool use_down_gen_k_suffix = (grp == "down-gen") && !split_M_only;
                std::string group_dtype_folder = dtype + (use_down_gen_k_suffix ? "_K" : "_M");
                std::string bin_folder = fw_path + dims_str + "/" + group_dtype_folder + "/";
                std::string filename_suffix = "_" + dims_str + "_" + gen_tile_size + "_" + col + "_" + dtype;
                std::string inst_file = bin_folder + "insts" + filename_suffix + ".txt";
                std::string xclbin_file = bin_folder + "final" + filename_suffix + ".pdi";

                // PDI Reuse Logic (Added)
                bool reuse_pdi = false;
                int reused_hw_idx = -1;
                // Use gen_tile_size in key to distinguish from prefill kernels
                std::tuple<int, std::string, std::string, std::string> cache_key = {K, gen_tile_size, col, group_dtype_folder};

                // K-Split Detection
                int split_type_val = 1;
                if (npu_dim > 0 && K > npu_dim) {
                    split_type_val = 2; // K-split
                }

                // Load HW
                int hw_idx = hw_ctx_count;
                int inst_idx = inst_ctx_count; // Define here for scope

                if (gops > 0 || split_type_val == 2) {
                    if (gops > 0 && debug_verbosity >= 1)
                        std::cout << "Simulation Mode (GOPS=" << gops << "): Skipping Gen PDI/Inst load for " << grp << std::endl;

                    if (split_type_val == 2 && debug_verbosity >= 1)
                        std::cout << "K-Split Kernel Detect (K=" << K << " > npu_dim=" << npu_dim << "): Skipping NPU HW Load for " << grp
                                  << " (Marking as split_type=2)" << std::endl;

                    hw_idx = -1;
                    inst_idx = -1;
                } else {
                    // Normal Gen HW Loading
                    if (minimal_pdi) {
                        if (pdi_cache.count(cache_key)) {
                            reuse_pdi = true;
                            reused_hw_idx = pdi_cache[cache_key];
                            if (debug_verbosity >= 1)
                                std::cout << "Reusing Gen PDI for group " << grp << " (hw_idx=" << reused_hw_idx << ")" << std::endl;
                        }
                    }

                    // Load HW

                    std::string full_xclbin_path = root_dir + "/" + xclbin_file;
                    std::string full_inst_path = root_dir + "/" + inst_file;

                    if (reuse_pdi) {
                        hw_idx = reused_hw_idx;
                    } else {
                        if (debug_verbosity >= 1)
                            std::cout << "Loading Gen PDI: " << full_xclbin_path << std::endl;

                        if (createHWctxt(drv_fd, hwctxt_array[hw_idx], full_xclbin_path.c_str(), 32) != 0) {
                            if (debug_verbosity >= 2)
                                std::cerr << "Failed to create Gen HW context for " << grp << " (" << full_xclbin_path << ")" << std::endl;
                            continue;
                        }
                        FlushCpuCache((const void *)hwctxt_array[hw_idx].pdi_vaddr, 0, hwctxt_array[hw_idx].pdi_size);
                        hw_ctx_count++;

                        if (minimal_pdi) {
                            pdi_cache[cache_key] = hw_idx;
                        }
                    }

                    // Load Inst
                    if (debug_verbosity >= 1)
                        std::cout << "Loading Gen Instructions: " << full_inst_path << std::endl;

                    if (createInstctxt(drv_fd, instctxt_array[inst_idx], full_inst_path.c_str(), true) != 0) {
                        std::cerr << "Failed to create Gen Inst context for " << grp << std::endl;
                        continue;
                    }
                    FlushCpuCache((const void *)instctxt_array[inst_idx].dpu_0_vaddr, 0,
                                  instctxt_array[inst_idx].num_dpu_0_insts * sizeof(uint32_t));
                    inst_ctx_count++;
                } // End Gen Hardware Load

                // Map
                for (int layer_id : layers) {
                    config_map[{1, K, N, layer_id, -1}] = {hw_idx, inst_idx, 1, K, N, -1, split_type_val, gops};
                }

                if (debug_verbosity >= 1) {
                    std::cout << "Loaded Gen Kernel: " << grp << " (" << dims_str << ") layers: " << layers.size() << std::endl;
                    std::cout << "Loaded Gen Kernel: " << grp << " (" << dims_str << ") layers: " << layers.size() << " to HW[" << hw_idx
                              << "] Inst[" << inst_idx << "]" << std::endl;
                }
            }
        }

        // Resize
        hwctxt_array.resize(hw_ctx_count);
        instctxt_array.resize(inst_ctx_count);

        if (debug_verbosity >= 1)
            std::cout << "Loaded NPU-only context: " << hw_ctx_count << " HW, " << inst_ctx_count << " Inst" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "Error parsing config for NPU-only: " << e.what() << std::endl;
    }
}
