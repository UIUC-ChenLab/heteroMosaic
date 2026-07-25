#include "unified_llm_w4a16_hetero/helper.hpp"
#include "unified_llm_w4a16_hetero/npuSetup.hpp"
#include "unified_llm_w4a16_hetero/unified_llm_w4a16.hpp"
#include "hipkernels/softmax.hpp"

#include <algorithm>
#include <c10/hip/HIPStream.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <map>
#include <omp.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

// Helper function to sample from logits
int64_t sample_token(const torch::Tensor &logits, float temperature, float top_p, int64_t top_k) {
    // Operate in logits dtype and avoid unnecessary full-vocab allocations.
    torch::Tensor scores = logits;
    if (temperature != 1.0f) {
        scores = scores / temperature;
    }

    auto softmax_1d = [&](const torch::Tensor &x) -> torch::Tensor {
        // Keep custom kernel focused on small vectors where launch overhead stays low.
        if (x.is_cuda() && x.is_contiguous() && x.dim() == 1 && x.numel() <= 16384) {
            auto out = torch::empty_like(x);
            if (hipkernels::launch_softmax_1d(x, out, c10::hip::getCurrentHIPStream().stream())) {
                return out;
            }
        }
        return torch::softmax(x, 0);
    };

    // Candidate map from local sampling indices to original vocab ids.
    torch::Tensor candidate_indices;
    bool has_candidate_map = false;

    // top-k first: this shrinks work for top-p + softmax + multinomial.
    const int64_t vocab_size = scores.size(0);
    if (top_k > 0 && top_k < vocab_size) {
        auto topk_result = torch::topk(scores, top_k, 0, true, true);
        scores = std::get<0>(topk_result).contiguous();
        candidate_indices = std::get<1>(topk_result);
        has_candidate_map = true;
    }

    int64_t sampled_local_idx = -1;
    if (top_p < 1.0f) {
        auto sorted_result = torch::sort(scores, 0, true);
        torch::Tensor sorted_scores = std::get<0>(sorted_result).contiguous();
        torch::Tensor sorted_indices = std::get<1>(sorted_result);

        // Compute cumulative probability over sorted distribution.
        torch::Tensor sorted_probs = softmax_1d(sorted_scores);
        torch::Tensor cumulative_probs = torch::cumsum(sorted_probs, 0);

        auto sorted_indices_to_remove = cumulative_probs > top_p;
        // Keep at least one token (the highest-probability one).
        if (sorted_indices_to_remove.numel() > 1) {
            sorted_indices_to_remove.slice(0, 1).copy_(sorted_indices_to_remove.slice(0, 0, -1).clone());
        }
        if (sorted_indices_to_remove.numel() > 0) {
            sorted_indices_to_remove.index_put_({0}, false);
        }

        sorted_scores = sorted_scores.masked_fill(sorted_indices_to_remove, -std::numeric_limits<float>::infinity());
        torch::Tensor filtered_probs = softmax_1d(sorted_scores);
        int64_t sampled_sorted_idx = torch::multinomial(filtered_probs, 1).item<int64_t>();
        sampled_local_idx = sorted_indices.index({sampled_sorted_idx}).item<int64_t>();
    } else {
        torch::Tensor probs = softmax_1d(scores);
        sampled_local_idx = torch::multinomial(probs, 1).item<int64_t>();
    }

    if (has_candidate_map) {
        return candidate_indices.index({sampled_local_idx}).item<int64_t>();
    }
    return sampled_local_idx;
}

// Parse safetensors header and extract tensor metadata
extern bool use_packed_weights;

// Parse safetensors header from mapped memory
std::map<std::string, SafetensorsTensorInfo> parse_safetensors_header_from_map(const char *data_ptr, uint64_t file_size,
                                                                               uint64_t &header_size) {
    std::map<std::string, SafetensorsTensorInfo> tensor_map;

    if (file_size < 8)
        return tensor_map;

    // Read header size (first 8 bytes)
    uint64_t header_size_le = *reinterpret_cast<const uint64_t *>(data_ptr);
    header_size = header_size_le;

    if (file_size < 8 + header_size)
        return tensor_map;

    // Parse JSON header
    std::string header(data_ptr + 8, header_size);

    // Simple JSON parsing for safetensors format
    size_t pos = 0;
    while (pos < header.size()) {
        // Find next tensor name
        size_t name_start = header.find('"', pos);
        if (name_start == std::string::npos)
            break;
        size_t name_end = header.find('"', name_start + 1);
        if (name_end == std::string::npos)
            break;
        std::string tensor_name = header.substr(name_start + 1, name_end - name_start - 1);

        // Find the object for this tensor
        size_t obj_start = header.find('{', name_end);
        if (obj_start == std::string::npos)
            break;
        size_t obj_end = obj_start + 1;
        int depth = 1;
        while (obj_end < header.size() && depth > 0) {
            if (header[obj_end] == '{')
                depth++;
            else if (header[obj_end] == '}')
                depth--;
            obj_end++;
        }

        std::string obj = header.substr(obj_start, obj_end - obj_start);
        SafetensorsTensorInfo info;

        // Parse dtype
        size_t dtype_pos = obj.find("\"dtype\"");
        if (dtype_pos != std::string::npos) {
            size_t dtype_start = obj.find('"', dtype_pos + 7);
            size_t dtype_end = obj.find('"', dtype_start + 1);
            if (dtype_end != std::string::npos) {
                info.dtype = obj.substr(dtype_start + 1, dtype_end - dtype_start - 1);
            }
        }

        // Parse shape
        size_t shape_pos = obj.find("\"shape\"");
        if (shape_pos != std::string::npos) {
            size_t lb = obj.find('[', shape_pos);
            size_t rb = obj.find(']', lb);
            if (lb != std::string::npos && rb != std::string::npos) {
                std::string shape_str = obj.substr(lb + 1, rb - lb - 1);
                std::istringstream iss(shape_str);
                int64_t val;
                while (iss >> val) {
                    info.shape.push_back(val);
                    if (iss.peek() == ',')
                        iss.ignore();
                }
            }
        }

        // Parse data_offsets
        size_t offset_pos = obj.find("\"data_offsets\"");
        if (offset_pos != std::string::npos) {
            size_t lb = obj.find('[', offset_pos);
            size_t rb = obj.find(']', lb);
            if (lb != std::string::npos && rb != std::string::npos) {
                std::string offset_str = obj.substr(lb + 1, rb - lb - 1);
                std::istringstream iss(offset_str);
                uint64_t offset1, offset2;
                if (iss >> offset1 && iss.peek() == ',' && (iss.ignore(), iss >> offset2)) {
                    info.offset_begin = offset1;
                    info.offset_end = offset2;
                }
            }
        }

        if (!info.dtype.empty() && !info.shape.empty() && info.offset_end > info.offset_begin) {
            info.valid = true;
            tensor_map[tensor_name] = info;
        }

        pos = obj_end;
    }

    return tensor_map;
}

// Load a tensor from memory mapped pointer
torch::Tensor load_tensor_from_ptr(const char *data_ptr, const SafetensorsTensorInfo &info, uint64_t header_size) {
    // Calculate absolute offset in mmap
    uint64_t data_start = 8 + header_size;
    const char *tensor_data = data_ptr + data_start + info.offset_begin;
    uint64_t bytes = info.offset_end - info.offset_begin;

    // Determine dtype
    torch::ScalarType torch_dtype = torch::kFloat32;
    if (info.dtype == "BF16")
        torch_dtype = torch::kBFloat16;
    else if (info.dtype == "F16")
        torch_dtype = torch::kFloat16;
    else if (info.dtype == "F32")
        torch_dtype = torch::kFloat32;
    else if (info.dtype == "I64")
        torch_dtype = torch::kInt64;
    else if (info.dtype == "I32")
        torch_dtype = torch::kInt32;
    else if (info.dtype == "U8" || info.dtype == "UINT8")
        torch_dtype = torch::kUInt8;
    else {
        std::cerr << "Unsupported dtype: " << info.dtype << std::endl;
        return torch::Tensor();
    }

    // Create tensor from blob (no copy) - unsafe generally, but safe here because we manage mmap
    // Note: We deliberately do NOT clone here. The clone happens when we copy_ to the model parameter later.
    // This tensor is just a view into the file.
    torch::Tensor tensor = torch::from_blob((void *)tensor_data, info.shape, torch_dtype);
    return tensor;
}

// Helper to unpack GPTQ int32 qweight to 4-bit (stored as uint8)
// Input: [in_features/8, out_features] int32
// Output: [out_features, in_features/2] uint8 (packed)
torch::Tensor unpack_gptq_qweight(torch::Tensor qweight) {
    auto device = qweight.device();
    int64_t in_features_div_8 = qweight.size(0);
    int64_t out_features = qweight.size(1);
    int64_t in_features = in_features_div_8 * 8;

    // 1. Unpack int32 to 8x 4-bit values
    // qweight: [K/8, N]

    std::vector<torch::Tensor> unpacked_parts;
    for (int i = 0; i < 8; ++i) {
        auto shift = torch::tensor(i * 4, torch::kInt32).to(device);
        auto mask = torch::tensor(0xF, torch::kInt32).to(device);
        auto part = torch::bitwise_and(torch::bitwise_right_shift(qweight, shift), mask).to(torch::kUInt8);
        unpacked_parts.push_back(part);
    }

    // Stack: [K/8, N, 8]
    auto unpacked = torch::stack(unpacked_parts, 2);

    // Permute to [K/8, 8, N] to make K contiguous
    unpacked = unpacked.permute({0, 2, 1}).contiguous();

    // Flatten: [K, N]
    unpacked = unpacked.view({in_features, out_features});

    // Transpose: [N, K]
    unpacked = unpacked.t().contiguous();

    // Repack to [N, K/2] uint8
    // Even indices: low bits
    // Odd indices: high bits
    auto low = unpacked.slice(1, 0, in_features, 2);  // [N, K/2]
    auto high = unpacked.slice(1, 1, in_features, 2); // [N, K/2]

    auto packed = torch::bitwise_or(low, torch::bitwise_left_shift(high, torch::tensor(4, torch::kUInt8).to(device)));

    return packed;
}

// Helper to unpack AWQ-packed int32 tensor to 4-bit values (as uint8)
// Applies the {0, 4, 1, 5, 2, 6, 3, 7} permutation to transform ZigZag order to Contiguous order.
// Input: Tensor of int32 (ZigZag packed)
// Output: Tensor of uint8 with shape [..., 8] (Contiguous logical order)
torch::Tensor unpack_awq_zigzag_to_contiguous(torch::Tensor packed_int32) {
    auto device = packed_int32.device();
    std::vector<torch::Tensor> unpacked_parts;

    // AWQ permutation: [0, 4, 1, 5, 2, 6, 3, 7]
    // This maps output index k to the shift amount (permutation[k] * 4)
    // This ensures that unpacked_parts[k] corresponds to the k-th logical element.
    const int permutation[8] = {0, 4, 1, 5, 2, 6, 3, 7};

    for (int k = 0; k < 8; ++k) {
        int shift_amount = permutation[k] * 4;
        // Use scalar operations to avoid tensor allocation
        torch::Tensor part;
        if (shift_amount > 0) {
            part = torch::bitwise_right_shift(packed_int32, shift_amount);
        } else {
            part = packed_int32;
        }
        part = torch::bitwise_and(part, 0x0F).to(torch::kUInt8);
        unpacked_parts.push_back(part);
    }

    // Stack along the last dimension
    return torch::stack(unpacked_parts, -1);
}

// Helper to unpack AWQ int32 qweight to 4-bit (stored as uint8)
// Input: [in_features, out_features/8] int32
// Output: [in_features, out_features] uint8 (Unpacked [In, Out])
torch::Tensor unpack_awq_qweight(torch::Tensor qweight) {
    auto device = qweight.device();
    // Transpose to [Out/8, In]
    qweight = qweight.t().contiguous();

    int64_t out_features_div_8 = qweight.size(0);
    int64_t in_features = qweight.size(1);
    int64_t out_features = out_features_div_8 * 8;

    // Unpack: [Out/8, In] -> [Out/8, In, 8]
    // This step converts ZigZag packing to Contiguous unpacking
    auto unpacked = unpack_awq_zigzag_to_contiguous(qweight);

    // Permute to [Out/8, 8, In]
    unpacked = unpacked.permute({0, 2, 1}).contiguous();

    // Flatten: [Out, In]
    unpacked = unpacked.view({out_features, in_features});

    // Return unpacked [In, Out] uint8 (Transpose here)
    return unpacked.t().contiguous().to(torch::kUInt8);
}

// Helper to unpack AWQ qzeros
// Input: [n_groups, out_features/8] int32
// Output: [n_groups, out_features] int8 (Unpacked [Groups, Out])
torch::Tensor unpack_awq_qzeros(torch::Tensor qzeros) {
    auto device = qzeros.device();
    // qzeros is [Groups, Out/8] (int32)

    // Unpack: [Groups, Out/8] -> [Groups, Out/8, 8]
    // Convert ZigZag to Contiguous
    auto unpacked = unpack_awq_zigzag_to_contiguous(qzeros);

    // Flatten last two dims: [Groups, Out]
    int64_t n_groups = qzeros.size(0);
    int64_t out_features = qzeros.size(1) * 8;
    unpacked = unpacked.view({n_groups, out_features});

    // Return [Groups, Out] int8 (No Transpose)
    return unpacked.contiguous().to(torch::kInt8);
}

// Helper to unpack GPTQ int32 qzeros
// Input: [n_groups, out_features/8] int32
// Output: [out_features, n_groups] bf16
torch::Tensor unpack_gptq_qzeros(torch::Tensor qzeros, int64_t out_features, bool add_one) {
    auto device = qzeros.device();
    int64_t n_groups = qzeros.size(0);

    // 1. Unpack int32 to 8x 4-bit values
    // qzeros: [G, N/8]

    std::vector<torch::Tensor> unpacked_parts;
    for (int i = 0; i < 8; ++i) {
        int shift_amount = i * 4;
        torch::Tensor part;
        if (shift_amount > 0) {
            part = torch::bitwise_right_shift(qzeros, shift_amount);
        } else {
            part = qzeros;
        }
        part = torch::bitwise_and(part, 0x0F).to(torch::kInt32);
        unpacked_parts.push_back(part);
    }

    // Stack: [G, N/8, 8]
    auto unpacked = torch::stack(unpacked_parts, 2);

    // Flatten: [G, N]
    unpacked = unpacked.view({n_groups, out_features});

    // Transpose: [N, G]
    unpacked = unpacked.t().contiguous();

    // Convert to Int8 and add 1 (GPTQ zero point offset)
    auto zeros = unpacked.to(torch::kInt8);
    if (add_one) {
        zeros = zeros + 1;
    }

    return zeros;
}

// Load quantized weights from safetensors file with mmap
// GPTQ format: layer.weight.qweight (uint8 packed), layer.weight.scales (bf16), layer.weight.qzeros (uint8 packed)
void UnifiedLLMW4A16Impl::load_quantized_weights_from_safetensors(const std::string &filename) {
    // Safetensors path currently loads separate q/k/v projections.
    gemma_use_qkv_fused_ = false;
    qwen_use_qkv_fused_ = false;

    if (debug_verbosity >= 1) {
        std::cout << "Loading weights from safetensors (mmap): " << filename << std::endl;
    }

    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        std::perror("open");
        return;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::perror("fstat");
        close(fd);
        return;
    }

    size_t file_size = sb.st_size;
    const char *map = (const char *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd); // Can close fd after mmap

    if (map == MAP_FAILED) {
        std::perror("mmap");
        return;
    }

    // Prefetch pages
    if (madvise((void *)map, file_size, MADV_WILLNEED) != 0) {
        std::perror("madvise");
    }

    // Parse header
    uint64_t header_size = 0;
    auto tensor_map = parse_safetensors_header_from_map(map, file_size, header_size);

    if (tensor_map.empty()) {
        std::cerr << "Failed to parse safetensors header" << std::endl;
        munmap((void *)map, file_size);
        return;
    }

    if (debug_verbosity >= 1) {
        std::cout << "Found " << tensor_map.size() << " tensors in safetensors file" << std::endl;
    }

    this->eval();
    torch::NoGradGuard no_grad;

    size_t loaded = 0;
    size_t skipped = 0;

    // Helper loading lambda using mmap ptr
    auto load_tensor = [&](const SafetensorsTensorInfo &info) { return load_tensor_from_ptr(map, info, header_size); };

    // Load token embedding (not quantized)
    auto it_embed = tensor_map.find("model.embed_tokens.weight");
    if (it_embed != tensor_map.end() && it_embed->second.valid) {
        torch::Tensor loaded_tensor = load_tensor(it_embed->second);
        if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
            token_embedding->weight.set_requires_grad(false);
            token_embedding->weight.copy_(loaded_tensor.to(token_embedding->weight.dtype()).to(token_embedding->weight.device()));
            loaded++;
            if (debug_verbosity >= 1) {
                std::cout << "Safetensors: model.embed_tokens.weight  Shape: " << loaded_tensor.sizes()
                          << "  Dtype: " << it_embed->second.dtype << " (" << loaded_tensor.dtype() << ")" << std::endl;
                std::cout << "Model param: token_embedding.weight  Shape: " << token_embedding->weight.sizes()
                          << "  Dtype: " << token_embedding->weight.dtype() << std::endl;
            }
            if (debug_verbosity >= 1) {
                std::cout << "  -> Loaded directly from model.embed_tokens.weight" << std::endl;
            }
        }
    } else {
        std::cerr << "Warning: model.embed_tokens.weight not found!" << std::endl;
        skipped++;
    }

    // Load quantized linear layers
    // Optimize: Find which layers are present in this safetensors file first
    std::set<int64_t> present_layers;
    std::string prefix = "model.layers.";
    for (const auto &pair : tensor_map) {
        const std::string &name = pair.first;
        if (name.compare(0, prefix.length(), prefix) == 0) {
            size_t end_pos = name.find('.', prefix.length());
            if (end_pos != std::string::npos) {
                std::string layer_num_str = name.substr(prefix.length(), end_pos - prefix.length());
                try {
                    int64_t layer_idx = std::stoll(layer_num_str);
                    if (layer_idx >= 0 && layer_idx < num_hidden_layers_) {
                        present_layers.insert(layer_idx);
                    }
                } catch (...) {
                }
            }
        }
    }

    // Load quantized linear layers
    // Optimize: Find which layers are present in this safetensors file first
    auto load_linear_layer = [&](QuantizedLinear &layer, const std::string &base_name) {
        auto it_qweight = tensor_map.find(base_name + ".qweight");
        auto it_scales = tensor_map.find(base_name + ".scales");
        auto it_qzeros = tensor_map.find(base_name + ".qzeros");
        auto it_g_idx = tensor_map.find(base_name + ".g_idx");
        auto it_bias = tensor_map.find(base_name + ".bias");

        // Alternate naming (compressed-tensors / AutoGPTQ)
        auto it_packed = tensor_map.find(base_name + ".weight_packed");
        auto it_scale = tensor_map.find(base_name + ".weight_scale");
        auto it_packed_g_idx = tensor_map.find(base_name + ".weight_g_idx");

        torch::Tensor qweight, scales, qzeros, g_idx;
        bool is_compressed_tensors = false;

        if (it_packed != tensor_map.end() && it_packed->second.valid && it_scale != tensor_map.end() && it_scale->second.valid) {

            is_compressed_tensors = true;
            // qweight and scales are load_tensor'd but they are just views, so cost is negligible
            qweight = load_tensor(it_packed->second);
            scales = load_tensor(it_scale->second);

            if (it_packed_g_idx != tensor_map.end() && it_packed_g_idx->second.valid) {
                g_idx = load_tensor(it_packed_g_idx->second);
            }

            // Qzeros often missing in symmetric quantization - handled below
        } else if (it_qweight != tensor_map.end() && it_qweight->second.valid && it_scales != tensor_map.end() && it_scales->second.valid &&
                   it_qzeros != tensor_map.end() && it_qzeros->second.valid) {

            qweight = load_tensor(it_qweight->second);
            scales = load_tensor(it_scales->second);
            qzeros = load_tensor(it_qzeros->second);

            if (it_g_idx != tensor_map.end() && it_g_idx->second.valid) {
                g_idx = load_tensor(it_g_idx->second);
            }
        } else {
            std::cerr << "Warning: Missing quantized weights for " << base_name << std::endl;
            skipped++;
            return;
        }

        if (qweight.defined() && scales.defined()) {
            auto device = token_embedding->weight.device();

            std::string qweight_dtype;
            std::string scales_dtype;
            std::string qzeros_dtype;

            std::string qweight_name_str = base_name + ".qweight";
            std::string scales_name_str = base_name + ".scales";
            std::string qzeros_name_str = base_name + ".qzeros";

            if (is_compressed_tensors) {
                qweight_dtype = it_packed->second.dtype;
                scales_dtype = it_scale->second.dtype;
                qzeros_dtype = "N/A (Synthetic)"; // Usually synthetic for symmetric

                qweight_name_str = base_name + ".weight_packed";
                scales_name_str = base_name + ".weight_scale";
                // qzeros not loaded directly if missing

                // qweight: [N, K/8] -> needs [K/8, N] for unpack
                qweight = qweight.to(device).t().contiguous();
                qweight = unpack_gptq_qweight(qweight); // -> [N, K/2]

                // scales: [N, G] -> Already in [N, G] format for C++ backend
                scales = scales.to(torch::kBFloat16).to(device).contiguous();

                // qzeros
                if (!qzeros.defined()) {
                    // Symmetric 4-bit compressed-tensors checkpoints omit explicit zero points.
                    // Model them as centered int4 with zero point 8.
                    qzeros = torch::full({scales.size(0), scales.size(1)}, 8, torch::TensorOptions().dtype(torch::kInt8).device(device));
                }
            } else {
                qweight_dtype = it_qweight->second.dtype;
                scales_dtype = it_scales->second.dtype;
                qzeros_dtype = it_qzeros->second.dtype;

                // Detect layout based on shape matches with expected dimensions
                // GPTQ: [In/8, Out]
                // AWQ: [In, Out/8]

                bool is_awq = false;
                bool is_gptq = false;

                int64_t in_feat = layer->in_features();
                int64_t out_feat = layer->out_features();

                // Check AWQ match
                if (qweight.size(0) == in_feat && qweight.size(1) == out_feat / 8) {
                    is_awq = true;
                }
                // Check GPTQ match
                else if (qweight.size(0) == in_feat / 8 && qweight.size(1) == out_feat) {
                    is_gptq = true;
                }
                // Ambiguous case (square packed) or Fallback heuristic
                else {
                    // Fallback to heuristic if dimensions don't match expected (could happen if layer dims are wrong?)
                    // But usually we trust layer dims.
                    // If ambiguous (e.g. In=2048, Out=16384 for AWQ -> [2048, 2048])
                    // vs (In=16384, Out=2048 for GPTQ -> [2048, 2048])
                    // We know which layer we are loading, so using layer dims resolves it.

                    if (qweight.size(0) > qweight.size(1)) {
                        is_awq = true;
                    } else if (qweight.size(1) > qweight.size(0)) {
                        is_gptq = true;
                    } else {
                        // Square case [2048, 2048]
                        // If we are here, it means exact match check failed?
                        // No, if exact match checks passed, we wouldn't be here.
                        // So if we are here, the loaded weight doesn't match the layer dimensions.
                        std::cerr << "Warning: Weight shape " << qweight.sizes()
                                  << " does not match expected layer dimensions In=" << in_feat << " Out=" << out_feat << std::endl;

                        // Heuristic fallback: Assume AWQ if we are defaulting to AWQ models, but better to check
                        // typical patterns
                        if (qweight.size(0) == 2048 && qweight.size(1) == 2048) {
                            // Could be Up/Gate (AWQ) or Down (GPTQ)
                            std::cerr << "Ambiguous square weight. Defaulting to AWQ based on recent usage." << std::endl;
                            is_awq = true;
                        } else {
                            // Default to GPTQ for safety/legacy
                            is_gptq = true;
                        }
                    }
                }

                if (is_awq) {
                    if (debug_verbosity >= 1) {
                        std::cout << "  -> Detected AWQ layout [In, Out/8], using AWQ unpack" << std::endl;
                    }
                    qweight = unpack_awq_qweight(qweight.to(device));

                    int64_t out_features_check = qweight.size(1); // unpack_awq_qweight now returns [In, Out]

                    // Dynamic check for scales layout
                    // AWQ scales are usually [Groups, Out], which matches scales.size(1) == Out (out_features_check)
                    // We want to keep them as [Groups, Out] for the debug path

                    // Detected [Groups, Out] scales -> Keep as [Groups, Out]
                    if (debug_verbosity >= 1)
                        std::cout << "  -> Detected [Groups, Out] scales, keeping as [Groups, Out]" << std::endl;
                    scales = scales.to(torch::kBFloat16).to(device).contiguous();

                    // AWQ zeros are packed [In/G, Out/8] (usually), unpack using AWQ permutation
                    torch::Tensor qzeros_dev = qzeros.to(device);
                    qzeros = unpack_awq_qzeros(qzeros_dev);
                } else {
                    // Standard GPTQ
                    std::cout << "  -> Detected GPTQ layout [In/8, Out], using GPTQ unpack" << std::endl;
                    qweight = unpack_gptq_qweight(qweight.to(device));

                    // GPTQ scales are [In/G, Out], transpose to [Out, In/G]
                    scales = scales.to(torch::kBFloat16).to(device).t().contiguous();

                    // GPTQ zeros are packed, unpack them
                    qzeros = unpack_gptq_qzeros(qzeros.to(device), scales.size(0), true);
                }
            }

            if (debug_verbosity >= 1) {
                std::cout << base_name << std::endl;
                std::cout << "  -> qweight shape: " << qweight.sizes() << ", dtype: " << qweight.dtype() << std::endl;
                std::cout << "  -> scales shape: " << scales.sizes() << ", dtype: " << scales.dtype() << std::endl;
                std::cout << "  -> qzeros shape: " << qzeros.sizes() << ", dtype: " << qzeros.dtype() << std::endl;
            }

            if (g_idx.defined()) {
                g_idx = g_idx.to(torch::kInt32).to(device);
            }
            layer->set_quantized_weights(qweight, scales, qzeros, g_idx);
            loaded++;

            // Bias
            if (it_bias != tensor_map.end() && it_bias->second.valid) {
                torch::Tensor bias_tensor = load_tensor(it_bias->second);
                if (bias_tensor.defined()) {
                    for (auto &param : layer->named_parameters()) {
                        if (param.key() == "bias") {
                            param.value().set_requires_grad(false);
                            param.value().copy_(bias_tensor.to(param.value().dtype()).to(device));
                            if (debug_verbosity >= 1) {
                                std::cout << "  -> Loaded bias for " << base_name << std::endl;
                            }
                            break;
                        }
                    }
                }
            }

            if (debug_verbosity >= 1) {
                std::cout << "Safetensors: " << qweight_name_str << "  Shape: " << qweight.sizes() << "  Dtype: " << qweight_dtype << " ("
                          << qweight.dtype() << ")" << std::endl;
                std::cout << "Safetensors: " << scales_name_str << "  Shape: " << scales.sizes() << "  Dtype: " << scales_dtype << " ("
                          << scales.dtype() << ")" << std::endl;
                std::cout << "Safetensors: " << qzeros_name_str << "  Shape: " << qzeros.sizes() << "  Dtype: " << qzeros_dtype << " ("
                          << qzeros.dtype() << ")" << std::endl;
                std::cout << "Model param: " << base_name << " (quantized)" << std::endl;
                std::cout << "  -> Loaded" << std::endl;
            }
        } else {
            std::cerr << "Warning: Failed to load (undefined tensors) for " << base_name << std::endl;
            skipped++;
        }
    };

    std::vector<int64_t> layers_vec(present_layers.begin(), present_layers.end());

#pragma omp parallel for
    for (size_t idx = 0; idx < layers_vec.size(); ++idx) {
        int64_t i = layers_vec[idx];
        std::string layer_prefix = "model.layers." + std::to_string(i);

        // Attention layers: q_proj, k_proj, v_proj, o_proj
        std::vector<std::pair<QuantizedLinear, std::string>> attn_layers = {{q_layers[i], "self_attn.q_proj"},
                                                                            {k_layers[i], "self_attn.k_proj"},
                                                                            {v_layers[i], "self_attn.v_proj"},
                                                                            {o_layers[i], "self_attn.o_proj"}};

        for (auto &[layer, suffix] : attn_layers) {
            load_linear_layer(layer, layer_prefix + "." + suffix);
        }

        // MLP layers: gate_proj, up_proj, down_proj
        std::vector<std::pair<QuantizedLinear, std::string>> mlp_layers = {
            {gate_layers[i], "mlp.gate_proj"}, {up_layers[i], "mlp.up_proj"}, {down_layers[i], "mlp.down_proj"}};

        for (auto &[layer, suffix] : mlp_layers) {
            load_linear_layer(layer, layer_prefix + "." + suffix);
        }

        // RMSNorm layers (not quantized)
        auto it_input_norm = tensor_map.find(layer_prefix + ".input_layernorm.weight");
        if (it_input_norm != tensor_map.end() && it_input_norm->second.valid) {
            torch::Tensor loaded_tensor = load_tensor(it_input_norm->second);
            if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
                input_norms[i]->set_weight(loaded_tensor);
                loaded++;
                if (debug_verbosity >= 1) {
                    std::cout << "Safetensors: " << layer_prefix << ".input_layernorm.weight  Shape: " << loaded_tensor.sizes()
                              << "  Dtype: " << it_input_norm->second.dtype << " (" << loaded_tensor.dtype() << ")" << std::endl;
                    std::cout << "Model param: input_norms[" << i << "].weight" << std::endl;
                    std::cout << "  -> Loaded" << std::endl;
                }
            }
        }

        auto it_post_norm = tensor_map.find(layer_prefix + ".post_attention_layernorm.weight");
        if (it_post_norm != tensor_map.end() && it_post_norm->second.valid) {
            torch::Tensor loaded_tensor = load_tensor(it_post_norm->second);
            if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
                post_attn_norms[i]->set_weight(loaded_tensor);
                loaded++;
                if (debug_verbosity >= 1) {
                    std::cout << "Safetensors: " << layer_prefix << ".post_attention_layernorm.weight  Shape: " << loaded_tensor.sizes()
                              << "  Dtype: " << it_post_norm->second.dtype << " (" << loaded_tensor.dtype() << ")" << std::endl;
                    std::cout << "Model param: post_attn_norms[" << i << "].weight" << std::endl;
                    std::cout << "  -> Loaded" << std::endl;
                }
            }
        }
    }

    // Final norm
    auto it_final_norm = tensor_map.find("model.norm.weight");
    if (it_final_norm != tensor_map.end() && it_final_norm->second.valid) {
        torch::Tensor loaded_tensor = load_tensor(it_final_norm->second);
        if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
            final_norm->set_weight(loaded_tensor);
            loaded++;
            if (debug_verbosity >= 1) {
                std::cout << "Safetensors: model.norm.weight  Shape: " << loaded_tensor.sizes()
                          << "  Dtype: " << it_final_norm->second.dtype << " (" << loaded_tensor.dtype() << ")" << std::endl;
                std::cout << "Model param: final_norm.weight" << std::endl;
                std::cout << "  -> Loaded" << std::endl;
            }
        }
    }

    // LM head (unquantized)
    auto it_lm_weight = tensor_map.find("lm_head.weight");
    if (it_lm_weight != tensor_map.end() && it_lm_weight->second.valid) {
        torch::Tensor loaded_tensor = load_tensor(it_lm_weight->second);
        if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
            lm_head->weight.set_requires_grad(false);
            lm_head->weight.copy_(loaded_tensor.to(lm_head->weight.dtype()).to(lm_head->weight.device()));
            loaded++;
            if (debug_verbosity >= 1) {
                std::cout << "Safetensors: lm_head.weight  Shape: " << loaded_tensor.sizes() << "  Dtype: " << it_lm_weight->second.dtype
                          << " (" << loaded_tensor.dtype() << ")" << std::endl;
                std::cout << "Model param: lm_head.weight  Shape: " << lm_head->weight.sizes() << "  Dtype: " << lm_head->weight.dtype()
                          << std::endl;
                std::cout << "  -> Loaded" << std::endl;
            }
        }
    } else {
        // Check if we should tie weights (Gemma/Qwen often tie weights)
        // Llama usually doesn't, but if safetensors is missing lm_head, it likely means tied.
        // We check if token_embedding has been loaded (it's loaded at the start).
        // Note: We don't strictly check arch_type_ here because if the file is missing lm_head,
        // it's the best guess anyway.

        // Check if token_embedding matches lm_head shape (vocab_size, hidden_size)
        if (token_embedding->weight.size(0) == lm_head->weight.size(0) && token_embedding->weight.size(1) == lm_head->weight.size(1)) {

            std::cout << "Model param: lm_head.weight" << std::endl;
            std::cout << "  -> Tied to token_embedding.weight" << std::endl;

            lm_head->weight.set_requires_grad(false);
            lm_head->weight.copy_(token_embedding->weight);
            loaded++;
        } else {
            std::cerr << "Warning: Missing lm_head.weight and could not tie to token_embedding" << std::endl;
            skipped++;
        }
    }

    std::cout << "Loaded " << loaded << " parameters, skipped " << skipped << std::endl;

    munmap((void *)map, file_size);
}

void UnifiedLLMW4A16Impl::load_non_quantized_weights_from_safetensors(const std::string &filename) {
    if (debug_verbosity >= 1) {
        std::cout << "Loading non-quantized weights from safetensors (mmap): " << filename << std::endl;
    }

    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        std::perror("open");
        return;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::perror("fstat");
        close(fd);
        return;
    }

    size_t file_size = sb.st_size;
    const char *map = (const char *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (map == MAP_FAILED) {
        std::perror("mmap");
        return;
    }

    if (madvise((void *)map, file_size, MADV_WILLNEED) != 0) {
        std::perror("madvise");
    }

    uint64_t header_size = 0;
    auto tensor_map = parse_safetensors_header_from_map(map, file_size, header_size);
    if (tensor_map.empty()) {
        std::cerr << "Failed to parse safetensors header" << std::endl;
        munmap((void *)map, file_size);
        return;
    }

    this->eval();
    torch::NoGradGuard no_grad;

    size_t loaded = 0;
    size_t skipped = 0;

    auto load_tensor = [&](const SafetensorsTensorInfo &info) { return load_tensor_from_ptr(map, info, header_size); };

    // Token embedding
    auto it_embed = tensor_map.find("model.embed_tokens.weight");
    if (it_embed != tensor_map.end() && it_embed->second.valid) {
        torch::Tensor loaded_tensor = load_tensor(it_embed->second);
        if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
            token_embedding->weight.set_requires_grad(false);
            token_embedding->weight.copy_(loaded_tensor.to(token_embedding->weight.dtype()).to(token_embedding->weight.device()));
            loaded++;
        }
    } else {
        std::cerr << "Warning: model.embed_tokens.weight not found!" << std::endl;
        skipped++;
    }

    // Per-layer norms
    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        std::string layer_prefix = "model.layers." + std::to_string(i);

        auto it_input_norm = tensor_map.find(layer_prefix + ".input_layernorm.weight");
        if (it_input_norm != tensor_map.end() && it_input_norm->second.valid) {
            torch::Tensor loaded_tensor = load_tensor(it_input_norm->second);
            if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
                input_norms[i]->set_weight(loaded_tensor);
                loaded++;
            }
        }

        auto it_post_norm = tensor_map.find(layer_prefix + ".post_attention_layernorm.weight");
        if (it_post_norm != tensor_map.end() && it_post_norm->second.valid) {
            torch::Tensor loaded_tensor = load_tensor(it_post_norm->second);
            if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
                post_attn_norms[i]->set_weight(loaded_tensor);
                loaded++;
            }
        }
    }

    // Quantized layer biases (q/k/v/o and MLP) are stored as regular tensors in some models (e.g., Qwen).
    // When using pre-saved quantized bins, load these biases here to match the full safetensors path.
    auto load_bias = [&](QuantizedLinear &layer, const std::string &base_name) {
        auto it_bias = tensor_map.find(base_name + ".bias");
        if (it_bias == tensor_map.end() || !it_bias->second.valid) {
            return;
        }

        torch::Tensor bias_tensor = load_tensor(it_bias->second);
        if (!bias_tensor.defined() || bias_tensor.numel() == 0) {
            return;
        }

        auto device = token_embedding->weight.device();
        for (auto &param : layer->named_parameters()) {
            if (param.key() == "bias") {
                param.value().set_requires_grad(false);
                param.value().copy_(bias_tensor.to(param.value().dtype()).to(device));
                loaded++;
                if (debug_verbosity >= 1) {
                    std::cout << "  -> Loaded bias for " << base_name << std::endl;
                }
                break;
            }
        }
    };

    auto load_fused_qkv_bias = [&](QuantizedLinear &layer, const std::string &q_base_name, const std::string &k_base_name,
                                   const std::string &v_base_name) {
        auto it_q_bias = tensor_map.find(q_base_name + ".bias");
        auto it_k_bias = tensor_map.find(k_base_name + ".bias");
        auto it_v_bias = tensor_map.find(v_base_name + ".bias");
        if (it_q_bias == tensor_map.end() || !it_q_bias->second.valid || it_k_bias == tensor_map.end() || !it_k_bias->second.valid ||
            it_v_bias == tensor_map.end() || !it_v_bias->second.valid) {
            return;
        }

        torch::Tensor q_bias_tensor = load_tensor(it_q_bias->second);
        torch::Tensor k_bias_tensor = load_tensor(it_k_bias->second);
        torch::Tensor v_bias_tensor = load_tensor(it_v_bias->second);
        if (!q_bias_tensor.defined() || q_bias_tensor.numel() == 0 || !k_bias_tensor.defined() || k_bias_tensor.numel() == 0 ||
            !v_bias_tensor.defined() || v_bias_tensor.numel() == 0) {
            return;
        }

        auto fused_bias = torch::cat({q_bias_tensor, k_bias_tensor, v_bias_tensor}, 0).contiguous();
        auto device = token_embedding->weight.device();
        for (auto &param : layer->named_parameters()) {
            if (param.key() == "bias") {
                param.value().set_requires_grad(false);
                param.value().copy_(fused_bias.to(param.value().dtype()).to(device));
                loaded++;
                if (debug_verbosity >= 1) {
                    std::cout << "  -> Loaded fused qkv bias for " << q_base_name << ", " << k_base_name << ", " << v_base_name << std::endl;
                }
                break;
            }
        }
    };

    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        std::string layer_prefix = "model.layers." + std::to_string(i);

        // Attention layers
        if (arch_type_ == ArchitectureType::QWEN25SMALL && i < static_cast<int64_t>(qkv_layers.size())) {
            load_fused_qkv_bias(qkv_layers[i], layer_prefix + ".self_attn.q_proj", layer_prefix + ".self_attn.k_proj",
                                layer_prefix + ".self_attn.v_proj");
        }
        load_bias(q_layers[i], layer_prefix + ".self_attn.q_proj");
        load_bias(k_layers[i], layer_prefix + ".self_attn.k_proj");
        load_bias(v_layers[i], layer_prefix + ".self_attn.v_proj");
        load_bias(o_layers[i], layer_prefix + ".self_attn.o_proj");

        // MLP layers
        load_bias(gate_layers[i], layer_prefix + ".mlp.gate_proj");
        load_bias(up_layers[i], layer_prefix + ".mlp.up_proj");
        load_bias(down_layers[i], layer_prefix + ".mlp.down_proj");
    }

    // Final norm
    auto it_final_norm = tensor_map.find("model.norm.weight");
    if (it_final_norm != tensor_map.end() && it_final_norm->second.valid) {
        torch::Tensor loaded_tensor = load_tensor(it_final_norm->second);
        if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
            final_norm->set_weight(loaded_tensor);
            loaded++;
        }
    }

    // LM head
    auto it_lm_weight = tensor_map.find("lm_head.weight");
    if (it_lm_weight != tensor_map.end() && it_lm_weight->second.valid) {
        torch::Tensor loaded_tensor = load_tensor(it_lm_weight->second);
        if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
            lm_head->weight.set_requires_grad(false);
            lm_head->weight.copy_(loaded_tensor.to(lm_head->weight.dtype()).to(lm_head->weight.device()));
            loaded++;
        }
    } else {
        if (token_embedding->weight.size(0) == lm_head->weight.size(0) && token_embedding->weight.size(1) == lm_head->weight.size(1)) {
            lm_head->weight.set_requires_grad(false);
            lm_head->weight.copy_(token_embedding->weight);
            loaded++;
        } else {
            std::cerr << "Warning: Missing lm_head.weight and could not tie to token_embedding" << std::endl;
            skipped++;
        }
    }

    if (debug_verbosity >= 1) {
        std::cout << "Loaded " << loaded << " non-quantized parameters, skipped " << skipped << std::endl;
    }

    munmap((void *)map, file_size);
}

static torch::Tensor read_bin_tensor(const std::string &path, torch::ScalarType dtype, const std::vector<int64_t> &shape) {
    std::ifstream input_file(path, std::ios::binary);
    if (!input_file) {
        throw std::runtime_error("Could not open file: " + path);
    }

    auto tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
    size_t expected_bytes = tensor.numel() * tensor.element_size();

    input_file.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(input_file.tellg());
    input_file.seekg(0, std::ios::beg);

    if (file_size != expected_bytes) {
        throw std::runtime_error("File size mismatch for " + path + " (expected " + std::to_string(expected_bytes) + ", got " +
                                 std::to_string(file_size) + ")");
    }

    input_file.read(reinterpret_cast<char *>(tensor.data_ptr()), file_size);
    if (!input_file) {
        throw std::runtime_error("Failed to read file: " + path);
    }
    return tensor;
}

void UnifiedLLMW4A16Impl::load_quantized_weights_from_bins(const std::string &weights_dir) {
    if (debug_verbosity >= 1) {
        std::cout << "Loading quantized weights from bins: " << weights_dir << std::endl;
    }

    this->eval();
    torch::NoGradGuard no_grad;

    auto load_layer = [&](QuantizedLinear &layer, const std::string &prefix) {
        try {
            if (use_packed_weights) {
                if (layer->is_k_split()) {
                    int64_t split_k = layer->split_k();
                    std::string path0 = weights_dir + "/" + prefix + ".k" + std::to_string(split_k) + ".packed0.bin";
                    std::string path1 = weights_dir + "/" + prefix + ".k" + std::to_string(split_k) + ".packed1.bin";
                    if (!std::filesystem::exists(path0) || !std::filesystem::exists(path1)) {
                        // Backward-compat: fall back to legacy names if split-tagged bins are absent.
                        path0 = weights_dir + "/" + prefix + ".packed0.bin";
                        path1 = weights_dir + "/" + prefix + ".packed1.bin";
                    }

                    size_t size0 = std::filesystem::file_size(path0);
                    size_t size1 = std::filesystem::file_size(path1);

                    auto t0 = read_bin_tensor(path0, torch::kUInt8, {static_cast<int64_t>(size0)});
                    auto t1 = read_bin_tensor(path1, torch::kUInt8, {static_cast<int64_t>(size1)});
                    layer->set_packed_params_split(t0, t1);
                } else {
                    std::string path = weights_dir + "/" + prefix + ".packed.bin";
                    size_t size = std::filesystem::file_size(path);
                    auto t = read_bin_tensor(path, torch::kUInt8, {static_cast<int64_t>(size)});
                    layer->set_packed_params(t);
                }
            } else {
                int64_t out_features = layer->out_features();
                int64_t in_features = layer->in_features();
                int64_t packed_in = (in_features + 1) / 2;

                std::string q_path = weights_dir + "/" + prefix + ".qweight.bin";
                std::string s_path = weights_dir + "/" + prefix + ".scales.bin";
                std::string z_path = weights_dir + "/" + prefix + ".zeros.bin";

                auto q = read_bin_tensor(q_path, torch::kUInt8, {out_features, packed_in});

                size_t s_bytes = std::filesystem::file_size(s_path);
                int64_t s_numel = static_cast<int64_t>(s_bytes / 2); // bf16
                int64_t s_groups = s_numel / out_features;
                if (s_groups * out_features != s_numel) {
                    throw std::runtime_error("Invalid scales shape for " + s_path);
                }
                std::vector<int64_t> s_shape = (s_groups <= 1) ? std::vector<int64_t>{out_features}
                                                               : std::vector<int64_t>{out_features, s_groups};
                auto s = read_bin_tensor(s_path, torch::kBFloat16, s_shape);

                size_t z_bytes = std::filesystem::file_size(z_path);
                int64_t z_numel = static_cast<int64_t>(z_bytes);
                int64_t z_groups = z_numel / out_features;
                if (z_groups * out_features != z_numel) {
                    throw std::runtime_error("Invalid zeros shape for " + z_path);
                }
                std::vector<int64_t> z_shape = (z_groups <= 1) ? std::vector<int64_t>{out_features}
                                                               : std::vector<int64_t>{out_features, z_groups};
                auto z = read_bin_tensor(z_path, torch::kInt8, z_shape);

                layer->set_unpacked_params(q, s, z);
            }
        } catch (const std::exception &e) {
            std::cerr << "Error loading layer " << prefix << ": " << e.what() << std::endl;
            throw;
        }
    };

    struct UnpackedTriplet {
        torch::Tensor qweight;
        torch::Tensor scales;
        torch::Tensor zeros;
    };

    auto load_unpacked_triplet = [&](const std::string &prefix, int64_t out_features, int64_t in_features) -> UnpackedTriplet {
        int64_t packed_in = (in_features + 1) / 2;
        std::string q_path = weights_dir + "/" + prefix + ".qweight.bin";
        std::string s_path = weights_dir + "/" + prefix + ".scales.bin";
        std::string z_path = weights_dir + "/" + prefix + ".zeros.bin";

        auto q = read_bin_tensor(q_path, torch::kUInt8, {out_features, packed_in});

        size_t s_bytes = std::filesystem::file_size(s_path);
        int64_t s_numel = static_cast<int64_t>(s_bytes / 2); // bf16
        int64_t s_groups = s_numel / out_features;
        if (s_groups * out_features != s_numel) {
            throw std::runtime_error("Invalid scales shape for " + s_path);
        }
        std::vector<int64_t> s_shape =
            (s_groups <= 1) ? std::vector<int64_t>{out_features} : std::vector<int64_t>{out_features, s_groups};
        auto s = read_bin_tensor(s_path, torch::kBFloat16, s_shape);

        size_t z_bytes = std::filesystem::file_size(z_path);
        int64_t z_numel = static_cast<int64_t>(z_bytes);
        int64_t z_groups = z_numel / out_features;
        if (z_groups * out_features != z_numel) {
            throw std::runtime_error("Invalid zeros shape for " + z_path);
        }
        std::vector<int64_t> z_shape =
            (z_groups <= 1) ? std::vector<int64_t>{out_features} : std::vector<int64_t>{out_features, z_groups};
        auto z = read_bin_tensor(z_path, torch::kInt8, z_shape);

        return {q, s, z};
    };

    const bool enable_gemma_qkv_fused = (arch_type_ == ArchitectureType::GEMMA && !qkv_layers.empty());
    const bool enable_qwen_qkv_fused = (arch_type_ == ArchitectureType::QWEN25SMALL && !qkv_layers.empty());
    gemma_use_qkv_fused_ = false;
    qwen_use_qkv_fused_ = false;

    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        if (enable_gemma_qkv_fused || enable_qwen_qkv_fused) {
            const std::string q_prefix = "layer_" + std::to_string(i) + "_q";
            const std::string k_prefix = "layer_" + std::to_string(i) + "_k";
            const std::string v_prefix = "layer_" + std::to_string(i) + "_v";
            const std::string arch_label = enable_gemma_qkv_fused ? "Gemma" : "Qwen";

            try {
                if (use_packed_weights) {
                    if (qkv_layers[i]->is_k_split()) {
                        throw std::runtime_error(arch_label + " fused qkv path does not support K-split packed bins for qkv layer.");
                    }
                    std::string qkv_path = weights_dir + "/layer_" + std::to_string(i) + "_qkv.packed.bin";
                    if (!std::filesystem::exists(qkv_path)) {
                        throw std::runtime_error("Missing fused " + arch_label + " qkv bin: " + qkv_path +
                                                 ". Rebuild pre-saved bins so layer_*_qkv.packed.bin files are generated.");
                    }
                    auto qkv_blob = read_bin_tensor(qkv_path, torch::kUInt8, {static_cast<int64_t>(std::filesystem::file_size(qkv_path))});
                    qkv_layers[i]->set_packed_params(qkv_blob);
                } else {
                    auto q_triplet = load_unpacked_triplet(q_prefix, q_layers[i]->out_features(), q_layers[i]->in_features());
                    auto k_triplet = load_unpacked_triplet(k_prefix, k_layers[i]->out_features(), k_layers[i]->in_features());
                    auto v_triplet = load_unpacked_triplet(v_prefix, v_layers[i]->out_features(), v_layers[i]->in_features());

                    auto qweight = torch::cat({q_triplet.qweight, k_triplet.qweight, v_triplet.qweight}, 0);
                    auto scales = torch::cat({q_triplet.scales, k_triplet.scales, v_triplet.scales}, 0);
                    auto zeros = torch::cat({q_triplet.zeros, k_triplet.zeros, v_triplet.zeros}, 0);
                    qkv_layers[i]->set_unpacked_params(qweight, scales, zeros);
                }
            } catch (const std::exception &e) {
                std::cerr << "Error loading fused qkv layer " << i << ": " << e.what() << std::endl;
                throw;
            }
        } else {
            load_layer(q_layers[i], "layer_" + std::to_string(i) + "_q");
            load_layer(k_layers[i], "layer_" + std::to_string(i) + "_k");
            load_layer(v_layers[i], "layer_" + std::to_string(i) + "_v");
        }

        load_layer(o_layers[i], "layer_" + std::to_string(i) + "_o");
        load_layer(gate_layers[i], "layer_" + std::to_string(i) + "_gate");
        load_layer(up_layers[i], "layer_" + std::to_string(i) + "_up");
        load_layer(down_layers[i], "layer_" + std::to_string(i) + "_down");
    }

    gemma_use_qkv_fused_ = enable_gemma_qkv_fused;
    qwen_use_qkv_fused_ = enable_qwen_qkv_fused;
    if (debug_verbosity >= 1 && gemma_use_qkv_fused_) {
        std::cout << "Gemma fused qkv projections enabled." << std::endl;
    }
    if (debug_verbosity >= 1 && qwen_use_qkv_fused_) {
        std::cout << "Qwen fused qkv projections enabled." << std::endl;
    }
}

GemmaScratchSpace UnifiedLLMW4A16Impl::get_scratch_space() {
    GemmaScratchSpace scratch;
    scratch.x_buffer = x_buffer;
    scratch.qkv_buffer = qkv_buffer;
    scratch.gate_buffer = gate_buffer;
    scratch.up_buffer = up_buffer;
    scratch.output_buffer = output_buffer;
    scratch.hidden_states_buffer = hidden_states_buffer;
    scratch.queries_buffer = queries_buffer;
    scratch.keys_buffer = keys_buffer;
    scratch.values_buffer = values_buffer;
    scratch.attn_output_buffer = attn_output_buffer;
    scratch.attn_output_proj_buffer = attn_output_proj_buffer;
    scratch.norm_buffer = norm_buffer;
    return scratch;
}

void UnifiedLLMW4A16Impl::initialize_dummy_weights(int seed) {
    torch::NoGradGuard no_grad;
    torch::manual_seed(seed);
    if (debug_verbosity >= 1) {
        std::cout << "Initializing dummy weights (seed=" << seed << ")..." << std::endl;
    }

    auto device = token_embedding->weight.device();

    // Initialize embeddings and heads
    token_embedding->weight.uniform_(-0.1, 0.1);
    final_norm->set_weight(torch::ones({hidden_size_}).to(device));
    lm_head->weight.uniform_(-0.1, 0.1);

    gemma_use_qkv_fused_ = (arch_type_ == ArchitectureType::GEMMA && !qkv_layers.empty());
    qwen_use_qkv_fused_ = (arch_type_ == ArchitectureType::QWEN25SMALL && !qkv_layers.empty());

    for (int i = 0; i < num_hidden_layers_; ++i) {
        // Initialize norms
        input_norms[i]->set_weight(torch::ones({hidden_size_}).to(device));
        post_attn_norms[i]->set_weight(torch::ones({hidden_size_}).to(device));

        // Helper to init quantized linear
        auto init_layer = [&](QuantizedLinear &layer) {
            int64_t in_feat = layer->in_features();
            int64_t out_feat = layer->out_features();

            // qweight: [In, Out] uint8 (values 0-15)
            auto qweight = torch::randint(0, 16, {in_feat, out_feat}, torch::TensorOptions().dtype(torch::kUInt8).device(device));

            // scales: [Groups, Out] bf16
            int64_t groups = in_feat / groupsize_;
            if (groups < 1)
                groups = 1;

            auto scales = torch::rand({groups, out_feat}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));

            // qzeros: [Groups, Out] int8 (values 0-15, usually around 8)
            auto qzeros = torch::full({groups, out_feat}, 8, torch::TensorOptions().dtype(torch::kInt8).device(device));

            layer->set_quantized_weights(qweight, scales, qzeros);
        };

        // Attention
        if (!qkv_layers.empty()) {
            init_layer(qkv_layers[i]);
        }
        init_layer(q_layers[i]);
        init_layer(k_layers[i]);
        init_layer(v_layers[i]);
        init_layer(o_layers[i]);

        // MLP
        init_layer(gate_layers[i]);
        init_layer(up_layers[i]);
        init_layer(down_layers[i]);
    }

    if (debug_verbosity >= 1) {
        std::cout << "Dummy weights initialized." << std::endl;
    }
}
