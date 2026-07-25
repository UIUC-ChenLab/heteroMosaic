#pragma once

#include <map>
#include <string>
#include <torch/torch.h>
#include <vector>

// Helper function to sample from logits
int64_t sample_token(const torch::Tensor &logits, float temperature, float top_p, int64_t top_k);

// Safetensors parsing structures and functions
struct SafetensorsTensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t offset_begin;
    uint64_t offset_end;
    bool valid;

    SafetensorsTensorInfo() : valid(false) {}
};

// Parse safetensors header from mapped memory
std::map<std::string, SafetensorsTensorInfo> parse_safetensors_header_from_map(const char *data_ptr, uint64_t file_size,
                                                                               uint64_t &header_size);

// Load a tensor from memory mapped pointer
torch::Tensor load_tensor_from_ptr(const char *data_ptr, const SafetensorsTensorInfo &info, uint64_t header_size);

// Helper to unpack GPTQ int32 qweight to 4-bit (stored as uint8)
torch::Tensor unpack_gptq_qweight(torch::Tensor qweight);

// Helper to unpack AWQ-packed int32 tensor to 4-bit values (as uint8)
torch::Tensor unpack_awq_zigzag_to_contiguous(torch::Tensor packed_int32);

// Helper to unpack AWQ int32 qweight to 4-bit (stored as uint8)
torch::Tensor unpack_awq_qweight(torch::Tensor qweight);

// Helper to unpack AWQ qzeros
torch::Tensor unpack_awq_qzeros(torch::Tensor qzeros);

// Helper to unpack GPTQ int32 qzeros
torch::Tensor unpack_gptq_qzeros(torch::Tensor qzeros, int64_t out_features, bool add_one = true);
