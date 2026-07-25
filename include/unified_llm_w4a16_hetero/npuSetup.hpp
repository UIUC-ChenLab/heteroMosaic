#pragma once

#include <array>
#include <cstdint>
// #include <hip/hip_runtime.h>
#include "npu_matmul_func/npu_matmul_func.hpp"
#include <hip/hip_runtime.h>
#include <map>
#include <mutex>
#include <string>
#include <torch/torch.h>
#include <tuple>
#include <vector>

// Use standard types in header, convert to __u32/__u64 in implementation
// This avoids conflicts with system headers that may define these types differently

// XDNA driver file descriptor
extern int xdna_drv_fd;
// (global variable accessible to all functions)

// Global map to cache imported handles: ptr -> handle
extern std::map<void *, uint32_t> ptr_to_handle_map;

// Global hardware target setting from kernels.json ("gpu", "npu", or "hetero")
extern std::string hw_target;

// Global debug verbosity level from kernels.json (1=minimal, 2=moderate, 3=verbose)
extern int debug_verbosity;

// Global dummy weights flag
extern bool dummy_weights_enabled;

// Global warmup flag
extern bool warmup_enabled;

// Global minimal PDI flag
extern bool minimal_pdi;

// Global NPU dim
extern int npu_dim;
extern int64_t chunking_tokens;
extern std::vector<int64_t> chunking_token_schedule;
extern int64_t chunking_inflight;
extern bool chunking_scheduled;
extern bool async_chunking;

extern bool use_packed_weights;
extern int64_t pad_packed_weights;
extern bool cpu_decode;
extern bool gemv_driven_split_K;
struct StageBubbleSpec {
    int64_t chunk_id;
    int64_t layer_id;
    std::string stage;
    int64_t delay_us;
};
extern std::string trace_output_path;
extern std::string trace_run_tag;
extern bool trace_sync_stages;
extern std::vector<StageBubbleSpec> stage_bubbles;
// Global RoPE scaling settings (e.g., Llama3)
extern bool rope_scaling_enabled;
extern std::string rope_scaling_type;
extern float rope_scaling_factor;
extern float rope_scaling_low_freq_factor;
extern float rope_scaling_high_freq_factor;
extern float rope_scaling_original_max_position_embeddings;

// Global NPU mutex for synchronization
extern std::mutex npu_mutex;

// Maximum number of NPU contexts
#define MAX_NPU_HW_CTX 6
#define MAX_NPU_INST_CTX 64

// Initialize NPU context arrays
void init_npu();

// Orchestration function to initialize everything
int initialize_npu();

// Initialize XDNA driver (opens /dev/accel/accel0)
// Returns 0 on success, -1 on failure
int initialize_xdna_driver(const char *drv_path = "/dev/accel/accel0");

// Function to export DMA-BUF and import to xdna
// Returns handle (uint32_t) or 0 on error
uint32_t import_dma_buf_to_xdna(void *hip_managed_ptr, size_t size, int dataTypeinBytes);

// Function to import all weights and parameters of a PyTorch module to XDNA
void import_all_weights_to_xdna(torch::nn::Module &module);

// Key: {forM, forK, forN, layer_id, chunk_id} for lookup.
// chunk_id defaults to -1 when configs are not chunk-specific.
using NPUKey = std::array<int, 5>;
using PackedWeightPadSpec = std::array<int64_t, 2>;

// Value: hardware/instruction context indices, actual kernel dimensions, and config
struct NPUValue {
    int hw_idx;
    int inst_idx;
    int npuM;
    int npuK;
    int npuN;
    int cpuN;
    int cpuThreads;
    int config;
    int split_type; // 0=None, 1=M-split, 2=K-split, 3=N-split
};

extern std::vector<hwctxt> hwctxt_array;
extern std::vector<instctxt> instctxt_array;
extern std::map<NPUKey, NPUValue> config_map;
extern std::map<std::string, PackedWeightPadSpec> packed_weight_pad_map;

// Helper to convert layer string to ID
int get_layer_id(const std::string &layer_type);

// Read NPU configuration (debug verbosity, heterogeneity) from kernels.json
void read_npu_config(const std::string &config_path = "");

// Load NPU kernels from JSON configuration
void load_npu_kernels(int drv_fd, const std::string &config_path = "");

// Load NPU-only kernels (strict mode)
void load_npu_only_kernels(int drv_fd, const std::string &config_path = "");

// Helper to determine split type (0=None, 1=M, 2=K, 3=N) for a given layer config
// helper to check if K-split is needed for allocation
int get_split_type(int64_t K, int64_t N, std::string layer_type);

// Get NPU context indices for given dimensions
// Returns {hw_idx, inst_idx} or {-1, -1} if not found
std::pair<int, int> get_npu_context(int M, int K, int N, const std::string &layer_type = "");
PackedWeightPadSpec get_packed_weight_pad_alignment(const std::string &layer_type);

int npuMatmul_zero(int hwctx_numb, int instctx_numb, void *output_pointer, void *input_pointer, void *weight_pointer,
                   uint32_t output_xdna_handle, uint32_t input_xdna_handle, uint32_t weight_xdna_handle, hipEvent_t hip_event = nullptr);
