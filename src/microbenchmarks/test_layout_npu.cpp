#include "unified_llm_w4a16_hetero/npuSetup.hpp"
#include "unified_llm_w4a16_hetero/unified_llm_w4a16.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <torch/torch.h>
#include <unistd.h> // For access()
#include <vector>

// TEST_CASE options:
// 0 = Deterministic test with synthetic data
// 1 = Random test with synthetic data
// 2 = Real Llama-3-8B weights (q_proj for 4Kx4Kx4K, mlp.up_proj for 4Kx4Kx14336)
#define TEST_CASE 1
// Set to 1 to enable golden reference dequantization (enables slow reference path for correctness checks).
#define TEST_REFERENCE 0

// Helper to load Safetensors
struct SafetensorsTensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t offset_begin;
    uint64_t offset_end;
    bool valid;
    SafetensorsTensorInfo() : valid(false) {}
};

// Forward declare from unified_llm_w4a16.cpp
std::map<std::string, SafetensorsTensorInfo> parse_safetensors_header(const std::string &filename, uint64_t &header_size);
torch::Tensor load_tensor_from_safetensors(const std::string &filename, const SafetensorsTensorInfo &info, uint64_t header_size);
torch::Tensor unpack_awq_qweight(torch::Tensor qweight);
torch::Tensor unpack_awq_qzeros(torch::Tensor qzeros);

// Reference Dequantization Function (Golden Standard)
// Modified to accept unpacked weights (kInt8) directly
torch::Tensor reference_dequantize(torch::Tensor unpacked_weight_, torch::Tensor scale_, torch::Tensor zero_point_, int64_t in_features_,
                                   int64_t out_features_) {
    // torch::NoGradGuard no_grad; // Not needed in simple test
    auto device = unpacked_weight_.device();

    const int64_t in_features = in_features_;
    const int64_t out_features = out_features_;

    // 2) Apply scale / zero via broadcasting-friendly views.
    // Match debug_dequantize_weights logic which uses [In, Out] and [Groups, Out]

    // qweight passed in is [Out, In], but debug path expects [In, Out]
    auto debug_qweight = unpacked_weight_.contiguous().to(torch::kInt8);
    auto debug_scalefactor = scale_;
    auto debug_zeropoint = zero_point_;

    if (debug_scalefactor.dim() == 2) {
        auto apply_group = [&](const torch::Tensor &scales, const torch::Tensor &zeros) {
            // Scales are [Groups, Out] (User Request)
            int64_t n_groups = scales.size(0); // Dimm 0 is Groups
            int64_t group_size = in_features / n_groups;

            // w: [In, Out] -> [Groups, GroupSize, Out]
            auto w_view = debug_qweight.view({n_groups, group_size, out_features});

            // scales: [Groups, Out] -> [Groups, 1, Out] for broadcasting
            auto s_view = scales.unsqueeze(1);
            auto z_view = zeros.unsqueeze(1);

            // Convert Int8 to Int8 for subtraction (-15..15 range)
            // Result will be Int8
            auto w_sub_z = w_view.sub(z_view);

            // Convert to BF16 for multiplication with scales
            auto w_float = w_sub_z.to(torch::kBFloat16);

            // Result: [Groups, GroupSize, Out] -> [In, Out]
            return w_float.mul_(s_view).view({in_features, out_features});
        };

        return apply_group(debug_scalefactor, debug_zeropoint);
    } else {
        std::cout << "Per-channel dequantization: not implemented" << std::endl;
        std::exit(1);
    }
}

// ==========================================
// Custom NPU Packing / Unpacking Implementation
// Derived from mlir-aie-w4a16 test.cpp
// ==========================================

constexpr int LARGE_TILE_SIZE_ROW = 128;
constexpr int LARGE_TILE_SIZE_COL = 64;
constexpr int SMALL_TILE_SIZE = 8;

// Simple progress bar helper
void print_progress(int current, int total, const char *prefix) {
    int bar_width = 40;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(bar_width * progress);

    std::cout << "\r" << prefix << " [";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "%" << std::flush;
    if (current == total)
        std::cout << std::endl;
}

// Pack weights, scales, and zeros into NPU-friendly format
std::vector<uint8_t> pack_weights_custom(const torch::Tensor &weights_in, // [K, N] uint8
                                         const torch::Tensor &scales_in,  // [N, Groups] bf16
                                         const torch::Tensor &zeros_in    // [N, Groups] uint8
) {
    int64_t K = weights_in.size(0);
    int64_t N = weights_in.size(1);

    int num_large_tiles = (K / LARGE_TILE_SIZE_ROW) * (N / LARGE_TILE_SIZE_COL);
    // New Size: 128*32 (weights) + 64*2 (scales) + 64*2 (zeros) = 4096 + 128 + 128 = 4352 bytes
    size_t B_SIZE = num_large_tiles * (128 * 32 + 64 * 2 + 64 * 2);

    std::vector<uint8_t> BVec(B_SIZE);
    char *BVec_bytes = reinterpret_cast<char *>(BVec.data());

    // Accessors
    auto weights_acc = weights_in.accessor<uint8_t, 2>();
    auto scales_acc = scales_in.accessor<at::BFloat16, 2>(); // [Groups, N]
    auto zeros_acc = zeros_in.accessor<uint8_t, 2>();        // [Groups, N]

    int large_tile_index = 0;
    int last_progress = -1;
    for (int col_mod = 0; col_mod < 8; col_mod++) {
        for (int large_tile_col = col_mod; large_tile_col < N / LARGE_TILE_SIZE_COL; large_tile_col += 8) {
            for (int large_tile_row = 0; large_tile_row < K / LARGE_TILE_SIZE_ROW; large_tile_row++) {
                // Progress bar update
                int progress_pct = (large_tile_index * 100) / num_large_tiles;
                if (progress_pct != last_progress) {
                    print_progress(large_tile_index, num_large_tiles, "Packing");
                    last_progress = progress_pct;
                }

                // Calculate byte offset for this large tile
                size_t large_tile_offset = large_tile_index * (128 * 32 + 64 * 2 + 64 * 2);

                // Pack int4 weights (128x64 weights -> 128x32 bytes)
                for (int small_tile_row = 0; small_tile_row < LARGE_TILE_SIZE_ROW / SMALL_TILE_SIZE; small_tile_row++) {
                    for (int small_tile_col = 0; small_tile_col < LARGE_TILE_SIZE_COL / SMALL_TILE_SIZE; small_tile_col++) {
                        int small_tile_index = small_tile_row * (LARGE_TILE_SIZE_COL / SMALL_TILE_SIZE) + small_tile_col;
                        size_t small_tile_offset = small_tile_index * (SMALL_TILE_SIZE * SMALL_TILE_SIZE / 2);

                        for (int i = 0; i < SMALL_TILE_SIZE; i++) {
                            for (int j = 0; j < SMALL_TILE_SIZE; j += 2) {
                                int matrix_row1 = large_tile_row * LARGE_TILE_SIZE_ROW + small_tile_row * SMALL_TILE_SIZE + i;
                                int matrix_col1 = large_tile_col * LARGE_TILE_SIZE_COL + small_tile_col * SMALL_TILE_SIZE + j;
                                int matrix_row2 = matrix_row1;
                                int matrix_col2 = matrix_col1 + 1;

                                if (matrix_row1 < K && matrix_col1 < N) {
                                    uint8_t w1 = weights_acc[matrix_row1][matrix_col1];
                                    uint8_t w2 = weights_acc[matrix_row2][matrix_col2];

                                    uint8_t packed_byte = (w1 & 0x0F) | ((w2 & 0x0F) << 4);

                                    size_t byte_offset = large_tile_offset + small_tile_offset + (i * SMALL_TILE_SIZE + j) / 2;
                                    BVec_bytes[byte_offset] = packed_byte;
                                }
                            }
                        }
                    }
                }

                // Pack Scales: 64 scales * 2 bytes = 128 bytes
                size_t scale_offset = large_tile_offset + 128 * 32;

                for (int i = 0; i < 64; i++) {
                    int g = large_tile_row;
                    int col = large_tile_col * 64 + i;

                    at::BFloat16 val = scales_acc[g][col];

                    // Write 1 BF16 (2 bytes)
                    at::BFloat16 *scale_ptr = reinterpret_cast<at::BFloat16 *>(BVec_bytes + scale_offset + i * sizeof(at::BFloat16));
                    *scale_ptr = val;
                }

                // Pack Zeros: 64 zeros * 1 byte... duplicated to align -> 128 bytes
                size_t zero_point_offset = large_tile_offset + 128 * 32 + 64 * 2;

                for (int group = 0; group < LARGE_TILE_SIZE_COL / SMALL_TILE_SIZE; group++) {
                    for (int repeat = 0; repeat < 2; repeat++) {
                        for (int i = 0; i < 8; i++) {
                            int g = large_tile_row;
                            int col = large_tile_col * 64 + group * 8 + i;

                            uint8_t z = zeros_acc[g][col];

                            // Offset: group * 16 + repeat * 8 + i
                            BVec_bytes[zero_point_offset + group * 16 + repeat * 8 + i] = z;
                        }
                    }
                }

                large_tile_index++;
            }
        }
    }
    print_progress(num_large_tiles, num_large_tiles, "Packing");
    return BVec;
}

// Unpack function (Reverse of above)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> unpack_weights_custom(const std::vector<uint8_t> &packed_data, int64_t K,
                                                                              int64_t N) {
    auto weights_out = torch::zeros({K, N}, torch::kUInt8);
    auto weights_acc = weights_out.accessor<uint8_t, 2>();

    auto scales_out = torch::zeros({K / 128, N}, torch::kBFloat16); // [Groups, N]
    auto zeros_out = torch::zeros({K / 128, N}, torch::kUInt8);     // [Groups, N]

    const char *BVec_bytes = reinterpret_cast<const char *>(packed_data.data());

    int num_large_tiles = (K / LARGE_TILE_SIZE_ROW) * (N / LARGE_TILE_SIZE_COL);
    int large_tile_index = 0;
    int last_progress = -1;
    for (int col_mod = 0; col_mod < 8; col_mod++) {
        for (int large_tile_col = col_mod; large_tile_col < N / LARGE_TILE_SIZE_COL; large_tile_col += 8) {
            for (int large_tile_row = 0; large_tile_row < K / LARGE_TILE_SIZE_ROW; large_tile_row++) {
                // Progress bar update
                int progress_pct = (large_tile_index * 100) / num_large_tiles;
                if (progress_pct != last_progress) {
                    print_progress(large_tile_index, num_large_tiles, "Unpacking");
                    last_progress = progress_pct;
                }

                size_t large_tile_offset = large_tile_index * (128 * 32 + 64 * 2 + 64 * 2);

                // 1. Unpack Weights (Same as before)
                for (int small_tile_row = 0; small_tile_row < 16; small_tile_row++) {
                    for (int small_tile_col = 0; small_tile_col < 8; small_tile_col++) {
                        int small_tile_index = small_tile_row * 8 + small_tile_col;
                        size_t small_tile_offset = small_tile_index * 32;

                        for (int i = 0; i < 8; i++) {
                            for (int j = 0; j < 8; j += 2) {
                                size_t byte_offset = large_tile_offset + small_tile_offset + (i * 8 + j) / 2;
                                uint8_t packed = BVec_bytes[byte_offset];
                                uint8_t w1 = packed & 0x0F;
                                uint8_t w2 = (packed >> 4) & 0x0F;

                                int r1 = large_tile_row * 128 + small_tile_row * 8 + i;
                                int c1 = large_tile_col * 64 + small_tile_col * 8 + j;
                                int r2 = r1;
                                int c2 = c1 + 1;

                                if (r1 < K && c1 < N)
                                    weights_acc[r1][c1] = w1;
                                if (r2 < K && c2 < N)
                                    weights_acc[r2][c2] = w2;
                            }
                        }
                    }
                }

                // 2. Unpack Scales
                size_t scale_offset = large_tile_offset + 4096;
                for (int i = 0; i < 64; i++) {
                    const at::BFloat16 *src = reinterpret_cast<const at::BFloat16 *>(BVec_bytes + scale_offset + i * 2);
                    int c = large_tile_col * 64 + i;
                    int g = large_tile_row;

                    if (g < K / 128 && c < N) {
                        scales_out[g][c] = *src;
                    }
                }

                // 3. Unpack Zeros
                size_t zero_point_offset = large_tile_offset + 4096 + 128;
                for (int group = 0; group < 8; group++) { // group < LARGE_TILE_SIZE_COL / SMALL_TILE_SIZE
                    // repeat = 0
                    for (int i = 0; i < 8; i++) {
                        int c = large_tile_col * 64 + group * 8 + i;
                        int g = large_tile_row;

                        uint8_t z = BVec_bytes[zero_point_offset + group * 16 + i];

                        if (g < K / 128 && c < N) {
                            zeros_out.index_put_({g, c}, (int)z);
                        }
                    }
                }

                large_tile_index++;
            }
        }
    }
    print_progress(num_large_tiles, num_large_tiles, "Unpacking");

    // Return [K, N], [Groups, N], [Groups, N]
    return std::make_tuple(weights_out.contiguous().to(torch::kInt8), scales_out, zeros_out);
}

int main() {
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA/HIP not available! Cannot run NPU test." << std::endl;
        return 1;
    }
    auto device = torch::kCUDA;

    // Initialize XDNA
    std::cout << "Initializing XDNA driver..." << std::endl;
    if (initialize_xdna_driver() != 0) {
        std::cerr << "Failed to initialize XDNA driver!" << std::endl;
        return 1;
    }
    init_npu();

    // Dimensions
    int64_t M = 4096;
    int64_t K = 8192;  // In Features: 4096, 8192
    int64_t N = 28672; // Out Features: 14336, 28672 (4096 for q_proj, 14336 for mlp.up_proj)
    int64_t group_size = 128;
    int64_t num_groups = K / group_size;

    std::cout << "Dimensions: M=" << M << ", K=" << K << ", N=" << N << ", Groups=" << num_groups << std::endl;

    torch::Tensor B_in_out;
    torch::Tensor scales;
    torch::Tensor zeros;
    torch::Tensor x_test;

    // qweight will be [Out, In] after processing
    torch::Tensor qweight;

    if (TEST_CASE == 1) {
        std::cout << "Starting Random Test (TEST_CASE 1)..." << std::endl;

        // Set seed for reproducibility
        torch::manual_seed(42);

        // Random Weights (uint8, 0-15)
        B_in_out = torch::randint(0, 16, {K, N}, torch::kUInt8);

        // Random Zeros (uint8, 0-15), [Groups, N]
        zeros = torch::randint(0, 16, {num_groups, N}, torch::kUInt8);

        // Random Scales (BF16 compatible), [Groups, N]
        auto scales_float = torch::rand({num_groups, N}, torch::kFloat32); // 0..1
        auto scales_bf16 = scales_float.to(torch::kBFloat16) * 0.1f;
        scales = scales_bf16.clone();

        // Random Input (BF16)
        x_test = torch::rand({M, K}, torch::kBFloat16) * 0.01f;

        // Convert to [Out, In] for packing (Standard layout)
        qweight = B_in_out.contiguous(); // [Out, In]

    } else if (TEST_CASE == 2) {
        // Load real Llama-3-8B weights from safetensors
        std::cout << "Starting Real Weights Test (TEST_CASE 2)..." << std::endl;

        // Set seed for reproducibility
        torch::manual_seed(42);

        const char *env_root_weight = std::getenv("HETEROMOSAIC_ROOT");
        std::string root_weight = env_root_weight ? env_root_weight : "/home/greg/Desktop/heteroMosaic";
        std::string safetensors_path = root_weight + "/py/unified_llm_w4a16/model_weights/llama-3-8b-instruct-awq.safetensors";

        // Check if file exists
        if (access(safetensors_path.c_str(), F_OK) == -1) {
            std::cerr << "Error: Safetensors file not found at " << safetensors_path << std::endl;
            std::cerr << "Please download the model weights first." << std::endl;
            return 1;
        }

        uint64_t header_size = 0;
        auto tensor_map = parse_safetensors_header(safetensors_path, header_size);
        if (tensor_map.empty()) {
            std::cerr << "Failed to parse safetensors header" << std::endl;
            return 1;
        }

        // Select layer based on dimensions
        std::string layer_prefix;
        if (N == 4096) {
            // Q projection: 4096 -> 4096
            layer_prefix = "model.layers.0.self_attn.q_proj";
            std::cout << "Loading layer0 Q projection (4096x4096)..." << std::endl;
        } else if (N == 14336) {
            // MLP up projection: 4096 -> 14336
            layer_prefix = "model.layers.0.mlp.up_proj";
            std::cout << "Loading layer0 MLP up projection (4096x14336)..." << std::endl;
        } else {
            std::cerr << "Error: No Llama-3-8B layer matches dimensions K=" << K << " N=" << N << std::endl;
            std::cerr << "Supported: 4096x4096 (q_proj) or 4096x14336 (mlp.up_proj)" << std::endl;
            return 1;
        }

        // Load raw tensors
        auto qweight_raw = load_tensor_from_safetensors(safetensors_path, tensor_map[layer_prefix + ".qweight"], header_size);
        auto scales_raw = load_tensor_from_safetensors(safetensors_path, tensor_map[layer_prefix + ".scales"], header_size);
        auto qzeros_raw = load_tensor_from_safetensors(safetensors_path, tensor_map[layer_prefix + ".qzeros"], header_size);

        if (!qweight_raw.defined() || !scales_raw.defined() || !qzeros_raw.defined()) {
            std::cerr << "Failed to load tensors for " << layer_prefix << std::endl;
            return 1;
        }
        std::cout << "Loaded weights: qweight=" << qweight_raw.sizes() << " scales=" << scales_raw.sizes()
                  << " zeros=" << qzeros_raw.sizes() << std::endl;

        // 1. Unpack Weights: AWQ format [In, Out/8] int32 -> [Out, In] uint8
        std::cout << "Unpacking AWQ weights..." << std::endl;
        auto qweight_unpacked = unpack_awq_qweight(qweight_raw); // Returns [Out, In] uint8
        qweight = qweight_unpacked;

        // 2. Unpack Zeros: AWQ format [Groups, Out/8] int32 -> [Out, Groups] uint8
        std::cout << "Unpacking AWQ zeros..." << std::endl;
        auto zeros_unpacked = unpack_awq_qzeros(qzeros_raw).to(torch::kUInt8);
        zeros = zeros_unpacked.contiguous(); // Transpose to [Out, Groups]

        // 3. Scales: [Groups, Out] fp16 -> [Groups, Out] bf16
        scales = scales_raw.to(torch::kBFloat16).contiguous(); // [Groups, Out]

        // Update dimensions from actual weights
        K = qweight.size(0); // In
        N = qweight.size(1); // Out
        num_groups = K / group_size;

        std::cout << "Actual dimensions: K=" << K << " N=" << N << " Groups=" << num_groups << std::endl;
        std::cout << "qweight: " << qweight.sizes() << " scales: " << scales.sizes() << " zeros: " << zeros.sizes() << std::endl;

        // Random Input (BF16) - use seeded random for reproducibility
        torch::manual_seed(42);
        x_test = torch::rand({M, K}, torch::kBFloat16) * 0.2f;

    } else {
        std::cout << "Starting Deterministic Test (TEST_CASE 0)..." << std::endl;

        // B Matrix: Cyclic int4 values 0-15
        B_in_out = torch::zeros({K, N}, torch::kUInt8);
        for (int64_t i = 0; i < K; ++i) {
            uint8_t val = (i % 16);
            B_in_out[i].fill_(val);
        }

        // Scales: 0.1, 0.2, ... 0.8 (Per-Input)
        auto scales_per_group = torch::arange(1, num_groups + 1, torch::kFloat32) * 0.1f;
        scales = scales_per_group.unsqueeze(1).expand({num_groups, N}).clone().to(torch::kBFloat16);

        // Zeros: 0
        zeros = torch::zeros({num_groups, N}, torch::kUInt8);

        // X: 0.1
        x_test = torch::full({M, K}, 0.1, torch::kBFloat16);

        // Convert to [Out, In] for packing (Standard layout)
        qweight = B_in_out.contiguous(); // [Out, In]
    }
    std::cout << "qweight: shape=" << qweight.sizes() << ", type=" << qweight.dtype() << std::endl;
    std::cout << "scales:  shape=" << scales.sizes() << ", type=" << scales.dtype() << std::endl;
    std::cout << "zeros:   shape=" << zeros.sizes() << ", type=" << zeros.dtype() << std::endl;

    // Set up input tensor for NPU
    auto x_test_gpu = x_test.to(device);
    uint32_t handle_in = import_dma_buf_to_xdna(x_test_gpu.data_ptr(), x_test_gpu.numel(), 2);

    // Reference tensor for comparison (will be set based on TEST_REFERENCE)
    torch::Tensor y_ref;

    // Manual calculation disabled/guarded
    if (TEST_CASE == 0) {
        // manual calculation
        float activation_value = 0.1f;
        float expected_val = 0.0f;
        for (int i = 0; i < K; ++i) {
            uint8_t val = (i % 16);
            uint8_t zero = 0;
            float scale = (i / 128 + 1) * 0.1f;

            expected_val += (val - zero) * scale * activation_value;
        }
        std::cout << "Expected value: " << expected_val << std::endl;
    }

    torch::Tensor w_npu_dequant;
#if TEST_REFERENCE
    // 3. Dequantize Reference
    std::cout << "Running Reference Dequantization..." << std::endl;
    // scales/zeros are [Groups, N]. reference_dequantize matches [Groups, N].
    auto w_ref = reference_dequantize(qweight, scales.contiguous(), zeros.contiguous(), K, N); // [In, Out]

    // Copy for GPU Reference
    w_ref = w_ref.to(device);
    y_ref = torch::matmul(x_test_gpu, w_ref); // [M, Out]

    // Print first 8 values of y_ref with 2 decimal digits
    std::cout << "First 8 values of y_ref: ";
    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < 8; ++i) {
        std::cout << y_ref[0][i].item<float>() << " ";
    }
    std::cout << std::endl;
    std::cout.unsetf(std::ios::fixed);

    // Test Custom Packing/Unpacking
    std::cout << "\nTesting Custom NPU Packing/Unpacking..." << std::endl;
    // B_in_out [K, N], scales [N, Groups], zeros [N, Groups]
    auto packed_npu = pack_weights_custom(qweight, scales, zeros);

    auto [unpacked_w_npu, unpacked_s_npu, unpacked_z_npu] = unpack_weights_custom(packed_npu, K, N);

    // Simple dense dequantize: (w - z) * s
    // w is [K, N] (In, Out).
    // z, s are [Groups, N].
    // We need to broadcast z/s to [K, N].
    // [Groups, N] -> [Groups*128, N] -> [K, N]

    // Repeat interleave rows (dim 0)
    auto s_expanded = unpacked_s_npu.repeat_interleave(128, 0);
    auto z_expanded = unpacked_z_npu.repeat_interleave(128, 0);

    w_npu_dequant = (unpacked_w_npu.to(torch::kInt8) - z_expanded.to(torch::kInt8)).to(torch::kBFloat16) * s_expanded;

    // GEMM with NPU unpacked weights
    // w_npu_dequant is [In, Out].
    w_npu_dequant = w_npu_dequant.to(device);
    auto y_custom = torch::matmul(x_test_gpu, w_npu_dequant);

    std::cout << "First 8 values of y_custom: ";
    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < 8; ++i) {
        std::cout << y_custom[0][i].item<float>() << " ";
    }
    std::cout << std::endl;

    // Compare
    std::cout << "Shape of y_ref: " << y_ref.sizes() << std::endl;
    std::cout << "Shape of y_custom: " << y_custom.sizes() << std::endl;
    bool match = torch::allclose(y_ref, y_custom, 0.00, 0.00);
    if (match) {
        std::cout << "SUCCESS: y_custom matches y_ref!" << std::endl;
    } else {
        std::cout << "FAILURE: y_custom mismatch!" << std::endl;
        std::cout << "Max diff: " << (y_ref - y_custom).abs().max().item<float>() << std::endl;
    }
#else
    std::cout << "Skipping reference dequantization test (TEST_REFERENCE=0)..." << std::endl;
#endif

    // Test QuantizedLinear (always run - this becomes the reference when TEST_REFERENCE=0)
    std::cout << "\nTesting QuantizedLinear..." << std::endl;
    QuantizedLinearImpl layer(K, N, false);
    // Transpose [Groups, N] -> [N, Groups] for the layer (No longer needed as layer expects [Groups, N])
    layer.set_quantized_weights(qweight, scales, zeros, torch::Tensor());

    auto w_method = layer.dequantize_weights_packed().to(torch::kBFloat16);
    auto w_method_gpu = w_method.to(device);
    auto y_method = torch::matmul(x_test_gpu, w_method_gpu.t());

    std::cout << "First 8 values of y_method: ";
    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < 8; ++i) {
        std::cout << y_method[0][i].item<float>() << " ";
    }
    std::cout << std::endl;

#if TEST_REFERENCE
    std::cout << std::endl;
    std::cout << "Comparing layer.get_packed_params() vs packed_npu" << std::endl;
    // Get packed params on CPU for comparison
    auto packed_layer = layer.get_packed_params().cpu();
    auto packed_npu_tensor = torch::from_blob(packed_npu.data(), {(long)packed_npu.size()}, torch::kUInt8).clone(); // [Size]

    std::cout << "Shape of packed_layer: " << packed_layer.sizes() << std::endl;
    std::cout << "Shape of packed_npu:   " << packed_npu_tensor.sizes() << std::endl;

    if (packed_layer.sizes() != packed_npu_tensor.sizes()) {
        std::cout << "Packed params Shape mismatch!" << std::endl;
        std::cout << "Layer: " << packed_layer.sizes() << " vs NPU: " << packed_npu_tensor.sizes() << std::endl;
    } else {
        bool match_packed = torch::equal(packed_layer, packed_npu_tensor);
        bool almost_match = torch::allclose(packed_layer.to(torch::kFloat32), packed_npu_tensor.to(torch::kFloat32));

        if (match_packed) {
            std::cout << "SUCCESS: Packed params match exact!" << std::endl;
        } else if (almost_match) {
            std::cout << "SUCCESS: Packed params match (allclose)!" << std::endl;
        } else {
            std::cout << "FAILURE: Packed params mismatch!" << std::endl;
            auto diff = (packed_layer.to(torch::kFloat32) - packed_npu_tensor.to(torch::kFloat32)).abs();
            std::cout << "Max diff: " << diff.max().item<float>() << std::endl;
        }
    }

    std::cout << "Comparing y_method with y_ref..." << std::endl;
    bool match_method = torch::allclose(y_ref, y_method, 0.05, 0.05);
    if (match_method) {
        std::cout << "SUCCESS: y_method matches y_ref!" << std::endl;
    } else {
        std::cout << "FAILURE: y_method mismatch!" << std::endl;
        std::cout << "Max diff: " << (y_ref - y_method).abs().max().item<float>() << std::endl;
    }

#else
    // Use method output as reference for NPU comparison
    y_ref = y_method.clone();
    auto packed_params = layer.get_packed_params();
    std::cout << "Using QuantizedLinear output as reference for NPU comparison." << std::endl;
#endif

    // 8. NPU Execution (only if kernel exists for this dimension)
    std::cout << "\n============================================" << std::endl;
    std::cout << "Starting NPU Integration Test (Layout Verification)" << std::endl;
    std::cout << "============================================" << std::endl;

    // Check if we have a kernel for this dimension
    // Check if we have a kernel for this dimension
    const char *env_root = std::getenv("HETEROMOSAIC_ROOT");
    std::string root_dir = env_root ? env_root : "/home/greg/Desktop/heteroMosaic";
    std::string xclbin_path, inst_path;
    bool kernel_available = false;

    if (M == 4096 && K == 8192 && N == 28672) {
        xclbin_path = root_dir + "/hw_bins/npu2/8192x8192x28672/bf16_int4AWQ_bf16_M/final_4096x8192x28672_64x128x64_8c_bf16_int4AWQ_bf16.pdi";
        inst_path = root_dir + "/hw_bins/npu2/8192x8192x28672/bf16_int4AWQ_bf16_M/insts_4096x8192x28672_64x128x64_8c_bf16_int4AWQ_bf16.txt";
        kernel_available = true;
    } else if (M == 4096 && K == 4096 && N == 14336) {
        xclbin_path = root_dir + "/hw_bins/npu2/8192x4096x14336/bf16_int4AWQ_bf16_M/final_2048x4096x14336_64x128x64_8c_bf16_int4AWQ_bf16.pdi";
        inst_path = root_dir + "/hw_bins/npu2/8192x4096x14336/bf16_int4AWQ_bf16_M/insts_4096x4096x14336_64x128x64_8c_bf16_int4AWQ_bf16.txt";
        kernel_available = true;
    } else if (M == 4096 && K == 4096 && N == 4096) {
        xclbin_path = root_dir + "/hw_bins/w4a16/final_4096x4096x4096_64x128x64_8c.pdi";
        inst_path = root_dir + "/hw_bins/w4a16/insts_4096x4096x4096_64x128x64_8c.txt";
        kernel_available = true;
    } else if (M == 1024 && K == 1024 && N == 2048) {
        xclbin_path = root_dir + "/hw_bins/w4a16/final_1024x1024x2048_64x128x64_8c.pdi";
        inst_path = root_dir + "/hw_bins/w4a16/insts_1024x1024x2048_64x128x64_8c.txt";
        kernel_available = true;
    }

    if (!kernel_available) {
        std::cout << "Skipping NPU Execution: No kernel available for dimensions " << M << "x" << K << "x" << N << std::endl;
        std::cout << "\nTest completed (NPU execution skipped)." << std::endl;
        return 0;
    }

    std::cout << "Loading PDI: " << xclbin_path << std::endl;
    if (createHWctxt(xdna_drv_fd, hwctxt_array[0], xclbin_path.c_str(), 32) != 0) {
        std::cerr << "Failed to create HW context!" << std::endl;
        return 1;
    }

    std::cout << "Loading Instructions: " << inst_path << std::endl;
    if (createInstctxt(xdna_drv_fd, instctxt_array[0], inst_path.c_str(), true) != 0) {
        std::cerr << "Failed to create Inst context!" << std::endl;
        return 1;
    }

    // Packed Weights for NPU
#if TEST_REFERENCE
    // Use packed_npu from pack_weights_custom (std::vector -> tensor)
    auto packed_tensor_cpu = torch::from_blob(packed_npu.data(), {(long)packed_npu.size()}, torch::kUInt8);
    auto packed_cuda = packed_tensor_cpu.to(device);
#else
    // Use packed_params from QuantizedLinearImpl (already a tensor)
    auto packed_cuda = packed_params.to(device);
#endif
    uint32_t handle_w = import_dma_buf_to_xdna(packed_cuda.data_ptr(), packed_cuda.numel(), 1);

    // Output Y
    auto y_npu = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));
    uint32_t handle_out = import_dma_buf_to_xdna(y_npu.data_ptr(), y_npu.numel(), 2);

    std::cout << "Running NPU GEMM (8 Iterations)..." << std::endl;
    // Flush caches for CPU written data before execution
    FlushCpuCache((const void *)instctxt_array[0].dpu_0_vaddr, 0, instctxt_array[0].num_dpu_0_insts * sizeof(uint32_t));
    FlushCpuCache((const void *)hwctxt_array[0].pdi_vaddr, 0, hwctxt_array[0].pdi_size);

    int num_runs = 8;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        // Cast nullptr to hipEvent_t to resolve overload ambiguity
        int ret = npuMatmul_zero(0, 0, y_npu.data_ptr(), x_test_gpu.data_ptr(), packed_cuda.data_ptr(), handle_out, handle_in, handle_w,
                                 (hipEvent_t) nullptr);

        if (ret != 0) {
            std::cerr << "NPU Execution Failed on iteration " << i << "!" << std::endl;
            return 1;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "NPU Execution Successful!" << std::endl;

    std::chrono::duration<double> diff = end - start;
    double avg_latency = diff.count() / num_runs;

    // TOPS Calculation
    // Ops = 2 * M * N * K
    // TOPS = Ops / (Time * 1e12)
    double ops = 2.0 * M * K * N;
    double tops = ops / (avg_latency * 1e12);

    std::cout << "Average Latency: " << avg_latency * 1000.0 << " ms" << std::endl;
    std::cout << "Performance: " << tops << " TOPS" << std::endl;

    // Comparison
    std::cout << "Comparing NPU output with Reference..." << std::endl;
    auto y_npu_cpu = y_npu.cpu();

    // Check first 8 elements
    std::cout << "First 8 values of y_npu: ";
    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < 8; ++i) {
        std::cout << y_npu_cpu[0][i].item<float>() << " ";
    }
    std::cout << std::endl;

    bool match_npu = torch::allclose(y_ref.cpu(), y_npu_cpu, 0.2, 0.2);
    if (match_npu) {
        std::cout << "SUCCESS: y_npu matches y_ref!" << std::endl;
    } else {
        std::cout << "FAILURE: y_npu mismatch!" << std::endl;
        auto diff = (y_ref.cpu() - y_npu_cpu).abs();
        std::cout << "Max diff: " << diff.max().item<float>() << std::endl;
    }

    return 0;
}