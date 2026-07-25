/**
 * @file w4a16_gemm.h
 * @brief Fused W4A16 GEMM kernel for custom NPU-style quantized layout
 *
 * This kernel performs fused dequantization and matrix multiplication
 * for 4-bit quantized weights with 16-bit (bf16) activations.
 *
 * Weight Layout (Custom NPU Tile Layout):
 * - Weights are stored in 128x64 tiles (K_tile=128, N_tile=64)
 * - Each tile has packed_params buffer of 4352 bytes:
 *   - 4096 bytes: packed 4-bit weights (128*64/2)
 *   - 128 bytes: bf16 scales (64 scales, one per column)
 *   - 128 bytes: uint8 zeros (64 zeros, duplicated to 128 bytes)
 * - Tiles are stored in ZigZag order for NPU access patterns
 */

#pragma once

#include <hip/hip_runtime.h>

#include <torch/torch.h>

namespace hipkernels {

/**
 * @brief Fused W4A16 quantized GEMM
 *
 * Performs: output = input @ dequant(packed_weights).T
 * Where dequant(w) = (w - zero) * scale
 *
 * @param output Output tensor [batch, seq_len, out_features], bf16
 * @param input Input tensor [batch, seq_len, in_features], bf16
 * @param packed_params Packed quantized parameters (weights, scales, zeros), uint8
 * @param in_features K dimension (input features)
 * @param out_features N dimension (output features)
 * @param stream HIP stream to launch kernel on (default 0)
 */
void w4a16_gemm_fused_packed(torch::Tensor &output, const torch::Tensor &input, const torch::Tensor &packed_params, int64_t in_features,
                             int64_t out_features, hipStream_t stream = 0);

void w4a16_gemv_fused_packed(torch::Tensor &output, const torch::Tensor &input, const torch::Tensor &packed_params, int64_t in_features,
                             int64_t out_features, int64_t total_out_features = -1, int64_t start_col = 0, hipStream_t stream = 0);

/**
 * @brief WMMA-accelerated W4A16 GEMM with tile dequantization
 *
 * Uses RDNA3 tensor cores (WMMA) for ~3x higher throughput.
 * Dequantizes weights to BF16 per tile, then uses WMMA for matrix multiply.
 *
 * @param output Output tensor [batch, seq_len, out_features], bf16
 * @param input Input tensor [batch, seq_len, in_features], bf16
 * @param packed_params Packed quantized parameters (weights, scales, zeros), uint8
 * @param in_features K dimension (input features)
 * @param out_features N dimension (output features)
 */
void w4a16_gemm_wmma_packed(torch::Tensor &output, const torch::Tensor &input, const torch::Tensor &packed_params, int64_t in_features,
                            int64_t out_features);

/**
 * @brief Check if HIP kernel is available for current device
 */
bool is_hip_available();

/**
 * @brief Synchronize HIP stream
 */
void hip_synchronize();

} // namespace hipkernels
