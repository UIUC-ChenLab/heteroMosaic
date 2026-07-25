#include "hipkernels/flash_attn_decode.hpp"
#include <algorithm>
#include <c10/hip/HIPStream.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <torch/torch.h>

#define HIP_CHECK(call)                                                                                                                    \
    do {                                                                                                                                   \
        hipError_t _e = (call);                                                                                                            \
        if (_e != hipSuccess) {                                                                                                            \
            std::cerr << "HIP error: " << hipGetErrorString(_e) << " at " << __FILE__ << ":" << __LINE__ << std::endl;                     \
            std::exit(1);                                                                                                                  \
        }                                                                                                                                  \
    } while (0)

static int getenv_int(const char *name, int default_value) {
    const char *value = std::getenv(name);
    if (!value) {
        return default_value;
    }
    return std::atoi(value);
}

static std::string format_tensor_1d(const torch::Tensor &t) {
    auto cpu = t.to(torch::kFloat32).cpu().contiguous();
    const auto numel = cpu.numel();
    const float *data = cpu.data_ptr<float>();
    std::ostringstream oss;
    oss << "[";
    for (int64_t i = 0; i < numel; ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << std::fixed << std::setprecision(6) << data[i];
    }
    oss << "]";
    return oss.str();
}

static torch::Tensor repeat_kv(const torch::Tensor &x, int64_t n_rep) {
    if (n_rep == 1) {
        return x;
    }
    auto sizes = x.sizes();
    int64_t batch = sizes[0];
    int64_t num_kv_heads = sizes[1];
    int64_t seq_len = sizes[2];
    int64_t head_dim = sizes[3];

    auto expanded = x.unsqueeze(2).expand({batch, num_kv_heads, n_rep, seq_len, head_dim});
    return expanded.reshape({batch, num_kv_heads * n_rep, seq_len, head_dim});
}

static void benchmark_prefill(int seq_len, bool use_bf16, bool check_correctness, int iters = 100, bool model_timing = true,
                              int warmup_iters = 0) {
    auto device = torch::kCUDA;
    auto dtype = use_bf16 ? torch::kBFloat16 : torch::kFloat16;
    auto options = torch::TensorOptions().device(device).dtype(dtype);

    const int batch_size = 1;
    const int n_heads_q = 32;
    const int n_heads_kv = 8;
    const int gqa = n_heads_q / n_heads_kv;
    const int head_dim = 128;
    const int q_len = seq_len;

    torch::manual_seed(1234);
    torch::Tensor q;
    torch::Tensor k;
    torch::Tensor v;

    q = torch::rand({batch_size, n_heads_q, q_len, head_dim}, options);
    k = torch::rand({batch_size, n_heads_kv, seq_len, head_dim}, options);
    v = torch::rand({batch_size, n_heads_kv, seq_len, head_dim}, options);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int element_size = q.element_size();
    auto stream = c10::hip::getCurrentHIPStream().stream();

    torch::Tensor attn_output = torch::empty_like(q);
    const int stride_q_seq = q.stride(2) * element_size;
    const int stride_q_head = q.stride(1) * element_size;
    const int stride_q_batch = q.stride(0) * element_size;
    const int stride_k_seq = k.stride(2) * element_size;
    const int stride_k_head = k.stride(1) * element_size;
    const int stride_k_batch = k.stride(0) * element_size;
    const int stride_v_seq = v.stride(2) * element_size;
    const int stride_v_head = v.stride(1) * element_size;
    const int stride_v_batch = v.stride(0) * element_size;
    const int stride_o_seq = attn_output.stride(2) * element_size;
    const int stride_o_head = attn_output.stride(1) * element_size;
    const int stride_o_batch = attn_output.stride(0) * element_size;

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        // commented for now do not delete
        // auto k_attn = repeat_kv(k, gqa);
        // auto v_attn = repeat_kv(v, gqa);
        // attn_output = torch::scaled_dot_product_attention(q, k_attn, v_attn, c10::nullopt, 0.0, true, std::nullopt, false);
        launch_flash_attn_prefill_hip(q.data_ptr(), k.data_ptr(), v.data_ptr(), attn_output.data_ptr(), batch_size, n_heads_q, n_heads_kv,
                                      head_dim, q_len, seq_len, scale, stride_q_seq, stride_q_head, stride_q_batch, stride_k_seq,
                                      stride_k_head, stride_k_batch, stride_v_seq, stride_v_head, stride_v_batch, stride_o_seq,
                                      stride_o_head, stride_o_batch, /*q_start=*/0, use_bf16, stream);
    }
    HIP_CHECK(hipDeviceSynchronize());

    float avg_ms = 0.0f;
    if (model_timing) {
        torch::cuda::synchronize();
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            // commented for now do not delete
            // auto k_attn = repeat_kv(k, gqa);
            // auto v_attn = repeat_kv(v, gqa);
            // attn_output = torch::scaled_dot_product_attention(q, k_attn, v_attn, c10::nullopt, 0.0, true, std::nullopt, false);
            launch_flash_attn_prefill_hip(q.data_ptr(), k.data_ptr(), v.data_ptr(), attn_output.data_ptr(), batch_size, n_heads_q,
                                          n_heads_kv, head_dim, q_len, seq_len, scale, stride_q_seq, stride_q_head, stride_q_batch,
                                          stride_k_seq, stride_k_head, stride_k_batch, stride_v_seq, stride_v_head, stride_v_batch,
                                          stride_o_seq, stride_o_head, stride_o_batch, /*q_start=*/0, use_bf16, stream);
        }
        torch::cuda::synchronize();
        auto end = std::chrono::high_resolution_clock::now();
        avg_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f;
        avg_ms /= static_cast<float>(iters);
    } else {
        hipEvent_t start;
        hipEvent_t stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        HIP_CHECK(hipEventRecord(start, stream));
        for (int i = 0; i < iters; ++i) {
            // commented for now do not delete
            // auto k_attn = repeat_kv(k, gqa);
            // auto v_attn = repeat_kv(v, gqa);
            // attn_output = torch::scaled_dot_product_attention(q, k_attn, v_attn, c10::nullopt, 0.0, true, std::nullopt, false);
            launch_flash_attn_prefill_hip(q.data_ptr(), k.data_ptr(), v.data_ptr(), attn_output.data_ptr(), batch_size, n_heads_q,
                                          n_heads_kv, head_dim, q_len, seq_len, scale, stride_q_seq, stride_q_head, stride_q_batch,
                                          stride_k_seq, stride_k_head, stride_k_batch, stride_v_seq, stride_v_head, stride_v_batch,
                                          stride_o_seq, stride_o_head, stride_o_batch, /*q_start=*/0, use_bf16, stream);
        }
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        avg_ms = ms / static_cast<float>(iters);

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    std::cout << "Attention Shapes:" << " (dtype=" << (use_bf16 ? "bf16" : "fp16") << ")" << std::endl;
    std::cout << "Q: " << q.sizes() << std::endl;
    std::cout << "K: " << k.sizes() << std::endl;
    std::cout << "V: " << k.sizes() << std::endl;

    if (check_correctness) {
        HIP_CHECK(hipDeviceSynchronize());

        const int64_t check_q = std::min<int64_t>(8, q.size(2));
        auto q_sub = q.slice(2, 0, check_q);

        torch::Tensor k_rep = k;
        torch::Tensor v_rep = v;
        if (n_heads_q != n_heads_kv) {
            const int gqa = n_heads_q / n_heads_kv;
            k_rep = repeat_kv(k, gqa);
            v_rep = repeat_kv(v, gqa);
        }
        std::cout << "k_rep: " << k_rep.sizes() << std::endl;
        std::cout << "v_rep: " << v_rep.sizes() << std::endl;

        auto scores = torch::matmul(q_sub, k_rep.transpose(-2, -1)) * scale; // [B, Hq, Q, S]
        auto mask = torch::full({check_q, seq_len}, -std::numeric_limits<float>::infinity(),
                                torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));
        mask = torch::triu(mask, 1);

        auto scores_f = scores.to(torch::kFloat32) + mask;
        auto probs = torch::softmax(scores_f, -1).to(q.dtype());
        auto ref = torch::matmul(probs, v_rep); // [B, Hq, Q, D]

        auto out_sub = attn_output.slice(2, 0, check_q).to(torch::kFloat32);
        auto ref_fp32 = ref.to(torch::kFloat32);
        auto diff = (out_sub - ref_fp32).abs();
        auto max_val = diff.max().item<float>();
        double atol = use_bf16 ? 5e-2 : 1e-2;
        double rtol = use_bf16 ? 5e-2 : 1e-2;
        bool ok = torch::allclose(out_sub, ref_fp32, rtol, atol);

        auto out_flat = out_sub.flatten();
        auto ref_flat = ref_fp32.flatten();
        const int64_t n = std::min<int64_t>(8, out_flat.numel());
        auto out_head = out_flat.index({torch::indexing::Slice(0, n)});
        auto ref_head = ref_flat.index({torch::indexing::Slice(0, n)});

        std::cout << "Attention Time: " << std::fixed << std::setprecision(3) << avg_ms << " ms" << std::endl;
        std::cout << "Correctness (slice q_len=" << check_q << "): " << (ok ? "PASS" : "FAIL") << " | max_abs_diff=" << std::setprecision(6)
                  << max_val << " | rtol=" << rtol << " atol=" << atol << "\n";
        std::cout << "ref[0:8] = " << format_tensor_1d(ref_head) << "\n";
        std::cout << "out[0:8] = " << format_tensor_1d(out_head) << "\n\n";
    }
}

static void benchmark_decode(int seq_len_kv, bool use_bf16, bool check_correctness, int iters = 100, bool model_layout = false,
                             bool model_timing = false, int warmup_iters = 0) {
    auto device = torch::kCUDA;
    auto dtype = use_bf16 ? torch::kBFloat16 : torch::kFloat16;
    auto options = torch::TensorOptions().device(device).dtype(dtype);

    const int batch_size = 1;
    const int n_heads_q = 32;
    const int n_heads_kv = 8;
    const int head_dim = getenv_int("FLASH_ATTN_DECODE_HEAD_DIM", 128);

    torch::manual_seed(1234);
    torch::Tensor q;
    torch::Tensor k;
    torch::Tensor v;
    torch::Tensor out;

    torch::Tensor q_buf;
    torch::Tensor k_cache;
    torch::Tensor v_cache;

    if (model_layout) {
        const int q_len = 1;
        int cache_seq_len = getenv_int("FLASH_ATTN_DECODE_CACHE_SEQ_LEN", 9216);
        if (cache_seq_len < seq_len_kv) {
            cache_seq_len = seq_len_kv;
        }

        q_buf = torch::rand({batch_size, q_len, n_heads_q * head_dim}, options);
        q = q_buf.view({batch_size, q_len, n_heads_q, head_dim}).transpose(1, 2);

        k_cache = torch::rand({batch_size, n_heads_kv, cache_seq_len, head_dim}, options);
        v_cache = torch::rand({batch_size, n_heads_kv, cache_seq_len, head_dim}, options);
        k = k_cache.narrow(2, 0, seq_len_kv);
        v = v_cache.narrow(2, 0, seq_len_kv);

        out = torch::zeros({batch_size, n_heads_q, q_len, head_dim}, options);
    } else {
        q = torch::rand({batch_size, n_heads_q, 1, head_dim}, options);
        k = torch::rand({batch_size, n_heads_kv, seq_len_kv, head_dim}, options);
        v = torch::rand({batch_size, n_heads_kv, seq_len_kv, head_dim}, options);
        out = torch::empty_like(q);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int element_size = q.element_size();
    auto stream = c10::hip::getCurrentHIPStream().stream();

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        if (head_dim == 256) {
            launch_flash_attn_decode_hip_hd256(
                q.data_ptr(), k.data_ptr(), v.data_ptr(), nullptr, out.data_ptr(), batch_size, n_heads_q, n_heads_kv, head_dim, seq_len_kv,
                scale, q.stride(2) * element_size, q.stride(1) * element_size, q.stride(0) * element_size, k.stride(2) * element_size,
                k.stride(1) * element_size, k.stride(0) * element_size, v.stride(2) * element_size, v.stride(1) * element_size,
                v.stride(0) * element_size, 0, use_bf16, stream);
        } else {
            launch_flash_attn_decode_hip(q.data_ptr(), k.data_ptr(), v.data_ptr(), nullptr, out.data_ptr(), batch_size, n_heads_q,
                                         n_heads_kv, head_dim, seq_len_kv, scale, q.stride(2) * element_size, q.stride(1) * element_size,
                                         q.stride(0) * element_size, k.stride(2) * element_size, k.stride(1) * element_size,
                                         k.stride(0) * element_size, v.stride(2) * element_size, v.stride(1) * element_size,
                                         v.stride(0) * element_size, 0, use_bf16, stream);
        }
    }
    HIP_CHECK(hipDeviceSynchronize());

    float avg_ms = 0.0f;
    if (model_timing) {
        // Model-style timing: host sync around a loop, then average.
        torch::cuda::synchronize();
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            if (head_dim == 256) {
                launch_flash_attn_decode_hip_hd256(
                    q.data_ptr(), k.data_ptr(), v.data_ptr(), nullptr, out.data_ptr(), batch_size, n_heads_q, n_heads_kv, head_dim,
                    seq_len_kv, scale, q.stride(2) * element_size, q.stride(1) * element_size, q.stride(0) * element_size,
                    k.stride(2) * element_size, k.stride(1) * element_size, k.stride(0) * element_size, v.stride(2) * element_size,
                    v.stride(1) * element_size, v.stride(0) * element_size, 0, use_bf16, stream);
            } else {
                launch_flash_attn_decode_hip(q.data_ptr(), k.data_ptr(), v.data_ptr(), nullptr, out.data_ptr(), batch_size, n_heads_q,
                                             n_heads_kv, head_dim, seq_len_kv, scale, q.stride(2) * element_size,
                                             q.stride(1) * element_size, q.stride(0) * element_size, k.stride(2) * element_size,
                                             k.stride(1) * element_size, k.stride(0) * element_size, v.stride(2) * element_size,
                                             v.stride(1) * element_size, v.stride(0) * element_size, 0, use_bf16, stream);
            }
        }
        torch::cuda::synchronize();
        auto end = std::chrono::high_resolution_clock::now();
        avg_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f;
        avg_ms /= static_cast<float>(iters);
    } else {
        hipEvent_t start;
        hipEvent_t stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        HIP_CHECK(hipEventRecord(start, stream));
        for (int i = 0; i < iters; ++i) {
            if (head_dim == 256) {
                launch_flash_attn_decode_hip_hd256(
                    q.data_ptr(), k.data_ptr(), v.data_ptr(), nullptr, out.data_ptr(), batch_size, n_heads_q, n_heads_kv, head_dim,
                    seq_len_kv, scale, q.stride(2) * element_size, q.stride(1) * element_size, q.stride(0) * element_size,
                    k.stride(2) * element_size, k.stride(1) * element_size, k.stride(0) * element_size, v.stride(2) * element_size,
                    v.stride(1) * element_size, v.stride(0) * element_size, 0, use_bf16, stream);
            } else {
                launch_flash_attn_decode_hip(q.data_ptr(), k.data_ptr(), v.data_ptr(), nullptr, out.data_ptr(), batch_size, n_heads_q,
                                             n_heads_kv, head_dim, seq_len_kv, scale, q.stride(2) * element_size,
                                             q.stride(1) * element_size, q.stride(0) * element_size, k.stride(2) * element_size,
                                             k.stride(1) * element_size, k.stride(0) * element_size, v.stride(2) * element_size,
                                             v.stride(1) * element_size, v.stride(0) * element_size, 0, use_bf16, stream);
            }
        }
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        avg_ms = ms / static_cast<float>(iters);

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    std::cout << "Attention Shapes:" << " (dtype=" << (use_bf16 ? "bf16" : "fp16") << ")" << std::endl;
    std::cout << "Q: " << q.sizes() << std::endl;
    std::cout << "K: " << k.sizes() << std::endl;
    std::cout << "V: " << v.sizes() << std::endl;

    if (check_correctness) {
        HIP_CHECK(hipDeviceSynchronize());
        // IMPORTANT: this decode microbench uses q_len = 1 and DOES NOT apply a causal mask.
        // Using SDPA with is_causal=true would only attend to the first key when q_len=1 (no positional offset),
        // and on ROCm can also trigger long one-time compilation. So we build the reference explicitly:
        // ref = softmax((q @ k^T) * scale) @ v, with GQA handled by repeating KV heads.

        auto q_bf16 = q;
        auto k_bf16 = k;
        auto v_bf16 = v;

        torch::Tensor k_rep = k_bf16;
        torch::Tensor v_rep = v_bf16;
        if (n_heads_q != n_heads_kv) {
            const int gqa = n_heads_q / n_heads_kv;
            k_rep = repeat_kv(k_bf16, gqa);
            v_rep = repeat_kv(v_bf16, gqa);
        }
        std::cout << "k_rep: " << k_rep.sizes() << std::endl;
        std::cout << "v_rep: " << v_rep.sizes() << std::endl;

        auto scores = torch::matmul(q_bf16, k_rep.transpose(-2, -1)) * scale; // [B, Hq, 1, S]
        auto probs = torch::softmax(scores, -1);
        auto ref = torch::matmul(probs, v_rep); // [B, Hq, 1, D]

        auto diff = (out - ref).abs();
        auto max_val = diff.max().item<float>();
        double atol = use_bf16 ? 5e-2 : 1e-2;
        double rtol = use_bf16 ? 5e-2 : 1e-2;
        bool ok = torch::allclose(out, ref, rtol, atol);

        auto out_flat = out.flatten();
        auto ref_flat = ref.flatten();
        const int64_t n = std::min<int64_t>(8, out_flat.numel());
        auto out_head = out_flat.index({torch::indexing::Slice(0, n)});
        auto ref_head = ref_flat.index({torch::indexing::Slice(0, n)});

        std::cout << "Attention Time: " << std::fixed << std::setprecision(3) << avg_ms << " ms" << std::endl;
        std::cout << "Correctness: " << (ok ? "PASS" : "FAIL") << " | max_abs_diff=" << std::setprecision(6) << max_val
                  << " | rtol=" << rtol << " atol=" << atol << "\n";
        std::cout << "ref[0:8] = " << format_tensor_1d(ref_head) << "\n";
        std::cout << "out[0:8] = " << format_tensor_1d(out_head) << "\n\n";
    }
}

int main() {
    if (!torch::cuda::is_available()) {
        std::cerr << "HIP available check failed" << std::endl;
        return 1;
    }

    std::cout << "Flash Attention Prefill microbenchmark" << std::endl;
    benchmark_prefill(4096, true, true, 16, true, 0);

    std::cout << "Flash Attention Decode microbenchmark" << std::endl;
    // benchmark_decode(27, true, true, 1, true, true);
    benchmark_decode(4111, true, true, 16, true, true, 0);

    std::cout << std::flush;
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(0);
}
