#include "unified_llm_w4a16_hetero/npuSetup.hpp"
#include "unified_llm_w4a16_hetero/unified_llm_w4a16.hpp"
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

PYBIND11_MODULE(unified_llm_w4a16_hetero_libtorch, m) {
    m.doc() = "Pybind11 bindings for Unified LLM W4A16 Quantized LibTorch backend";
    m.attr("GET_TRACES_ENABLED") = py::bool_(GET_TRACES != 0);

    py::enum_<ArchitectureType>(m, "ArchitectureType")
        .value("LLAMA3", ArchitectureType::LLAMA3)
        .value("GEMMA", ArchitectureType::GEMMA)
        .value("QWEN15", ArchitectureType::QWEN15)
        .value("QWEN25SMALL", ArchitectureType::QWEN25SMALL)
        .value("PHI3", ArchitectureType::PHI3)
        .export_values();

    py::class_<UnifiedLLMW4A16Impl, std::shared_ptr<UnifiedLLMW4A16Impl>>(m, "UnifiedLLMW4A16")
        .def(py::init([](ArchitectureType arch_type, int64_t vocab_size, int64_t hidden_size, int64_t intermediate_size,
                         int64_t num_hidden_layers, int64_t num_attention_heads, int64_t num_key_value_heads, int64_t head_dim,
                         float rms_norm_eps, float rope_theta, int64_t max_seq_len, int64_t max_batch_size, int64_t groupsize,
                         std::string device_str, std::string config_path, float partial_rotary_factor,
                         int64_t original_max_position_embeddings, std::vector<float> rope_short_factors,
                         std::vector<float> rope_long_factors, int64_t model_max_position_embeddings) {
                 torch::Device device = (device_str == "cuda") ? torch::kCUDA : torch::kCPU;
                 return std::make_shared<UnifiedLLMW4A16Impl>(arch_type, vocab_size, hidden_size, intermediate_size, num_hidden_layers,
                                                              num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps, rope_theta,
                                                              max_seq_len, max_batch_size, groupsize, device, config_path,
                                                              partial_rotary_factor, original_max_position_embeddings,
                                                              std::move(rope_short_factors), std::move(rope_long_factors),
                                                              model_max_position_embeddings);
             }),
             py::arg("arch_type"), py::arg("vocab_size"), py::arg("hidden_size"), py::arg("intermediate_size"),
             py::arg("num_hidden_layers"), py::arg("num_attention_heads"), py::arg("num_key_value_heads"), py::arg("head_dim"),
             py::arg("rms_norm_eps"), py::arg("rope_theta"), py::arg("max_seq_len") = 8192, py::arg("max_batch_size") = 1,
             py::arg("groupsize") = 128, py::arg("device") = "cpu", py::arg("config_path") = "",
             py::arg("partial_rotary_factor") = 1.0f, py::arg("original_max_position_embeddings") = 0,
             py::arg("rope_short_factors") = std::vector<float>{}, py::arg("rope_long_factors") = std::vector<float>{},
             py::arg("model_max_position_embeddings") = 0)
        .def(
            "forward",
            [](UnifiedLLMW4A16Impl &self, torch::Tensor input_ids, int64_t start_pos) -> torch::Tensor {
                // Ensure input is contiguous
                if (!input_ids.is_contiguous())
                    input_ids = input_ids.contiguous();
                return self.forward(input_ids, start_pos);
            },
            py::arg("input_ids"), py::arg("start_pos") = 0)
        .def(
            "generate",
            [](UnifiedLLMW4A16Impl &self, torch::Tensor input_ids, int64_t max_new_tokens, float temperature, float top_p, int64_t top_k,
               int64_t eos_token_id) -> torch::Tensor {
                // Ensure input is contiguous
                if (!input_ids.is_contiguous())
                    input_ids = input_ids.contiguous();
                return self.generate(input_ids, max_new_tokens, temperature, top_p, top_k, eos_token_id);
            },
            py::arg("input_ids"), py::arg("max_new_tokens"), py::arg("temperature") = 1.0f, py::arg("top_p") = 0.9f, py::arg("top_k") = 50,
            py::arg("eos_token_id") = -1)
        .def("to", &UnifiedLLMW4A16Impl::to, py::arg("device"))
        .def("load_quantized_weights_from_safetensors", &UnifiedLLMW4A16Impl::load_quantized_weights_from_safetensors, py::arg("filename"))
        .def("load_non_quantized_weights_from_safetensors", &UnifiedLLMW4A16Impl::load_non_quantized_weights_from_safetensors,
             py::arg("filename"))
        .def("load_quantized_weights_from_bins", &UnifiedLLMW4A16Impl::load_quantized_weights_from_bins, py::arg("weights_dir"))
        .def("initialize_npu", &UnifiedLLMW4A16Impl::initialize_npu, "Initialize NPU driver and resources")
        .def("import_weights", &UnifiedLLMW4A16Impl::import_weights, "Import weights to NPU")
        .def("initialize_dummy_weights", &UnifiedLLMW4A16Impl::initialize_dummy_weights, py::arg("seed") = 42,
             "Initialize dummy weights for testing");
}
