"""
Phi-3.5-mini-instruct-awq w4a16 quantized Python frontend.
Keeps the hetero backend execution unfused by splitting fused AWQ
checkpoint tensors into the backend's standard q/k/v/gate/up layout first.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

import torch
from transformers import AutoTokenizer

# Add build directory to Python path
_script_dir = Path(__file__).parent.resolve()
_project_root = _script_dir.parent.parent
_build_dir = _project_root / "build" / "py" / "unified_llm_w4a16"

if _build_dir.exists():
    sys.path.insert(0, str(_build_dir))
else:
    _alt_build = _project_root / "build" / "py"
    if _alt_build.exists():
        sys.path.insert(0, str(_alt_build))
    _local_build = Path("build") / "py" / "unified_llm_w4a16"
    if _local_build.exists():
        sys.path.insert(0, str(_local_build.resolve()))

ArchitectureType = None

try:
    from .llama3_8b_w4a16_model import (
        LLaMA3W4A16Model as _LlamaFrontendBase,
        _align_zeros_to_scales,
        _detect_lscpu_model_name,
        _ensure_default_config_file,
        _get_quantized_tensors,
        _load_wikitext2_raw_text,
        _model_name_from_path,
        _normalize_cpu_model_name,
        _normalize_group_tensor,
        _pack_qweight_out_in2,
        _pack_weights_packed,
        _save_used_prompt,
        _select_prompt_test_prompt_len,
        _set_prompt_len_in_config,
        _split_packed_bin_from_file,
        _split_packed_paths,
        _unpack_awq_qweight,
        _unpack_awq_qzeros,
        _unpack_gptq_qweight,
        _unpack_packed_out_in2_to_in_out,
        _write_tensor_raw,
        load_config_with_comments,
        measure_generation_power,
    )
except ImportError:
    from llama3_8b_w4a16_model import (
        LLaMA3W4A16Model as _LlamaFrontendBase,
        _align_zeros_to_scales,
        _detect_lscpu_model_name,
        _ensure_default_config_file,
        _get_quantized_tensors,
        _load_wikitext2_raw_text,
        _model_name_from_path,
        _normalize_cpu_model_name,
        _normalize_group_tensor,
        _pack_qweight_out_in2,
        _pack_weights_packed,
        _save_used_prompt,
        _select_prompt_test_prompt_len,
        _set_prompt_len_in_config,
        _split_packed_bin_from_file,
        _split_packed_paths,
        _unpack_awq_qweight,
        _unpack_awq_qzeros,
        _unpack_gptq_qweight,
        _unpack_packed_out_in2_to_in_out,
        _write_tensor_raw,
        load_config_with_comments,
        measure_generation_power,
    )


DEFAULT_MODEL_REPO = "thesven/Phi-3.5-mini-instruct-awq"
DEFAULT_MODEL_FILENAME = "model.safetensors"
DEFAULT_MODEL_PATH = f"{DEFAULT_MODEL_REPO}/{DEFAULT_MODEL_FILENAME}"
DEFAULT_TOKENIZER_PATH = DEFAULT_MODEL_REPO
DEFAULT_MODEL_LABEL = "Phi-3.5-mini-instruct-awq"
DEFAULT_CONFIG_STEM = "phi3.5_3.8b"
DEFAULT_RUNTIME_MAX_SEQ_LEN = 16512
PHI35_MINI_MAX_POSITION_EMBEDDINGS = 131072
DEFAULT_PROMPT_TEST_PREFIX = (
    "Please provide a comprehensive summary of the following document. "
    "The summary should capture the main points, key developments, and important themes discussed in the text."
)
DEFAULT_PROMPT_TEST_SUFFIX = "\n\nSummary:"

PHI35_MINI_SHORT_ROPE_FACTORS = [
    1.0,
    1.0199999809265137,
    1.0299999713897705,
    1.0299999713897705,
    1.0499999523162842,
    1.0499999523162842,
    1.0499999523162842,
    1.0499999523162842,
    1.0499999523162842,
    1.0699999332427979,
    1.0999999046325684,
    1.1099998950958252,
    1.1599998474121094,
    1.1599998474121094,
    1.1699998378753662,
    1.2899998426437378,
    1.339999794960022,
    1.679999828338623,
    1.7899998426437378,
    1.8199998140335083,
    1.8499997854232788,
    1.8799997568130493,
    1.9099997282028198,
    1.9399996995925903,
    1.9899996519088745,
    2.0199997425079346,
    2.0199997425079346,
    2.0199997425079346,
    2.0199997425079346,
    2.0199997425079346,
    2.0199997425079346,
    2.0299997329711914,
    2.0299997329711914,
    2.0299997329711914,
    2.0299997329711914,
    2.0299997329711914,
    2.0299997329711914,
    2.0299997329711914,
    2.0299997329711914,
    2.0299997329711914,
    2.0799996852874756,
    2.0899996757507324,
    2.189999580383301,
    2.2199995517730713,
    2.5899994373321533,
    2.729999542236328,
    2.749999523162842,
    2.8399994373321533,
]
PHI35_MINI_LONG_ROPE_FACTORS = [
    1.0800000429153442,
    1.1100000143051147,
    1.1399999856948853,
    1.340000033378601,
    1.5899999141693115,
    1.600000023841858,
    1.6200000047683716,
    2.620000123977661,
    3.2300000190734863,
    3.2300000190734863,
    4.789999961853027,
    7.400000095367432,
    7.700000286102295,
    9.09000015258789,
    12.199999809265137,
    17.670000076293945,
    24.46000099182129,
    28.57000160217285,
    30.420001983642578,
    30.840002059936523,
    32.590003967285156,
    32.93000411987305,
    42.320003509521484,
    44.96000289916992,
    50.340003967285156,
    50.45000457763672,
    57.55000305175781,
    57.93000411987305,
    58.21000289916992,
    60.1400032043457,
    62.61000442504883,
    62.62000274658203,
    62.71000289916992,
    63.1400032043457,
    63.1400032043457,
    63.77000427246094,
    63.93000411987305,
    63.96000289916992,
    63.970001220703125,
    64.02999877929688,
    64.06999969482422,
    64.08000183105469,
    64.12000274658203,
    64.41000366210938,
    64.4800033569336,
    64.51000213623047,
    64.52999877929688,
    64.83999633789062,
]

DEFAULT_HETERO_CONFIG = {
    "heterogeneity": "gpu",
    "warmup": False,
    "dummy_weights": False,
    "debug_verbosity": 0,
    "usePackedWeights": True,
    "padPackedWeights": False,
    "usePreSavedWeights": True,
    "prompt_len": 1,
    "cpu_decode": False,
    "minimal_pdi": True,
    "gemv_driven_split_K": False,
    "gemv_npu_col": "8c",
    "chunking": True,
    "chunking_scheduled": False,
    "gpu_chunking": {
        "gpu_chunk_size": 2048,
        "gpu_chunking_inflight": 1,
    },
    "kernels_gemm_chunked": [],
    "kernels_gemm": [],
    "kernels_gemv": [],
    "npuOnlydefault": [
        {
            "qo": [3072, 3072, -1],
            "kv": [3072, 3072, -1],
            "upgate": [3072, 8192, -1],
            "down": [8192, 3072, -1],
            "fw_path": "hw_bins/npu2/",
            "max_ctx_len": 8192,
            "num_tiles": 32,
            "tile_size": "64x128x64",
            "col": "8c",
            "dtype": "bf16_int4AWQ_bf16",
        }
    ],
}


def _ensure_phi35_default_config_file(config_path: Union[str, Path]) -> Path:
    path = Path(config_path).expanduser().resolve()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_HETERO_CONFIG, f, indent=4)
        f.write("\n")
    print(f"Created missing Phi-3.5 config file with default values: {path}")
    return path


def _tokenize_prompt_test_tokens(tokenizer: AutoTokenizer, long_prompt: str, target_tokens: int) -> List[int]:
    prefix_tokens = tokenizer.encode(DEFAULT_PROMPT_TEST_PREFIX + "\n\nDocument:\n", add_special_tokens=False)
    suffix_tokens = tokenizer.encode(DEFAULT_PROMPT_TEST_SUFFIX, add_special_tokens=False)
    document_tokens = tokenizer.encode(long_prompt, add_special_tokens=False)

    fixed_tokens = prefix_tokens + suffix_tokens
    if target_tokens <= len(fixed_tokens):
        return fixed_tokens[:target_tokens]

    doc_budget = target_tokens - len(prefix_tokens) - len(suffix_tokens)
    if len(document_tokens) >= doc_budget:
        trimmed_document = document_tokens[:doc_budget]
    elif document_tokens:
        repeats = (doc_budget + len(document_tokens) - 1) // len(document_tokens)
        trimmed_document = (document_tokens * repeats)[:doc_budget]
    else:
        trimmed_document = []

    return prefix_tokens + trimmed_document + suffix_tokens


def _resolve_phi35_mini_config_path(config_path: Optional[str]) -> str:
    if config_path:
        resolved = _ensure_phi35_default_config_file(config_path)
        return str(resolved)

    configs_dir = Path(__file__).parent.resolve() / "configs"
    cpu_model_name = _detect_lscpu_model_name()
    normalized = _normalize_cpu_model_name(cpu_model_name) if cpu_model_name else ""

    filename = f"configs_strixP_{DEFAULT_CONFIG_STEM}.json5"
    mapping = [
        (("RYZEN AI 7 350", "RADEON 860M"), f"configs_krackanP_{DEFAULT_CONFIG_STEM}.json5"),
        (("RYZEN AI 9 HX 370", "RADEON 890M"), f"configs_strixP_{DEFAULT_CONFIG_STEM}.json5"),
        (("RYZEN AI MAX+ 395", "RADEON 8060S"), f"configs_strixH_{DEFAULT_CONFIG_STEM}.json5"),
    ]
    for tokens, candidate in mapping:
        if all(token in normalized for token in tokens):
            filename = candidate
            break

    selected = _ensure_phi35_default_config_file(configs_dir / filename)
    if cpu_model_name:
        print(f"Auto-selected config based on lscpu model '{cpu_model_name}': {selected}")
    else:
        print(f"Auto-selected fallback config: {selected}")
    return str(selected)


def _split_tensor_on_output_axis(tensor: torch.Tensor, split_sizes: Sequence[int], base_name: str) -> List[torch.Tensor]:
    total = sum(split_sizes)
    if tensor.dim() == 1:
        if tensor.numel() != total:
            raise ValueError(f"{base_name}: expected {total} elements, got shape {tuple(tensor.shape)}")
        return [part.contiguous() for part in torch.split(tensor, list(split_sizes), dim=0)]

    if tensor.size(-1) == total:
        return [part.contiguous() for part in torch.split(tensor, list(split_sizes), dim=-1)]

    if tensor.size(0) == total:
        return [part.contiguous() for part in torch.split(tensor, list(split_sizes), dim=0)]

    raise ValueError(f"{base_name}: could not split tensor with shape {tuple(tensor.shape)} into {list(split_sizes)}")


def _split_awq_packed_output_tensor(tensor: torch.Tensor, split_sizes: Sequence[int], base_name: str) -> List[torch.Tensor]:
    if any((size % 8) != 0 for size in split_sizes):
        raise ValueError(f"{base_name}: split sizes must be divisible by 8 for AWQ packed tensors: {list(split_sizes)}")

    packed_sizes = [size // 8 for size in split_sizes]
    total_packed = sum(packed_sizes)
    if tensor.dim() != 2 or tensor.size(-1) != total_packed:
        raise ValueError(
            f"{base_name}: expected packed AWQ tensor with trailing dim {total_packed}, got shape {tuple(tensor.shape)}"
        )
    return [part.contiguous() for part in torch.split(tensor, packed_sizes, dim=-1)]


def _copy_or_merge_raw_safetensors(model_path: str, saved_safetensors: Path) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    import glob

    def _is_valid_safetensors_file(path: Path) -> bool:
        try:
            with safe_open(str(path), framework="pt"):
                return True
        except Exception:
            return False

    def _atomic_copy_file(src: Union[str, Path], dst: Path) -> None:
        tmp_path = dst.with_name(dst.name + ".tmp")
        shutil.copy2(str(src), tmp_path)
        os.replace(tmp_path, dst)

    if saved_safetensors.exists():
        if _is_valid_safetensors_file(saved_safetensors):
            print(f"Using saved safetensors: {saved_safetensors}")
            return saved_safetensors
        print(f"Cached safetensors is invalid, recreating: {saved_safetensors}")
        saved_safetensors.unlink()

    resolved_model_path = model_path
    if os.path.isfile(resolved_model_path):
        if not resolved_model_path.endswith(".safetensors"):
            raise NotImplementedError("Only safetensors checkpoints are supported for Phi-3.5-mini loading.")
        print(f"Saving safetensors to {saved_safetensors}...")
        _atomic_copy_file(resolved_model_path, saved_safetensors)
        return saved_safetensors

    if resolved_model_path.endswith(".safetensors") and not os.path.exists(resolved_model_path):
        repo_id, filename = resolved_model_path.rsplit("/", 1)
        print(f"Downloading safetensors file {filename} from {repo_id}...")
        downloaded_file = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"Model downloaded to {downloaded_file}")
        print(f"Saving safetensors to {saved_safetensors}...")
        _atomic_copy_file(downloaded_file, saved_safetensors)
        return saved_safetensors

    if not os.path.exists(resolved_model_path):
        print(f"Model path {resolved_model_path} not found locally, downloading from Hub...")
        resolved_model_path = snapshot_download(repo_id=resolved_model_path)
        print(f"Model downloaded to {resolved_model_path}")

    safetensors_files = glob.glob(os.path.join(resolved_model_path, "*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors weight files found in {resolved_model_path}")

    if len(safetensors_files) == 1:
        print(f"Saving safetensors to {saved_safetensors}...")
        _atomic_copy_file(safetensors_files[0], saved_safetensors)
        return saved_safetensors

    print(f"Found {len(safetensors_files)} safetensors files, merging...")
    merged_state_dict = {}
    for st_file in safetensors_files:
        print(f"  Loading {st_file}...")
        merged_state_dict.update(load_file(st_file))

    print(f"Saving merged safetensors to {saved_safetensors}...")
    tmp_path = saved_safetensors.with_name(saved_safetensors.name + ".tmp")
    save_file(merged_state_dict, str(tmp_path))
    os.replace(tmp_path, saved_safetensors)
    return saved_safetensors


def _phi35_model_cache_name(model_path: str) -> str:
    normalized = str(model_path).replace("\\", "/")
    if normalized.endswith(".safetensors"):
        parts = [part for part in normalized.split("/") if part]
        if not parts:
            return "phi3.5_mini_instruct_awq_model"
        file_stem = Path(parts[-1]).stem
        if len(parts) >= 2:
            return f"{parts[-2]}_{file_stem}"
        return file_stem
    return _model_name_from_path(normalized)


def _default_tokenizer_path_for_model(model_path: Optional[str]) -> str:
    if not model_path:
        return DEFAULT_TOKENIZER_PATH

    normalized = str(model_path).replace("\\", "/")

    if os.path.isdir(normalized):
        return normalized

    if normalized.endswith(".safetensors"):
        if os.path.isfile(normalized):
            local_parent = Path(normalized).resolve().parent
            if (local_parent / "tokenizer.json").exists() or (local_parent / "tokenizer_config.json").exists():
                return str(local_parent)
            return DEFAULT_TOKENIZER_PATH
        parts = [part for part in normalized.split("/") if part]
        if len(parts) >= 3:
            return "/".join(parts[:-1])
        return DEFAULT_TOKENIZER_PATH

    if "/" in normalized:
        return normalized
    return DEFAULT_TOKENIZER_PATH


def _compute_effective_max_seq_len(
    explicit_max_seq_len: Optional[int],
    *,
    prompt_token_count: Optional[int] = None,
    max_new_tokens: int = 0,
    fallback: int = DEFAULT_RUNTIME_MAX_SEQ_LEN,
    auto_floor: int = 2048,
    hard_cap: int = PHI35_MINI_MAX_POSITION_EMBEDDINGS,
) -> int:
    if explicit_max_seq_len is not None:
        value = int(explicit_max_seq_len)
        if value <= 0:
            raise ValueError(f"max_seq_len must be positive, got {value}")
        return min(value, hard_cap)

    required = None
    if prompt_token_count is not None:
        required = int(prompt_token_count) + max(0, int(max_new_tokens))
        if required <= 0:
            required = 1

    if required is None:
        required = int(fallback)
    else:
        required = max(required, int(auto_floor))

    return min(required, hard_cap)


def _infer_prompt_token_count(tokenizer_path: str, text: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    encoded = tokenizer(text, return_tensors="pt", padding=False, truncation=False)
    return int(encoded["input_ids"].size(1))


def _split_phi35_fused_awq_state_dict(
    state_dict: Dict[str, torch.Tensor],
    num_hidden_layers: int,
    q_proj_size: int,
    kv_proj_size: int,
    intermediate_size: int,
) -> Dict[str, torch.Tensor]:
    split_state_dict: Dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        if ".self_attn.qkv_proj." in key or ".mlp.gate_up_proj." in key:
            continue
        split_state_dict[key] = tensor

    attn_split_sizes = [q_proj_size, kv_proj_size, kv_proj_size]
    attn_names = ["q_proj", "k_proj", "v_proj"]
    mlp_split_sizes = [intermediate_size, intermediate_size]
    mlp_names = ["gate_proj", "up_proj"]

    def _split_compressed_packed_linear(
        fused_prefix: str,
        split_sizes: Sequence[int],
        split_names: Sequence[str],
        split_base_prefix: str,
    ) -> None:
        packed_key = fused_prefix + ".weight_packed"
        scale_key = fused_prefix + ".weight_scale"
        if packed_key not in state_dict or scale_key not in state_dict:
            return

        packed_parts = _split_tensor_on_output_axis(state_dict[packed_key], split_sizes, packed_key)
        scale_parts = _split_tensor_on_output_axis(state_dict[scale_key], split_sizes, scale_key)

        in_features = None
        shape_dtype = torch.int64
        shape_device = "cpu"
        shape_key = fused_prefix + ".weight_shape"
        if shape_key in state_dict:
            weight_shape = state_dict[shape_key]
            if weight_shape.numel() >= 2:
                in_features = int(weight_shape.reshape(-1)[1].item())
            shape_dtype = weight_shape.dtype
            shape_device = weight_shape.device
        elif packed_parts:
            in_features = int(packed_parts[0].size(1) * 8)

        if in_features is None:
            raise ValueError(f"{fused_prefix}: could not infer in_features for compressed packed split")

        for name, split_size, packed_part, scale_part in zip(split_names, split_sizes, packed_parts, scale_parts):
            base_name = f"{split_base_prefix}.{name}"
            split_state_dict[base_name + ".weight_packed"] = packed_part
            split_state_dict[base_name + ".weight_scale"] = scale_part
            split_state_dict[base_name + ".weight_shape"] = torch.tensor(
                [split_size, in_features],
                dtype=shape_dtype,
                device=shape_device,
            )

    for layer_idx in range(num_hidden_layers):
        attn_prefix = f"model.layers.{layer_idx}.self_attn.qkv_proj"
        if attn_prefix + ".qweight" in state_dict:
            qweight_parts = _split_awq_packed_output_tensor(
                state_dict[attn_prefix + ".qweight"],
                attn_split_sizes,
                attn_prefix + ".qweight",
            )
            scales_parts = _split_tensor_on_output_axis(
                state_dict[attn_prefix + ".scales"],
                attn_split_sizes,
                attn_prefix + ".scales",
            )
            qzeros_parts = _split_awq_packed_output_tensor(
                state_dict[attn_prefix + ".qzeros"],
                attn_split_sizes,
                attn_prefix + ".qzeros",
            )

            for name, qweight_part, scales_part, qzeros_part in zip(attn_names, qweight_parts, scales_parts, qzeros_parts):
                base_name = f"model.layers.{layer_idx}.self_attn.{name}"
                split_state_dict[base_name + ".qweight"] = qweight_part
                split_state_dict[base_name + ".scales"] = scales_part
                split_state_dict[base_name + ".qzeros"] = qzeros_part

            if attn_prefix + ".bias" in state_dict:
                bias_parts = _split_tensor_on_output_axis(
                    state_dict[attn_prefix + ".bias"],
                    attn_split_sizes,
                    attn_prefix + ".bias",
                )
                for name, bias_part in zip(attn_names, bias_parts):
                    split_state_dict[f"model.layers.{layer_idx}.self_attn.{name}.bias"] = bias_part

            if attn_prefix + ".g_idx" in state_dict:
                g_idx = state_dict[attn_prefix + ".g_idx"]
                for name in attn_names:
                    split_state_dict[f"model.layers.{layer_idx}.self_attn.{name}.g_idx"] = g_idx

        _split_compressed_packed_linear(
            attn_prefix,
            attn_split_sizes,
            attn_names,
            f"model.layers.{layer_idx}.self_attn",
        )

        mlp_prefix = f"model.layers.{layer_idx}.mlp.gate_up_proj"
        if mlp_prefix + ".qweight" in state_dict:
            qweight_parts = _split_awq_packed_output_tensor(
                state_dict[mlp_prefix + ".qweight"],
                mlp_split_sizes,
                mlp_prefix + ".qweight",
            )
            scales_parts = _split_tensor_on_output_axis(
                state_dict[mlp_prefix + ".scales"],
                mlp_split_sizes,
                mlp_prefix + ".scales",
            )
            qzeros_parts = _split_awq_packed_output_tensor(
                state_dict[mlp_prefix + ".qzeros"],
                mlp_split_sizes,
                mlp_prefix + ".qzeros",
            )

            for name, qweight_part, scales_part, qzeros_part in zip(mlp_names, qweight_parts, scales_parts, qzeros_parts):
                base_name = f"model.layers.{layer_idx}.mlp.{name}"
                split_state_dict[base_name + ".qweight"] = qweight_part
                split_state_dict[base_name + ".scales"] = scales_part
                split_state_dict[base_name + ".qzeros"] = qzeros_part

            if mlp_prefix + ".bias" in state_dict:
                bias_parts = _split_tensor_on_output_axis(
                    state_dict[mlp_prefix + ".bias"],
                    mlp_split_sizes,
                    mlp_prefix + ".bias",
                )
                for name, bias_part in zip(mlp_names, bias_parts):
                    split_state_dict[f"model.layers.{layer_idx}.mlp.{name}.bias"] = bias_part

            if mlp_prefix + ".g_idx" in state_dict:
                g_idx = state_dict[mlp_prefix + ".g_idx"]
                for name in mlp_names:
                    split_state_dict[f"model.layers.{layer_idx}.mlp.{name}.g_idx"] = g_idx

        _split_compressed_packed_linear(
            mlp_prefix,
            mlp_split_sizes,
            mlp_names,
            f"model.layers.{layer_idx}.mlp",
        )

    return split_state_dict


def _ensure_split_phi35_safetensors(
    raw_safetensors: Path,
    split_safetensors: Path,
    num_hidden_layers: int,
    q_proj_size: int,
    kv_proj_size: int,
    intermediate_size: int,
) -> Path:
    from safetensors.torch import load_file, save_file

    manifest_path = split_safetensors.with_suffix(".manifest.json")
    expected_manifest = {
        "format_version": 1,
        "source_safetensors": raw_safetensors.name,
        "split_layout": "phi3.5_mini_instruct_awq_unfused",
        "num_hidden_layers": num_hidden_layers,
        "q_proj_size": q_proj_size,
        "kv_proj_size": kv_proj_size,
        "intermediate_size": intermediate_size,
    }

    if split_safetensors.exists() and manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing_manifest = json.load(f)
            if all(existing_manifest.get(k) == v for k, v in expected_manifest.items()):
                print(f"Using cached split safetensors: {split_safetensors}")
                return split_safetensors
        except Exception:
            pass

    print(f"Creating unfused Phi-3.5 split cache: {split_safetensors}")
    state_dict = load_file(str(raw_safetensors))
    split_state_dict = _split_phi35_fused_awq_state_dict(
        state_dict,
        num_hidden_layers=num_hidden_layers,
        q_proj_size=q_proj_size,
        kv_proj_size=kv_proj_size,
        intermediate_size=intermediate_size,
    )
    save_file(split_state_dict, str(split_safetensors))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(expected_manifest, f, indent=2)
        f.write("\n")
    return split_safetensors


class Phi35MiniInstructAWQW4A16Model:
    """Phi-3.5-mini-instruct-awq w4a16 quantized model wrapper for the hetero backend."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        vocab_size: int = 32064,
        hidden_size: int = 3072,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        head_dim: int = 96,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        max_seq_len: Optional[int] = None,
        max_batch_size: int = 1,
        groupsize: int = 128,
        device: str = "cuda",
        backend: str = "hetero",
        config_path: Optional[str] = None,
        model_max_position_embeddings: int = PHI35_MINI_MAX_POSITION_EMBEDDINGS,
        partial_rotary_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        rope_short_factors: Optional[List[float]] = None,
        rope_long_factors: Optional[List[float]] = None,
    ):
        self.device = device
        self.head_dim = head_dim
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.max_seq_len = _compute_effective_max_seq_len(max_seq_len)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.groupsize = groupsize
        self.model_max_position_embeddings = model_max_position_embeddings
        self.partial_rotary_factor = partial_rotary_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.rope_short_factors = list(rope_short_factors or PHI35_MINI_SHORT_ROPE_FACTORS)
        self.rope_long_factors = list(rope_long_factors or PHI35_MINI_LONG_ROPE_FACTORS)

        if tokenizer_path is None:
            tokenizer_path = _default_tokenizer_path_for_model(self.model_path)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Warning: Could not load tokenizer from {tokenizer_path}: {e}")
            self.tokenizer = None

        if backend != "hetero":
            raise NotImplementedError("Phi-3.5-mini support is implemented for the hetero backend only in this pass.")

        try:
            import unified_llm_w4a16_hetero_libtorch

            self.backend_module = unified_llm_w4a16_hetero_libtorch
        except ImportError as e:
            raise ImportError(f"Could not import hetero backend: {e}")

        ArchitectureType = self.backend_module.ArchitectureType
        config_path = _resolve_phi35_mini_config_path(config_path)
        self.config_path = config_path

        constructor_args = [
            ArchitectureType.PHI3,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rope_theta,
            self.max_seq_len,
            max_batch_size,
            groupsize,
            device,
            config_path,
            partial_rotary_factor,
            original_max_position_embeddings,
            self.rope_short_factors,
            self.rope_long_factors,
            model_max_position_embeddings,
        ]
        self.model = self.backend_module.UnifiedLLMW4A16(*constructor_args)

        self.config = {}
        self.use_packed_weights = False
        self.use_pre_saved_weights = False
        self.pad_packed_weights = False
        self.debug_verbosity = 0

        use_dummy = False
        try:
            self.config = load_config_with_comments(config_path)
            self.use_packed_weights = bool(self.config.get("usePackedWeights", False))
            self.use_pre_saved_weights = bool(self.config.get("usePreSavedWeights", False))
            self.pad_packed_weights = bool(self.config.get("padPackedWeights", False))
            self.debug_verbosity = int(self.config.get("debug_verbosity", 0))
            if self.config.get("dummy_weights", False):
                use_dummy = True
                print("Dummy weights enabled in config.")
        except Exception as e:
            print(f"Error reading config: {e}")

        if use_dummy:
            if hasattr(self.model, "initialize_dummy_weights"):
                print("Initializing dummy weights...")
                self.model.initialize_dummy_weights()
            else:
                print("Warning: Backend does not support dummy weights initialization.")
        else:
            self._load_quantized_weights(self.model_path, weights_folder="model_weights")

        print("Importing weights to NPU...")
        self.model.import_weights()

    def tokenize(self, text: Union[str, List[str]]) -> torch.Tensor:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        if isinstance(text, str):
            text = [text]

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        return encoded["input_ids"].to(self.device)

    def _load_quantized_weights(self, model_path: str, weights_folder: str = "model_weights") -> None:
        print(f"Loading quantized weights from {model_path}...")
        try:
            script_dir = Path(__file__).parent
            weights_dir = script_dir / weights_folder
            os.makedirs(weights_dir, exist_ok=True)

            model_name = _phi35_model_cache_name(model_path)
            raw_safetensors = weights_dir / f"{model_name}.safetensors"
            split_safetensors = weights_dir / f"{model_name}_unfused.safetensors"

            _copy_or_merge_raw_safetensors(model_path, raw_safetensors)
            split_safetensors = _ensure_split_phi35_safetensors(
                raw_safetensors,
                split_safetensors,
                num_hidden_layers=self.num_hidden_layers,
                q_proj_size=self.num_attention_heads * self.head_dim,
                kv_proj_size=self.num_key_value_heads * self.head_dim,
                intermediate_size=self.intermediate_size,
            )

            use_packed = self.use_packed_weights
            use_presaved = self.use_pre_saved_weights
            pad_packed = self.pad_packed_weights

            if use_presaved:
                presaved_dir = weights_dir / f"{split_safetensors.stem}_{'packed' if use_packed else 'unpacked'}"
                self._prepare_presaved_weights(split_safetensors, presaved_dir, use_packed, pad_packed)

                t0 = time.time()
                if hasattr(self.model, "load_non_quantized_weights_from_safetensors"):
                    self.model.load_non_quantized_weights_from_safetensors(str(split_safetensors))
                else:
                    print("Warning: Backend missing load_non_quantized_weights_from_safetensors; falling back to full load.")
                    self.model.load_quantized_weights_from_safetensors(str(split_safetensors))
                    self.load_time = time.time() - t0
                    return

                if hasattr(self.model, "load_quantized_weights_from_bins"):
                    self.model.load_quantized_weights_from_bins(str(presaved_dir))
                else:
                    print("Warning: Backend missing load_quantized_weights_from_bins; falling back to full load.")
                    self.model.load_quantized_weights_from_safetensors(str(split_safetensors))
                    self.load_time = time.time() - t0
                    return

                self.load_time = time.time() - t0
                print(f"Weights loaded from pre-saved bins in {self.load_time:.2f} seconds")
            else:
                t0 = time.time()
                self.model.load_quantized_weights_from_safetensors(str(split_safetensors))
                self.load_time = time.time() - t0
                print(f"Weights loaded in {self.load_time:.2f} seconds")
        except Exception as e:
            print(f"Error loading quantized weights: {e}")
            import traceback

            traceback.print_exc()
            print("\nNote: Falling back to randomly initialized weights.")
            print("The model will not produce meaningful output without proper weights.")


Phi35MiniInstructAWQW4A16Model._prepare_presaved_weights = _LlamaFrontendBase._prepare_presaved_weights
Phi35MiniInstructAWQW4A16Model.generate = _LlamaFrontendBase.generate
Phi35MiniInstructAWQW4A16Model.__call__ = _LlamaFrontendBase.__call__
Phi35MiniInstructAWQW4A16Model.perplexity = _LlamaFrontendBase.perplexity


def run_prompt_test(
    target_tokens,
    model_path=None,
    tokenizer_path=None,
    device="cuda",
    backend="hetero",
    max_seq_len: Optional[int] = None,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    generate=True,
    config_path=None,
    perplexity=False,
    measure_power=False,
    save_used_prompt=False,
):
    script_dir = Path(__file__).parent
    prompts_file = script_dir.parent / "prompts.txt"

    if not prompts_file.exists():
        print(f"Error: Prompts file not found at {prompts_file}")
        return 1

    print("=" * 60)
    print(f"PROMPT TEST: Target token count = {target_tokens}")
    print("=" * 60 + "\n")

    print(f"Reading long prompt from: {prompts_file}")
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts_content = f.read()

    long_prompt = prompts_content.replace("<|begin_of_text|>", "").strip()

    print("Initializing tokenizer...")
    try:
        from transformers import AutoTokenizer

        if tokenizer_path is None:
            tokenizer_path = DEFAULT_TOKENIZER_PATH
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer initialized successfully!\n")
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        return 1

    base_prompt_tokens = tokenizer.encode(DEFAULT_PROMPT_TEST_PREFIX + "\n\nDocument:\n", add_special_tokens=False)
    summary_suffix_tokens = tokenizer.encode(DEFAULT_PROMPT_TEST_SUFFIX, add_special_tokens=False)
    long_prompt_tokens = tokenizer.encode(long_prompt, add_special_tokens=False)

    print("Token counts:")
    print(f"  Base prompt: {len(base_prompt_tokens)} tokens")
    print(f"  Long prompt: {len(long_prompt_tokens)} tokens")
    print(f"  Summary suffix: {len(summary_suffix_tokens)} tokens")
    print(f"  Combined (before truncation): {len(base_prompt_tokens) + len(long_prompt_tokens) + len(summary_suffix_tokens)} tokens")
    print(f"  Target: {target_tokens} tokens\n")

    full_token_count = len(base_prompt_tokens) + len(long_prompt_tokens) + len(summary_suffix_tokens)
    truncated_tokens = _tokenize_prompt_test_tokens(tokenizer, long_prompt, int(target_tokens))
    actual_token_count = len(truncated_tokens)

    if full_token_count > target_tokens:
        print(f"Truncated from {full_token_count} to {actual_token_count} tokens while preserving the summary cue")
    elif full_token_count < target_tokens and long_prompt_tokens:
        print(f"Extended from {full_token_count} to {actual_token_count} tokens while preserving the summary cue")
    else:
        print(f"Prompt is exactly {actual_token_count} tokens (target: {target_tokens})")

    if save_used_prompt:
        used_prompt_text = tokenizer.decode(truncated_tokens, skip_special_tokens=False)
        _save_used_prompt(used_prompt_text)

    print("\n" + "=" * 60)
    print("INITIALIZING MODEL:")
    print("=" * 60 + "\n")

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    if backend == "hetero":
        config_path = _resolve_phi35_mini_config_path(config_path)
        selected_prompt_len = _select_prompt_test_prompt_len(config_path, int(target_tokens))
        _set_prompt_len_in_config(config_path, selected_prompt_len)

    effective_max_seq_len = _compute_effective_max_seq_len(
        max_seq_len,
        prompt_token_count=actual_token_count,
        max_new_tokens=max_new_tokens if generate else 0,
    )
    if max_seq_len is None:
        print(
            f"Auto-selected backend max_seq_len={effective_max_seq_len} "
            f"from prompt={actual_token_count} and max_new_tokens={max_new_tokens if generate else 0}"
        )

    print(f"Initializing {DEFAULT_MODEL_LABEL} w4a16 quantized model...")
    try:
        model = Phi35MiniInstructAWQW4A16Model(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            max_seq_len=effective_max_seq_len,
            device=device,
            backend=backend,
            config_path=config_path,
        )
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    if perplexity:
        print("PERPLEXITY EVALUATION:")
    elif generate:
        print("GENERATING OUTPUT:")
    else:
        print("FORWARD PASS (NO GENERATION):")
    print("=" * 60 + "\n")

    try:
        input_ids = torch.tensor([truncated_tokens], dtype=torch.long, device=device)

        if perplexity:
            print("Running perplexity evaluation...")
            start_time = time.time()
            metrics = model.perplexity(input_ids)
            end_time = time.time()
            print(f"Eval time: {end_time - start_time:.4f} seconds")
            print(f"Tokens evaluated: {metrics['num_tokens']}")
            print(f"Cross-entropy loss: {metrics['loss']:.6f}")
            print(f"Perplexity: {metrics['perplexity']:.6f}")
            return 0

        if not generate:
            print("Running forward pass only...")
            with torch.no_grad():
                start_time = time.time()
                logits = model(input_ids)
                end_time = time.time()
            print(f"Prefill time: {end_time - start_time:.4f} seconds")
            print(f"Logits shape: {logits.shape}")
            print(f"First 4 logits (last token in batch 0): {logits[0, -1, :4].tolist()}")
            return 0

        print(f"Input prompt length: {actual_token_count} tokens")
        print(f"Generating up to {max_new_tokens} new tokens...")
        print(f"Temperature: {temperature}, Top-p: {top_p}, Top-k: {top_k}\n")
        print(f"Input token IDs shape: {input_ids.shape}")

        if measure_power:
            generated = measure_generation_power(
                model.generate,
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        else:
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

        if model.tokenizer is not None:
            prompt_len = input_ids.size(1)
            generated_tokens = generated[0, prompt_len:].tolist()
            decoded_generated = model.tokenizer.decode(generated_tokens, skip_special_tokens=False)

            print(f"\n{'=' * 60}")
            print("GENERATED TEXT:")
            print(f"{'=' * 60}")
            print(decoded_generated)
            print(f"{'=' * 60}")
            print(f"\nGenerated {len(generated_tokens)} tokens")
        else:
            print(f"\nGenerated token IDs: {generated}")
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print(f"\n{'=' * 60}")
    print("Test completed!")
    if hasattr(model, "load_time"):
        print(f"Weight loading time: {model.load_time:.2f} seconds")
    print(f"{'=' * 60}\n")
    return 0


def run_wikitext2_perplexity(
    model_path=None,
    tokenizer_path=None,
    device="cuda",
    backend="hetero",
    config_path=None,
    split: str = "test",
    max_seq_len: Optional[int] = None,
    max_length: int = 2048,
    stride: int = 2048,
):
    if max_length < 2:
        raise ValueError("max_length must be >= 2")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    def _coerce_chunk_size(raw_value):
        if raw_value is None:
            return None
        if isinstance(raw_value, (list, tuple)):
            for item in raw_value:
                try:
                    val = int(item)
                except Exception:
                    continue
                if val > 0:
                    return val
            return None
        try:
            val = int(raw_value)
            return val if val > 0 else None
        except Exception:
            return None

    script_dir = Path(__file__).parent
    model_weights_dir = script_dir / "model_weights"
    model_weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"WIKITEXT-2 PERPLEXITY ({split} split)")
    print("=" * 60 + "\n")

    text = _load_wikitext2_raw_text(model_weights_dir, split=split)

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    effective_max_seq_len = _compute_effective_max_seq_len(
        max_seq_len,
        prompt_token_count=max_length,
    )
    if max_seq_len is None:
        print(f"Auto-selected backend max_seq_len={effective_max_seq_len} from WikiText window max_length={max_length}")

    print(f"Initializing {DEFAULT_MODEL_LABEL} w4a16 quantized model...")
    try:
        model = Phi35MiniInstructAWQW4A16Model(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            max_seq_len=effective_max_seq_len,
            device=device,
            backend=backend,
            config_path=config_path,
        )
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback

        traceback.print_exc()
        return 1

    if model.tokenizer is None:
        print("Error: tokenizer is required for WikiText-2 perplexity.")
        return 1

    print("Tokenizing WikiText-2 corpus...")
    encoded = model.tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids_full = encoded["input_ids"]
    if input_ids_full.size(1) < 2:
        print("Error: tokenized WikiText-2 corpus is too short.")
        return 1

    token_cache_path = model_weights_dir / f"wikitext-2-raw-v1_{split}_tokens.pt"
    torch.save(input_ids_full.cpu(), token_cache_path)
    print(f"Saved tokenized WikiText-2 tensor: {token_cache_path}")
    print(f"Total tokens: {input_ids_full.size(1)}")
    print(f"Eval max_length: {max_length}, stride: {stride}")

    backend_prefill_chunk = None
    if hasattr(model, "model") and hasattr(model.model, "get_prefill_chunk_size"):
        try:
            backend_prefill_chunk = int(model.model.get_prefill_chunk_size())
        except Exception:
            backend_prefill_chunk = None
    if (backend_prefill_chunk is None or backend_prefill_chunk <= 0) and isinstance(getattr(model, "config", None), dict):
        try:
            gpu_chunk_cfg = model.config.get("gpu_chunking")
            if isinstance(gpu_chunk_cfg, dict):
                backend_prefill_chunk = _coerce_chunk_size(gpu_chunk_cfg.get("gpu_chunk_size"))
                if backend_prefill_chunk is None:
                    backend_prefill_chunk = _coerce_chunk_size(gpu_chunk_cfg.get("chunk_size"))
            elif model.config.get("chunking", False) and not isinstance(model.config.get("chunking"), bool):
                backend_prefill_chunk = _coerce_chunk_size(model.config.get("chunking"))
        except Exception:
            backend_prefill_chunk = None
    if backend_prefill_chunk is None or backend_prefill_chunk <= 0:
        backend_prefill_chunk = min(int(getattr(model, "max_seq_len", 4096)), 4096)
    forward_chunk_size = max(1, min(backend_prefill_chunk, max_length))
    print(f"Internal forward chunk size: {forward_chunk_size}")

    total_nll = 0.0
    total_tokens = 0
    prev_end_loc = 0
    seq_len = input_ids_full.size(1)
    start_time = time.time()
    window_starts = list(range(0, seq_len, stride))
    total_windows = len(window_starts)

    for window_idx, begin_loc in enumerate(window_starts):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids_window = input_ids_full[:, begin_loc:end_loc]
        window_len = input_ids_window.size(1)
        input_ids_window_dev = input_ids_window.to(device)
        tokens_to_ignore = max(0, (window_len - 1) - trg_len)
        window_loss_sum = 0.0
        window_valid_tokens = 0

        try:
            with torch.no_grad():
                if window_len <= forward_chunk_size:
                    chunk_ranges = [(0, window_len)]
                else:
                    chunk_ranges = [
                        (i, min(i + forward_chunk_size, window_len))
                        for i in range(0, window_len, forward_chunk_size)
                    ]

                for chunk_begin, chunk_end in chunk_ranges:
                    chunk_input = input_ids_window_dev[:, chunk_begin:chunk_end]
                    if chunk_begin == 0:
                        chunk_logits = model(chunk_input)
                    else:
                        chunk_logits = model(chunk_input, start_pos=chunk_begin)

                    target_begin = chunk_begin + 1
                    target_end = min(chunk_end + 1, window_len)
                    if target_begin >= target_end:
                        continue

                    labels = input_ids_window_dev[:, target_begin:target_end].to(chunk_logits.device).contiguous()
                    chunk_token_count = labels.size(1)
                    if chunk_logits.dim() != 3 or chunk_logits.size(0) != labels.size(0):
                        raise RuntimeError(
                            f"Unexpected logits shape from backend: {tuple(chunk_logits.shape)} "
                            f"for labels shape {tuple(labels.shape)}"
                        )
                    available_logits = int(chunk_logits.size(1))
                    if available_logits <= 0:
                        continue
                    if available_logits < chunk_token_count:
                        raise RuntimeError(
                            "Backend returned fewer logits than expected for perplexity. "
                            f"chunk_begin={chunk_begin}, chunk_end={chunk_end}, "
                            f"chunk_input_len={chunk_input.size(1)}, labels_len={chunk_token_count}, "
                            f"logits_len={available_logits}. "
                            "This usually means prefill chunking returned last-token-only logits; "
                            "reduce effective chunk size or adjust backend chunk config."
                        )
                    logits_for_loss = chunk_logits[:, :chunk_token_count, :].float().contiguous()
                    vocab_size = logits_for_loss.size(-1)

                    ignore_prefix = max(0, tokens_to_ignore - chunk_begin)
                    if ignore_prefix >= chunk_token_count:
                        continue
                    if ignore_prefix > 0:
                        labels = labels.clone()
                        labels[:, :ignore_prefix] = -100

                    loss_sum = torch.nn.functional.cross_entropy(
                        logits_for_loss.view(-1, vocab_size),
                        labels.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    valid_tokens_chunk = int((labels != -100).sum().item())
                    window_loss_sum += float(loss_sum.item())
                    window_valid_tokens += valid_tokens_chunk
        except Exception as e:
            print(
                f"Error during forward pass at window begin={begin_loc}, end={end_loc}, "
                f"window_len={end_loc - begin_loc}, stride={stride}: {e}"
            )
            return 1

        total_nll += window_loss_sum
        total_tokens += window_valid_tokens

        progress_ratio = float(window_idx + 1) / float(total_windows)
        bar_width = 28
        filled = int(progress_ratio * bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)
        elapsed = time.time() - start_time
        running_ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("nan")
        print(
            f"\rProgress [{bar}] {window_idx + 1}/{total_windows} "
            f"({progress_ratio * 100.0:5.1f}%) | begin={begin_loc}, end={end_loc}, "
            f"tokens_evaluated={total_tokens}, running_ppl={running_ppl:.4f}, elapsed={elapsed:.1f}s",
            end="",
            flush=True,
        )

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    print()

    if total_tokens <= 0:
        print("Error: no valid tokens were evaluated.")
        return 1

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    end_time = time.time()

    print("\n" + "=" * 60)
    print("WIKITEXT-2 PERPLEXITY RESULT")
    print("=" * 60)
    print(f"Split: {split}")
    print(f"Tokens evaluated: {total_tokens}")
    print(f"Average NLL (loss): {avg_nll:.6f}")
    print(f"Perplexity: {ppl:.6f}")
    print(f"Eval time: {end_time - start_time:.2f} seconds")
    print("=" * 60 + "\n")
    return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description=f"{DEFAULT_MODEL_LABEL} W4A16 Quantized Model - Unified Hetero Backend")
    parser.add_argument(
        "--text",
        type=str,
        default="What is the meaning of life the universe and everything?",
        help="Input text to process.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer or HuggingFace model name",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to quantized model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="hetero",
        choices=["hetero"],
        help="Backend to use (default: hetero)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to hetero config JSON (default: auto-detect from lscpu)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Maximum context length to preallocate in the backend. If omitted, auto-size from the prompt and generation budget.",
    )
    parser.add_argument(
        "--generate",
        dest="generate",
        action="store_true",
        default=True,
        help="Generate text instead of just getting logits (default: True)",
    )
    parser.add_argument(
        "--no-generate",
        dest="generate",
        action="store_false",
        help="Disable generation, just get logits",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--prompt-test",
        type=int,
        default=None,
        help="Run prompt test case with specified token count.",
    )
    parser.add_argument(
        "--measure-power",
        action="store_true",
        default=False,
        help="Enable power measurement during generation.",
    )
    parser.add_argument(
        "--save-used-prompt",
        action="store_true",
        default=False,
        help="Save the exact prompt used for the run to used_prompt.txt.",
    )
    parser.add_argument(
        "--perplexity",
        action="store_true",
        default=False,
        help="Compute perplexity for the input text (or prompt-test sequence) instead of generation.",
    )
    parser.add_argument(
        "--wikitext2-perplexity",
        action="store_true",
        help="Compute perplexity on WikiText-2 and save downloaded/tokenized files under model_weights.",
    )
    parser.add_argument(
        "--wikitext2-split",
        type=str,
        default="test",
        choices=["train", "valid", "validation", "test"],
        help="WikiText-2 split to evaluate.",
    )
    parser.add_argument(
        "--wikitext2-max-length",
        type=int,
        default=2048,
        help="Max context length per evaluation window for WikiText-2 perplexity.",
    )
    parser.add_argument(
        "--wikitext2-stride",
        type=int,
        default=2048,
        help="Stride for sliding-window WikiText-2 perplexity.",
    )

    args = parser.parse_args()

    if args.wikitext2_perplexity:
        return run_wikitext2_perplexity(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            backend=args.backend,
            max_seq_len=args.max_seq_len,
            config_path=args.config_path,
            split=args.wikitext2_split,
            max_length=args.wikitext2_max_length,
            stride=args.wikitext2_stride,
        )

    if args.prompt_test is not None:
        return run_prompt_test(
            args.prompt_test,
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            backend=args.backend,
            max_seq_len=args.max_seq_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            generate=args.generate,
            config_path=args.config_path,
            perplexity=args.perplexity,
            measure_power=args.measure_power,
            save_used_prompt=args.save_used_prompt,
        )

    args.config_path = _resolve_phi35_mini_config_path(args.config_path)
    _set_prompt_len_in_config(args.config_path, 1)

    resolved_model_path = args.model_path or DEFAULT_MODEL_PATH
    resolved_tokenizer_path = args.tokenizer_path or _default_tokenizer_path_for_model(resolved_model_path)
    prompt_token_count = None
    try:
        prompt_token_count = _infer_prompt_token_count(resolved_tokenizer_path, args.text)
    except Exception as e:
        print(f"Warning: Could not infer prompt token count from tokenizer {resolved_tokenizer_path}: {e}")
    effective_max_seq_len = _compute_effective_max_seq_len(
        args.max_seq_len,
        prompt_token_count=prompt_token_count,
        max_new_tokens=args.max_new_tokens if args.generate else 0,
    )
    if args.max_seq_len is None:
        if prompt_token_count is None:
            print(f"Auto-selected backend max_seq_len={effective_max_seq_len} using fallback runtime default")
        else:
            print(
                f"Auto-selected backend max_seq_len={effective_max_seq_len} "
                f"from prompt={prompt_token_count} and max_new_tokens={args.max_new_tokens if args.generate else 0}"
            )

    print("=" * 60)
    print(f"Initializing {DEFAULT_MODEL_LABEL} w4a16 quantized model...")
    print("=" * 60)

    try:
        model = Phi35MiniInstructAWQW4A16Model(
            model_path=resolved_model_path,
            tokenizer_path=resolved_tokenizer_path,
            max_seq_len=effective_max_seq_len,
            device=args.device,
            backend=args.backend,
            config_path=args.config_path,
        )
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print(f"Processing text: '{args.text}'")
    if args.save_used_prompt:
        _save_used_prompt(args.text)

    if args.perplexity:
        print("\nRunning perplexity evaluation...")
        try:
            input_ids = model.tokenize(args.text)
            start_time = time.time()
            metrics = model.perplexity(input_ids)
            end_time = time.time()
            print(f"Eval time: {end_time - start_time:.4f} seconds")
            print(f"Tokens evaluated: {metrics['num_tokens']}")
            print(f"Cross-entropy loss: {metrics['loss']:.6f}")
            print(f"Perplexity: {metrics['perplexity']:.6f}")
        except Exception as e:
            print(f"Error during perplexity evaluation: {e}")
            import traceback

            traceback.print_exc()
            return 1
    elif args.generate:
        print(f"Generating {args.max_new_tokens} tokens...\n")
        try:
            input_ids = model.tokenize(args.text)

            if args.measure_power:
                generated = measure_generation_power(
                    model.generate,
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
            else:
                generated = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )

            if model.tokenizer is not None:
                decoded_full = model.tokenizer.decode(generated[0].tolist(), skip_special_tokens=False)
                prompt_len = input_ids.size(1)
                generated_tokens = generated[0, prompt_len:].tolist()
                decoded_generated = model.tokenizer.decode(generated_tokens, skip_special_tokens=False)

                print(f"\n{'=' * 60}")
                print("Full output (prompt + generated):")
                print(f"{'=' * 60}")
                print(decoded_full)
                print(f"{'=' * 60}")
                print("Generated text only:")
                print(f"{'=' * 60}")
                print(decoded_generated)
                print(f"{'=' * 60}")
            else:
                print(f"\nGenerated token IDs: {generated}")
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback

            traceback.print_exc()
            return 1
    else:
        print("\nRunning forward pass (getting logits)...")
        try:
            start_time = time.time()
            logits = model(args.text)
            end_time = time.time()
            print(f"Prefill time: {end_time - start_time:.4f} seconds")
            print(f"Logits shape: {logits.shape}")
            print(f"Logits dtype: {logits.dtype}")
            print(f"Logits device: {logits.device}")
            print("\nLogits statistics:")
            print(f"  Min: {logits.min().item():.4f}")
            print(f"  Max: {logits.max().item():.4f}")
            print(f"  Mean: {logits.mean().item():.4f}")
            print(f"  Std: {logits.std().item():.4f}")
        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback

            traceback.print_exc()
            return 1

    print(f"\n{'=' * 60}")
    if hasattr(model, "load_time"):
        print(f"Weight loading time: {model.load_time:.2f} seconds")
    print("Done!")
    print(f"{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
