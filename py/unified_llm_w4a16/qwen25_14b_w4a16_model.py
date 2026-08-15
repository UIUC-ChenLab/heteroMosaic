"""
Qwen2.5-14B-Instruct-AWQ w4a16 Quantized Python frontend.
Handles tokenization and interfaces with the C++ backend.
"""

import torch
import numpy as np
from typing import Optional, List, Union, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
import json
import torch.nn.functional as F
from pathlib import Path
import gc

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

# We will import the backend dynamically in __init__ based on the selected backend
ArchitectureType = None

import subprocess
import time
import re
import os

DEFAULT_SYSTEM_PROFILE = "strixP"
SYSTEM_PROFILE_TOKEN_MAP = [
    (("RYZEN AI 7 350", "RADEON 860M"), "krackanP"),
    (("RYZEN AI 9 HX 370", "RADEON 890M"), "strixP"),
    (("RYZEN AI MAX+ 395", "RADEON 8060S"), "strixH"),
]
QWEN25_14B_PROFILE_CONFIGS = {
    "krackanP": "configs_krackanP_qwen25_14b.json5",
    "strixP": "configs_strixP_qwen25_14b.json5",
    "strixH": "configs_strixH_qwen25_14b.json5",
}
QWEN25_14B_DEFAULT_HETERO_CONFIG = {
    "heterogeneity": "gpu",
    "warmup": False,
    "dummy_weights": False,
    "debug_verbosity": 1,
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
        "gpu_chunk_size": [2048],
        "gpu_chunk_shedule": [4096, 2048, 1024],
        "gpu_chunking_inflight": 2,
    },
    "kernels_gemm_chunked": [],
    "kernels_gemm": [],
    "kernels_gemv": [],
    "kernels": [],
    "npuOnlydefault": [
        {
            "qo": [5120, 5120],
            "kv": [5120, 1024],
            "upgate": [5120, 13824],
            "down": [13824, 5120],
            "fw_path": "hw_bins/npu2/",
            "max_ctx_len": 8192,
            "num_tiles": 32,
            "tile_size": "64x128x64",
            "col": "8c",
            "dtype": "bf16_int4AWQ_bf16",
        }
    ],
}

DEFAULT_QWEN25_14B_MAX_PROMPT_LEN = 16384
DEFAULT_QWEN25_14B_MAX_SEQ_LEN = 16512
LEGACY_PAD_PACKED_ALIGNMENT = 2048
PACKED_PAD_GROUP_TO_LAYER_TYPES = {
    "qkv": ("qkv",),
    "o": ("o",),
    "qo": ("q", "o"),
    "kv": ("k", "v"),
    "upgate": ("gate", "up"),
    "down": ("down",),
}


def _normalize_pad_packed_weights_setting(value: object) -> Union[bool, int]:
    if isinstance(value, bool):
        return value
    if value is None:
        return False

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "0", "false", "off", "no"}:
            return False
        if lowered in {"true", "on", "yes"}:
            return True
        try:
            value = int(lowered)
        except ValueError:
            return False

    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return False

    return numeric_value if numeric_value > 0 else False


def _resolve_pad_packed_alignment(pad_packed: Union[bool, int]) -> int:
    if isinstance(pad_packed, bool):
        return LEGACY_PAD_PACKED_ALIGNMENT if pad_packed else 0

    try:
        numeric_value = int(pad_packed)
    except (TypeError, ValueError):
        return 0

    return numeric_value if numeric_value > 0 else 0


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _normalize_dim_pad_pair(value: object) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            pad_k = int(value[0])
        except (TypeError, ValueError):
            pad_k = 0
        try:
            pad_n = int(value[1])
        except (TypeError, ValueError):
            pad_n = 0
        return (pad_k if pad_k > 0 else 0, pad_n if pad_n > 0 else 0)
    return (0, 0)


def _canonical_pad_group_name(group_name: str) -> str:
    return group_name[:-4] if group_name.endswith("-gen") else group_name


def _extract_packed_weight_pad_by_layer(config: dict) -> Dict[str, tuple[int, int]]:
    pad_by_layer: Dict[str, tuple[int, int]] = {}
    if not isinstance(config, dict):
        return pad_by_layer

    entries = config.get("npuOnlydefault")
    if not isinstance(entries, list):
        return pad_by_layer

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for group_name, group_value in entry.items():
            if str(group_name).endswith("-gen"):
                continue
            canonical_group = _canonical_pad_group_name(str(group_name))
            layer_names = PACKED_PAD_GROUP_TO_LAYER_TYPES.get(canonical_group)
            if not layer_names:
                continue
            if not isinstance(group_value, dict):
                continue

            pad_spec = _normalize_dim_pad_pair(group_value.get("pad"))
            if pad_spec == (0, 0):
                continue

            for layer_name in layer_names:
                pad_by_layer[layer_name] = pad_spec

    return pad_by_layer


def _resolve_layer_pad_packed_alignment(
    default_pad_packed: Union[bool, int, tuple[int, int]],
    pad_by_layer: Optional[Dict[str, tuple[int, int]]],
    layer_name: str,
) -> tuple[int, int]:
    if isinstance(default_pad_packed, tuple):
        enabled_pad = _normalize_dim_pad_pair(default_pad_packed)
        if enabled_pad == (0, 0):
            return (0, 0)
    else:
        scalar_align = _resolve_pad_packed_alignment(default_pad_packed)
        if scalar_align <= 0:
            return (0, 0)
        enabled_pad = (scalar_align, scalar_align)

    if isinstance(pad_by_layer, dict):
        pad_spec = pad_by_layer.get(layer_name)
        if pad_spec is not None:
            return _normalize_dim_pad_pair(pad_spec)

    return enabled_pad


def _normalize_cpu_model_name(model_name: str) -> str:
    return re.sub(r"\s+", " ", model_name.strip().upper())


def _detect_lscpu_model_name() -> str:
    try:
        output = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.STDOUT)
    except Exception as e:
        print(f"Warning: could not run lscpu for config auto-detection: {e}")
        return ""

    match = re.search(r"^Model name:\s*(.+)$", output, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()

    print("Warning: could not find 'Model name' in lscpu output for config auto-detection.")
    return ""


def _detect_system_profile() -> tuple[str, str]:
    cpu_model_name = _detect_lscpu_model_name()
    normalized = _normalize_cpu_model_name(cpu_model_name) if cpu_model_name else ""
    for tokens, profile in SYSTEM_PROFILE_TOKEN_MAP:
        if all(token in normalized for token in tokens):
            return profile, cpu_model_name
    return DEFAULT_SYSTEM_PROFILE, cpu_model_name


def _ensure_default_config_file(config_path: Union[str, Path], default_config: dict) -> Path:
    path = Path(config_path).expanduser().resolve()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=4)
        f.write("\n")
    print(f"Created missing config file with default values: {path}")
    return path


def _resolve_qwen25_14b_config_path(config_path: Optional[str]) -> str:
    if config_path:
        resolved = _ensure_default_config_file(config_path, QWEN25_14B_DEFAULT_HETERO_CONFIG)
        print(f"Using user-provided config for qwen25_14b: {resolved}")
        return str(resolved)

    profile, cpu_model_name = _detect_system_profile()
    filename = QWEN25_14B_PROFILE_CONFIGS.get(profile, QWEN25_14B_PROFILE_CONFIGS[DEFAULT_SYSTEM_PROFILE])
    selected = _ensure_default_config_file(
        Path(__file__).parent.resolve() / "configs" / filename,
        QWEN25_14B_DEFAULT_HETERO_CONFIG,
    )
    if cpu_model_name:
        print(
            f"Auto-selected qwen25_14b config profile '{profile}' based on "
            f"lscpu model '{cpu_model_name}': {selected}"
        )
    else:
        print(f"Auto-selected qwen25_14b fallback profile '{profile}': {selected}")
    return str(selected)


def set_prompt_len_in_config(config_path: str, prompt_len: int) -> None:
    path = Path(config_path).expanduser().resolve()
    if prompt_len <= 0:
        raise ValueError(f"prompt_len must be a positive integer, got: {prompt_len}")

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    updated_raw, replacements = re.subn(
        r'("prompt_len"\s*:\s*)(-?\d+)',
        rf"\g<1>{prompt_len}",
        raw,
        count=1,
    )

    if replacements == 0:
        updated_raw, replacements = re.subn(
            r'"npu_dim"\s*:\s*(-?\d+)',
            f"\"prompt_len\": {prompt_len}",
            raw,
            count=1,
        )

    if replacements == 0:
        cfg = load_config_with_comments(str(path))
        cfg["prompt_len"] = prompt_len
        if "npu_dim" in cfg:
            del cfg["npu_dim"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
            f.write("\n")
        print(f"Set prompt_len={prompt_len} in config: {path}")
        return

    with open(path, "w", encoding="utf-8") as f:
        f.write(updated_raw)
    print(f"Set prompt_len={prompt_len} in config: {path}")


def select_prompt_test_prompt_len(config_path: str, target_tokens: int) -> int:
    selected_prompt_len = int(target_tokens)
    try:
        cfg = load_config_with_comments(config_path)
        chunk_size = 0
        chunking_enabled = False
        if isinstance(cfg, dict):
            chunking_raw = cfg.get("chunking", False)
            chunking_enabled = bool(chunking_raw)

            gpu_chunk_cfg = cfg.get("gpu_chunking")
            if isinstance(gpu_chunk_cfg, dict):
                try:
                    chunk_size = int(gpu_chunk_cfg.get("gpu_chunk_size", gpu_chunk_cfg.get("chunk_size", 0)))
                except Exception:
                    chunk_size = 0

            if chunk_size <= 0 and not isinstance(chunking_raw, bool):
                try:
                    chunk_size = int(chunking_raw)
                except Exception:
                    chunk_size = 0

        if chunking_enabled and chunk_size > 0:
            print(
                f"Chunking active for prompt-test (target_tokens={target_tokens}, chunk_size={chunk_size}); "
                f"keeping prompt_len={selected_prompt_len}"
            )
    except Exception as e:
        print(f"Warning: could not read chunking from config for prompt_len selection: {e}")
    return selected_prompt_len


def measure_generation_power(model_generate_func, *args, **kwargs):
    """Wrapper to measure power during generation using minimal overhead subprocess."""
    log_file = "power_monitor_log.txt"
    if os.path.exists(log_file):
        os.remove(log_file)
        
    start_time = time.time()
    
    # Start background process using shell to loop and log
    # This avoids python interpreter overhead for the monitoring
    monitor_cmd = f"while true; do rocm-smi --showpower >> {log_file}; sleep 0.1; done"
    monitor_process = subprocess.Popen(monitor_cmd, shell=True, preexec_fn=os.setsid)
    
    try:
        # Run the actual generation
        result = model_generate_func(*args, **kwargs)
    finally:
        # Stop monitoring
        try:
            import signal
            os.killpg(os.getpgid(monitor_process.pid), signal.SIGTERM)
        except Exception:
            pass
            
    duration = time.time() - start_time
    
    # Parse log file
    readings = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                matches = re.findall(r'Power \(W\):\s*([\d\.]+)', content)
                # Sum power across all reported devices (assuming single node or sum desired)
                readings = [float(m) for m in matches]
                
        except Exception as e:
            print(f"Error parsing power log: {e}")
            
    print(f"\n{'='*60}")
    print(f"POWER CONSUMPTION REPORT:")
    print(f"{'='*60}")
    if readings and len(readings) > 0:
        # Calculate average power and total energy
        avg_power = sum(readings) / len(readings)
        avg_power = sum(readings) / len(readings) 
        total_energy = avg_power * duration 
        
        print(f"Average Power: {avg_power:.2f} W")
        print(f"Total Energy:  {total_energy:.2f} J")
        print(f"Duration:      {duration:.2f} s")
        print(f"Samples:       {len(readings)}")
        print(f"Min Power:     {min(readings):.2f} W")
        print(f"Max Power:     {max(readings):.2f} W")
    else:
        print("No power readings collected.")
    print(f"{'='*60}\n")
    
    if os.path.exists(log_file):
        os.remove(log_file)
        
    return result


def load_config_with_comments(path: str) -> dict:
    """Load JSON/JSON5-like config with // and /* */ comments stripped."""
    try:
        with open(path, "r") as f:
            content = f.read()
        content = re.sub(r"//.*", "", content)
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        return json.loads(content)
    except Exception as e:
        print(f"Error loading config with comment stripping: {e}")
        with open(path, "r") as f:
            return json.load(f)


def _model_name_from_path(model_path: str) -> str:
    if "/" in model_path:
        model_name = model_path.split("/")[-1]
    else:
        model_name = model_path
    return model_name.replace("/", "_").replace("\\", "_")


def _write_tensor_raw(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = tensor.contiguous().cpu()
    t_bytes = t.view(torch.uint8)
    t_bytes.numpy().tofile(str(path))


def _unpack_awq_zigzag_to_contiguous(packed_int32: torch.Tensor) -> torch.Tensor:
    device = packed_int32.device
    permutation = [0, 4, 1, 5, 2, 6, 3, 7]
    parts = []
    for k in range(8):
        shift_amount = permutation[k] * 4
        if shift_amount > 0:
            part = torch.bitwise_right_shift(packed_int32, shift_amount)
        else:
            part = packed_int32
        part = torch.bitwise_and(part, 0x0F).to(torch.uint8)
        parts.append(part)
    return torch.stack(parts, dim=-1).to(device)


def _unpack_awq_qweight(qweight: torch.Tensor) -> torch.Tensor:
    # Input: [In, Out/8] int32 -> Output: [In, Out] uint8
    qweight = qweight.t().contiguous()
    out_features_div_8 = qweight.size(0)
    in_features = qweight.size(1)
    out_features = out_features_div_8 * 8
    unpacked = _unpack_awq_zigzag_to_contiguous(qweight)
    unpacked = unpacked.permute(0, 2, 1).contiguous()
    unpacked = unpacked.view(out_features, in_features)
    return unpacked.t().contiguous().to(torch.uint8)


def _unpack_awq_qzeros(qzeros: torch.Tensor) -> torch.Tensor:
    # Input: [G, Out/8] int32 -> Output: [G, Out] int8
    unpacked = _unpack_awq_zigzag_to_contiguous(qzeros)
    n_groups = qzeros.size(0)
    out_features = qzeros.size(1) * 8
    unpacked = unpacked.view(n_groups, out_features)
    return unpacked.contiguous().to(torch.int8)


def _unpack_gptq_qweight(qweight: torch.Tensor) -> torch.Tensor:
    # Input: [In/8, Out] int32 -> Output: [Out, In/2] uint8 (packed)
    in_features_div_8 = qweight.size(0)
    out_features = qweight.size(1)
    in_features = in_features_div_8 * 8
    parts = []
    for i in range(8):
        shift = i * 4
        part = torch.bitwise_and(torch.bitwise_right_shift(qweight, shift), 0x0F).to(torch.uint8)
        parts.append(part)
    unpacked = torch.stack(parts, dim=2)
    unpacked = unpacked.permute(0, 2, 1).contiguous()
    unpacked = unpacked.view(in_features, out_features)
    unpacked = unpacked.t().contiguous()
    low = unpacked[:, 0:in_features:2]
    high = unpacked[:, 1:in_features:2]
    packed = torch.bitwise_or(low, torch.bitwise_left_shift(high, 4))
    return packed.to(torch.uint8)


def _unpack_gptq_qzeros(qzeros: torch.Tensor, out_features: int, add_one: bool = True) -> torch.Tensor:
    # Input: [G, Out/8] int32 -> Output: [Out, G] int8
    parts = []
    for i in range(8):
        shift = i * 4
        part = torch.bitwise_and(torch.bitwise_right_shift(qzeros, shift), 0x0F).to(torch.int32)
        parts.append(part)
    unpacked = torch.stack(parts, dim=2)
    n_groups = qzeros.size(0)
    unpacked = unpacked.view(n_groups, out_features)
    unpacked = unpacked.t().contiguous()
    zeros = unpacked.to(torch.int8)
    if add_one:
        zeros = zeros + 1
    return zeros


def _unpack_packed_out_in2_to_in_out(qweight_packed: torch.Tensor) -> torch.Tensor:
    # Input: [Out, In/2] uint8 -> Output: [In, Out] int8
    out_features = qweight_packed.size(0)
    in_features = qweight_packed.size(1) * 2
    q_view = qweight_packed.view(out_features, in_features // 2, 1)
    w_low = torch.bitwise_and(q_view, 0x0F).to(torch.int8)
    w_high = torch.bitwise_right_shift(q_view, 4).to(torch.int8)
    w_cat = torch.cat([w_low, w_high], dim=2)
    w_out_in = w_cat.view(out_features, in_features)
    return w_out_in.t().contiguous()


def _pack_qweight_out_in2(unpacked_qweight: torch.Tensor) -> torch.Tensor:
    # Input: [In, Out] int8 -> Output: [Out, In/2] uint8
    w_out_in = unpacked_qweight.t().contiguous()
    out_features = w_out_in.size(0)
    in_features = w_out_in.size(1)
    w_view = w_out_in.view(out_features, in_features // 2, 2)
    w_low = w_view.select(-1, 0).to(torch.uint8)
    w_high = w_view.select(-1, 1).to(torch.uint8)
    packed = torch.bitwise_or(torch.bitwise_and(w_low, 0x0F), torch.bitwise_left_shift(torch.bitwise_and(w_high, 0x0F), 4))
    return packed.to(torch.uint8)


def _normalize_group_tensor(t: torch.Tensor, out_feat: int) -> torch.Tensor:
    if t is None:
        return t
    if t.dim() == 1:
        if t.numel() == out_feat:
            return t
        if t.numel() % out_feat == 0:
            return t.view(out_feat, -1)
        return t
    if t.dim() == 2:
        if t.size(0) == out_feat:
            return t
        if t.size(1) == out_feat:
            return t.t().contiguous()
        if t.numel() % out_feat == 0:
            return t.view(out_feat, -1)
        return t
    if t.numel() % out_feat == 0:
        return t.view(out_feat, -1)
    return t


def _align_zeros_to_scales(zeros: torch.Tensor, scales_out: torch.Tensor, out_feat: int) -> torch.Tensor:
    zeros_out = _normalize_group_tensor(zeros, out_feat)
    if scales_out is None:
        return zeros_out
    if scales_out.dim() == 2:
        target_groups = scales_out.size(1)
        if zeros_out.dim() == 2:
            if zeros_out.size(0) == target_groups and zeros_out.size(1) == out_feat:
                zeros_out = zeros_out.t().contiguous()
            elif zeros_out.size(0) == out_feat and zeros_out.size(1) != target_groups and zeros_out.numel() == out_feat * target_groups:
                zeros_out = zeros_out.reshape(out_feat, target_groups)
        elif zeros_out.numel() == out_feat * target_groups:
            zeros_out = zeros_out.view(out_feat, target_groups)
    return zeros_out


def _split_packed_paths(presaved_dir: Path, layer_idx: int, short_name: str, split_k: int) -> tuple[Path, Path]:
    tag = f"k{int(split_k)}"
    return (
        presaved_dir / f"layer_{layer_idx}_{short_name}.{tag}.packed0.bin",
        presaved_dir / f"layer_{layer_idx}_{short_name}.{tag}.packed1.bin",
    )


def _split_packed_bin_from_file(
    packed_path: Path,
    out0_path: Path,
    out1_path: Path,
    split_k: int,
    in_feat: int,
    out_feat: int,
    pad_packed: Union[bool, int, tuple[int, int]],
) -> bool:
    if not packed_path.exists():
        return False

    # Compute padded sizes to match packing layout
    K_npu = in_feat
    N_npu = out_feat
    pad_align_k, pad_align_n = _resolve_layer_pad_packed_alignment(pad_packed, None, "")
    if pad_align_n > 0:
        N_npu = _round_up_to_multiple(N_npu, pad_align_n)
    if pad_align_k > 0:
        K_npu = _round_up_to_multiple(K_npu, pad_align_k)

    K0 = int(split_k)
    K1 = int(in_feat) - K0
    K0_npu = K0
    K1_npu = K1
    if pad_align_k > 0:
        K0_npu = _round_up_to_multiple(K0_npu, pad_align_k)
        K1_npu = _round_up_to_multiple(K1_npu, pad_align_k)
        if K0_npu + K1_npu != K_npu:
            return False

    if K0_npu % 128 != 0 or K1_npu % 128 != 0 or N_npu % 64 != 0:
        return False

    num_tiles_row = (K_npu + 127) // 128
    num_tiles_col = (N_npu + 63) // 64
    num_tiles = num_tiles_row * num_tiles_col
    expected_size = num_tiles * 4352

    packed = np.fromfile(str(packed_path), dtype=np.uint8)
    if packed.size != expected_size:
        return False

    packed = packed.reshape(num_tiles, 4352)
    tile_indices = _get_tile_indices(K_npu, N_npu).cpu().numpy()
    tile_rows = tile_indices // num_tiles_col
    num_tiles_row_0 = K0_npu // 128
    mask0 = tile_rows < num_tiles_row_0

    packed0 = packed[mask0]
    packed1 = packed[~mask0]
    if packed0.size == 0 or packed1.size == 0:
        return False

    _write_tensor_raw(out0_path, torch.from_numpy(packed0))
    _write_tensor_raw(out1_path, torch.from_numpy(packed1))
    return True


def _get_tile_indices(rows: int, cols: int) -> torch.Tensor:
    LARGE_TILE_SIZE_ROW = 128
    LARGE_TILE_SIZE_COL = 64
    num_tiles_row = (rows + LARGE_TILE_SIZE_ROW - 1) // LARGE_TILE_SIZE_ROW
    num_tiles_col = (cols + LARGE_TILE_SIZE_COL - 1) // LARGE_TILE_SIZE_COL
    tile_indices = []
    for col_mod in range(8):
        for c in range(col_mod, num_tiles_col, 8):
            for r in range(num_tiles_row):
                tile_indices.append(r * num_tiles_col + c)
    return torch.tensor(tile_indices, dtype=torch.long)


def _pack_weights_packed(
    qw: torch.Tensor,
    s: torch.Tensor,
    z: torch.Tensor,
    K_in: int,
    N_in: int,
    pad_packed: Union[bool, int, tuple[int, int]],
) -> torch.Tensor:
    LARGE_TILE_SIZE_ROW = 128
    LARGE_TILE_SIZE_COL = 64

    K_npu = K_in
    N_npu = N_in

    qw_p = qw
    s_p = s
    z_p = z

    pad_align_K, pad_align_N = _resolve_layer_pad_packed_alignment(pad_packed, None, "")
    if pad_align_K > 0 or pad_align_N > 0:

        if pad_align_N > 0 and N_in % pad_align_N != 0:
            padded_N = _round_up_to_multiple(N_in, pad_align_N)
            pad_val = padded_N - N_in
            N_npu = padded_N
            qw_p = F.pad(qw_p, (0, pad_val))

            if s.size(0) == N_in:
                if s.dim() == 1:
                    s_p = F.pad(s_p, (0, pad_val))
                    z_p = F.pad(z_p, (0, pad_val))
                else:
                    s_p = F.pad(s_p, (0, 0, 0, pad_val))
                    z_p = F.pad(z_p, (0, 0, 0, pad_val))
            elif s.size(1) == N_in:
                s_p = F.pad(s_p, (0, pad_val, 0, 0))
                z_p = F.pad(z_p, (0, pad_val, 0, 0))

        if pad_align_K > 0 and K_in % pad_align_K != 0:
            padded_K = _round_up_to_multiple(K_in, pad_align_K)
            pad_val_k = padded_K - K_in
            K_npu = padded_K
            qw_p = F.pad(qw_p, (0, 0, 0, pad_val_k))

            is_grouped = (s.numel() > s.size(0) and s.numel() > s.size(1))
            if s.dim() == 1 and s.size(0) == N_in:
                is_grouped = False

            if is_grouped:
                num_groups = -1
                group_dim = -1
                if s_p.size(0) == N_npu and s_p.dim() == 2:
                    num_groups = s_p.size(1)
                    group_dim = 1
                elif s_p.size(1) == N_npu and s_p.dim() == 2:
                    num_groups = s_p.size(0)
                    group_dim = 0

                if num_groups > 0:
                    group_size = 128
                    expected_groups = (padded_K + group_size - 1) // group_size
                    pad_groups = expected_groups - num_groups
                    if pad_groups > 0:
                        if group_dim == 1:
                            s_p = F.pad(s_p, (0, pad_groups, 0, 0))
                            z_p = F.pad(z_p, (0, pad_groups, 0, 0))
                        else:
                            s_p = F.pad(s_p, (0, 0, 0, pad_groups))
                            z_p = F.pad(z_p, (0, 0, 0, pad_groups))

    scale_flat = s_p.contiguous().view(-1)
    zero_flat = z_p.contiguous().view(-1)
    num_scales = scale_flat.size(0)
    total_weights = N_npu * K_npu
    group_size = (total_weights // num_scales) if num_scales > 0 else 128

    num_tiles_row = (K_npu + LARGE_TILE_SIZE_ROW - 1) // LARGE_TILE_SIZE_ROW
    num_tiles_col = (N_npu + LARGE_TILE_SIZE_COL - 1) // LARGE_TILE_SIZE_COL
    K_padded = num_tiles_row * LARGE_TILE_SIZE_ROW
    N_padded = num_tiles_col * LARGE_TILE_SIZE_COL

    w_padded = torch.zeros((K_padded, N_padded), dtype=torch.int8)
    w_padded[:K_npu, :N_npu].copy_(qw_p)

    w_tiles = w_padded.view(num_tiles_row, 128, num_tiles_col, 64)
    w_tiles = w_tiles.permute(0, 2, 1, 3).contiguous().view(-1, 128, 64)

    tile_indices = _get_tile_indices(K_npu, N_npu)
    w_ordered = w_tiles.index_select(0, tile_indices)

    total_tiles = w_ordered.size(0)
    w_reshaped = w_ordered.view(total_tiles, 16, 8, 8, 8)
    w_permuted = w_reshaped.permute(0, 1, 3, 2, 4).contiguous()
    w_even = w_permuted[..., 0::2]
    w_odd = w_permuted[..., 1::2]
    w_packed_blk = torch.bitwise_or(torch.bitwise_and(w_even, 0x0F), torch.bitwise_left_shift(torch.bitwise_and(w_odd, 0x0F), 4)).to(
        torch.uint8
    )
    packed_weights_flat = w_packed_blk.view(-1, 4096)

    tile_rows = torch.div(tile_indices, num_tiles_col, rounding_mode="floor")
    tile_cols = torch.remainder(tile_indices, num_tiles_col)
    col_offsets = torch.arange(64).unsqueeze(0)
    global_col_indices = tile_cols.unsqueeze(1) * 64 + col_offsets
    global_rows_start = tile_rows.unsqueeze(1) * 128
    group_idx = torch.div(global_rows_start, group_size, rounding_mode="floor")
    scale_indices = group_idx * N_npu + global_col_indices
    scale_indices = torch.clamp(scale_indices, 0, num_scales - 1).to(torch.long)

    gathered_scales = scale_flat.index_select(0, scale_indices.view(-1)).view(-1, 64)
    gathered_zeros = zero_flat.index_select(0, scale_indices.view(-1)).view(-1, 64)

    scales_uint8 = gathered_scales.view(torch.uint8).view(-1, 128)
    zeros_dup = gathered_zeros.view(torch.uint8).view(-1, 8, 8).repeat_interleave(2, 1).view(-1, 128)

    packed_final = torch.cat([packed_weights_flat, scales_uint8, zeros_dup], dim=1)
    return packed_final.contiguous().view(-1)


def _get_quantized_tensors(state_dict: Dict[str, torch.Tensor], base_name: str):
    packed_key = base_name + ".weight_packed"
    scale_key = base_name + ".weight_scale"
    if packed_key in state_dict and scale_key in state_dict:
        qweight = state_dict[packed_key]
        scales = state_dict[scale_key]
        qzeros = state_dict.get(base_name + ".weight_qzeros", None)
        g_idx = state_dict.get(base_name + ".weight_g_idx", None)
        return qweight, scales, qzeros, g_idx, True

    qweight = state_dict.get(base_name + ".qweight")
    scales = state_dict.get(base_name + ".scales")
    qzeros = state_dict.get(base_name + ".qzeros")
    g_idx = state_dict.get(base_name + ".g_idx")
    if qweight is None or scales is None or qzeros is None:
        missing = [k for k in (base_name + ".qweight", base_name + ".scales", base_name + ".qzeros") if k not in state_dict]
        raise KeyError(f"Missing quantized tensors for {base_name}: {missing}")
    return qweight, scales, qzeros, g_idx, False


def _resolve_safetensors_sources(model_path: str) -> Tuple[List[Path], Dict[str, Path]]:
    model_path_obj = Path(model_path).expanduser().resolve()
    if model_path_obj.is_file() and model_path_obj.suffix == ".safetensors":
        return [model_path_obj], {}

    if not model_path_obj.exists() or not model_path_obj.is_dir():
        raise FileNotFoundError(f"Model path does not exist or is not a directory: {model_path_obj}")

    index_candidates = sorted(model_path_obj.glob("*.safetensors.index.json"))
    if index_candidates:
        index_path = index_candidates[0]
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid safetensors index format in {index_path}")

        ordered_files: List[Path] = []
        seen = set()
        tensor_to_file: Dict[str, Path] = {}
        for key, rel_file in weight_map.items():
            resolved = (model_path_obj / rel_file).resolve()
            tensor_to_file[key] = resolved
            if resolved not in seen:
                seen.add(resolved)
                ordered_files.append(resolved)

        if ordered_files:
            return ordered_files, tensor_to_file

    safetensors_files = sorted(model_path_obj.glob("*.safetensors"))
    return [p.resolve() for p in safetensors_files], {}


def _build_tensor_to_file_map(
    safetensors_files: List[Path],
    seed_map: Optional[Dict[str, Path]] = None,
) -> Dict[str, Path]:
    from safetensors import safe_open

    tensor_to_file: Dict[str, Path] = {}
    if seed_map:
        for key, value in seed_map.items():
            tensor_to_file[key] = Path(value).expanduser().resolve()

    for st_path in safetensors_files:
        st_path = Path(st_path).expanduser().resolve()
        with safe_open(str(st_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor_to_file.setdefault(key, st_path)
    return tensor_to_file


def _load_selected_tensors_from_file(safetensors_path: Path, keys: List[str]) -> Dict[str, torch.Tensor]:
    from safetensors import safe_open

    selected: Dict[str, torch.Tensor] = {}
    if not keys:
        return selected

    unique_keys = list(dict.fromkeys(keys))
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        for key in unique_keys:
            if key in available:
                selected[key] = handle.get_tensor(key)
    return selected


def _load_state_slice_for_base_name(base_name: str, tensor_to_file: Dict[str, Path]) -> Dict[str, torch.Tensor]:
    packed_key = base_name + ".weight_packed"
    scale_key = base_name + ".weight_scale"
    packed_qzeros_key = base_name + ".weight_qzeros"
    packed_g_idx_key = base_name + ".weight_g_idx"

    qweight_key = base_name + ".qweight"
    scales_key = base_name + ".scales"
    qzeros_key = base_name + ".qzeros"
    g_idx_key = base_name + ".g_idx"

    keys_to_load: List[str] = []
    if packed_key in tensor_to_file and scale_key in tensor_to_file:
        keys_to_load.extend([packed_key, scale_key])
        if packed_qzeros_key in tensor_to_file:
            keys_to_load.append(packed_qzeros_key)
        if packed_g_idx_key in tensor_to_file:
            keys_to_load.append(packed_g_idx_key)
    elif qweight_key in tensor_to_file and scales_key in tensor_to_file and qzeros_key in tensor_to_file:
        keys_to_load.extend([qweight_key, scales_key, qzeros_key])
        if g_idx_key in tensor_to_file:
            keys_to_load.append(g_idx_key)
    else:
        candidates = [packed_key, scale_key, qweight_key, scales_key, qzeros_key]
        found = [k for k in candidates if k in tensor_to_file]
        raise KeyError(f"Could not resolve quantized tensors for {base_name}. Found keys: {found}")

    keys_by_file: Dict[Path, List[str]] = {}
    for key in keys_to_load:
        file_path = tensor_to_file.get(key)
        if file_path is None:
            continue
        keys_by_file.setdefault(file_path, []).append(key)

    state_slice: Dict[str, torch.Tensor] = {}
    for file_path, file_keys in keys_by_file.items():
        state_slice.update(_load_selected_tensors_from_file(file_path, file_keys))
    return state_slice



class Qwen25_14BW4A16Model:
    """Qwen2.5-14B-Instruct-AWQ w4a16 quantized model wrapper."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        vocab_size: int = 152064,
        hidden_size: int = 5120,
        intermediate_size: int = 13824,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 40,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        max_seq_len: int = DEFAULT_QWEN25_14B_MAX_SEQ_LEN,
        max_batch_size: int = 1,
        groupsize: int = 128,
        device: str = "cuda",
        backend: str = "base",
        config_path: Optional[str] = None
    ):


        """
        Initialize Qwen2.5-14B-Instruct-AWQ w4a16 quantized model.
        Defaults are for Qwen2.5-14B-Instruct-AWQ.
        """
        self.device = device
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.groupsize = groupsize
        self.enable_fused_qkv_bins = False
        
        # Initialize tokenizer
        if tokenizer_path is None:
            tokenizer_path = "Qwen/Qwen2.5-14B-Instruct"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Warning: Could not load tokenizer from {tokenizer_path}: {e}")
            self.tokenizer = None
        
        # Initialize C++ backend
        if backend == "base":
            try:
                import unified_llm_w4a16_libtorch
                self.backend_module = unified_llm_w4a16_libtorch
            except ImportError as e:
                raise ImportError(f"Could not import base backend: {e}")
        elif backend == "hetero":
            try:
                import unified_llm_w4a16_hetero_libtorch
                self.backend_module = unified_llm_w4a16_hetero_libtorch
            except ImportError as e:
                raise ImportError(f"Could not import hetero backend: {e}")
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Get ArchitectureType from the selected backend
        ArchitectureType = self.backend_module.ArchitectureType

        # Prepare constructor arguments
        constructor_args = [
            ArchitectureType.QWEN15,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rope_theta,
            max_seq_len,
            max_batch_size,
            groupsize,
            device
        ]

        if backend == "hetero":
            config_path = _resolve_qwen25_14b_config_path(config_path)
            constructor_args.append(config_path)

        self.model = self.backend_module.UnifiedLLMW4A16(*constructor_args)
        
        # Check dummy weights flag (regardless of backend)
        self.config = {}
        self.use_packed_weights = False
        self.use_pre_saved_weights = False
        self.pad_packed_weights = False
        self.packed_weight_pad_by_layer: Dict[str, tuple[int, int]] = {}
        self.debug_verbosity = 0

        use_dummy = False
        if config_path:
            try:
                self.config = load_config_with_comments(config_path)
                self.use_packed_weights = bool(self.config.get("usePackedWeights", False))
                self.use_pre_saved_weights = bool(self.config.get("usePreSavedWeights", False))
                self.pad_packed_weights = _normalize_pad_packed_weights_setting(self.config.get("padPackedWeights", False))
                self.packed_weight_pad_by_layer = _extract_packed_weight_pad_by_layer(self.config)
                self.debug_verbosity = int(self.config.get("debug_verbosity", 0))
                if self.config.get("dummy_weights", False):
                    use_dummy = True
                    print("Dummy weights enabled in config.")
            except Exception as e:
                print(f"Error reading config: {e}")

        # Load quantized weights if model_path is provided and not using dummy
        if use_dummy:
            if hasattr(self.model, "initialize_dummy_weights"):
                print("Initializing dummy weights...")
                self.model.initialize_dummy_weights()
            else:
                 print("Warning: Backend does not support dummy weights initialization. Falling back to file loading or random init.")
                 if model_path:
                    self._load_quantized_weights(model_path, weights_folder="model_weights")
        elif model_path:
            self._load_quantized_weights(model_path, weights_folder="model_weights")

        if backend == "hetero":
            print("Importing weights to NPU...")
            self.model.import_weights()
    
    def _load_quantized_weights(self, model_path: str, weights_folder: str = "model_weights"):
        """Load quantized weights from neuralmagic model using safetensors directly."""
        print(f"Loading quantized weights from {model_path}...")
        try:
            from huggingface_hub import snapshot_download
            import shutil
            
            # Create weights folder if it doesn't exist
            script_dir = Path(__file__).parent
            weights_dir = script_dir / weights_folder
            os.makedirs(weights_dir, exist_ok=True)
            
            # Extract model name for filename
            model_name = _model_name_from_path(model_path)
            saved_safetensors = weights_dir / f"{model_name}.safetensors"
            
            safetensors_paths: List[Path] = []
            tensor_to_file: Dict[str, Path] = {}

            if saved_safetensors.exists():
                print(f"Using saved safetensors: {saved_safetensors}")
                safetensors_paths = [saved_safetensors.resolve()]
            else:
                # Download model if needed
                if not os.path.exists(model_path):
                    print(f"Model path {model_path} not found locally, downloading from Hub...")
                    model_path = snapshot_download(repo_id=model_path)
                    print(f"Model downloaded to {model_path}")

                safetensors_paths, tensor_to_file = _resolve_safetensors_sources(model_path)
                if not safetensors_paths:
                    model_dir = Path(model_path)
                    bin_files = sorted(model_dir.glob("*.bin")) if model_dir.exists() else []
                    if not bin_files:
                        raise FileNotFoundError(f"No weight files found in {model_path}")
                    print("Warning: Found .bin files but not .safetensors. This loader is optimized for safetensors.")
                    raise NotImplementedError("Only safetensors are supported for this direct loading method.")

                if len(safetensors_paths) == 1:
                    src = safetensors_paths[0]
                    if src.resolve() != saved_safetensors.resolve():
                        print(f"Saving safetensors to {saved_safetensors}...")
                        shutil.copy2(src, saved_safetensors)
                    print(f"Saved safetensors file: {saved_safetensors}")
                    print(f"File size: {os.path.getsize(saved_safetensors) / (1024**3):.2f} GB")
                    safetensors_paths = [saved_safetensors.resolve()]
                else:
                    print(
                        f"Found {len(safetensors_paths)} safetensors shards; "
                        "using shard-streaming path (no merge) to avoid OOM."
                    )

            # Load into C++ backend from safetensors file
            use_packed = self.use_packed_weights
            use_presaved = self.use_pre_saved_weights
            pad_packed = self.pad_packed_weights

            if use_presaved:
                presaved_dir = weights_dir / f"{model_name}_{'packed' if use_packed else 'unpacked'}"
                self._prepare_presaved_weights(
                    safetensors_paths,
                    presaved_dir,
                    use_packed,
                    pad_packed,
                    model_name=model_name,
                    tensor_to_file=tensor_to_file,
                )

                t0 = time.time()
                if hasattr(self.model, "load_non_quantized_weights_from_safetensors"):
                    self._load_non_quantized_weights_from_safetensors_shards(safetensors_paths, tensor_to_file)
                else:
                    print("Warning: Backend missing load_non_quantized_weights_from_safetensors; falling back to full safetensors load.")
                    ordered_paths = self._order_shards_for_lm_head_last(safetensors_paths, tensor_to_file)
                    for st_path in ordered_paths:
                        self.model.load_quantized_weights_from_safetensors(str(st_path))
                    t1 = time.time()
                    self.load_time = t1 - t0
                    print(f"Weights loaded in {self.load_time:.2f} seconds (fallback)")
                    return

                if hasattr(self.model, "load_quantized_weights_from_bins"):
                    self.model.load_quantized_weights_from_bins(str(presaved_dir))
                else:
                    print("Warning: Backend missing load_quantized_weights_from_bins; falling back to full safetensors load.")
                    ordered_paths = self._order_shards_for_lm_head_last(safetensors_paths, tensor_to_file)
                    for st_path in ordered_paths:
                        self.model.load_quantized_weights_from_safetensors(str(st_path))
                    t1 = time.time()
                    self.load_time = t1 - t0
                    print(f"Weights loaded in {self.load_time:.2f} seconds (fallback)")
                    return
                t1 = time.time()
                self.load_time = t1 - t0
                print(f"Weights loaded from pre-saved bins in {self.load_time:.2f} seconds")
            else:
                # Load into C++ backend from safetensors file(s)
                t0 = time.time()
                ordered_paths = self._order_shards_for_lm_head_last(safetensors_paths, tensor_to_file)
                for st_path in ordered_paths:
                    self.model.load_quantized_weights_from_safetensors(str(st_path))
                t1 = time.time()
                self.load_time = t1 - t0
                print(f"Weights loaded in {self.load_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error loading quantized weights: {e}")
            import traceback
            traceback.print_exc()
            print("\nNote: Falling back to randomly initialized weights.")
            print("The model will not produce meaningful output without proper weights.")

    def _order_shards_for_lm_head_last(
        self,
        safetensors_paths: List[Path],
        tensor_to_file: Optional[Dict[str, Path]] = None,
    ) -> List[Path]:
        if len(safetensors_paths) <= 1:
            return list(safetensors_paths)

        resolved_paths = [Path(p).expanduser().resolve() for p in safetensors_paths]
        tensor_to_file = dict(tensor_to_file or {})
        if "lm_head.weight" not in tensor_to_file:
            try:
                tensor_to_file = _build_tensor_to_file_map(resolved_paths, tensor_to_file)
            except Exception:
                return resolved_paths
        lm_head_file = tensor_to_file.get("lm_head.weight")
        if lm_head_file is None:
            return resolved_paths

        lm_head_file = Path(lm_head_file).expanduser().resolve()
        if lm_head_file not in resolved_paths:
            return resolved_paths

        ordered = [p for p in resolved_paths if p != lm_head_file]
        ordered.append(lm_head_file)
        return ordered

    def _load_non_quantized_weights_from_safetensors_shards(
        self,
        safetensors_paths: List[Path],
        tensor_to_file: Optional[Dict[str, Path]] = None,
    ) -> None:
        ordered_paths = self._order_shards_for_lm_head_last(safetensors_paths, tensor_to_file)
        if len(ordered_paths) > 1:
            print(f"Loading non-quantized weights from {len(ordered_paths)} safetensors shards...")
        for st_path in ordered_paths:
            self.model.load_non_quantized_weights_from_safetensors(str(st_path))

    def _prepare_presaved_weights(
        self,
        safetensors_paths: Union[Path, List[Path]],
        presaved_dir: Path,
        use_packed: bool,
        pad_packed: Union[bool, int, tuple[int, int]],
        model_name: Optional[str] = None,
        tensor_to_file: Optional[Dict[str, Path]] = None,
    ) -> None:
        if isinstance(safetensors_paths, (str, Path)):
            safetensors_files = [Path(safetensors_paths).expanduser().resolve()]
        else:
            safetensors_files = [Path(p).expanduser().resolve() for p in safetensors_paths]
        if not safetensors_files:
            raise ValueError("No safetensors files provided for pre-save preparation.")

        manifest_path = presaved_dir / "manifest.json"
        model_name = model_name or safetensors_files[0].stem
        hetero_mode = ""
        split_k_by_layer: Dict[str, int] = {}
        if use_packed:
            hetero_mode = str(self.config.get("heterogeneity", "")).lower() if isinstance(self.config, dict) else ""
            if hetero_mode in ("hetero", "gpu_split"):
                kernels = self.config.get("kernels_gemm") if isinstance(self.config, dict) else None
                if not kernels and isinstance(self.config, dict):
                    kernels = self.config.get("kernels")
                kernels = kernels or []
                for entry in kernels:
                    try:
                        layer = str(entry.get("layer", "")).strip()
                        forK = int(entry.get("forK"))
                        npuK = int(entry.get("npuK"))
                        if layer and forK > 0 and npuK > 0 and forK != npuK:
                            split_k_by_layer[layer] = npuK
                    except Exception:
                        continue

        fused_qkv_bins = bool(use_packed and getattr(self, "enable_fused_qkv_bins", False))

        expected_manifest = {
            "format_version": 2 if fused_qkv_bins else 1,
            "model_name": model_name,
            "use_packed_weights": use_packed,
            "pad_packed_weights": pad_packed,
            "pad_packed_weights_by_layer": {k: list(v) for k, v in sorted(self.packed_weight_pad_by_layer.items())},
            "heterogeneity": hetero_mode,
            "split_k_by_layer": split_k_by_layer if use_packed else {},
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "groupsize": self.groupsize,
        }
        if fused_qkv_bins:
            expected_manifest["qwen_fused_qkv_bins"] = True
        if not use_packed:
            expected_manifest["unpacked_layout"] = "out_groups_v2"

        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    existing = json.load(f)
                if all(existing.get(k) == v for k, v in expected_manifest.items()):
                    if self.debug_verbosity >= 1:
                        print(f"Using existing pre-saved weights in {presaved_dir}")
                    return
            except Exception:
                pass

        print(f"Preprocessing weights into bin files under {presaved_dir}...")

        layer_specs = [
            ("self_attn.q_proj", "q", self.hidden_size, self.num_attention_heads * self.head_dim),
            ("self_attn.k_proj", "k", self.hidden_size, self.num_key_value_heads * self.head_dim),
            ("self_attn.v_proj", "v", self.hidden_size, self.num_key_value_heads * self.head_dim),
            ("self_attn.o_proj", "o", self.num_attention_heads * self.head_dim, self.hidden_size),
            ("mlp.gate_proj", "gate", self.hidden_size, self.intermediate_size),
            ("mlp.up_proj", "up", self.hidden_size, self.intermediate_size),
            ("mlp.down_proj", "down", self.intermediate_size, self.hidden_size),
        ]

        # Fast-path: if base manifest matches and packed bins exist, only split K if needed.
        base_match = False
        if use_packed and manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    existing = json.load(f)
                base_keys = [
                    "format_version",
                    "model_name",
                    "use_packed_weights",
                    "pad_packed_weights",
                    "num_hidden_layers",
                    "hidden_size",
                    "intermediate_size",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "head_dim",
                    "groupsize",
                ]
                base_match = all(existing.get(k) == expected_manifest.get(k) for k in base_keys)
            except Exception:
                base_match = False

        if use_packed and split_k_by_layer and presaved_dir.exists() and base_match:
            split_ok = True
            for layer_idx in range(self.num_hidden_layers):
                for _suffix, short_name, in_feat, out_feat in layer_specs:
                    split_k = split_k_by_layer.get(short_name, 0)
                    layer_pad_spec = _resolve_layer_pad_packed_alignment(pad_packed, self.packed_weight_pad_by_layer, short_name)
                    if split_k > 0:
                        out0_path, out1_path = _split_packed_paths(presaved_dir, layer_idx, short_name, split_k)
                        if out0_path.exists() and out1_path.exists():
                            continue
                        packed_path = presaved_dir / f"layer_{layer_idx}_{short_name}.packed.bin"
                        ok = _split_packed_bin_from_file(
                            packed_path,
                            out0_path,
                            out1_path,
                            split_k,
                            in_feat,
                            out_feat,
                            layer_pad_spec,
                        )
                        if not ok:
                            split_ok = False
                            break
                    else:
                        packed_path = presaved_dir / f"layer_{layer_idx}_{short_name}.packed.bin"
                        if not packed_path.exists():
                            split_ok = False
                            break
                if split_ok and fused_qkv_bins:
                    qkv_path = presaved_dir / f"layer_{layer_idx}_qkv.packed.bin"
                    if not qkv_path.exists():
                        split_ok = False
                if not split_ok:
                    break
            if split_ok:
                presaved_dir.mkdir(parents=True, exist_ok=True)
                with open(manifest_path, "w") as f:
                    json.dump(expected_manifest, f, indent=2)
                print(f"Pre-saved weights written to {presaved_dir}")
                return

        tensor_to_file_resolved = _build_tensor_to_file_map(safetensors_files, tensor_to_file)

        for layer_idx in range(self.num_hidden_layers):
            attn_parts: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]] = {}
            for suffix, short_name, in_feat, out_feat in layer_specs:
                base_name = f"model.layers.{layer_idx}.{suffix}"
                state_slice = _load_state_slice_for_base_name(base_name, tensor_to_file_resolved)
                qweight, scales, qzeros, _g_idx, is_compressed = _get_quantized_tensors(state_slice, base_name)

                qweight = qweight.cpu()
                scales = scales.cpu()
                if qzeros is not None:
                    qzeros = qzeros.cpu()

                qweight_packed = None
                if is_compressed:
                    qweight = qweight.t().contiguous()
                    qweight_packed = _unpack_gptq_qweight(qweight.to(torch.int32))
                    qweight_unpacked = _unpack_packed_out_in2_to_in_out(qweight_packed)
                    scales = scales.to(torch.bfloat16)
                    if qzeros is None:
                        qzeros = torch.full_like(scales, 8, dtype=torch.int8)
                    zeros = qzeros.to(torch.int8)
                else:
                    is_awq = qweight.size(0) == in_feat and qweight.size(1) == out_feat // 8
                    is_gptq = qweight.size(0) == in_feat // 8 and qweight.size(1) == out_feat

                    if not is_awq and not is_gptq:
                        if qweight.size(0) > qweight.size(1):
                            is_awq = True
                        else:
                            is_gptq = True

                    if is_awq:
                        qweight_unpacked = _unpack_awq_qweight(qweight.to(torch.int32))
                        scales = scales.to(torch.bfloat16)
                        zeros = _unpack_awq_qzeros(qzeros.to(torch.int32))
                    else:
                        qweight_packed = _unpack_gptq_qweight(qweight.to(torch.int32))
                        qweight_unpacked = _unpack_packed_out_in2_to_in_out(qweight_packed)
                        scales = scales.to(torch.bfloat16).t().contiguous()
                        zeros = _unpack_gptq_qzeros(qzeros.to(torch.int32), out_feat, add_one=True)

                scales_pack = scales.to(torch.bfloat16).contiguous()
                zeros_pack = zeros.to(torch.int8).contiguous()
                if short_name in ("q", "k", "v"):
                    attn_parts[short_name] = (
                        qweight_unpacked.to(torch.int8).contiguous(),
                        scales_pack,
                        zeros_pack,
                        in_feat,
                        out_feat,
                    )

                if use_packed:
                    split_k = split_k_by_layer.get(short_name, 0)
                    layer_pad_spec = _resolve_layer_pad_packed_alignment(pad_packed, self.packed_weight_pad_by_layer, short_name)
                    if split_k > 0 and split_k < in_feat:
                        out0_path, out1_path = _split_packed_paths(presaved_dir, layer_idx, short_name, split_k)
                        if out0_path.exists() and out1_path.exists():
                            continue

                        group_size = self.groupsize if self.groupsize > 0 else 128
                        groups0 = (split_k + group_size - 1) // group_size

                        def _split_groups(t: torch.Tensor):
                            if t.dim() != 2:
                                return t, t
                            if t.size(0) == out_feat:
                                # [N, G]
                                g_total = t.size(1)
                                g0 = min(groups0, g_total)
                                return t[:, :g0], t[:, g0:]
                            if t.size(1) == out_feat:
                                # [G, N]
                                g_total = t.size(0)
                                g0 = min(groups0, g_total)
                                return t[:g0, :], t[g0:, :]
                            # Fallback: split last dim
                            g_total = t.size(-1)
                            g0 = min(groups0, g_total)
                            return t.narrow(-1, 0, g0), t.narrow(-1, g0, g_total - g0)

                        q0 = qweight_unpacked[:split_k, :]
                        q1 = qweight_unpacked[split_k:, :]
                        s0, s1 = _split_groups(scales_pack)
                        z0, z1 = _split_groups(zeros_pack)

                        packed0 = _pack_weights_packed(q0.to(torch.int8), s0, z0, split_k, out_feat, layer_pad_spec)
                        packed1 = _pack_weights_packed(q1.to(torch.int8), s1, z1, in_feat - split_k, out_feat, layer_pad_spec)
                        _write_tensor_raw(out0_path, packed0)
                        _write_tensor_raw(out1_path, packed1)
                    else:
                        packed = _pack_weights_packed(
                            qweight_unpacked.to(torch.int8),
                            scales_pack,
                            zeros_pack,
                            in_feat,
                            out_feat,
                            layer_pad_spec,
                        )
                        out_path = presaved_dir / f"layer_{layer_idx}_{short_name}.packed.bin"
                        _write_tensor_raw(out_path, packed)
                else:
                    if qweight_packed is None:
                        qweight_packed = _pack_qweight_out_in2(qweight_unpacked.to(torch.int8))

                    scales_out = _normalize_group_tensor(scales, out_feat)
                    zeros_out = _align_zeros_to_scales(zeros, scales_out, out_feat)

                    scales_out = scales_out.to(torch.bfloat16)
                    zeros_out = zeros_out.to(torch.int8)

                    _write_tensor_raw(presaved_dir / f"layer_{layer_idx}_{short_name}.qweight.bin", qweight_packed.to(torch.uint8))
                    _write_tensor_raw(presaved_dir / f"layer_{layer_idx}_{short_name}.scales.bin", scales_out)
                    _write_tensor_raw(presaved_dir / f"layer_{layer_idx}_{short_name}.zeros.bin", zeros_out)

                del state_slice
                del qweight
                del scales
                if qzeros is not None:
                    del qzeros

            if fused_qkv_bins and all(k in attn_parts for k in ("q", "k", "v")):
                q_qw, q_s, q_z, q_in, q_out = attn_parts["q"]
                k_qw, k_s, k_z, k_in, k_out = attn_parts["k"]
                v_qw, v_s, v_z, v_in, v_out = attn_parts["v"]

                if not (q_in == k_in == v_in):
                    raise RuntimeError(f"Inconsistent Qwen qkv in_features on layer {layer_idx}: q={q_in}, k={k_in}, v={v_in}")

                qkv_qw = torch.cat([q_qw, k_qw, v_qw], dim=1).contiguous()

                def _concat_qkv_group_tensors(
                    q_t: torch.Tensor,
                    k_t: torch.Tensor,
                    v_t: torch.Tensor,
                    q_out_f: int,
                    k_out_f: int,
                    v_out_f: int,
                    tensor_name: str,
                ) -> torch.Tensor:
                    if q_t.dim() == 2 and k_t.dim() == 2 and v_t.dim() == 2:
                        if (
                            q_t.size(0) == q_out_f
                            and k_t.size(0) == k_out_f
                            and v_t.size(0) == v_out_f
                            and q_t.size(1) == k_t.size(1) == v_t.size(1)
                        ):
                            return torch.cat([q_t, k_t, v_t], dim=0).contiguous()
                        if (
                            q_t.size(1) == q_out_f
                            and k_t.size(1) == k_out_f
                            and v_t.size(1) == v_out_f
                            and q_t.size(0) == k_t.size(0) == v_t.size(0)
                        ):
                            return torch.cat([q_t, k_t, v_t], dim=1).contiguous()
                    raise RuntimeError(
                        f"Unsupported {tensor_name} layout for fused qkv at layer {layer_idx}: "
                        f"q={tuple(q_t.shape)} k={tuple(k_t.shape)} v={tuple(v_t.shape)}"
                    )

                qkv_s = _concat_qkv_group_tensors(q_s, k_s, v_s, q_out, k_out, v_out, "scales")
                qkv_z = _concat_qkv_group_tensors(q_z, k_z, v_z, q_out, k_out, v_out, "zeros")
                qkv_out = q_out + k_out + v_out

                qkv_pad_spec = _resolve_layer_pad_packed_alignment(pad_packed, self.packed_weight_pad_by_layer, "qkv")
                qkv_packed = _pack_weights_packed(qkv_qw, qkv_s, qkv_z, q_in, qkv_out, qkv_pad_spec)
                _write_tensor_raw(presaved_dir / f"layer_{layer_idx}_qkv.packed.bin", qkv_packed)

            if layer_idx % 2 == 1:
                gc.collect()

        presaved_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(expected_manifest, f, indent=2)
        print(f"Pre-saved weights written to {presaved_dir}")
    
    def tokenize(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Tokenize input text."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        if isinstance(text, str):
            text = [text]
        
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len
        )
        
        return encoded["input_ids"].to(self.device)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: int = 50,
        start_pos: int = 0
    ) -> torch.Tensor:
        """Generate tokens from input using C++ backend."""
        eos_token_id = -1
        if self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
            eos_token_id = self.tokenizer.eos_token_id
        
        return self.model.generate(
            input_ids,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            eos_token_id
        )
    
    def __call__(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Forward pass."""
        if isinstance(input_ids, str):
            input_ids = self.tokenize(input_ids)
        
        return self.model.forward(input_ids, start_pos)


def run_prompt_test(target_tokens, model_path=None, tokenizer_path=None, device="cuda", backend="base",
                    max_new_tokens=512, temperature=0.7, top_p=0.9, top_k=50, generate=True,
                    config_path=None, measure_power=False):
    """Run prompt test case."""
    from pathlib import Path
    
    script_dir = Path(__file__).parent
    prompts_file = script_dir.parent / "prompts.txt"
    
    if not prompts_file.exists():
        print(f"Error: Prompts file not found at {prompts_file}")
        return 1
    
    print("=" * 60)
    print(f"PROMPT TEST: Target token count = {target_tokens}")
    print("=" * 60 + "\n")

    if int(target_tokens) > DEFAULT_QWEN25_14B_MAX_PROMPT_LEN:
        print(
            f"Error: prompt-test target_tokens={target_tokens} exceeds "
            f"Qwen25_14B maximum prompt length={DEFAULT_QWEN25_14B_MAX_PROMPT_LEN}."
        )
        return 1
    
    # Read the single long prompt from file
    print(f"Reading long prompt from: {prompts_file}")
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts_content = f.read()
    
    # Extract the prompt (remove <|begin_of_text|> marker)
    long_prompt = prompts_content.replace("<|begin_of_text|>", "").strip()
    
    print("Initializing tokenizer...")
    try:
        from transformers import AutoTokenizer
        if tokenizer_path is None:
            tokenizer_path = "Qwen/Qwen2.5-14B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer initialized successfully!\n")
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        return 1
    
    # Base prompt to prepend
    base_prompt = "Please provide a comprehensive summary of the following document.\n\nDocument:\n"
    
    # Tokenize base prompt
    base_prompt_tokens = tokenizer.encode(base_prompt, add_special_tokens=False)
    
    # Tokenize the long prompt
    long_prompt_tokens = tokenizer.encode(long_prompt, add_special_tokens=False)
    
    # Concatenate
    full_prompt_tokens = base_prompt_tokens + long_prompt_tokens
    full_token_count = len(full_prompt_tokens)
    
    # Adjust to target token count
    if full_token_count > target_tokens:
        truncated_tokens = full_prompt_tokens[:target_tokens]
    elif full_token_count < target_tokens:
        tokens_needed = target_tokens - full_token_count
        long_prompt_token_count = len(long_prompt_tokens)
        if long_prompt_token_count > 0:
            additional_repeats = (tokens_needed + long_prompt_token_count - 1) // long_prompt_token_count
            if additional_repeats == 0: additional_repeats = 1
            extended_tokens = full_prompt_tokens + (long_prompt_tokens * additional_repeats)
            truncated_tokens = extended_tokens[:target_tokens]
        else:
            truncated_tokens = full_prompt_tokens
    else:
        truncated_tokens = full_prompt_tokens
    
    actual_token_count = len(truncated_tokens)
    print(f"Actual tokens: {actual_token_count}")
    generation_capacity = max(1, int(max_new_tokens)) if generate else 0
    effective_max_seq_len = max(
        DEFAULT_QWEN25_14B_MAX_SEQ_LEN,
        actual_token_count + generation_capacity,
    )
    print(
        f"Backend max_seq_len={effective_max_seq_len} from prompt={actual_token_count}, "
        f"requested max_new_tokens={max_new_tokens if generate else 0}, "
        f"and generation_capacity={generation_capacity}"
    )
    
    # Initialize model and generate
    print("\n" + "=" * 60)
    print("INITIALIZING MODEL:")
    print("=" * 60 + "\n")
    
    if model_path is None:
        model_path = "Qwen/Qwen2.5-14B-Instruct-AWQ"

    if backend == "hetero":
        config_path = _resolve_qwen25_14b_config_path(config_path)
        selected_prompt_len = select_prompt_test_prompt_len(
            config_path,
            int(target_tokens),
        )
        set_prompt_len_in_config(
            config_path,
            selected_prompt_len,
        )
    
    print("Initializing Qwen2.5-14B w4a16 quantized model...")
    try:
        model = Qwen25_14BW4A16Model(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            max_seq_len=effective_max_seq_len,
            device=device,
            backend=backend,
            config_path=config_path
        )
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Generate output
    print("\n" + "=" * 60)
    if generate:
        print("GENERATING OUTPUT:")
    else:
        print("FORWARD PASS (NO GENERATION):")
    print("=" * 60 + "\n")
    
    try:
        input_ids = torch.tensor([truncated_tokens], dtype=torch.long, device=device)
        
        if not generate:
            # Just run forward pass to test prefill/prompt processing
            print("Running forward pass only...")
            import time
            start_time = time.time()
            logits = model(input_ids)
            end_time = time.time()
            print(f"Forward pass completed in {end_time - start_time:.4f} seconds")
            print(f"Logits shape: {logits.shape}")
            return 0

        if measure_power:
            generated = measure_generation_power(
                model.generate,
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        else:
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        
        if model.tokenizer is not None:
            prompt_len = input_ids.size(1)
            generated_tokens = generated[0, prompt_len:].tolist()
            decoded_generated = model.tokenizer.decode(generated_tokens, skip_special_tokens=False)
            
            print(f"\n{'='*60}")
            print(f"GENERATED TEXT:")
            print(f"{'='*60}")
            print(decoded_generated)
            print(f"{'='*60}")
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
    print(f"{'=' * 60}\n")
    
    return 0


def main():
    """Example usage of Qwen25_14BW4A16Model when run as a script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen2.5-14B W4A16 Quantized Model - Unified LibTorch Backend")
    parser.add_argument(
        "--text",
        type=str,
        default="What is the meaning of life the universe and everything?",
        help="Input text to process (default: 'What is the meaning of life?')"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer or HuggingFace model name"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct-AWQ",
        help="Path to AWQ quantized model (default: Qwen/Qwen2.5-14B-Instruct-AWQ)"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to NPU config JSON (default: auto-detect from lscpu for hetero backend)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="hetero",
        choices=["base", "hetero"],
        help="Backend to use (default: base)"
    )
    parser.add_argument(
        "--generate",
        dest="generate",
        action="store_true",
        default=True,
        help="Generate text instead of just getting logits (default: True)"
    )
    parser.add_argument(
        "--no-generate",
        dest="generate",
        action="store_false",
        help="Disable generation, just get logits"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum number of tokens to generate (if --generate is used, default: 16)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy decoding, always picks most likely token; higher values increase randomness and diversity)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter (0.0-1.0): considers tokens with cumulative probability up to this value. Lower values = more focused, higher = more diverse"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter: only considers the top k most likely tokens. Lower values = more focused, higher = more diverse. Set to 0 to disable"
    )
    parser.add_argument(
        "--prompt-test",
        type=int,
        default=None,
        help="Run prompt test case with specified token count. Extracts prompt from prompts.txt, truncates to this length, and tests tokenizer chunking. Can be any positive integer."
    )
    parser.add_argument(
        "--measure-power",
        action="store_true",
        default=False,
        help="Enable power measurement during generation (default: off)."
    )
    
    args = parser.parse_args()
    
    # Handle prompt test case
    if args.prompt_test is not None:
        return run_prompt_test(
            args.prompt_test,
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            backend=args.backend,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            generate=args.generate,
            config_path=args.config_path,
            measure_power=args.measure_power
        )

    if args.backend == "hetero":
        args.config_path = _resolve_qwen25_14b_config_path(args.config_path)
        set_prompt_len_in_config(
            args.config_path,
            1,
        )
    
    print("=" * 60)
    print("Initializing Qwen2.5-14B W4A16 Quantized Model...")
    print("=" * 60)
    
    try:
        model = Qwen25_14BW4A16Model(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            backend=args.backend,
            config_path=args.config_path
        )

        print("Model initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Make sure you have:")
        print("  1. Built the C++ backend (run 'make' in build directory)")
        print("  2. The unified_llm_w4a16_libtorch module is in your Python path")
        print("  3. Model weights are loaded (if required)")
        return 1
    
    print(f"Processing text: '{args.text}'")
    
    if args.generate:
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
                    top_k=args.top_k
                )
            else:
                generated = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k
                )
            
            if model.tokenizer is not None:
                # Decode the full sequence (prompt + generated)
                decoded_full = model.tokenizer.decode(generated[0].tolist(), skip_special_tokens=False)
                
                # Decode only the generated part (excluding prompt)
                prompt_len = input_ids.size(1)
                generated_tokens = generated[0, prompt_len:].tolist()
                decoded_generated = model.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                
                print(f"\n{'='*60}")
                print(f"Full output (prompt + generated):")
                print(f"{'='*60}")
                print(decoded_full)
                print(f"{'='*60}")
                print(f"Generated text only:")
                print(f"{'='*60}")
                print(decoded_generated)
                print(f"{'='*60}")
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
            logits = model(args.text)
            print(f"Logits shape: {logits.shape}")
            print(f"Logits dtype: {logits.dtype}")
            print(f"Logits device: {logits.device}")
            
            print(f"\nLogits statistics:")
            print(f"  Min: {logits.min().item():.4f}")
            print(f"  Max: {logits.max().item():.4f}")
            print(f"  Mean: {logits.mean().item():.4f}")
            print(f"  Std: {logits.std().item():.4f}")
        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    print("\nDone!")
    return 0

if __name__ == "__main__":
    exit(main())
