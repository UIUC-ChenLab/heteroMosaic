"""
LLaMA3.1-8B-Instruct w4a16 Quantized Python frontend.
Handles tokenization and interfaces with the C++ backend.
"""

import torch
from typing import Optional, List, Union, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
import json
import torch.nn.functional as F
from pathlib import Path

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

# The hetero backend is imported dynamically in __init__.
ArchitectureType = None

import subprocess
import time
import re
import os
import math
from urllib.request import urlopen

DEFAULT_HETERO_CONFIG = {
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
    "gpu_chunking": {
        "gpu_chunk_size": 2048,
        "gpu_chunking_inflight": 1,
    },
    "kernels_gemm_chunked": [],
    "kernels_gemm": [],
    "kernels_gemv": [],
    "npuOnlydefault": [
        {
            "qo": [4096, 4096, -1],
            "kv": [4096, 1024, -1],
            "upgate": [4096, 14336, -1],
            "down": [14336, 4096, -1],
            "fw_path": "hw_bins/npu2/",
            "max_ctx_len": 8192,
            "num_tiles": 32,
            "tile_size": "64x128x64",
            "col": "8c",
            "dtype": "bf16_int4AWQ_bf16",
        }
    ],
}


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


def _ensure_default_config_file(config_path: Union[str, Path]) -> Path:
    path = Path(config_path).expanduser().resolve()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(DEFAULT_HETERO_CONFIG, f, indent=4)
        f.write("\n")
    print(f"Created missing config file with default values: {path}")
    return path


def _resolve_llama3_config_path(config_path: Optional[str]) -> str:
    if config_path:
        resolved = _ensure_default_config_file(config_path)
        return str(resolved)

    configs_dir = Path(__file__).parent.resolve() / "configs"
    cpu_model_name = _detect_lscpu_model_name()
    normalized = _normalize_cpu_model_name(cpu_model_name) if cpu_model_name else ""

    filename = "configs_strixP_llama3_8b.json5"
    mapping = [
        (("RYZEN AI 7 350", "RADEON 860M"), "configs_krackanP_llama3_8b.json5"),
        (("RYZEN AI 9 HX 370", "RADEON 890M"), "configs_strixP_llama3_8b.json5"),
        (("RYZEN AI MAX+ 395", "RADEON 8060S"), "configs_strixH_llama3_8b.json5"),
    ]
    for tokens, candidate in mapping:
        if all(token in normalized for token in tokens):
            filename = candidate
            break

    selected = _ensure_default_config_file(configs_dir / filename)
    if cpu_model_name:
        print(f"Auto-selected config based on lscpu model '{cpu_model_name}': {selected}")
    else:
        print(f"Auto-selected fallback config: {selected}")
    return str(selected)


def _save_used_prompt(prompt_text: str, output_path: Optional[Union[str, Path]] = None) -> Path:
    """Persist the prompt text used for a run."""
    path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else (Path(__file__).parent.resolve() / "used_prompt.txt")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt_text, encoding="utf-8")
    print(f"Saved used prompt to: {path}")
    return path


def _set_prompt_len_in_config(config_path: str, prompt_len: int) -> None:
    """Update prompt_len in the selected config file while preserving formatting/comments when possible."""
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
        # Backward-compat rewrite: migrate legacy npu_dim key to prompt_len.
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


def _select_prompt_test_prompt_len(config_path: str, target_tokens: int) -> int:
    """Choose prompt_len for prompt-test.

    prompt_len must stay equal to the actual prompt-test token count so grouped
    chunked kernel configs can be selected by prompt_len (e.g., 4096).
    """
    selected_prompt_len = int(target_tokens)
    try:
        cfg = load_config_with_comments(config_path)
        chunk_size = 0
        chunking_enabled = False
        if isinstance(cfg, dict):
            chunking_raw = cfg.get("chunking", False)
            chunking_enabled = bool(chunking_raw)

            # New schema: chunking is boolean and chunk size lives in gpu_chunking.
            gpu_chunk_cfg = cfg.get("gpu_chunking")
            if isinstance(gpu_chunk_cfg, dict):
                try:
                    chunk_size = int(gpu_chunk_cfg.get("gpu_chunk_size", gpu_chunk_cfg.get("chunk_size", 0)))
                except Exception:
                    chunk_size = 0

            # Legacy fallback: chunking as integer token size.
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
            os.killpg(os.getpgid(monitor_process.pid), signal.SIGKILL)
            monitor_process.wait()
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
    pad_packed: bool,
) -> bool:
    if not packed_path.exists():
        return False

    # Compute padded sizes to match packing layout
    K_npu = in_feat
    N_npu = out_feat
    if pad_packed:
        pad_align = 2048
        if N_npu % pad_align != 0:
            N_npu = ((N_npu + pad_align - 1) // pad_align) * pad_align
        if K_npu % pad_align != 0:
            K_npu = ((K_npu + pad_align - 1) // pad_align) * pad_align

    K0 = int(split_k)
    K1 = int(in_feat) - K0
    K0_npu = K0
    K1_npu = K1
    if pad_packed:
        pad_align = 2048
        if K0_npu % pad_align != 0:
            K0_npu = ((K0_npu + pad_align - 1) // pad_align) * pad_align
        if K1_npu % pad_align != 0:
            K1_npu = ((K1_npu + pad_align - 1) // pad_align) * pad_align
        if K0_npu + K1_npu != K_npu:
            return False

    if K0_npu % 128 != 0 or K1_npu % 128 != 0 or N_npu % 64 != 0:
        return False

    num_tiles_row = (K_npu + 127) // 128
    num_tiles_col = (N_npu + 63) // 64
    num_tiles = num_tiles_row * num_tiles_col
    expected_size = num_tiles * 4352

    import numpy as np

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


def _pack_weights_packed(qw: torch.Tensor, s: torch.Tensor, z: torch.Tensor, K_in: int, N_in: int, pad_packed: bool) -> torch.Tensor:
    LARGE_TILE_SIZE_ROW = 128
    LARGE_TILE_SIZE_COL = 64

    K_npu = K_in
    N_npu = N_in

    qw_p = qw
    s_p = s
    z_p = z

    if pad_packed:
        pad_align_N = 2048
        pad_align_K = 2048

        if N_in % pad_align_N != 0:
            padded_N = ((N_in + pad_align_N - 1) // pad_align_N) * pad_align_N
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

        if K_in % pad_align_K != 0:
            padded_K = ((K_in + pad_align_K - 1) // pad_align_K) * pad_align_K
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

class LLaMA3W4A16Model:
    """LLaMA3.1-8B-Instruct w4a16 quantized model wrapper."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        vocab_size: int = 128256,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 500000.0,
        max_seq_len: int = 16512,
        max_batch_size: int = 1,
        groupsize: int = 128,
        device: str = "cuda",
        backend: str = "hetero",
        config_path: Optional[str] = None
    ):


        """
        Initialize LLaMA3.1-8B-Instruct w4a16 quantized model.
        
        Args:
            model_path: Path to quantized model (e.g., "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
            tokenizer_path: Path to tokenizer or model name from HuggingFace
            vocab_size: Vocabulary size (default: 128256 for LLaMA3.1)
            hidden_size: Hidden dimension (default: 4096 for LLaMA3.1-8B)
            intermediate_size: FFN intermediate size (default: 14336)
            num_hidden_layers: Number of transformer layers (default: 32)
            num_attention_heads: Number of attention heads (default: 32)
            num_key_value_heads: Number of KV heads for GQA (default: 8)
            head_dim: Dimension per head (default: 128)
            rms_norm_eps: RMSNorm epsilon (default: 1e-5)
            rope_theta: RoPE theta parameter (default: 500000.0)
            max_seq_len: Maximum sequence length (default: 8192)
            max_batch_size: Maximum batch size (default: 1)
            groupsize: Quantization group size (default: 128)
            device: Device to run on ("cpu" or "cuda")
            device: Device to run on ("cpu" or "cuda")
            backend: Backend to use ("hetero")
            config_path: Path to NPU config JSON
        """


        self.device = device
        self.head_dim = head_dim
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.groupsize = groupsize
        
        # Initialize tokenizer
        if tokenizer_path is None:
            tokenizer_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Warning: Could not load tokenizer from {tokenizer_path}: {e}")
            self.tokenizer = None
        
        if backend != "hetero":
            raise ValueError(f"Unknown backend: {backend}")
        try:
            import unified_llm_w4a16_hetero_libtorch
            self.backend_module = unified_llm_w4a16_hetero_libtorch
        except ImportError as e:
            raise ImportError(f"Could not import hetero backend: {e}")

        # Get ArchitectureType from the selected backend
        ArchitectureType = self.backend_module.ArchitectureType

        # Prepare constructor arguments
        constructor_args = [
            ArchitectureType.LLAMA3,
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
            config_path = _resolve_llama3_config_path(config_path)
            constructor_args.append(config_path)

        self.model = self.backend_module.UnifiedLLMW4A16(*constructor_args)
        
        # Check dummy weights flag (regardless of backend)
        self.config = {}
        self.use_packed_weights = False
        self.use_pre_saved_weights = False
        self.pad_packed_weights = False
        self.debug_verbosity = 0

        use_dummy = False
        if config_path:
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
            from safetensors.torch import load_file, save_file
            import glob
            import shutil
            
            # Create weights folder if it doesn't exist
            script_dir = Path(__file__).parent
            weights_dir = script_dir / weights_folder
            os.makedirs(weights_dir, exist_ok=True)
            
            # Extract model name for filename
            model_name = _model_name_from_path(model_path)
            saved_safetensors = weights_dir / f"{model_name}.safetensors"
            
            # Check if we already have the safetensors saved locally
            if saved_safetensors.exists():
                print(f"Using saved safetensors: {saved_safetensors}")
                safetensors_files = [str(saved_safetensors)]
            else:
                # Download model if needed
                if not os.path.exists(model_path):
                    print(f"Model path {model_path} not found locally, downloading from Hub...")
                    try:
                        model_path = snapshot_download(repo_id=model_path)
                        print(f"Model downloaded to {model_path}")
                    except Exception as e:
                        print(f"Error downloading model: {e}")
                        pass
                
                # Find safetensors files
                safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
                if not safetensors_files:
                    bin_files = glob.glob(os.path.join(model_path, "*.bin"))
                    if not bin_files:
                        raise FileNotFoundError(f"No weight files found in {model_path}")
                    print("Warning: Found .bin files but not .safetensors. This loader is optimized for safetensors.")
                    raise NotImplementedError("Only safetensors are supported for this direct loading method.")
                
                # If multiple safetensors files, merge them or use the main one
                if len(safetensors_files) == 1:
                    # Copy the safetensors file to model_weights folder
                    print(f"Saving safetensors to {saved_safetensors}...")
                    shutil.copy2(safetensors_files[0], saved_safetensors)
                    print(f"Saved safetensors file: {saved_safetensors}")
                    print(f"File size: {os.path.getsize(saved_safetensors) / (1024**3):.2f} GB")
                else:
                    # Multiple files - merge them into one
                    print(f"Found {len(safetensors_files)} safetensors files, merging...")
                    merged_state_dict = {}
                    for st_file in safetensors_files:
                        print(f"  Loading {st_file}...")
                        merged_state_dict.update(load_file(st_file))
                    
                    print(f"Saving merged safetensors to {saved_safetensors}...")
                    save_file(merged_state_dict, str(saved_safetensors))
                    print(f"Saved merged safetensors file: {saved_safetensors}")
                    print(f"File size: {os.path.getsize(saved_safetensors) / (1024**3):.2f} GB")
                    safetensors_files = [str(saved_safetensors)]
            use_packed = self.use_packed_weights
            use_presaved = self.use_pre_saved_weights
            pad_packed = self.pad_packed_weights

            if use_presaved:
                presaved_dir = weights_dir / f"{model_name}_{'packed' if use_packed else 'unpacked'}"
                self._prepare_presaved_weights(saved_safetensors, presaved_dir, use_packed, pad_packed)

                t0 = time.time()
                if hasattr(self.model, "load_non_quantized_weights_from_safetensors"):
                    self.model.load_non_quantized_weights_from_safetensors(str(saved_safetensors))
                else:
                    print("Warning: Backend missing load_non_quantized_weights_from_safetensors; falling back to full safetensors load.")
                    self.model.load_quantized_weights_from_safetensors(str(saved_safetensors))
                    t1 = time.time()
                    self.load_time = t1 - t0
                    print(f"Weights loaded in {self.load_time:.2f} seconds (fallback)")
                    return

                if hasattr(self.model, "load_quantized_weights_from_bins"):
                    self.model.load_quantized_weights_from_bins(str(presaved_dir))
                else:
                    print("Warning: Backend missing load_quantized_weights_from_bins; falling back to full safetensors load.")
                    self.model.load_quantized_weights_from_safetensors(str(saved_safetensors))
                    t1 = time.time()
                    self.load_time = t1 - t0
                    print(f"Weights loaded in {self.load_time:.2f} seconds (fallback)")
                    return
                t1 = time.time()
                self.load_time = t1 - t0
                print(f"Weights loaded from pre-saved bins in {self.load_time:.2f} seconds")
            else:
                # Load into C++ backend from safetensors file
                t0 = time.time()
                self.model.load_quantized_weights_from_safetensors(str(saved_safetensors))
                t1 = time.time()
                self.load_time = t1 - t0
                print(f"Weights loaded in {self.load_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error loading quantized weights: {e}")
            import traceback
            traceback.print_exc()
            print("\nNote: Falling back to randomly initialized weights.")
            print("The model will not produce meaningful output without proper weights.")

    def _prepare_presaved_weights(self, saved_safetensors: Path, presaved_dir: Path, use_packed: bool, pad_packed: bool) -> None:
        manifest_path = presaved_dir / "manifest.json"
        model_name = saved_safetensors.stem
        hetero_mode = ""
        split_k_by_layer: Dict[str, int] = {}
        if use_packed:
            hetero_mode = str(self.config.get("heterogeneity", "")).lower() if isinstance(self.config, dict) else ""
            if hetero_mode in ("hetero", "gpu_split") and isinstance(self.config, dict):
                gemv_driven_split_k = bool(self.config.get("gemv_driven_split_K", False))

                def _collect_split_k(entries) -> None:
                    for entry in (entries or []):
                        try:
                            layer = str(entry.get("layer", "")).strip()
                            forK = int(entry.get("forK"))
                            npuK = int(entry.get("npuK"))
                            if layer and forK > 0 and npuK > 0 and npuK < forK:
                                split_k_by_layer[layer] = npuK
                        except Exception:
                            continue

                # Always seed from GEMM configs (or legacy kernels[]), then override from GEMV
                # when gemv_driven_split_K is enabled.
                kernels_gemm = self.config.get("kernels_gemm")
                if not kernels_gemm:
                    kernels_gemm = self.config.get("kernels")
                _collect_split_k(kernels_gemm)

                if gemv_driven_split_k:
                    _collect_split_k(self.config.get("kernels_gemv"))

        expected_manifest = {
            "format_version": 1,
            "model_name": model_name,
            "use_packed_weights": use_packed,
            "pad_packed_weights": pad_packed,
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
        from safetensors.torch import load_file

        state_dict = load_file(str(saved_safetensors))

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
                            pad_packed,
                        )
                        if not ok:
                            split_ok = False
                            break
                    else:
                        packed_path = presaved_dir / f"layer_{layer_idx}_{short_name}.packed.bin"
                        if not packed_path.exists():
                            split_ok = False
                            break
                if not split_ok:
                    break
            if split_ok:
                presaved_dir.mkdir(parents=True, exist_ok=True)
                with open(manifest_path, "w") as f:
                    json.dump(expected_manifest, f, indent=2)
                print(f"Pre-saved weights written to {presaved_dir}")
                return

        for layer_idx in range(self.num_hidden_layers):
            for suffix, short_name, in_feat, out_feat in layer_specs:
                base_name = f"model.layers.{layer_idx}.{suffix}"
                qweight, scales, qzeros, _g_idx, is_compressed = _get_quantized_tensors(state_dict, base_name)

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

                if use_packed:
                    split_k = split_k_by_layer.get(short_name, 0)
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
                        s0, s1 = _split_groups(scales.to(torch.bfloat16))
                        z0, z1 = _split_groups(zeros.to(torch.int8))

                        packed0 = _pack_weights_packed(q0.to(torch.int8), s0, z0, split_k, out_feat, pad_packed)
                        packed1 = _pack_weights_packed(q1.to(torch.int8), s1, z1, in_feat - split_k, out_feat, pad_packed)
                        _write_tensor_raw(out0_path, packed0)
                        _write_tensor_raw(out1_path, packed1)
                    else:
                        packed = _pack_weights_packed(
                            qweight_unpacked.to(torch.int8),
                            scales.to(torch.bfloat16),
                            zeros.to(torch.int8),
                            in_feat,
                            out_feat,
                            pad_packed,
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
            max_length=8192
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

    def perplexity(self, input_ids: Union[str, torch.Tensor]) -> dict:
        """
        Compute causal-LM perplexity for the provided sequence(s).
        Returns: loss, perplexity, num_tokens.
        """
        if isinstance(input_ids, str):
            input_ids = self.tokenize(input_ids)

        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids with shape [batch, seq_len], got {tuple(input_ids.shape)}")
        if input_ids.size(1) < 2:
            raise ValueError("Need at least 2 tokens to compute perplexity.")

        with torch.no_grad():
            logits = self.model.forward(input_ids, 0)

        shift_logits = logits[:, :-1, :].float().contiguous()
        shift_labels = input_ids[:, 1:].to(shift_logits.device).contiguous()
        vocab_size = shift_logits.size(-1)

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer is not None else None
        if pad_token_id is not None:
            valid_mask = shift_labels.ne(pad_token_id)
            num_tokens = int(valid_mask.sum().item())
            if num_tokens == 0:
                raise ValueError("No non-pad tokens available for perplexity computation.")
            labels_for_loss = shift_labels.masked_fill(~valid_mask, -100)
            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                labels_for_loss.view(-1),
                ignore_index=-100,
                reduction="mean",
            )
        else:
            num_tokens = int(shift_labels.numel())
            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                reduction="mean",
            )

        ppl = torch.exp(loss)
        return {
            "loss": float(loss.item()),
            "perplexity": float(ppl.item()),
            "num_tokens": num_tokens,
        }


def run_prompt_test(target_tokens, model_path=None, tokenizer_path=None, device="cuda", backend="hetero",
                    max_new_tokens=512, temperature=0.7, top_p=0.9, top_k=50, generate=True, config_path=None,
                    perplexity=False, measure_power=False, save_used_prompt=False):
    """
    Run prompt test case: load single long prompt from prompts.txt,
    concatenate base prompt, truncate to requested token count, and generate output.
    
    Args:
        target_tokens: Target token count (can be any positive integer)
        model_path: Path to neuralmagic quantized model
        tokenizer_path: Path to tokenizer
        device: Device to run on ("cpu" or "cuda")
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
    
    Returns:
        0 on success, 1 on error
    """
    from pathlib import Path
    
    script_dir = Path(__file__).parent
    prompts_file = script_dir.parent / "prompts.txt"
    
    if not prompts_file.exists():
        print(f"Error: Prompts file not found at {prompts_file}")
        return 1
    
    print("=" * 60)
    print(f"PROMPT TEST: Target token count = {target_tokens}")
    print("=" * 60 + "\n")
    
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
            tokenizer_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer initialized successfully!\n")
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        return 1
    
    # Base prompt to prepend
    base_prompt = """Please provide a comprehensive summary of the following document. The summary should capture the main points, key developments, and important themes discussed in the text."""
    
    # Tokenize base prompt
    base_prompt_tokens = tokenizer.encode(base_prompt, add_special_tokens=False)
    base_prompt_token_count = len(base_prompt_tokens)
    
    # Tokenize the long prompt (document + Summary:)
    long_prompt_tokens = tokenizer.encode(long_prompt, add_special_tokens=False)
    long_prompt_token_count = len(long_prompt_tokens)
    
    print(f"Token counts:")
    print(f"  Base prompt: {base_prompt_token_count} tokens")
    print(f"  Long prompt (doc + Summary:): {long_prompt_token_count} tokens")
    print(f"  Combined (before truncation): {base_prompt_token_count + long_prompt_token_count} tokens")
    print(f"  Target: {target_tokens} tokens\n")
    
    # Concatenate base prompt + long prompt tokens directly
    full_prompt_tokens = base_prompt_tokens + long_prompt_tokens
    full_token_count = len(full_prompt_tokens)
    
    # Always adjust to target token count
    if full_token_count > target_tokens:
        truncated_tokens = full_prompt_tokens[:target_tokens]
        actual_token_count = len(truncated_tokens)
        print(f"Truncated from {full_token_count} to {actual_token_count} tokens")
    elif full_token_count < target_tokens:
        tokens_needed = target_tokens - full_token_count
        long_prompt_token_count = len(long_prompt_tokens)
        
        if long_prompt_token_count > 0:
            additional_repeats = (tokens_needed + long_prompt_token_count - 1) // long_prompt_token_count
            if additional_repeats == 0:
                additional_repeats = 1
            
            extended_tokens = full_prompt_tokens + (long_prompt_tokens * additional_repeats)
            truncated_tokens = extended_tokens[:target_tokens]
            actual_token_count = len(truncated_tokens)
            print(f"Extended from {full_token_count} to {actual_token_count} tokens (target: {target_tokens}, added {additional_repeats} more document copies)")
        else:
            truncated_tokens = full_prompt_tokens
            actual_token_count = full_token_count
            print(f"Warning: Cannot extend prompt (long_prompt is empty). Using {actual_token_count} tokens")
    else:
        truncated_tokens = full_prompt_tokens
        actual_token_count = full_token_count
        print(f"Prompt is exactly {actual_token_count} tokens (target: {target_tokens})")

    if save_used_prompt:
        # Save the exact prompt text used after token-level truncation/extension.
        used_prompt_text = tokenizer.decode(truncated_tokens, skip_special_tokens=False)
        _save_used_prompt(used_prompt_text)
    
    # Initialize model and generate
    print("\n" + "=" * 60)
    print("INITIALIZING MODEL:")
    print("=" * 60 + "\n")
    
    if model_path is None:
        model_path = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

    if backend == "hetero":
        config_path = _resolve_llama3_config_path(config_path)
        selected_prompt_len = _select_prompt_test_prompt_len(config_path, int(target_tokens))
        _set_prompt_len_in_config(config_path, selected_prompt_len)
    
    print("Initializing LLaMA3.1-8B-Instruct w4a16 quantized model...")
    try:
        model = LLaMA3W4A16Model(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
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
            # Just run forward pass to test prefill/prompt processing
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
    if hasattr(model, 'load_time'):
        print(f"Weight loading time: {model.load_time:.2f} seconds")
    print(f"{'=' * 60}\n")
    
    return 0


def _load_wikitext2_raw_text(model_weights_dir: Path, split: str = "test") -> str:
    """
    Load WikiText-2 raw split and cache the plain text under model_weights_dir.
    Tries Hugging Face datasets first, then falls back to raw text URL.
    """
    model_weights_dir.mkdir(parents=True, exist_ok=True)
    text_cache_path = model_weights_dir / f"wikitext-2-raw-v1_{split}.txt"

    if text_cache_path.exists():
        print(f"Using cached WikiText-2 text: {text_cache_path}")
        return text_cache_path.read_text(encoding="utf-8")

    text = None
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        lines = [line for line in ds["text"] if line and line.strip()]
        text = "\n\n".join(lines)
        print(f"Downloaded WikiText-2 via datasets ({split} split).")
    except Exception as e:
        print(f"Could not load WikiText-2 via datasets ({e}). Falling back to raw text URL.")
        fallback_urls = {
            "train": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
            "validation": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
            "valid": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
            "test": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
        }
        if split not in fallback_urls:
            raise ValueError(f"Unsupported WikiText-2 split '{split}'. Use one of train/valid/validation/test.")
        with urlopen(fallback_urls[split]) as resp:
            text = resp.read().decode("utf-8")
        text = "\n\n".join([line for line in text.splitlines() if line.strip()])
        print(f"Downloaded WikiText-2 from fallback URL ({split} split).")

    text_cache_path.write_text(text, encoding="utf-8")
    print(f"Saved WikiText-2 text cache: {text_cache_path}")
    return text


def run_wikitext2_perplexity(
    model_path=None,
    tokenizer_path=None,
    device="cuda",
    backend="hetero",
    config_path=None,
    split: str = "test",
    max_length: int = 2048,
    stride: int = 2048,
):
    """
    Evaluate perplexity on WikiText-2 with sliding-window evaluation.
    Saves fetched text and tokenized IDs under model_weights.
    """
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
        model_path = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

    print("Initializing LLaMA3.1-8B-Instruct w4a16 quantized model...")
    try:
        model = LLaMA3W4A16Model(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
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

                    loss_sum = F.cross_entropy(
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
    """Example usage of LLaMA3W4A16Model when run as a script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLaMA3.1-8B-Instruct W4A16 Quantized Model - Unified LibTorch Backend")
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
        default="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        help="Path to quantized model (default: hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4)"
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
        choices=["hetero"],
        help="Backend to use (default: hetero)"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to NPU config JSON (default: auto-detect from lscpu for hetero backend)"
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
        help="Maximum number of tokens to generate (if --generate is used, default: 32)"
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
    parser.add_argument(
        "--save-used-prompt",
        action="store_true",
        default=False,
        help="Save the exact prompt used for the run to used_prompt.txt (default: off)."
    )
    parser.add_argument(
        "--perplexity",
        action="store_true",
        default=False,
        help="Compute perplexity for the input text (or prompt-test sequence) instead of generation."
    )
    parser.add_argument(
        "--wikitext2-perplexity",
        action="store_true",
        help="Compute perplexity on WikiText-2 and save downloaded/tokenized files under model_weights."
    )
    parser.add_argument(
        "--wikitext2-split",
        type=str,
        default="test",
        choices=["train", "valid", "validation", "test"],
        help="WikiText-2 split to evaluate."
    )
    parser.add_argument(
        "--wikitext2-max-length",
        type=int,
        default=2048,
        help="Max context length per evaluation window for WikiText-2 perplexity."
    )
    parser.add_argument(
        "--wikitext2-stride",
        type=int,
        default=2048,
        help="Stride for sliding-window WikiText-2 perplexity."
    )
    
    args = parser.parse_args()

    if args.wikitext2_perplexity:
        return run_wikitext2_perplexity(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            backend=args.backend,
            config_path=args.config_path,
            split=args.wikitext2_split,
            max_length=args.wikitext2_max_length,
            stride=args.wikitext2_stride,
        )
    
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
            perplexity=args.perplexity,
            measure_power=args.measure_power,
            save_used_prompt=args.save_used_prompt,
        )

    if args.backend == "hetero":
        args.config_path = _resolve_llama3_config_path(args.config_path)
        _set_prompt_len_in_config(args.config_path, 1)
    
    print("=" * 60)
    print("Initializing LLaMA3.1-8B-Instruct w4a16 quantized model...")
    print("=" * 60)
    
    try:
        model = LLaMA3W4A16Model(
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
        print("  2. The unified_llm_w4a16_hetero_libtorch module is in your Python path")
        print("  3. Model weights are loaded (if required)")
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
            start_time = time.time()
            logits = model(args.text)
            end_time = time.time()
            print(f"Prefill time: {end_time - start_time:.4f} seconds")
            
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
    
    print(f"\n{'=' * 60}")
    if hasattr(model, 'load_time'):
        print(f"Weight loading time: {model.load_time:.2f} seconds")
    print("Done!")
    print(f"{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    exit(main())

# First 4 logits (last token in batch 0): [5.28125, 4.625, 2.65625, 3.171875]
