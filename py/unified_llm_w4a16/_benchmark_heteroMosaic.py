import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

# PROMPT_SIZES = [1024, 2048, 4096, 8192, 16384]
PROMPT_SIZES = [256, 512, 1024, 2048, 4096, 8192]
MICRO_BATCH_SIZE = -2
DEFAULT_TIMEOUT_SEC = 300
WARMUP = False
DUMMY_WEIGHTS = True
RUN_GEN = False
COOLDOWN_SEC = 16
RETRY = 3
GEN_COMPARE_MAX_NEW_TOKENS = 16
SPECIAL_NPU_16K_CTX_LEN = 16384
DO_NOT_SEARCH_MP = True
TILE_FUSE = False
TILE_FUSE_NPU_MODE = "npu-sim"


# Default model target for benchmarks.
# Available values:
# MODEL = "gemma"
MODEL = "llama3_8b"
# MODEL = "llama3_70b"
# MODEL = "phi3.5_3.8b"
# MODEL = "qwen14b"
# MODEL = "qwen3b"

SYSTEM_PROFILE_TOKEN_MAP = [
    {
        "tokens": ("RYZEN AI 7 350", "RADEON 860M"),
        "profile": "krackanP",
        "gpu_chunking_inflight": 1,
    },
    {
        "tokens": ("RYZEN AI 9 HX 370", "RADEON 890M"),
        "profile": "strixP",
        "gpu_chunking_inflight": 1,
    },
    {
        "tokens": ("RYZEN AI MAX+ 395", "RADEON 8060S"),
        "profile": "strixH",
        "gpu_chunking_inflight": 2,
    },
]

MODELS = [
    {
        "id": "llama3_8b",
        "name": "Llama3",
        "script": "llama3_8b_w4a16_model.py",
        "MAX_CTX": 16384,
        "CHUNKING_S": True,
        "fallback_profile": "strixP",
        "profile_configs": {
            "krackanP": "configs_krackanP_llama3_8b.json5",
            "strixP": "configs_strixP_llama3_8b.json5",
            "strixH": "configs_strixH_llama3_8b.json5",
        },
    },
    {
        "id": "gemma",
        "name": "Gemma",
        "script": "gemma_w4a16_model.py",
        "MAX_CTX": 8192,
        "CHUNKING_S": True,
        "fallback_profile": "strixH",
        "profile_configs": {
            "krackanP": "configs_krackanP_gemma.json5",
            "strixP": "configs_strixP_gemma.json5",
            "strixH": "configs_strixH_gemma.json5",
        },
    },
    {
        "id": "qwen14b",
        "name": "Qwen2.5 14B",
        "script": "qwen25_14b_w4a16_model.py",
        "MAX_CTX": 16384,
        "CHUNKING_S": True,
        "fallback_profile": "strixP",
        "profile_configs": {
            "krackanP": "configs_krackanP_qwen25_14b.json5",
            "strixP": "configs_strixP_qwen25_14b.json5",
            "strixH": "configs_strixH_qwen25_14b.json5",
        },
    },
    {
        "id": "qwen3b",
        "name": "Qwen2.5 3B",
        "script": "qwen25_3b_w4a16_model.py",
        "MAX_CTX": 16384,
        "CHUNKING_S": True,
        "fallback_profile": "krackanP",
        "profile_configs": {
            "krackanP": "configs_krackanP_qwen25_3b.json5",
            "strixP": "configs_strixP_qwen25_3b.json5",
            "strixH": "configs_strixH_qwen25_3b.json5",
        },
    },
    {
        "id": "llama3_70b",
        "name": "Llama3 70B",
        "script": "llama3_70b_w4a16_model.py",
        "MAX_CTX": 16384,
        "CHUNKING_S": False,
        "fallback_profile": "strixP",
        "profile_configs": {
            "krackanP": "configs_krackanP_llama3_70b.json5",
            "strixP": "configs_strixP_llama3_70b.json5",
            "strixH": "configs_strixH_llama3_70b.json5",
        },
    },
    {
        "id": "phi3.5_3.8b",
        "name": "Phi-3.5 3.8B",
        "script": "phi35_3.8b_w4a16_model.py",
        "MAX_CTX": 16384,
        "CHUNKING_S": True,
        "fallback_profile": "strixP",
        "profile_configs": {
            "krackanP": "configs_krackanP_phi3.5_3.8b.json5",
            "strixP": "configs_strixP_phi3.5_3.8b.json5",
            "strixH": "configs_strixH_phi3.5_3.8b.json5",
        },
    },
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT_DIR, "benchmark.json")
LOG_FILE = os.path.join(
    SCRIPT_DIR,
    f"benchmark_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)


def log_print(message: str, file_handle=None) -> None:
    print(message)
    if file_handle:
        file_handle.write(message + "\n")
        file_handle.flush()


def iso_now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _normalize_cpu_model_name(model_name: str) -> str:
    return re.sub(r"\s+", " ", model_name.strip().upper())


def _detect_lscpu_model_name() -> str:
    try:
        output = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return ""

    match = re.search(r"^Model name:\s*(.+)$", output, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _detect_system_profile() -> Tuple[str, str]:
    cpu_model_name = _detect_lscpu_model_name()
    normalized = _normalize_cpu_model_name(cpu_model_name) if cpu_model_name else ""
    for rule in SYSTEM_PROFILE_TOKEN_MAP:
        tokens = tuple(rule.get("tokens", ()))
        profile = str(rule.get("profile", "")).strip()
        if all(token in normalized for token in tokens):
            return profile, cpu_model_name
    return "", cpu_model_name


def _detect_default_gpu_chunking_inflight() -> int:
    cpu_model_name = _detect_lscpu_model_name()
    normalized = _normalize_cpu_model_name(cpu_model_name) if cpu_model_name else ""
    for rule in SYSTEM_PROFILE_TOKEN_MAP:
        tokens = tuple(rule.get("tokens", ()))
        if all(token in normalized for token in tokens):
            return max(1, int(rule.get("gpu_chunking_inflight", 1)))
    return 1


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _model_max_ctx(model_conf: Optional[dict]) -> int:
    if not isinstance(model_conf, dict):
        return 0
    max_ctx = _safe_int(model_conf.get("MAX_CTX", 0), 0)
    return int(max_ctx) if int(max_ctx) > 0 else 0


def _model_chunking_s_enabled(model_conf: Optional[dict]) -> bool:
    if not isinstance(model_conf, dict):
        return True
    value = model_conf.get("CHUNKING_S", True)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _get_effective_prompt_sizes(prompt_sizes: List[int], model_conf: Optional[dict] = None) -> List[int]:
    max_ctx = _model_max_ctx(model_conf)
    if max_ctx <= 0:
        return [int(size) for size in prompt_sizes]

    capped = []
    seen = set()
    for size in prompt_sizes:
        effective_size = int(min(_safe_int(size, 0), max_ctx))
        if effective_size <= 0 or effective_size in seen:
            continue
        capped.append(effective_size)
        seen.add(effective_size)
    return capped


def resolve_model_config_path_for_platform(model: dict) -> Tuple[str, str]:
    configs_dir = os.path.join(SCRIPT_DIR, "configs")
    profile, cpu_model_name = _detect_system_profile()
    profile_configs = model.get("profile_configs", {})
    fallback_profile = str(model.get("fallback_profile", next(iter(profile_configs), "")))
    selected_profile = profile if profile in profile_configs else fallback_profile
    filename = profile_configs.get(selected_profile)
    if not filename:
        model_id = str(model.get("id", "")).strip()
        if not model_id:
            raise KeyError(f"Missing model id for benchmark config resolution: {model.get('name', 'unknown')}")
        filename = f"configs_{selected_profile}_{model_id}.json5"
    return os.path.join(configs_dir, filename), cpu_model_name


def load_config_with_comments(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r"//.*", "", content)
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    return json.loads(content)


def save_json_atomic(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def load_state(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def cleanup_temp_log_files(state_path: str, run_log_path: Optional[str] = None) -> Tuple[int, int, int, int]:
    removed_files = 0
    removed_refs = 0
    removed_baks = 0
    removed_state = 0

    # Cleanup-only mode: remove all benchmark run logs in script dir.
    if run_log_path is None:
        for name in os.listdir(SCRIPT_DIR):
            if re.fullmatch(r"benchmark_run_\d{8}_\d{6}\.log", name):
                path = os.path.join(SCRIPT_DIR, name)
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                        removed_files += 1
                    except Exception:
                        pass

        configs_dir = os.path.join(SCRIPT_DIR, "configs")
        if os.path.isdir(configs_dir):
            for name in os.listdir(configs_dir):
                if name.endswith(".bench.bak"):
                    path = os.path.join(configs_dir, name)
                    if os.path.isfile(path):
                        try:
                            os.remove(path)
                            removed_baks += 1
                        except Exception:
                            pass

        if os.path.isfile(state_path):
            try:
                os.remove(state_path)
                removed_state = 1
            except Exception:
                pass
    else:
        if os.path.exists(run_log_path):
            try:
                os.remove(run_log_path)
                removed_files += 1
            except Exception:
                pass

    state = load_state(state_path)
    if not isinstance(state, dict):
        return removed_files, removed_refs, removed_baks, removed_state

    logs = state.get("log_files")
    if isinstance(logs, list):
        if run_log_path is None:
            updated_logs = [
                p for p in logs
                if not re.fullmatch(r".*/benchmark_run_\d{8}_\d{6}\.log", str(p))
            ]
        else:
            updated_logs = [p for p in logs if p != run_log_path]
        removed_refs = len(logs) - len(updated_logs)
        if removed_refs > 0:
            state["log_files"] = updated_logs
            state["updated_at"] = iso_now()
            save_json_atomic(state_path, state)

    return removed_files, removed_refs, removed_baks, removed_state


def compute_ub(actual_tokens: int, micro_batch_size: int) -> int:
    if micro_batch_size > 0:
        return micro_batch_size
    if micro_batch_size == -2:
        return 4096 if actual_tokens > 4096 else max(1, actual_tokens // 2)
    return actual_tokens


def case_key(case: dict) -> str:
    return (
        f"{case['model']}|{case['prompt_size']}|{case['backend']}|"
        f"{int(case['chunking'])}|{int(case['chunking_scheduled'])}|"
        f"{int(case['minimal_pdi'])}|genv2|tok={int(GEN_COMPARE_MAX_NEW_TOKENS)}"
    )


def _search_minimal_pdi_values(backend: str, chunking: bool, scheduled: bool) -> List[bool]:
    # mp=0 corresponds to minimal_pdi=false. Allow opting out of those cases globally.
    if DO_NOT_SEARCH_MP:
        return [True]
    if (backend == "npu" and not chunking and not scheduled) or (
        backend == "hetero" and not chunking and not scheduled
    ):
        return [True, False]
    return [True]


def _build_scenarios_for_prompt_size(size: int, allow_chunking_s: bool) -> List[Tuple[str, bool, bool]]:
    scenarios: List[Tuple[str, bool, bool]] = []
    if size <= 8192 or size == SPECIAL_NPU_16K_CTX_LEN:
        scenarios.append(("npu", False, False))

    for chunking in (False, True):
        scenarios.append(("gpu", chunking, False))

    if size <= 8192 or size == SPECIAL_NPU_16K_CTX_LEN:
        scenarios.append(("hetero", False, False))
        scheduled_values = [False, True] if allow_chunking_s else [False]
        for scheduled in scheduled_values:
            scenarios.append(("hetero", True, scheduled))
    else:
        scheduled_values = [False, True] if allow_chunking_s else [False]
        for scheduled in scheduled_values:
            scenarios.append(("hetero", True, scheduled))

    return scenarios


def _should_keep_case_for_tile_fuse(
    backend: str, chunking: bool, scheduled: bool, minimal_pdi: bool
) -> bool:
    if not TILE_FUSE:
        return True
    # Tile-fuse benchmarking only keeps the baseline NPU prefill path.
    return backend == "npu" and not chunking and not scheduled and minimal_pdi


def _case_heterogeneity(case: dict) -> str:
    if (
        TILE_FUSE
        and case["backend"] == "npu"
        and not bool(case["chunking"])
        and not bool(case["chunking_scheduled"])
        and bool(case["minimal_pdi"])
    ):
        return TILE_FUSE_NPU_MODE
    return case["backend"]


def make_cases(model: dict, prompt_sizes: List[int], micro_batch_size: int) -> List[dict]:
    cases: List[dict] = []
    default_gpu_chunking_inflight = _detect_default_gpu_chunking_inflight()
    allow_chunking_s = _model_chunking_s_enabled(model)

    for size in prompt_sizes:
        actual_tokens = int(size)
        ub = int(compute_ub(actual_tokens, micro_batch_size))

        for backend, chunking, scheduled in _build_scenarios_for_prompt_size(size, allow_chunking_s):
            for minimal_pdi in _search_minimal_pdi_values(backend, bool(chunking), bool(scheduled)):
                if not _should_keep_case_for_tile_fuse(
                    backend, bool(chunking), bool(scheduled), bool(minimal_pdi)
                ):
                    continue
                case = {
                    "model": model["name"],
                    "script": model["script"],
                    "prompt_size": size,
                    "actual_tokens": actual_tokens,
                    "backend": backend,
                    "chunking": bool(chunking),
                    "chunking_scheduled": bool(scheduled),
                    "minimal_pdi": bool(minimal_pdi),
                    "gpu_chunk_size": ub,
                    "gpu_chunking_inflight": int(default_gpu_chunking_inflight),
                    "status": "pending",
                    "attempts": 0,
                    "last_start": None,
                    "last_end": None,
                    "returncode": None,
                    "error": None,
                    "metrics": None,
                    "command": None,
                    "last_output_tail": None,
                }
                case["key"] = case_key(case)
                cases.append(case)

    for idx, case in enumerate(cases):
        case["id"] = idx

    return cases


def expected_case_count(prompt_sizes: List[int], model: Optional[dict] = None) -> int:
    total = 0
    allow_chunking_s = _model_chunking_s_enabled(model)
    for size in prompt_sizes:
        for backend, chunking, scheduled in _build_scenarios_for_prompt_size(size, allow_chunking_s):
            for minimal_pdi in _search_minimal_pdi_values(backend, bool(chunking), bool(scheduled)):
                if _should_keep_case_for_tile_fuse(
                    backend, bool(chunking), bool(scheduled), bool(minimal_pdi)
                ):
                    total += 1
    return total


def build_initial_state(model: dict, prompt_sizes: List[int], micro_batch_size: int, timeout_sec: int, config_path: str, cpu_model_name: str) -> dict:
    return {
        "version": 2,
        "created_at": iso_now(),
        "updated_at": iso_now(),
        "model": model["name"],
        "script": model["script"],
        "config_path": config_path,
        "cpu_model_name": cpu_model_name,
        "prompt_sizes": prompt_sizes,
        "micro_batch_size": micro_batch_size,
        "gen_compare_max_new_tokens": int(GEN_COMPARE_MAX_NEW_TOKENS),
        "timeout_sec": timeout_sec,
        "cases": make_cases(model, prompt_sizes, micro_batch_size),
        "generation_compare": {
            "status": "skipped" if not RUN_GEN else "pending",
            "attempts": 0,
            "last_start": None,
            "last_end": None,
            "returncode": None,
            "error": None,
            "metrics": None,
            "command": None,
            "last_output_tail": None,
        },
        "log_files": [LOG_FILE],
    }


def merge_state(existing: dict, fresh: dict) -> dict:
    out = dict(fresh)
    out["created_at"] = existing.get("created_at", fresh["created_at"])
    out["log_files"] = list(dict.fromkeys((existing.get("log_files") or []) + [LOG_FILE]))

    old_cases = existing.get("cases", []) if isinstance(existing.get("cases"), list) else []
    old_by_key = {}
    for c in old_cases:
        k = c.get("key")
        if isinstance(k, str):
            old_by_key[k] = c

    merged_cases = []
    for new_case in fresh["cases"]:
        k = new_case["key"]
        old = old_by_key.get(k)
        if not old:
            merged_cases.append(new_case)
            continue

        merged = dict(new_case)
        for field in [
            "status",
            "attempts",
            "last_start",
            "last_end",
            "returncode",
            "error",
            "metrics",
            "command",
            "last_output_tail",
        ]:
            if field in old:
                merged[field] = old[field]

        if merged.get("status") == "running":
            merged["status"] = "timeout"
            merged["error"] = "Interrupted while running (likely reboot or kill)."
            merged["last_end"] = iso_now()

        merged_cases.append(merged)

    for idx, c in enumerate(merged_cases):
        c["id"] = idx

    out["cases"] = merged_cases
    old_generation_compare = existing.get("generation_compare")
    if isinstance(old_generation_compare, dict):
        merged_generation_compare = dict(fresh["generation_compare"])
        for field in [
            "status",
            "attempts",
            "last_start",
            "last_end",
            "returncode",
            "error",
            "metrics",
            "command",
            "last_output_tail",
        ]:
            if field in old_generation_compare:
                merged_generation_compare[field] = old_generation_compare[field]

        if merged_generation_compare.get("status") == "running":
            merged_generation_compare["status"] = "timeout"
            merged_generation_compare["error"] = "Interrupted while running (likely reboot or kill)."
            merged_generation_compare["last_end"] = iso_now()

        out["generation_compare"] = merged_generation_compare
    out["updated_at"] = iso_now()
    return out


def find_next_case_index(cases: List[dict]) -> Optional[int]:
    for idx, case in enumerate(cases):
        if case.get("status") != "done":
            return idx
    return None


def parse_metrics(output: str) -> Dict[str, float]:
    metrics = {
        "prefill_s": 0.0,
        "gen_s": 0.0,
        "avg_tok_s": 0.0,
        "avg_power": 0.0,
        "min_power": 0.0,
        "max_power": 0.0,
        "gen_gpu_s": 0.0,
        "gen_gpu_avg_tok_s": 0.0,
        "gen_hetero_s": 0.0,
        "gen_hetero_avg_tok_s": 0.0,
        "gen_vs_gpu_pct": 0.0,
    }

    m = re.search(r"Prefill time:\s*([\d\.]+)\s*seconds", output)
    if m:
        metrics["prefill_s"] = float(m.group(1))

    m = re.search(r"Total Generation Time:\s*([\d\.]+)\s*seconds", output)
    if m:
        metrics["gen_s"] = float(m.group(1))

    m = re.search(r"Average Time per Token:\s*([\d\.]+)\s*seconds", output)
    if m:
        metrics["avg_tok_s"] = float(m.group(1))

    m = re.search(r"Average Power:\s*([\d\.]+)\s*W", output)
    if m:
        metrics["avg_power"] = float(m.group(1))

    m = re.search(r"Min Power:\s*([\d\.]+)\s*W", output)
    if m:
        metrics["min_power"] = float(m.group(1))

    m = re.search(r"Max Power:\s*([\d\.]+)\s*W", output)
    if m:
        metrics["max_power"] = float(m.group(1))

    return metrics


def restore_config_from_backup(config_path: str, backup_path: str) -> None:
    shutil.copyfile(backup_path, config_path)


def ensure_backup(config_path: str, backup_path: str, log_handle) -> None:
    if os.path.exists(backup_path):
        restore_config_from_backup(config_path, backup_path)
        log_print(f"Using existing benchmark backup: {backup_path}", log_handle)
        return

    shutil.copyfile(config_path, backup_path)
    log_print(f"Created benchmark backup: {backup_path}", log_handle)


def _build_case_config_dict(backup_path: str, case: dict) -> dict:
    cfg = load_config_with_comments(backup_path)
    cfg["warmup"] = bool(WARMUP)
    cfg["dummy_weights"] = bool(DUMMY_WEIGHTS)
    cfg["prompt_len"] = int(case["prompt_size"])
    cfg["heterogeneity"] = _case_heterogeneity(case)
    cfg["chunking"] = bool(case["chunking"])
    cfg["chunking_scheduled"] = bool(case["chunking_scheduled"])
    cfg["minimal_pdi"] = bool(case["minimal_pdi"])
    gpu_cfg = cfg.get("gpu_chunking", {})
    if not isinstance(gpu_cfg, dict):
        gpu_cfg = {}
    existing_gpu_chunk_size = gpu_cfg.get("gpu_chunk_size", gpu_cfg.get("chunk_size"))
    if isinstance(existing_gpu_chunk_size, list):
        gpu_cfg["gpu_chunk_size"] = [int(case["gpu_chunk_size"])]
    else:
        gpu_cfg["gpu_chunk_size"] = int(case["gpu_chunk_size"])
    gpu_cfg["gpu_chunking_inflight"] = int(case["gpu_chunking_inflight"])
    cfg["gpu_chunking"] = gpu_cfg
    if case["backend"] == "npu" and int(case["prompt_size"]) == SPECIAL_NPU_16K_CTX_LEN:
        for entry in cfg.get("npuOnlydefault", []):
            if isinstance(entry, dict):
                entry["max_ctx_len"] = SPECIAL_NPU_16K_CTX_LEN
    return cfg


def _build_generation_compare_config_dict(backup_path: str, heterogeneity: str, gemv_driven_split_k: bool) -> dict:
    cfg = load_config_with_comments(backup_path)
    cfg["warmup"] = bool(WARMUP)
    cfg["dummy_weights"] = bool(DUMMY_WEIGHTS)
    cfg["heterogeneity"] = str(heterogeneity)
    cfg["gemv_driven_split_K"] = bool(gemv_driven_split_k)
    return cfg


def apply_case_config_from_backup(config_path: str, backup_path: str, case: dict) -> None:
    with open(backup_path, "r", encoding="utf-8") as f:
        base_text = f.read()

    text = base_text

    def bool_str(v: bool) -> str:
        return "true" if v else "false"

    def replace_one(pattern: str, replacement: str) -> bool:
        nonlocal text
        updated, count = re.subn(pattern, replacement, text, count=1)
        if count > 0:
            text = updated
            return True
        return False

    def replace_gpu_chunk_size(chunk_size: int) -> bool:
        nonlocal text

        def _replacement(match: re.Match[str]) -> str:
            prefix = match.group(1)
            value = match.group(2)
            if value.lstrip().startswith("["):
                return f"{prefix}[{int(chunk_size)}]"
            return f"{prefix}{int(chunk_size)}"

        updated, count = re.subn(
            r'("gpu_chunk_size"\s*:\s*)(\[[^\]]*\]|-?\d+)',
            _replacement,
            text,
            count=1,
        )
        if count > 0:
            text = updated
            return True
        return False

    ok = True
    ok &= replace_one(r'("warmup"\s*:\s*)(true|false|-?\d+)', rf'\g<1>{bool_str(WARMUP)}')
    ok &= replace_one(r'("dummy_weights"\s*:\s*)(true|false|-?\d+)', rf'\g<1>{bool_str(DUMMY_WEIGHTS)}')
    ok &= replace_one(r'("prompt_len"\s*:\s*)(-?\d+)', rf'\g<1>{int(case["prompt_size"])}')
    ok &= replace_one(r'("heterogeneity"\s*:\s*)"[^"]*"', rf'\g<1>"{_case_heterogeneity(case)}"')
    ok &= replace_one(r'("chunking"\s*:\s*)(true|false|-?\d+)', rf'\g<1>{bool_str(case["chunking"])}')
    ok &= replace_one(r'("chunking_scheduled"\s*:\s*)(true|false|-?\d+)', rf'\g<1>{bool_str(case["chunking_scheduled"])}')
    ok &= replace_one(r'("minimal_pdi"\s*:\s*)(true|false|-?\d+)', rf'\g<1>{bool_str(case["minimal_pdi"])}')
    ok &= replace_gpu_chunk_size(int(case["gpu_chunk_size"]))
    ok &= replace_one(r'("gpu_chunking_inflight"\s*:\s*)(-?\d+)', rf'\g<1>{int(case["gpu_chunking_inflight"])}')
    if case["backend"] == "npu" and int(case["prompt_size"]) == SPECIAL_NPU_16K_CTX_LEN:
        ok &= replace_one(
            r'("max_ctx_len"\s*:\s*)(-?\d+)',
            rf'\g<1>{SPECIAL_NPU_16K_CTX_LEN}',
        )

    if not ok:
        cfg = _build_case_config_dict(backup_path, case)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
            f.write("\n")
        return

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(text)


def _run_command(cmd_str: str, timeout_sec: int, log_handle, error_context: str) -> Tuple[str, int, Dict[str, float], str, str]:
    try:
        result = subprocess.run(
            cmd_str,
            shell=True,
            executable="/bin/bash",
            cwd=SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        output = result.stdout or ""
        if output:
            print(output)
            log_handle.write(output + "\n")
            log_handle.flush()

        metrics = parse_metrics(output)
        if result.returncode != 0:
            return "failed", result.returncode, metrics, output, f"{error_context}: process exited with code {result.returncode}."

        return "done", result.returncode, metrics, output, ""

    except subprocess.TimeoutExpired as e:
        output = (e.stdout or "") if isinstance(e.stdout, str) else ""
        if output:
            print(output)
            log_handle.write(output + "\n")
            log_handle.flush()
        return "timeout", -1, parse_metrics(output), output, f"{error_context}: timed out after {timeout_sec}s."


def _run_generation_compare(script: str, timeout_sec: int, log_handle, backup_path: str) -> Tuple[Optional[Dict[str, float]], int, str, Optional[str]]:
    compare_results = {}
    combined_output_parts = []
    compare_labels = [
        ("gpu", "gpu", False),
        ("hetero", "hetero", True),
    ]

    for label, heterogeneity, gemv_driven_split_k in compare_labels:
        cfg = _build_generation_compare_config_dict(backup_path, heterogeneity, gemv_driven_split_k)

        temp_config_path = os.path.join(
            SCRIPT_DIR,
            "configs",
            f".benchmark_{label}_gen_tmp.json5",
        )
        with open(temp_config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
            f.write("\n")

        cmd_str = (
            "pushd ../../ && source utils/setup.sh && popd && "
            f"python3 {script} "
            f"--max-new-tokens {int(GEN_COMPARE_MAX_NEW_TOKENS)} "
            f"--config-path {temp_config_path}"
        )
        log_print(
            f"Running generation compare ({label}): "
            f"heterogeneity={heterogeneity}, gemv_driven_split_K={str(bool(gemv_driven_split_k)).lower()}, "
            "prompt source=default text (no --prompt-test)",
            log_handle,
        )
        log_print(f"Generation compare command: {cmd_str}", log_handle)
        try:
            status, returncode, metrics, output, error = _run_command(
                cmd_str,
                timeout_sec,
                log_handle,
                f"Generation compare ({label})",
            )
        finally:
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

        if output:
            combined_output_parts.append(output)
        if status != "done":
            return None, returncode, "\n".join(combined_output_parts), error
        if float(metrics.get("avg_tok_s", 0.0)) <= 0.0:
            return None, returncode, "\n".join(combined_output_parts), (
                f"Generation compare ({label}): could not parse a valid Average Time per Token."
            )

        compare_results[label] = metrics

    gpu_avg = float(compare_results["gpu"].get("avg_tok_s", 0.0))
    hetero_avg = float(compare_results["hetero"].get("avg_tok_s", 0.0))
    gen_vs_gpu_pct = 0.0
    if gpu_avg > 0.0 and hetero_avg > 0.0:
        gen_vs_gpu_pct = ((gpu_avg / hetero_avg) - 1.0) * 100.0

    return {
        "gen_gpu_s": float(compare_results["gpu"].get("gen_s", 0.0)),
        "gen_gpu_avg_tok_s": gpu_avg,
        "gen_hetero_s": float(compare_results["hetero"].get("gen_s", 0.0)),
        "gen_hetero_avg_tok_s": hetero_avg,
        "gen_vs_gpu_pct": gen_vs_gpu_pct,
    }, 0, "\n".join(combined_output_parts), None


def run_single_case(case: dict, timeout_sec: int, log_handle, backup_path: str) -> Tuple[str, int, Dict[str, float], str, str]:
    scenario = (
        f"{case['backend']}|chunking={str(case['chunking']).lower()}|"
        f"scheduled={str(case['chunking_scheduled']).lower()}|"
        f"pdi={str(case['minimal_pdi']).lower()}"
    )
    if bool(case.get("chunking", False)):
        scenario += f"|ub={case['gpu_chunk_size']}"
    script = case["script"]
    size = int(case["prompt_size"])
    cmd_str = (
        "pushd ../../ && source utils/setup.sh && popd && "
        f"python3 {script} --prompt-test {size} --max-new-tokens 0"
    )

    log_print("-" * 100, log_handle)
    log_print(f"Running case {case['id']}: size={size}, scenario={scenario}", log_handle)
    log_print(f"Command: {cmd_str}", log_handle)

    status, returncode, metrics, output, error = _run_command(cmd_str, timeout_sec, log_handle, "Prefill benchmark")
    if status != "done":
        return status, returncode, metrics, output, error
    if metrics.get("prefill_s", 0.0) <= 0.0:
        return "failed", returncode, metrics, output, "Could not parse a valid prefill time."

    return "done", returncode, metrics, output, ""


def print_summary(state: dict, log_handle) -> None:
    cases = state.get("cases", [])
    log_print("\n" + "#" * 140, log_handle)
    log_print("FINAL BENCHMARK SUMMARY", log_handle)
    log_print("#" * 140, log_handle)
    log_print(
        "Legend: c=0 -> chunking=false, cs=0 -> chunking_scheduled=false, mp=1 -> minimal_pdi=true",
        log_handle,
    )
    log_print(
        f"{'ID':<4} | {'Size':<6} | {'Scenario':<58} | {'Status':<8} | {'Prefill(s)':<10}",
        log_handle,
    )
    log_print("-" * 140, log_handle)

    for c in cases:
        scenario = (
            f"{c['backend']}/c={int(bool(c['chunking']))}/"
            f"cs={int(bool(c['chunking_scheduled']))}/"
            f"mp={int(bool(c.get('minimal_pdi', True)))}"
        )
        if bool(c.get("chunking", False)):
            scenario += f"/ub={c['gpu_chunk_size']}"
        m = c.get("metrics") or {}
        log_print(
            f"{c['id']:<4} | {c['prompt_size']:<6} | {scenario:<58} | {c.get('status','?'):<8} | "
            f"{float(m.get('prefill_s', 0.0)):<10.4f}",
            log_handle,
        )

    generation_compare = state.get("generation_compare") or {}
    generation_metrics = generation_compare.get("metrics") or {}
    log_print("\nGENERATION COMPARE", log_handle)
    log_print("-" * 140, log_handle)
    if not RUN_GEN:
        log_print("Status: skipped | RUN_GEN is False", log_handle)
        return
    log_print(
        f"Status: {generation_compare.get('status', 'pending')} | "
        f"GPU Tok(s): {float(generation_metrics.get('gen_gpu_avg_tok_s', 0.0)):.4f} | "
        f"Het Tok(s): {float(generation_metrics.get('gen_hetero_avg_tok_s', 0.0)):.4f} | "
        f"HetVsGPU%: {float(generation_metrics.get('gen_vs_gpu_pct', 0.0)):.2f}",
        log_handle,
    )


def run_generation_compare_once(state: dict, model: dict, timeout_sec: int, log_handle, backup_path: str) -> bool:
    generation_compare = state.setdefault("generation_compare", {})
    if generation_compare.get("status") == "done":
        log_print("Generation compare already completed.", log_handle)
        return True

    generation_compare["status"] = "running"
    generation_compare["attempts"] = int(generation_compare.get("attempts", 0)) + 1
    generation_compare["last_start"] = iso_now()
    generation_compare["last_end"] = None
    generation_compare["error"] = None
    generation_compare["returncode"] = None
    generation_compare["command"] = (
        "pushd ../../ && source utils/setup.sh && popd && "
        f"python3 {model['script']} --max-new-tokens {int(GEN_COMPARE_MAX_NEW_TOKENS)} "
        "--config-path <temp benchmark generation config>"
    )
    state["updated_at"] = iso_now()
    save_json_atomic(STATE_FILE, state)

    metrics, returncode, output, error = _run_generation_compare(
        model["script"],
        timeout_sec,
        log_handle,
        backup_path,
    )

    generation_compare["last_end"] = iso_now()
    generation_compare["returncode"] = returncode
    generation_compare["metrics"] = metrics
    generation_compare["last_output_tail"] = (output[-2000:] if output else None)

    if metrics is not None and error is None:
        generation_compare["status"] = "done"
        generation_compare["error"] = None
        state["updated_at"] = iso_now()
        save_json_atomic(STATE_FILE, state)
        return True

    generation_compare["status"] = "timeout" if returncode == -1 else "failed"
    generation_compare["error"] = error or "Generation compare failed."
    state["updated_at"] = iso_now()
    save_json_atomic(STATE_FILE, state)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hetero Mosaic benchmark runner for prompt prefill plus generation comparison.")
    parser.add_argument("--dry-run", action="store_true", help="Generate/merge cases and exit without running benchmarks.")
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC, help="Per-case timeout in seconds.")
    parser.add_argument("--reset-state", action="store_true", help="Ignore existing benchmark.json and start a new state.")
    parser.add_argument("--clean-temp", action="store_true", help="Skip benchmarks and delete benchmark temp artifacts (benchmark_run_*.log, *.bench.bak, benchmark.json).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.clean_temp:
        removed_files, removed_refs, removed_baks, removed_state = cleanup_temp_log_files(STATE_FILE, run_log_path=None)
        print(
            f"Clean-temp mode complete. Removed logs={removed_files}, "
            f"backup_files={removed_baks}, benchmark_json={removed_state}, "
            f"pruned_log_refs={removed_refs}."
        )
        return

    if not MODELS:
        print("No models configured.")
        sys.exit(1)

    model = next((m for m in MODELS if str(m.get("id", "")).strip() == MODEL), None)
    if model is None:
        print(f"Selected MODEL '{MODEL}' is not present in MODELS.")
        sys.exit(1)
    config_path, cpu_model_name = resolve_model_config_path_for_platform(model)
    if not os.path.exists(config_path):
        print(f"Error: auto-detected config not found: {config_path}")
        sys.exit(1)

    script_path = os.path.join(SCRIPT_DIR, model["script"])
    if not os.path.exists(script_path):
        print(f"Error: model script not found: {script_path}")
        sys.exit(1)

    print(f"Logging all output to: {LOG_FILE}")
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as log_handle:
            log_print(f"Detected CPU model: {cpu_model_name or 'unknown'}", log_handle)
            log_print(f"Auto-selected config: {config_path}", log_handle)

            effective_prompt_sizes = _get_effective_prompt_sizes(PROMPT_SIZES, model)
            fresh = build_initial_state(
                model,
                effective_prompt_sizes,
                MICRO_BATCH_SIZE,
                int(args.timeout_sec),
                config_path,
                cpu_model_name,
            )
            existing = None if args.reset_state else load_state(STATE_FILE)
            state = merge_state(existing, fresh) if isinstance(existing, dict) else fresh
            state["updated_at"] = iso_now()
            if LOG_FILE not in state.get("log_files", []):
                state.setdefault("log_files", []).append(LOG_FILE)

            save_json_atomic(STATE_FILE, state)

            total_cases = len(state.get("cases", []))
            expected_cases = expected_case_count(effective_prompt_sizes, model)
            if effective_prompt_sizes != [int(size) for size in PROMPT_SIZES]:
                log_print(
                    f"Applied model MAX_CTX={_model_max_ctx(model)} to prompt sizes: {PROMPT_SIZES} -> {effective_prompt_sizes}",
                    log_handle,
                )
            log_print(f"Generated case count: {total_cases}", log_handle)
            if total_cases != expected_cases:
                log_print(f"Warning: expected {expected_cases} cases, found {total_cases}", log_handle)

            if args.dry_run:
                log_print("Dry run complete. benchmark.json written/merged.", log_handle)
                return

            backup_path = config_path + ".bench.bak"
            ensure_backup(config_path, backup_path, log_handle)

            stop_due_to_timeout = False
            stop_due_to_failure = False

            try:
                next_idx = find_next_case_index(state["cases"])
                if next_idx is None:
                    log_print("All cases already completed.", log_handle)
                else:
                    log_print(f"Resuming from case index {next_idx}", log_handle)

                idx = next_idx
                while idx is not None and idx < len(state["cases"]):
                    case = state["cases"][idx]
                    if case.get("status") == "done":
                        idx += 1
                        continue

                    restore_config_from_backup(config_path, backup_path)
                    apply_case_config_from_backup(config_path, backup_path, case)

                    case["status"] = "running"
                    case["attempts"] = int(case.get("attempts", 0)) + 1
                    case["last_start"] = iso_now()
                    case["last_end"] = None
                    case["error"] = None
                    case["returncode"] = None
                    case["command"] = (
                        "pushd ../../ && source utils/setup.sh && popd && "
                        f"python3 {case['script']} --prompt-test {case['prompt_size']} --max-new-tokens 0"
                    )
                    state["updated_at"] = iso_now()
                    save_json_atomic(STATE_FILE, state)

                    status, returncode, metrics, output, error = run_single_case(
                        case,
                        int(args.timeout_sec),
                        log_handle,
                        backup_path,
                    )

                    case["status"] = status
                    case["returncode"] = returncode
                    case["metrics"] = metrics
                    case["last_end"] = iso_now()
                    case["error"] = error if error else None
                    case["last_output_tail"] = (output[-2000:] if output else None)

                    state["updated_at"] = iso_now()
                    save_json_atomic(STATE_FILE, state)

                    if status == "timeout":
                        log_print(f"Timeout at case {case['id']}. Stopping run for reboot-safe resume.", log_handle)
                        stop_due_to_timeout = True
                        break

                    if status == "failed":
                        retries_used = max(0, int(case.get("attempts", 0)) - 1)
                        if retries_used < RETRY:
                            retries_left = RETRY - retries_used
                            log_print(
                                f"Failure at case {case['id']}: {error} "
                                f"Retrying ({retries_left} retries left).",
                                log_handle,
                            )
                            log_print(f"Cooldown: sleeping {COOLDOWN_SEC} seconds before retry.", log_handle)
                            time.sleep(COOLDOWN_SEC)
                            continue

                        log_print(
                            f"Failure at case {case['id']}: {error} "
                            f"Exhausted {RETRY} retries.",
                            log_handle,
                        )
                        stop_due_to_failure = True
                        break

                    if any(c.get("status") != "done" for c in state["cases"][idx + 1 :]):
                        log_print(f"Cooldown: sleeping {COOLDOWN_SEC} seconds before next case.", log_handle)
                        time.sleep(COOLDOWN_SEC)

                    idx += 1

                all_prompt_cases_done = all(c.get("status") == "done" for c in state.get("cases", []))
                if RUN_GEN and all_prompt_cases_done and not stop_due_to_timeout and not stop_due_to_failure:
                    restore_config_from_backup(config_path, backup_path)
                    generation_compare = state.get("generation_compare") or {}
                    if generation_compare.get("status") != "done":
                        generation_ok = run_generation_compare_once(
                            state,
                            model,
                            int(args.timeout_sec),
                            log_handle,
                            backup_path,
                        )
                        if not generation_ok:
                            if (state.get("generation_compare") or {}).get("status") == "timeout":
                                log_print(
                                    "Generation compare timed out. Stopping run for reboot-safe resume.",
                                    log_handle,
                                )
                                stop_due_to_timeout = True
                            else:
                                log_print(
                                    f"Generation compare failed: {(state.get('generation_compare') or {}).get('error')}",
                                    log_handle,
                                )
                                stop_due_to_failure = True

            finally:
                if os.path.exists(backup_path):
                    restore_config_from_backup(config_path, backup_path)
                    log_print("Restored config from benchmark backup baseline.", log_handle)

            all_done = all(c.get("status") == "done" for c in state.get("cases", [])) and (
                not RUN_GEN or (state.get("generation_compare") or {}).get("status") == "done"
            )
            if all_done and os.path.exists(backup_path):
                os.remove(backup_path)
                log_print(f"Removed benchmark backup: {backup_path}", log_handle)

            print_summary(state, log_handle)

            if stop_due_to_timeout:
                sys.exit(1)
            if stop_due_to_failure:
                sys.exit(1)
    finally:
        if args.clean_temp:
            removed_files, removed_refs, removed_baks, removed_state = cleanup_temp_log_files(STATE_FILE, LOG_FILE)
            print(
                f"Clean-temp mode complete. Removed logs={removed_files}, "
                f"backup_files={removed_baks}, benchmark_json={removed_state}, "
                f"pruned_log_refs={removed_refs}."
            )


if __name__ == "__main__":
    main()
