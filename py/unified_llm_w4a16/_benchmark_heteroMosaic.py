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

PROMPT_SIZES = [1024, 2048, 4096, 8192, 16384]
# PROMPT_SIZES = [256, 512, 1024, 2048, 4096, 8192]
MICRO_BATCH_SIZE = -2
DEFAULT_TIMEOUT_SEC = 300
WARMUP = True
DUMMY_WEIGHTS = True
RUN_GEN = True
COOLDOWN_SEC = 16
RETRY = 3
GEN_COMPARE_MAX_NEW_TOKENS = 16
SPECIAL_NPU_16K_CTX_LEN = 16384
DO_NOT_SEARCH_MP = True

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
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
BENCHMARK_DIR = os.path.join(REPO_ROOT, "build", "benchmarks", "heteroMosaic")
os.makedirs(BENCHMARK_DIR, exist_ok=True)
STATE_FILE = os.path.join(BENCHMARK_DIR, "benchmark.json")
LOG_FILE = os.path.join(
    BENCHMARK_DIR,
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

    # Cleanup-only mode: remove all benchmark run logs in the benchmark output dir.
    if run_log_path is None:
        for name in os.listdir(BENCHMARK_DIR):
            if re.fullmatch(r"benchmark_run_\d{8}_\d{6}\.log", name):
                path = os.path.join(BENCHMARK_DIR, name)
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


def make_cases(model: dict, prompt_sizes: List[int], micro_batch_size: int) -> List[dict]:
    cases: List[dict] = []
    default_gpu_chunking_inflight = _detect_default_gpu_chunking_inflight()
    allow_chunking_s = _model_chunking_s_enabled(model)

    for size in prompt_sizes:
        actual_tokens = int(size)
        ub = int(compute_ub(actual_tokens, micro_batch_size))

        for backend, chunking, scheduled in _build_scenarios_for_prompt_size(size, allow_chunking_s):
            for minimal_pdi in _search_minimal_pdi_values(backend, bool(chunking), bool(scheduled)):
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
            total += len(_search_minimal_pdi_values(backend, bool(chunking), bool(scheduled)))
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


def merge_point_update_state(existing: dict, fresh: dict) -> Tuple[dict, set, bool]:
    """Replace only selected fresh cases while preserving every other cached case."""
    if not isinstance(existing, dict):
        raise ValueError("Point update requires an existing benchmark.json state file.")
    if existing.get("model") != fresh.get("model") or existing.get("script") != fresh.get("script"):
        raise ValueError(
            "Point update model does not match the cached state; use a normal run or --reset-state."
        )
    if existing.get("config_path") != fresh.get("config_path"):
        raise ValueError(
            "Point update config path does not match the cached state; use a normal run or --reset-state."
        )

    fresh_by_key = {case["key"]: case for case in fresh.get("cases", [])}
    active_keys = set(fresh_by_key)
    existing_point_update = existing.get("point_update")
    resuming = (
        isinstance(existing_point_update, dict)
        and existing_point_update.get("status") == "running"
        and set(existing_point_update.get("active_case_keys") or []) == active_keys
    )
    merged_cases = []
    consumed_keys = set()
    preserved_fields = [
        "status",
        "attempts",
        "last_start",
        "last_end",
        "returncode",
        "error",
        "metrics",
        "command",
        "last_output_tail",
    ]

    old_cases = existing.get("cases", []) if isinstance(existing.get("cases"), list) else []
    for old_case in old_cases:
        key = old_case.get("key")
        if key in fresh_by_key:
            replacement = dict(fresh_by_key[key])
            if resuming:
                for field in preserved_fields:
                    if field in old_case:
                        replacement[field] = old_case[field]
                if replacement.get("status") == "running":
                    replacement["status"] = "timeout"
                    replacement["error"] = "Interrupted while running (likely reboot or kill)."
                    replacement["last_end"] = iso_now()
            merged_cases.append(replacement)
            consumed_keys.add(key)
        else:
            merged_cases.append(dict(old_case))

    for fresh_case in fresh.get("cases", []):
        if fresh_case["key"] not in consumed_keys:
            merged_cases.append(dict(fresh_case))

    for idx, case in enumerate(merged_cases):
        case["id"] = idx

    out = dict(existing)
    for field in [
        "version",
        "model",
        "script",
        "config_path",
        "cpu_model_name",
        "micro_batch_size",
        "gen_compare_max_new_tokens",
        "timeout_sec",
    ]:
        if field in fresh:
            out[field] = fresh[field]

    old_prompt_sizes = existing.get("prompt_sizes", [])
    out["prompt_sizes"] = list(
        dict.fromkeys(
            [int(size) for size in old_prompt_sizes]
            + [int(size) for size in fresh.get("prompt_sizes", [])]
        )
    )
    out["cases"] = merged_cases
    out["log_files"] = list(
        dict.fromkeys((existing.get("log_files") or []) + (fresh.get("log_files") or []))
    )
    if resuming:
        point_update = dict(existing_point_update)
        point_update["resumed_at"] = iso_now()
    else:
        point_update = {
            "started_at": iso_now(),
            "completed_at": None,
        }
    point_update["status"] = "running"
    point_update["prompt_sizes"] = [int(size) for size in fresh.get("prompt_sizes", [])]
    point_update["active_case_keys"] = sorted(active_keys)
    out["point_update"] = point_update
    out["updated_at"] = iso_now()
    return out, active_keys, resuming


def resolve_point_update_case_ids(existing: dict, requested_case_ids: List[int]) -> Tuple[List[int], set, List[int]]:
    """Resolve cached case IDs to stable case keys and their prompt sizes."""
    if not isinstance(existing, dict):
        raise ValueError("Selecting --case-ids requires an existing benchmark.json state file.")

    requested_ids = list(dict.fromkeys(int(case_id) for case_id in requested_case_ids))
    old_cases = existing.get("cases", []) if isinstance(existing.get("cases"), list) else []
    cases_by_id = {}
    for case in old_cases:
        try:
            case_id = int(case.get("id"))
        except (TypeError, ValueError):
            continue
        if case_id in cases_by_id:
            raise ValueError(f"Cached benchmark state contains duplicate case ID {case_id}.")
        cases_by_id[case_id] = case

    missing_ids = [case_id for case_id in requested_ids if case_id not in cases_by_id]
    if missing_ids:
        available_ids = " ".join(str(case_id) for case_id in sorted(cases_by_id)) or "none"
        missing_text = " ".join(str(case_id) for case_id in missing_ids)
        raise ValueError(
            f"Unknown cached case ID(s): {missing_text}. Available case IDs: {available_ids}."
        )

    selected_cases = [cases_by_id[case_id] for case_id in requested_ids]
    missing_keys = [case_id for case_id, case in zip(requested_ids, selected_cases) if not case.get("key")]
    if missing_keys:
        raise ValueError(
            "Cached case ID(s) are missing stable case keys: "
            + " ".join(str(case_id) for case_id in missing_keys)
            + ". Run a normal benchmark to rebuild the state."
        )

    selected_keys = {str(case["key"]) for case in selected_cases}
    prompt_sizes = list(dict.fromkeys(int(case["prompt_size"]) for case in selected_cases))
    return requested_ids, selected_keys, prompt_sizes


def filter_fresh_cases_for_point_update(fresh: dict, selected_keys: set) -> dict:
    """Limit a freshly generated state to the exact cached cases being refreshed."""
    filtered = dict(fresh)
    filtered_cases = [
        case for case in fresh.get("cases", []) if case.get("key") in selected_keys
    ]
    found_keys = {case.get("key") for case in filtered_cases}
    missing_keys = selected_keys - found_keys
    if missing_keys:
        raise ValueError(
            "The selected cached case(s) no longer exist in the current benchmark matrix. "
            "Run a normal benchmark to rebuild the state before using --case-ids."
        )
    filtered["cases"] = filtered_cases
    return filtered


def filter_fresh_cases_by_scenario(
    fresh: dict,
    backend: Optional[str] = None,
    chunking: Optional[bool] = None,
    chunking_scheduled: Optional[bool] = None,
    minimal_pdi: Optional[bool] = None,
) -> dict:
    """Limit a point update using human-readable benchmark fields."""
    filtered = dict(fresh)
    filtered_cases = []
    for case in fresh.get("cases", []):
        if backend is not None and case.get("backend") != backend:
            continue
        if chunking is not None and bool(case.get("chunking")) != chunking:
            continue
        if (
            chunking_scheduled is not None
            and bool(case.get("chunking_scheduled")) != chunking_scheduled
        ):
            continue
        if minimal_pdi is not None and bool(case.get("minimal_pdi")) != minimal_pdi:
            continue
        filtered_cases.append(case)

    if not filtered_cases:
        raise ValueError(
            "No benchmark cases match the selected prompt sizes and scenario fields."
        )
    filtered["cases"] = filtered_cases
    return filtered


def format_point_update_scenario(
    prompt_sizes: List[int],
    backend: Optional[str],
    chunking: Optional[bool],
    chunking_scheduled: Optional[bool],
    minimal_pdi: Optional[bool],
) -> str:
    fields = [f"prompt sizes={prompt_sizes}"]
    if backend is not None:
        fields.append(f"backend={backend}")
    if chunking is not None:
        fields.append(f"chunking={str(chunking).lower()}")
    if chunking_scheduled is not None:
        fields.append(f"chunking scheduled={str(chunking_scheduled).lower()}")
    if minimal_pdi is not None:
        fields.append(f"minimal PDI={str(minimal_pdi).lower()}")
    return ", ".join(fields)


def find_next_case_index(cases: List[dict], active_keys: Optional[set] = None) -> Optional[int]:
    for idx, case in enumerate(cases):
        if active_keys is not None and case.get("key") not in active_keys:
            continue
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
    cfg["heterogeneity"] = case["backend"]
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
    ok &= replace_one(r'("heterogeneity"\s*:\s*)"[^"]*"', rf'\g<1>"{case["backend"]}"')
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
        f"backend={case['backend']}, chunking={str(case['chunking']).lower()}, "
        f"chunking scheduled={str(case['chunking_scheduled']).lower()}, "
        f"minimal PDI={str(case['minimal_pdi']).lower()}"
    )
    if bool(case.get("chunking", False)):
        scenario += f", GPU chunk size={case['gpu_chunk_size']}"
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
    log_print("\n" + "#" * 180, log_handle)
    log_print("FINAL BENCHMARK SUMMARY", log_handle)
    log_print("#" * 180, log_handle)
    log_print(
        f"{'ID':<4} | {'Size':<6} | {'Scenario':<98} | {'Status':<8} | {'Prefill(s)':<10}",
        log_handle,
    )
    log_print("-" * 180, log_handle)

    for c in cases:
        scenario = (
            f"backend={c['backend']}, chunking={str(bool(c['chunking'])).lower()}, "
            f"chunking scheduled={str(bool(c['chunking_scheduled'])).lower()}, "
            f"minimal PDI={str(bool(c.get('minimal_pdi', True))).lower()}"
        )
        if bool(c.get("chunking", False)):
            scenario += f", GPU chunk size={c['gpu_chunk_size']}"
        m = c.get("metrics") or {}
        log_print(
            f"{c['id']:<4} | {c['prompt_size']:<6} | {scenario:<98} | {c.get('status','?'):<8} | "
            f"{float(m.get('prefill_s', 0.0)):<10.4f}",
            log_handle,
        )

    generation_compare = state.get("generation_compare") or {}
    generation_metrics = generation_compare.get("metrics") or {}
    log_print("\nGENERATION COMPARE", log_handle)
    log_print("-" * 180, log_handle)
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


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value}") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("prompt sizes must be positive integers")
    return result


def non_negative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value}") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("case IDs must be non-negative integers")
    return result


def boolean_arg(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes", "on"):
        return True
    if normalized in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hetero Mosaic benchmark runner for prompt prefill plus generation comparison.")
    parser.add_argument(
        "--model",
        choices=[str(model["id"]) for model in MODELS],
        default=MODEL,
        help=f"Model to benchmark (default: {MODEL}).",
    )
    parser.add_argument(
        "--prompt-sizes",
        nargs="+",
        type=positive_int,
        default=PROMPT_SIZES,
        metavar="TOKENS",
        help=(
            "Prompt token sizes to benchmark "
            f"(default: {' '.join(map(str, PROMPT_SIZES))})."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate/merge cases and exit without running benchmarks.")
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC, help="Per-case timeout in seconds.")
    parser.add_argument("--reset-state", action="store_true", help="Ignore existing benchmark.json and start a new state.")
    parser.add_argument(
        "--point-update",
        action="store_true",
        help="Rerun selected prompt sizes or exact --case-ids and update only those cached cases.",
    )
    parser.add_argument(
        "--case-ids",
        "--case-id",
        dest="case_ids",
        nargs="+",
        type=non_negative_int,
        default=None,
        metavar="ID",
        help=(
            "With --point-update, rerun only these case IDs from the existing benchmark.json. "
            "The case IDs determine the prompt sizes."
        ),
    )
    parser.add_argument(
        "--backend",
        "--heterogeneity",
        choices=["npu", "gpu", "hetero"],
        default=None,
        help="With --point-update, select a readable backend: npu, gpu, or hetero.",
    )
    parser.add_argument(
        "--chunking",
        type=boolean_arg,
        default=None,
        metavar="BOOL",
        help="With --point-update, select chunking=true or chunking=false.",
    )
    parser.add_argument(
        "--chunking-scheduled",
        type=boolean_arg,
        default=None,
        metavar="BOOL",
        help="With --point-update, select whether scheduled chunking is true or false.",
    )
    parser.add_argument(
        "--minimal-pdi",
        type=boolean_arg,
        default=None,
        metavar="BOOL",
        help="With --point-update, select minimal PDI=true or minimal PDI=false.",
    )
    parser.add_argument("--clean-temp", action="store_true", help="Skip benchmarks and delete benchmark temp artifacts (benchmark_run_*.log, *.bench.bak, benchmark.json).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.point_update and args.reset_state:
        print("Error: --point-update cannot be combined with --reset-state.")
        sys.exit(2)
    if args.point_update and args.dry_run:
        print("Error: --point-update cannot be combined with --dry-run because no benchmark would run.")
        sys.exit(2)
    if args.case_ids and not args.point_update:
        print("Error: --case-ids requires --point-update.")
        sys.exit(2)
    scenario_selectors = (
        args.backend,
        args.chunking,
        args.chunking_scheduled,
        args.minimal_pdi,
    )
    if any(value is not None for value in scenario_selectors) and not args.point_update:
        print("Error: scenario selectors require --point-update.")
        sys.exit(2)
    if args.case_ids and any(value is not None for value in scenario_selectors):
        print("Error: use either --case-ids or readable scenario selectors, not both.")
        sys.exit(2)

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

    model = next((m for m in MODELS if str(m.get("id", "")).strip() == args.model), None)
    if model is None:
        print(f"Selected model '{args.model}' is not present in MODELS.")
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

            existing = None if args.reset_state else load_state(STATE_FILE)
            selected_case_ids = None
            selected_case_keys = None
            if args.point_update and args.case_ids:
                try:
                    selected_case_ids, selected_case_keys, selected_prompt_sizes = (
                        resolve_point_update_case_ids(existing, args.case_ids)
                    )
                except ValueError as exc:
                    log_print(f"Error: {exc}", log_handle)
                    sys.exit(2)
                effective_prompt_sizes = _get_effective_prompt_sizes(selected_prompt_sizes, model)
            else:
                effective_prompt_sizes = _get_effective_prompt_sizes(args.prompt_sizes, model)

            fresh = build_initial_state(
                model,
                effective_prompt_sizes,
                MICRO_BATCH_SIZE,
                int(args.timeout_sec),
                config_path,
                cpu_model_name,
            )
            if selected_case_keys is not None:
                try:
                    fresh = filter_fresh_cases_for_point_update(fresh, selected_case_keys)
                except ValueError as exc:
                    log_print(f"Error: {exc}", log_handle)
                    sys.exit(2)
            elif args.point_update and any(value is not None for value in scenario_selectors):
                try:
                    fresh = filter_fresh_cases_by_scenario(
                        fresh,
                        backend=args.backend,
                        chunking=args.chunking,
                        chunking_scheduled=args.chunking_scheduled,
                        minimal_pdi=args.minimal_pdi,
                    )
                except ValueError as exc:
                    log_print(f"Error: {exc}", log_handle)
                    sys.exit(2)

            active_case_keys = None
            if args.point_update:
                try:
                    state, active_case_keys, point_update_resuming = merge_point_update_state(existing, fresh)
                except ValueError as exc:
                    log_print(f"Error: {exc}", log_handle)
                    sys.exit(2)
                if selected_case_ids is not None:
                    state["point_update"]["case_ids"] = selected_case_ids
                    state["point_update"].pop("selection", None)
                    point_update_target = f"case IDs: {selected_case_ids}"
                else:
                    state["point_update"].pop("case_ids", None)
                    point_update_target = format_point_update_scenario(
                        effective_prompt_sizes,
                        args.backend,
                        args.chunking,
                        args.chunking_scheduled,
                        args.minimal_pdi,
                    )
                    state["point_update"]["selection"] = {
                        "backend": args.backend,
                        "chunking": args.chunking,
                        "chunking_scheduled": args.chunking_scheduled,
                        "minimal_pdi": args.minimal_pdi,
                    }
                log_print(
                    f"Point update {'resuming' if point_update_resuming else 'started'} for "
                    f"{point_update_target}; "
                    f"preserved {len(state.get('cases', [])) - len(active_case_keys)} other cached cases.",
                    log_handle,
                )
            else:
                state = merge_state(existing, fresh) if isinstance(existing, dict) else fresh
            state["updated_at"] = iso_now()
            if LOG_FILE not in state.get("log_files", []):
                state.setdefault("log_files", []).append(LOG_FILE)

            save_json_atomic(STATE_FILE, state)

            total_cases = (
                len(active_case_keys)
                if active_case_keys is not None
                else len(state.get("cases", []))
            )
            expected_cases = (
                len(fresh.get("cases", []))
                if active_case_keys is not None
                else expected_case_count(effective_prompt_sizes, model)
            )
            if selected_case_ids is None and effective_prompt_sizes != [int(size) for size in args.prompt_sizes]:
                log_print(
                    f"Applied model MAX_CTX={_model_max_ctx(model)} to prompt sizes: {args.prompt_sizes} -> {effective_prompt_sizes}",
                    log_handle,
                )
            case_count_label = "Point-update case count" if active_case_keys is not None else "Generated case count"
            log_print(f"{case_count_label}: {total_cases}", log_handle)
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
                next_idx = find_next_case_index(state["cases"], active_case_keys)
                if next_idx is None:
                    log_print("All cases already completed.", log_handle)
                else:
                    log_print(f"Resuming from case index {next_idx}", log_handle)

                idx = next_idx
                while idx is not None and idx < len(state["cases"]):
                    case = state["cases"][idx]
                    if active_case_keys is not None and case.get("key") not in active_case_keys:
                        idx += 1
                        continue
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

                    if any(
                        c.get("status") != "done"
                        and (active_case_keys is None or c.get("key") in active_case_keys)
                        for c in state["cases"][idx + 1 :]
                    ):
                        log_print(f"Cooldown: sleeping {COOLDOWN_SEC} seconds before next case.", log_handle)
                        time.sleep(COOLDOWN_SEC)

                    idx += 1

                if active_case_keys is not None and all(
                    case.get("status") == "done"
                    for case in state.get("cases", [])
                    if case.get("key") in active_case_keys
                ):
                    point_update = state.get("point_update")
                    if isinstance(point_update, dict):
                        point_update["status"] = "complete"
                        point_update["completed_at"] = iso_now()
                        state["updated_at"] = iso_now()
                        save_json_atomic(STATE_FILE, state)
                        completed_target = (
                            f"case IDs: {selected_case_ids}"
                            if selected_case_ids is not None
                            else format_point_update_scenario(
                                effective_prompt_sizes,
                                args.backend,
                                args.chunking,
                                args.chunking_scheduled,
                                args.minimal_pdi,
                            )
                        )
                        log_print(
                            f"Point update complete for {completed_target}.",
                            log_handle,
                        )

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
