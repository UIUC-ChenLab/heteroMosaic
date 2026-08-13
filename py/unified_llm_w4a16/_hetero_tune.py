import os
import subprocess
import re
import shutil
import time
import datetime
import json
import sys
import math
import glob
import signal
import tempfile
import hashlib
from collections import defaultdict
from pathlib import Path

# Constants
PROMPT_SIZES = [512, 1024, 2048, 4096, 8192, 16384]
# PROMPT_SIZES = [8192]

# Default model target for tuning.
# Available values:
# MODEL = "gemma"
# MODEL = "llama3_8b"
# MODEL = "llama3_70b"
MODEL = "phi3.5_3.8b"
# MODEL = "qwen14b"

# Verbosity level for benchmarks (0=silent, 1=setup messages, 2=runtime messages)
VERBOSITY = 1
RUN_AVERAGE = 1
BENCHMARK_TIMEOUT_SEC = 300
RETRY_INVALID = 4
MAX_M_DIM = 8192
SPECIAL_GEMM_16K_DIM = 16384
DUMMY_WEIGHTS = True
WARMUP = False
LOG_FILE = f"hetero_tune_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
GEMM_LAYER_ORDER = ["qkv", "q", "k", "v", "o", "up", "gate", "down"]
GEMV_LAYER_ORDER = ["qkv", "q", "k", "v", "o", "up", "gate", "down"]
CW_STAGE_ORDER = ["G1", "A", "G2"]
CW_SHARE_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]
CW_SLACK_TOLERANCE_SEC = 1e-4
CW_BUBBLE_QUANTUM_US = 250
CW_MAX_BUBBLE_US = 5000
CW_MAX_WINDOWS = 8
CW_MAX_CRITICAL_PER_WINDOW = 2
CW_MAX_NONCRITICAL_PER_WINDOW = 2
CW_MAX_PROPOSALS_PER_ITER = 64
CW_RESERVED_SINGLE_TIGHTEN_PROPOSALS = 16
CW_SEARCH_BUDGET = 8
CW_LATENCY_GATE_PCT = 0.08
CW_LATENCY_GATE_FALLBACK_PCT = 0.12
CW_MIN_GATE_SURVIVORS = 2
CW_ATTENTION_WEIGHT = 1.0
CW_CRITICAL_ATTENTION_WEIGHT = 2.0
CW_ACCEPT_ABS_IMPROVEMENT_SEC = 0.005
CW_ACCEPT_REL_IMPROVEMENT_PCT = 0.005
CW_CHUNK_QUANTUM = 256
CW_MIN_CHUNK_SIZE = 512
CW_STRATEGY_VERSION = "lockstep_attn_proj_min512_v1"
GEMM_KERNEL_FIELD_ORDER = [
    "use",
    "layer",
    "forM",
    "forK",
    "forN",
    "npuM",
    "npuK",
    "npuN",
    "chunk_id",
    "config",
    "num_tiles",
    "fw_path",
    "tile_size",
    "col",
    "dtype",
]
CHUNK_SIZE_CANDIDATES = [512, 1024, 2048, 4096, 8192]
MAX_INFLIGHT = 3
MAX_SCHEDULE_CHUNKS = 8
MAX_SCHEDULE_CANDIDATES = 32
FORCH_INFLIGHT = -1
# [outer_cap, final_cap]; set either to -1 for exhaustive at that stage.
SEARCH_SPACE = [32, 4]
# When non-empty or commented out, gemm_chunkingS skips schedule-space exploration.
# FORCE_CHUNKING_SCHEDULE = [4096, 4096, 4096, 2048, 1024, 1024]
CHUNK_ID_EXPLORE_DESCEND = True
CHUNK_RECOVER = True
LAYER_REGRESSION_EPS_SEC = 0.1
LAYER_GUARD_DVFS_SETTLE_SEC = 15
GEMM_LAYER_BASELINE_DVFS_SLOWDOWN_PCT = 2.0
DEVICE_HURISTIC = [
    {
        "name": "krackanp_ai7_350",
        "tokens": ["RYZEN AI 7 350", "RADEON 860M"],
        "npuM_num": 7,
        "npuM_den": 8,
        "inflight_threads": 2,
    },
    {
        "name": "strixP_hx370_890m",
        "tokens": ["RYZEN AI 9 HX 370", "RADEON 890M"],
        "npuM_num": 1,
        "npuM_den": 2,
        "inflight_threads": 2,
    },
    {
        "name": "strixH_ai_max_395_8060s",
        "tokens": ["RYZEN AI MAX+ 395", "RADEON 8060S"],
        "npuM_num": 1,
        "npuM_den": 8,
        "inflight_threads": 3,
    }
]
_DEVICE_HURISTIC_CACHE = None
DEFAULT_CONFIG_PROFILE = "strixP"


class TuningAbortError(RuntimeError):
    pass


def _layer_guard_wait_sec(wait_sec, recovery_active=False):
    wait = max(0, int(wait_sec))
    if recovery_active:
        return 0
    return wait

models_config = [
    {
        "id": "gemma",
        "name": "Gemma 2B",
        "script": "gemma_w4a16_model.py",
        "config": "",
        "MAX_CTX": 8192,
        "MAX_K_SPLIT": 16384,
        "CHUNKING_SPLIT_K": True,
        "fallback_profile": "strixH",
        "profile_configs": {
            "krackanP": "configs_krackanP_gemma.json5",
            "strixP": "configs_strixP_gemma.json5",
            "strixH": "configs_strixH_gemma.json5",
        },
        "layers": [
            {"name": "qkv",  "K": 2048,  "N": 2560,  "gemm_split": "M",  "gemv_split": "K", "gemv_disable_npu_tuning": True},
            {"name": "o",    "K": 2048,  "N": 2048,  "gemm_split": "M",  "gemv_split": "K", "gemv_disable_npu_tuning": False},
            {"name": "up",   "K": 2048,  "N": 16384, "gemm_split": "M",  "gemv_split": "K", "gemv_disable_npu_tuning": False},
            {"name": "gate", "K": 2048,  "N": 16384, "gemm_split": "M",  "gemv_split": "K", "gemv_disable_npu_tuning": False},
            {"name": "down", "K": 16384, "N": 2048,  "gemm_split": "K",  "gemv_split": "K", "gemv_disable_npu_tuning": False}
        ]
    },
    {
        "id": "llama3_8b",
        "name": "Llama3 8B",
        "script": "llama3_8b_w4a16_model.py",
        "config": "",
        "MAX_CTX": 16384,
        "MAX_K_SPLIT": 14336,
        "CHUNKING_SPLIT_K": False,
        "fallback_profile": "strixP",
        "profile_configs": {
            "krackanP": "configs_krackanP_llama3_8b.json5",
            "strixP": "configs_strixP_llama3_8b.json5",
            "strixH": "configs_strixH_llama3_8b.json5",
        },
        "layers": [
            {"name": "q",    "K": 4096,  "N": 4096,  "gemm_split": "M",  "gemv_split": "K", "gemv_disable_npu_tuning": False},
            {"name": "k",    "K": 4096,  "N": 1024,  "gemm_split": "M",  "gemv_split": "M", "gemv_disable_npu_tuning": True},
            {"name": "v",    "K": 4096,  "N": 1024,  "gemm_split": "M",  "gemv_split": "M", "gemv_disable_npu_tuning": True},
            {"name": "o",    "K": 4096,  "N": 4096,  "gemm_split": "M",  "gemv_split": "K", "gemv_disable_npu_tuning": False},
            {"name": "up",   "K": 4096,  "N": 14336, "gemm_split": "M",  "gemv_split": "K", "gemv_disable_npu_tuning": False},
            {"name": "gate", "K": 4096,  "N": 14336, "gemm_split": "M",  "gemv_split": "K", "gemv_disable_npu_tuning": False},
            {"name": "down", "K": 14336, "N": 4096,  "gemm_split": "MK", "gemv_split": "K", "gemv_disable_npu_tuning": False}
        ]
    },
    {
        "id": "llama3_70b",
        "name": "Llama3 70B",
        "script": "llama3_70b_w4a16_model.py",
        "config": "",
        "MAX_CTX": 16384,
        "MAX_K_SPLIT": 22528,
        "CHUNKING_SPLIT_K": True,
        "fallback_profile": "strixP",
        "profile_configs": {
            "krackanP": "configs_krackanP_llama3_70b.json5",
            "strixP": "configs_strixP_llama3_70b.json5",
            "strixH": "configs_strixH_llama3_70b.json5",
        },
        "layers": [
            {"name": "q",    "K": 8192,  "N": 8192,  "gemm_split": "M",  "gemv_split": "K", "gemv_disable_npu_tuning": False},
            {"name": "k",    "K": 8192,  "N": 1024,  "gemm_split": "M",  "gemv_split": "K", "gemv_disable_npu_tuning": True},
            {"name": "v",    "K": 8192,  "N": 1024,  "gemm_split": "M",  "gemv_split": "K", "gemv_disable_npu_tuning": True},
            {"name": "o",    "K": 8192,  "N": 8192,  "gemm_split": "M",  "gemv_split": "K", "gemv_disable_npu_tuning": False},
            {"name": "up",   "K": 8192,  "N": 28672, "gemm_split": "M",  "gemv_split": "K", "gemv_disable_npu_tuning": False},
            {"name": "gate", "K": 8192,  "N": 28672, "gemm_split": "M",  "gemv_split": "K", "gemv_disable_npu_tuning": False},
            {"name": "down", "K": 28672, "N": 8192,  "gemm_split": "K", "gemv_split": "K", "gemv_disable_npu_tuning": False}
        ]
    },
    {
        "id": "qwen14b",
        "name": "Qwen2.5 14B",
        "script": "qwen25_14b_w4a16_model.py",
        "config": "",
        "MAX_CTX": 16384,
        "MAX_K_SPLIT": 13824,
        "CHUNKING_SPLIT_K": True,
        "fallback_profile": "strixP",
        "profile_configs": {
            "krackanP": "configs_krackanP_qwen25_14b.json5",
            "strixP": "configs_strixP_qwen25_14b.json5",
            "strixH": "configs_strixH_qwen25_14b.json5",
        },
        "layers": [
            {"name": "q",    "K": 5120,  "N": 5120,  "gemm_split": "M", "gemv_split": "K", "gemv_disable_npu_tuning": True},
            {"name": "k",    "K": 5120,  "N": 1024,  "gemm_split": "M", "gemv_split": "K", "gemv_disable_npu_tuning": True},
            {"name": "v",    "K": 5120,  "N": 1024,  "gemm_split": "M", "gemv_split": "K", "gemv_disable_npu_tuning": True},
            {"name": "o",    "K": 5120,  "N": 5120,  "gemm_split": "M", "gemv_split": "K", "gemv_disable_npu_tuning": True},
            {"name": "up",   "K": 5120,  "N": 13824, "gemm_split": "M", "gemv_split": "K", "gemv_disable_npu_tuning": True},
            {"name": "gate", "K": 5120,  "N": 13824, "gemm_split": "M", "gemv_split": "K", "gemv_disable_npu_tuning": True},
            {"name": "down", "K": 13824, "N": 5120,  "gemm_split": "M", "gemv_split": "K", "gemv_disable_npu_tuning": True}
        ]
    },
    {
        "id": "phi3.5_3.8b",
        "name": "Phi-3.5 3.8B",
        "script": "phi35_3.8b_w4a16_model.py",
        "config": "",
        "MAX_CTX": 16384,
        "MAX_K_SPLIT": 8192,
        "CHUNKING_SPLIT_K": False,
        "fallback_profile": "strixP",
        "profile_configs": {
            "krackanP": "configs_krackanP_phi3.5_3.8b.json5",
            "strixP": "configs_strixP_phi3.5_3.8b.json5",
            "strixH": "configs_strixH_phi3.5_3.8b.json5",
        },
        "layers": [
            {"name": "q",    "K": 3072, "N": 3072, "gemm_split": "M", "gemv_split": "K", "gemv_disable_npu_tuning": False},
            {"name": "k",    "K": 3072, "N": 3072, "gemm_split": "M", "gemv_split": "K", "gemv_disable_npu_tuning": False},
            {"name": "v",    "K": 3072, "N": 3072, "gemm_split": "M", "gemv_split": "K", "gemv_disable_npu_tuning": False},
            {"name": "o",    "K": 3072, "N": 3072, "gemm_split": "M", "gemv_split": "K", "gemv_disable_npu_tuning": False},
            {"name": "up",   "K": 3072, "N": 8192, "gemm_split": "M", "gemv_split": "K", "gemv_disable_npu_tuning": False},
            {"name": "gate", "K": 3072, "N": 8192, "gemm_split": "M", "gemv_split": "K", "gemv_disable_npu_tuning": False},
            {"name": "down", "K": 8192, "N": 3072, "gemm_split": "M", "gemv_split": "K", "gemv_disable_npu_tuning": False}
        ]
    }
]


def resolve_gemm_split_mode(layer_cfg=None):
    if layer_cfg and "gemm_split" in layer_cfg:
        mode = str(layer_cfg["gemm_split"]).strip().upper()
        if mode in ("M", "K", "MK"):
            return mode
    return "M"

def resolve_chunking_split_k_enabled(model_conf=None):
    if not isinstance(model_conf, dict):
        return False
    return _safe_bool(model_conf.get("CHUNKING_SPLIT_K", False), False)

def _model_max_k_split(model_conf):
    if not isinstance(model_conf, dict):
        return 0
    max_k = _safe_int(model_conf.get("MAX_K_SPLIT", 0), 0)
    return int(max_k) if int(max_k) > 0 else 0

def _model_max_ctx(model_conf):
    if not isinstance(model_conf, dict):
        return 0
    max_ctx = _safe_int(model_conf.get("MAX_CTX", 0), 0)
    return int(max_ctx) if int(max_ctx) > 0 else 0

def _k_tuning_upper_bound(model_conf, for_k):
    for_k = max(0, _safe_int(for_k, 0))
    max_k = _model_max_k_split(model_conf)
    if max_k <= 0:
        return int(for_k)
    return int(min(for_k, max_k))

def _cap_k_tuning_value(model_conf, for_k, candidate):
    upper = _k_tuning_upper_bound(model_conf, for_k)
    cand = _safe_int(candidate, 0)
    return int(max(0, min(cand, upper)))

DTYPE_BASE = "bf16_int4AWQ_bf16"
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_BUILD_DIR = _PROJECT_ROOT / "build" / "py" / "unified_llm_w4a16"
if _BUILD_DIR.exists():
    build_dir_str = str(_BUILD_DIR)
    if build_dir_str not in sys.path:
        sys.path.insert(0, build_dir_str)

_HETERO_BACKEND_MODULE = None

def _get_prompt_sizes_for_mode(mode, model_conf=None):
    sizes = list(PROMPT_SIZES)
    if len(sizes) <= 1:
        trimmed = sizes
    elif mode == "gemm":
        trimmed = sizes
    elif mode in ("gemm_chunking", "gemm_chunkingS", "gemm_CW"):
        trimmed = sizes[1:]
    else:
        trimmed = sizes

    max_ctx = _model_max_ctx(model_conf)
    if max_ctx <= 0:
        return trimmed

    capped = []
    seen = set()
    for size in trimmed:
        effective_size = int(min(int(size), max_ctx))
        if effective_size <= 0 or effective_size in seen:
            continue
        capped.append(effective_size)
        seen.add(effective_size)
    return capped


def _normalize_chunk_schedule(schedule):
    if not isinstance(schedule, (list, tuple)):
        return []
    normalized = []
    for item in schedule:
        value = _safe_int(item, 0)
        if value > 0:
            normalized.append(int(value))
    return normalized


def _format_chunk_schedule(schedule):
    normalized = _normalize_chunk_schedule(schedule)
    return "[" + ", ".join(str(v) for v in normalized) + "]"


def _chunk_schedule_key(schedule):
    normalized = _normalize_chunk_schedule(schedule)
    if not normalized:
        return ""
    return "x".join(str(v) for v in normalized)


def _get_forced_chunking_schedule():
    return _normalize_chunk_schedule(globals().get("FORCE_CHUNKING_SCHEDULE", []))


def _get_inflight_candidates(num_chunks):
    max_inflight = _safe_int(globals().get("MAX_INFLIGHT", 0), 0)
    capped_num_chunks = int(num_chunks)
    if max_inflight > 0:
        capped_num_chunks = min(capped_num_chunks, int(max_inflight))
    if capped_num_chunks <= 0:
        return []
    return list(range(1, capped_num_chunks + 1))


def _cw_get_device_policy(rule=None):
    selected_rule = rule if isinstance(rule, dict) else _get_device_heuristic_rule()
    inflight_threads = max(
        1,
        _safe_int(
            selected_rule.get(
                "inflight_threads",
                selected_rule.get("gpu_ingest_threads", 1),
            ),
            1,
        ),
    )
    max_inflight = _safe_int(globals().get("MAX_INFLIGHT", 0), 0)
    if max_inflight > 0:
        inflight_threads = min(int(inflight_threads), int(max_inflight))
    return {
        "rule_name": str(selected_rule.get("name", "unknown")),
        "inflight_threads": max(1, int(inflight_threads)),
    }


def _cw_target_inflight_for_rule(rule=None):
    return int(_cw_get_device_policy(rule).get("inflight_threads", 1))


def _cw_resolve_effective_inflight(num_chunks, forced_inflight=None, rule=None):
    num_chunks = int(num_chunks)
    if num_chunks <= 1:
        return None

    max_inflight = _safe_int(globals().get("MAX_INFLIGHT", 0), 0)
    capped_upper = int(num_chunks)
    if max_inflight > 0:
        capped_upper = min(capped_upper, int(max_inflight))
    if capped_upper <= 0:
        return None

    forced = _safe_int(forced_inflight, -1)
    if forced > 0:
        if forced <= capped_upper and num_chunks % forced == 0:
            return int(forced)
        return None

    required_inflight = int(_cw_target_inflight_for_rule(rule))
    if required_inflight <= 0 or required_inflight > capped_upper:
        return None
    if num_chunks % required_inflight != 0:
        return None
    return int(required_inflight)


def get_valid_chunk_schedules(prompt_len, max_chunks=MAX_SCHEDULE_CHUNKS, max_candidates=MAX_SCHEDULE_CANDIDATES):
    """
    Build coarse schedule candidates that exactly sum to prompt_len.
    Schedules are generated in non-increasing chunk order to avoid permutations.
    """
    prompt_len = int(prompt_len)
    if prompt_len <= 0:
        return []

    candidates = sorted(
        [int(c) for c in CHUNK_SIZE_CANDIDATES if int(c) > 0 and int(c) <= prompt_len],
        reverse=True,
    )
    if not candidates:
        return []

    results = []

    def _dfs(remaining, start_idx, acc):
        if len(results) >= int(max_candidates):
            return
        if remaining == 0:
            if 1 < len(acc) <= int(max_chunks):
                results.append(list(acc))
            return
        if len(acc) >= int(max_chunks):
            return

        for idx in range(start_idx, len(candidates)):
            chunk = candidates[idx]
            if chunk > remaining:
                continue
            if chunk == prompt_len:
                continue
            acc.append(chunk)
            _dfs(remaining - chunk, idx, acc)
            acc.pop()
            if len(results) >= int(max_candidates):
                return

    _dfs(prompt_len, 0, [])
    # Deterministic order: fewer chunks first, then larger front-loaded schedules.
    results.sort(key=lambda s: (len(s), tuple([-v for v in s])))
    return results


def _cw_schedule_proxy_metrics(chunk_plan, inflight):
    plan = _normalize_chunk_schedule(chunk_plan)
    inflight = max(1, _safe_int(inflight, 1))
    if not plan:
        return {
            "slot_projection_proxy": [],
            "slot_attention_proxy": [],
            "projection_proxy_spread": float("inf"),
            "attention_proxy_spread": float("inf"),
            "weighted_proxy_total": float("inf"),
            "distinct_chunk_sizes": 0,
            "chunk_size_transitions": 0,
        }

    slot_projection_proxy = [0.0 for _ in range(inflight)]
    slot_attention_proxy = [0.0 for _ in range(inflight)]
    start_pos = 0
    for chunk_id, chunk_len in enumerate(plan):
        slot_id = int(chunk_id) % int(inflight)
        chunk_len = int(chunk_len)
        slot_projection_proxy[slot_id] += float(chunk_len)
        slot_attention_proxy[slot_id] += float(chunk_len * (start_pos + chunk_len))
        start_pos += int(chunk_len)

    projection_proxy_spread = max(slot_projection_proxy) - min(slot_projection_proxy) if slot_projection_proxy else float("inf")
    attention_proxy_spread = max(slot_attention_proxy) - min(slot_attention_proxy) if slot_attention_proxy else float("inf")
    distinct_chunk_sizes = len(set(plan))
    chunk_size_transitions = sum(1 for idx in range(1, len(plan)) if int(plan[idx]) != int(plan[idx - 1]))
    return {
        "slot_projection_proxy": [float(value) for value in slot_projection_proxy],
        "slot_attention_proxy": [float(value) for value in slot_attention_proxy],
        "projection_proxy_spread": float(projection_proxy_spread),
        "attention_proxy_spread": float(attention_proxy_spread),
        "weighted_proxy_total": float((4.0 * float(attention_proxy_spread)) + float(projection_proxy_spread)),
        "distinct_chunk_sizes": int(distinct_chunk_sizes),
        "chunk_size_transitions": int(chunk_size_transitions),
    }


def _cw_schedule_candidate_rank_key(candidate_spec):
    schedule = _normalize_chunk_schedule(candidate_spec.get("chunk_schedule"))
    front_chunk = int(schedule[0]) if schedule else 0
    return (
        float(candidate_spec.get("attention_proxy_spread", float("inf"))),
        float(candidate_spec.get("projection_proxy_spread", float("inf"))),
        float(candidate_spec.get("weighted_proxy_total", float("inf"))),
        int(candidate_spec.get("distinct_chunk_sizes", len(set(schedule)) if schedule else 0)),
        int(candidate_spec.get("chunk_size_transitions", 0)),
        -int(front_chunk),
        _chunk_schedule_key(schedule),
    )


def _cw_best_schedule_for_chunk_count(prompt_len, num_chunks, inflight):
    prompt_len = int(prompt_len)
    num_chunks = int(num_chunks)
    inflight = max(1, int(inflight))
    if prompt_len <= 0 or num_chunks <= 1 or prompt_len % int(CW_CHUNK_QUANTUM) != 0:
        return None
    total_units = int(prompt_len // int(CW_CHUNK_QUANTUM))
    min_chunk_units = max(1, int(CW_MIN_CHUNK_SIZE) // int(CW_CHUNK_QUANTUM))
    if num_chunks > total_units:
        return None
    if num_chunks * min_chunk_units > total_units:
        return None

    best_spec = None
    best_key = None

    def _dfs(remaining_units, parts_left, max_units, acc_units):
        nonlocal best_spec, best_key
        if parts_left == 0:
            if remaining_units != 0:
                return
            schedule = [int(units) * int(CW_CHUNK_QUANTUM) for units in acc_units]
            metrics = _cw_schedule_proxy_metrics(schedule, inflight)
            candidate_spec = {
                "chunk_size": int(_schedule_fallback_chunk_size(schedule)),
                "chunk_schedule": list(schedule),
                "effective_inflight": int(inflight),
                "num_chunks": int(len(schedule)),
            }
            candidate_spec.update(metrics)
            candidate_key = _cw_schedule_candidate_rank_key(candidate_spec)
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_spec = candidate_spec
            return

        max_take = min(
            int(max_units),
            int(remaining_units - (parts_left - 1) * min_chunk_units),
        )
        min_take = max(
            min_chunk_units,
            int((remaining_units + parts_left - 1) // parts_left),
        )
        for take_units in range(max_take, min_take - 1, -1):
            next_remaining = int(remaining_units - take_units)
            if parts_left > 1 and next_remaining > int(take_units) * int(parts_left - 1):
                continue
            if parts_left > 1 and next_remaining < int(parts_left - 1) * min_chunk_units:
                continue
            acc_units.append(int(take_units))
            _dfs(next_remaining, parts_left - 1, take_units, acc_units)
            acc_units.pop()

    _dfs(total_units, num_chunks, total_units, [])
    return best_spec


def _cw_select_schedule_candidates_for_active_search(candidate_specs, max_pool=16):
    specs = []
    for spec in candidate_specs:
        schedule = _normalize_chunk_schedule(spec.get("chunk_schedule"))
        if len(schedule) <= 1:
            continue
        normalized = {
            "chunk_size": int(spec.get("chunk_size", _schedule_fallback_chunk_size(schedule))),
            "chunk_schedule": list(schedule),
            "effective_inflight": int(spec.get("effective_inflight", spec.get("inflight", 1))),
            "num_chunks": int(spec.get("num_chunks", len(schedule))),
        }
        if "attention_proxy_spread" in spec and "projection_proxy_spread" in spec:
            normalized["slot_projection_proxy"] = list(spec.get("slot_projection_proxy", []))
            normalized["slot_attention_proxy"] = list(spec.get("slot_attention_proxy", []))
            normalized["projection_proxy_spread"] = float(spec.get("projection_proxy_spread", float("inf")))
            normalized["attention_proxy_spread"] = float(spec.get("attention_proxy_spread", float("inf")))
            normalized["weighted_proxy_total"] = float(spec.get("weighted_proxy_total", float("inf")))
            normalized["distinct_chunk_sizes"] = int(spec.get("distinct_chunk_sizes", len(set(schedule))))
            normalized["chunk_size_transitions"] = int(spec.get("chunk_size_transitions", 0))
        else:
            normalized.update(_cw_schedule_proxy_metrics(schedule, normalized["effective_inflight"]))
        specs.append(normalized)

    specs.sort(key=_cw_schedule_candidate_rank_key)
    return specs[: max(1, int(max_pool))]


def _cw_generate_lockstep_candidate_specs(prompt_len, forced_inflight=None, max_chunks=MAX_SCHEDULE_CHUNKS):
    prompt_len = int(prompt_len)
    if prompt_len <= 0 or prompt_len % int(CW_CHUNK_QUANTUM) != 0:
        return []

    max_chunks = max(2, int(max_chunks))
    candidate_specs = []
    for num_chunks in range(2, max_chunks + 1):
        effective_inflight = _cw_resolve_effective_inflight(num_chunks, forced_inflight=forced_inflight)
        if effective_inflight is None:
            continue
        best_spec = _cw_best_schedule_for_chunk_count(prompt_len, num_chunks, effective_inflight)
        if best_spec is None:
            continue
        candidate_specs.append(best_spec)

    candidate_specs.sort(key=_cw_schedule_candidate_rank_key)
    return candidate_specs

def _schedule_shape_score(chunk_plan):
    plan = _normalize_chunk_schedule(chunk_plan)
    if not plan:
        return float("inf")
    num_chunks = len(plan)
    unique_chunks = len(set(plan))
    small_chunks = sum(1 for c in plan if int(c) <= 1024)
    changes = sum(1 for i in range(1, len(plan)) if int(plan[i]) != int(plan[i - 1]))
    # Lower is better: prefer fewer chunks, fewer small tails, and smoother schedules.
    return float(
        num_chunks * 1000
        + unique_chunks * 120
        + small_chunks * 80
        + changes * 40
        - int(plan[0]) / 16.0
    )

def _select_schedule_candidates_for_active_search(candidate_specs, max_pool=16):
    specs = []
    for spec in candidate_specs:
        schedule = _normalize_chunk_schedule(spec.get("chunk_schedule"))
        if len(schedule) <= 1:
            continue
        specs.append(
            {
                "chunk_size": int(spec.get("chunk_size", 0)),
                "chunk_schedule": list(schedule),
                "shape_score": _schedule_shape_score(schedule),
            }
        )
    if not specs:
        return []

    by_chunks = {}
    for spec in specs:
        by_chunks.setdefault(len(spec["chunk_schedule"]), []).append(spec)
    for bucket in by_chunks.values():
        bucket.sort(key=lambda s: (s["shape_score"], _chunk_schedule_key(s["chunk_schedule"])))

    selected = []
    seen = set()
    # First pass: keep one representative per chunk-count bucket for diversity.
    for chunk_count in sorted(by_chunks.keys()):
        top = by_chunks[chunk_count][0]
        key = _chunk_schedule_key(top["chunk_schedule"])
        if key not in seen:
            selected.append(top)
            seen.add(key)

    # Second pass: fill remaining slots with globally best-shaped schedules.
    remaining = sorted(
        [s for s in specs if _chunk_schedule_key(s["chunk_schedule"]) not in seen],
        key=lambda s: (s["shape_score"], _chunk_schedule_key(s["chunk_schedule"])),
    )
    for spec in remaining:
        if len(selected) >= int(max_pool):
            break
        selected.append(spec)
        seen.add(_chunk_schedule_key(spec["chunk_schedule"]))

    selected.sort(key=lambda s: (s["shape_score"], _chunk_schedule_key(s["chunk_schedule"])))
    return selected[: int(max_pool)]

def _normalize_cpu_model_name(model_name):
    return re.sub(r"\s+", " ", model_name.strip().upper())


def _detect_lscpu_model_name():
    try:
        result = subprocess.run(
            "lscpu",
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    for line in result.stdout.splitlines():
        if line.strip().startswith("Model name:"):
            parts = line.split(":", 1)
            if len(parts) > 1:
                return parts[1].strip()
    return ""


def _round_to_nearest_multiple(value, multiple):
    if multiple <= 0:
        return int(value)
    return int(round(float(value) / float(multiple)) * multiple)


def _get_device_heuristic_rule():
    global _DEVICE_HURISTIC_CACHE
    if _DEVICE_HURISTIC_CACHE is not None:
        return _DEVICE_HURISTIC_CACHE

    model_name = _normalize_cpu_model_name(_detect_lscpu_model_name())
    selected = None
    for rule in DEVICE_HURISTIC:
        tokens = [_normalize_cpu_model_name(str(t)) for t in rule.get("tokens", [])]
        if tokens and all(token in model_name for token in tokens):
            selected = rule
            break

    # Safe fallback if no device token match.
    if selected is None:
        selected = {
            "name": "fallback_half",
            "tokens": [],
            "npuM_num": 1,
            "npuM_den": 2,
        }

    _DEVICE_HURISTIC_CACHE = selected
    return _DEVICE_HURISTIC_CACHE


def _heuristic_npuM_for_split_k(forM):
    rule = _get_device_heuristic_rule()
    num = int(rule.get("npuM_num", 1))
    den = max(1, int(rule.get("npuM_den", 1)))
    raw = int((int(forM) * num) / den)
    rounded = _round_to_nearest_multiple(raw, 256)
    rounded = max(256, min(int(forM), int(rounded)))
    return int(rounded), str(rule.get("name", "unknown"))


def _detect_system_profile():
    cpu_model_name = _detect_lscpu_model_name()
    normalized = _normalize_cpu_model_name(cpu_model_name) if cpu_model_name else ""

    profile = ""
    mapping = [
        (("RYZEN AI 7 350", "RADEON 860M"), "krackanP"),
        (("RYZEN AI 9 HX 370", "RADEON 890M"), "strixP"),
        (("RYZEN AI MAX+ 395", "RADEON 8060S"), "strixH"),
    ]
    for tokens, candidate in mapping:
        if all(token in normalized for token in tokens):
            profile = candidate
            break
    return profile, cpu_model_name


def _resolve_model_config_path_for_platform(model_conf):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(script_dir, "configs")
    detected_profile, cpu_model_name = _detect_system_profile()
    profile_configs = model_conf.get("profile_configs", {})
    fallback_profile = str(model_conf.get("fallback_profile", DEFAULT_CONFIG_PROFILE))
    selected_profile = detected_profile if detected_profile in profile_configs else fallback_profile
    model_id = str(model_conf.get("id", "")).strip()
    if not model_id:
        raise KeyError(f"Missing model id for config resolution: {model_conf.get('name', 'unknown')}")
    filename = profile_configs.get(selected_profile)
    if not filename:
        filename = f"configs_{selected_profile}_{model_id}.json5"

    path = os.path.join(configs_dir, filename)
    return path, cpu_model_name, selected_profile


def _bootstrap_missing_config_with_submodel(model_conf, log_handle):
    config_path = model_conf["config"]
    if os.path.exists(config_path):
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = model_conf["script"]
    script_path = script_name if os.path.isabs(script_name) else os.path.join(script_dir, script_name)
    cmd = [sys.executable, script_path]

    log_print(
        f"Config not found at {config_path}. Running submodel once with no args to auto-create it...",
        log_handle,
    )
    try:
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
        if log_handle:
            log_handle.write("\n--- Submodel Bootstrap Output ---\n")
            log_handle.write(result.stdout)
            log_handle.write("\n--- End Submodel Bootstrap Output ---\n")
            log_handle.flush()
    except subprocess.TimeoutExpired as e:
        log_print("Submodel bootstrap timed out after 120s; checking whether config was created.", log_handle)
        if log_handle and e.stdout:
            log_handle.write("\n--- Submodel Bootstrap Output (Timeout) ---\n")
            log_handle.write(e.stdout)
            log_handle.write("\n--- End Submodel Bootstrap Output ---\n")
            log_handle.flush()
    except Exception as e:
        log_print(f"Submodel bootstrap failed to run: {e}", log_handle)

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config was not auto-created by submodel run: {config_path}"
        )

def clear_temp_artifacts(base_dir):
    """Delete tuner temp configs, chunk-recovery artifacts, and .log files."""
    removed = []
    root = os.path.abspath(base_dir)
    configs_dir = os.path.join(root, "configs")

    def _remove_pattern(directory, pattern):
        if not os.path.isdir(directory):
            return
        for path in sorted(glob.glob(os.path.join(directory, pattern))):
            if not os.path.isfile(path):
                continue
            try:
                os.remove(path)
                removed.append(path)
            except Exception:
                pass

    # Config temp files produced during tuning.
    _remove_pattern(configs_dir, "*.tmp.json5")
    _remove_pattern(configs_dir, "*_tmp.json5")
    _remove_pattern(configs_dir, "*.json5.tmp")
    _remove_pattern(configs_dir, "*.chunk_*_tmp.json5")

    # Chunk-recovery cache artifacts.
    _remove_pattern(configs_dir, "*.heuristics.json")
    _remove_pattern(configs_dir, "*.heuristics.json.tmp")

    # Log files in tuner directory.
    _remove_pattern(root, "*.log")

    return removed


def get_max_cpu_threads():
    """Query lscpu to get the maximum number of CPU threads."""
    try:
        # Run lscpu
        result = subprocess.run("lscpu", shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.strip().startswith("CPU(s):"):
                    # Format: "CPU(s):              32"
                    parts = line.split(":")
                    if len(parts) > 1:
                        val = int(parts[1].strip())
                        return val
    except Exception as e:
        print(f"Error querying lscpu: {e}")
    # Fallback default
    return 4

def log_print(message, file_handle=None):
    """Print to stdout and append to log file."""
    print(message)
    if file_handle:
        file_handle.write(message + "\n")
        file_handle.flush()
    else:
        with open(LOG_FILE, 'a') as f:
            f.write(message + "\n")

def remove_comments(json_str):
    """Removing comments from JSON5 to parse with standard json lib."""
    return re.sub(r'//.*', '', json_str)

def load_config(model_conf):
    with open(model_conf["config"], 'r') as f:
        content = f.read()
    
    try:
        data = json.loads(remove_comments(content))
        return data
    except Exception as e:
        log_print(f"Error parsing config: {e}")
        return None

def _safe_int(val, default=0):
    try:
        return int(val)
    except Exception:
        return default

def _get_search_space_limits():
    default_outer = 16
    default_final = 4
    raw = SEARCH_SPACE
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raw = [default_outer, default_final]

    def _normalize_limit(value, default_value):
        parsed = _safe_int(value, default_value)
        if parsed == -1:
            return -1
        if parsed <= 0:
            return int(default_value)
        return int(parsed)

    outer_cap = _normalize_limit(raw[0], default_outer)
    final_cap = _normalize_limit(raw[1], default_final)
    return int(outer_cap), int(final_cap)

def _safe_bool(val, default=False):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        lowered = val.strip().lower()
        if lowered in ("1", "true", "yes", "y", "on"):
            return True
        if lowered in ("0", "false", "no", "n", "off"):
            return False
    return bool(default)

def _is_grouped_kernels_gemm_chunked(kernels):
    return (
        isinstance(kernels, list)
        and len(kernels) > 0
        and isinstance(kernels[0], dict)
        and isinstance(kernels[0].get("kernels"), list)
    )

def _extract_chunk_scalar(raw):
    if isinstance(raw, (list, tuple)):
        for item in raw:
            value = _safe_int(item, 0)
            if value > 0:
                return int(value)
        return 0
    return _safe_int(raw, 0)

def _schedule_fallback_chunk_size(schedule, default=0):
    normalized = _normalize_chunk_schedule(schedule)
    if normalized:
        return int(min(normalized))
    return max(0, _safe_int(default, 0))

def _extract_chunk_schedule_from_map(payload):
    if not isinstance(payload, dict):
        return []
    raw = payload.get(
        "hetero_chunk_size_schedule",
        payload.get(
            "gpu_chunk_size_schedule",
            payload.get("gpu_chunk_schedule", payload.get("gpu_chunk_shedule", payload.get("chunk_schedule", []))),
        ),
    )
    return _normalize_chunk_schedule(raw)

def _is_chunking_scheduled_enabled(data):
    return _safe_bool(data.get("chunking_scheduled", False), False)

def _group_chunk_size(group):
    if not isinstance(group, dict):
        return 0
    chunk_size = _extract_chunk_scalar(
        group.get(
            "hetero_chunk_size",
            group.get("gpu_chunk_size", group.get("chunk_size", 0)),
        )
    )
    if chunk_size > 0:
        return int(chunk_size)
    return int(_schedule_fallback_chunk_size(_group_chunk_schedule(group), 0))


def _group_chunk_schedule(group):
    if not isinstance(group, dict):
        return []
    return _extract_chunk_schedule_from_map(group)

def _group_inflight(group):
    if not isinstance(group, dict):
        return 1
    return max(
        1,
        _safe_int(
            group.get(
                "hetero_inflight",
                group.get("gpu_chunking_inflight", group.get("chunking_inflight", group.get("inflight", 1))),
            ),
            1,
        ),
    )

def _group_prompt_len(group):
    if not isinstance(group, dict):
        return 0
    return _safe_int(group.get("prompt_len", 0), 0)


def _normalize_stage_bubbles(stage_bubbles):
    normalized = []
    if not isinstance(stage_bubbles, list):
        return normalized

    stage_order = {name: idx for idx, name in enumerate(CW_STAGE_ORDER)}
    for spec in stage_bubbles:
        if not isinstance(spec, dict):
            continue
        stage = str(spec.get("stage", "")).strip().upper()
        if stage not in ("G1", "G2"):
            continue
        delay_us = max(0, _safe_int(spec.get("delay_us", 0), 0))
        if delay_us <= 0:
            continue
        normalized.append(
            {
                "chunk_id": _safe_int(spec.get("chunk_id", -1), -1),
                "layer_id": _safe_int(spec.get("layer_id", -1), -1),
                "stage": stage,
                "delay_us": delay_us,
            }
        )

    normalized.sort(
        key=lambda item: (
            int(item.get("chunk_id", -1)),
            int(item.get("layer_id", -1)),
            stage_order.get(str(item.get("stage", "")), len(stage_order) + 1),
        )
    )
    return normalized

def _select_grouped_chunked_entry(
    groups,
    desired_chunk_size=0,
    desired_inflight=0,
    desired_prompt_len=0,
    desired_schedule=None,
    prefer_schedule=False,
):
    if not isinstance(groups, list) or len(groups) == 0:
        return None

    chunk_size_filter = _safe_int(desired_chunk_size, 0)
    inflight_filter = _safe_int(desired_inflight, 0)
    prompt_len_filter = _safe_int(desired_prompt_len, 0)
    schedule_filter = _normalize_chunk_schedule(desired_schedule)
    require_exact_schedule = len(schedule_filter) > 0

    ranked = []
    for idx, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        group_chunk = _group_chunk_size(group)
        group_inflight = _group_inflight(group)
        group_prompt_len = _group_prompt_len(group)
        group_schedule = _group_chunk_schedule(group)
        group_has_schedule = len(group_schedule) > 1

        if chunk_size_filter > 0 and int(group_chunk) != int(chunk_size_filter):
            continue
        if inflight_filter > 0 and int(group_inflight) != int(inflight_filter):
            continue

        if prompt_len_filter > 0:
            if int(group_prompt_len) == int(prompt_len_filter):
                prompt_rank = 0
            elif int(group_prompt_len) <= 0:
                prompt_rank = 1
            else:
                continue
        else:
            prompt_rank = 0 if int(group_prompt_len) <= 0 else 1

        if require_exact_schedule:
            if group_schedule != schedule_filter:
                continue
            schedule_rank = 0
        else:
            if prefer_schedule:
                schedule_rank = 0 if group_has_schedule else 1
            else:
                schedule_rank = 0 if not group_has_schedule else 1

        ranked.append((prompt_rank, schedule_rank, idx, group))

    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return ranked[0][3]

def _normalize_chunked_group_meta(group, desired_chunk_size, desired_inflight, desired_prompt_len, chunk_schedule=None):
    """Ensure grouped chunked-kernel metadata is explicit and up to date."""
    if not isinstance(group, dict):
        return
    if desired_prompt_len > 0:
        group["prompt_len"] = int(desired_prompt_len)
    elif "prompt_len" not in group:
        group["prompt_len"] = 0
    schedule = _normalize_chunk_schedule(chunk_schedule)
    if len(schedule) > 1:
        group["hetero_chunk_size_schedule"] = list(schedule)
        group.pop("hetero_chunk_size", None)
    else:
        group.pop("hetero_chunk_size_schedule", None)
        if desired_chunk_size > 0:
            group["hetero_chunk_size"] = int(desired_chunk_size)
    group["hetero_inflight"] = int(max(1, desired_inflight))
    if not isinstance(group.get("kernels"), list):
        group["kernels"] = []
    if "stage_bubbles" in group:
        group["stage_bubbles"] = _normalize_stage_bubbles(group.get("stage_bubbles"))
    # Keep metadata fields before kernels in saved JSON for readability.
    ordered = {}
    for key in (
        "prompt_len",
        "hetero_chunk_size",
        "hetero_chunk_size_schedule",
        "hetero_inflight",
        "stage_bubbles",
        "kernels",
    ):
        if key in group:
            ordered[key] = group[key]
    for key, value in group.items():
        if key not in ordered:
            ordered[key] = value
    group.clear()
    group.update(ordered)


def _get_chunked_group_ref(
    data,
    create=False,
    reset=False,
    chunk_size=None,
    inflight=None,
    prompt_len=None,
    chunk_schedule=None,
):
    desired_chunk_size, desired_inflight = _get_active_chunking_settings(data)
    desired_chunk_schedule = _normalize_chunk_schedule(chunk_schedule)
    if chunk_size is not None:
        desired_chunk_size = int(chunk_size)
    elif desired_chunk_size <= 0 and desired_chunk_schedule:
        desired_chunk_size = _schedule_fallback_chunk_size(desired_chunk_schedule)
    if inflight is not None:
        desired_inflight = int(inflight)
    desired_inflight = max(1, desired_inflight)
    desired_prompt_len = _safe_int(prompt_len if prompt_len is not None else data.get("prompt_len", 0), 0)
    if desired_chunk_size <= 0 and desired_chunk_schedule:
        desired_chunk_size = _schedule_fallback_chunk_size(desired_chunk_schedule)
    prefer_schedule = bool(desired_chunk_schedule) or _is_chunking_scheduled_enabled(data)

    kernels = data.get("kernels_gemm_chunked")
    if _is_grouped_kernels_gemm_chunked(kernels):
        selected_group = _select_grouped_chunked_entry(
            kernels,
            desired_chunk_size=desired_chunk_size,
            desired_inflight=desired_inflight,
            desired_prompt_len=desired_prompt_len,
            desired_schedule=desired_chunk_schedule if desired_chunk_schedule else None,
            prefer_schedule=prefer_schedule,
        )
        if selected_group is None and (not create):
            selected_group = _select_grouped_chunked_entry(
                kernels,
                desired_chunk_size=0,
                desired_inflight=desired_inflight,
                desired_prompt_len=desired_prompt_len,
                desired_schedule=desired_chunk_schedule if desired_chunk_schedule else None,
                prefer_schedule=prefer_schedule,
            )
        if selected_group is None and create:
            selected_group = {"kernels": [], "stage_bubbles": []}
            _normalize_chunked_group_meta(
                selected_group,
                desired_chunk_size,
                desired_inflight,
                desired_prompt_len,
                chunk_schedule=desired_chunk_schedule,
            )
            kernels.append(selected_group)
        if selected_group is None:
            return None
        if create:
            effective_chunk_schedule = (
                list(desired_chunk_schedule)
                if desired_chunk_schedule
                else ([] if desired_chunk_size > 0 else list(_group_chunk_schedule(selected_group)))
            )
            _normalize_chunked_group_meta(
                selected_group,
                desired_chunk_size,
                desired_inflight,
                desired_prompt_len,
                chunk_schedule=effective_chunk_schedule,
            )
        if reset:
            selected_group["kernels"] = []
        return selected_group

    if isinstance(kernels, list):
        if not create:
            return None
        grouped = {"kernels": [] if reset else kernels, "stage_bubbles": []}
        _normalize_chunked_group_meta(
            grouped,
            desired_chunk_size,
            desired_inflight,
            desired_prompt_len,
            chunk_schedule=desired_chunk_schedule,
        )
        data["kernels_gemm_chunked"] = [grouped]
        return grouped

    if create:
        grouped = {"kernels": [], "stage_bubbles": []}
        _normalize_chunked_group_meta(
            grouped,
            desired_chunk_size,
            desired_inflight,
            desired_prompt_len,
            chunk_schedule=desired_chunk_schedule,
        )
        data["kernels_gemm_chunked"] = [grouped]
        return grouped

    return None

def _get_active_chunking_settings(data):
    chunk_size = 0
    inflight = 1
    chunk_schedule = []

    # GPU chunk metadata should only drive active settings when running in GPU mode.
    if str(data.get("heterogeneity", "")).lower() == "gpu":
        gpu_cfg = data.get("gpu_chunking")
        if isinstance(gpu_cfg, dict):
            chunk_size = _extract_chunk_scalar(gpu_cfg.get("gpu_chunk_size", gpu_cfg.get("chunk_size", 0)))
            chunk_schedule = _extract_chunk_schedule_from_map(gpu_cfg)
            if chunk_size <= 0 and chunk_schedule:
                chunk_size = _schedule_fallback_chunk_size(chunk_schedule)
            inflight = max(
                1,
                _safe_int(gpu_cfg.get("gpu_chunking_inflight", gpu_cfg.get("chunking_inflight", gpu_cfg.get("inflight", 1))), 1),
            )

    if chunk_size <= 0 and not isinstance(data.get("chunking"), bool):
        chunk_size = _safe_int(data.get("chunking", 0), 0)

    inflight = max(1, _safe_int(data.get("chunking_inflight", inflight), inflight))

    kernels_grouped = data.get("kernels_gemm_chunked")
    if _is_grouped_kernels_gemm_chunked(kernels_grouped):
        current_prompt_len = _safe_int(data.get("prompt_len", 0), 0)
        selected_group = _select_grouped_chunked_entry(
            kernels_grouped,
            desired_chunk_size=chunk_size,
            desired_inflight=inflight,
            desired_prompt_len=current_prompt_len,
            desired_schedule=chunk_schedule if chunk_schedule else None,
            prefer_schedule=_is_chunking_scheduled_enabled(data),
        )
        if selected_group is None:
            selected_group = _select_grouped_chunked_entry(
                kernels_grouped,
                desired_chunk_size=0,
                desired_inflight=0,
                desired_prompt_len=current_prompt_len,
                desired_schedule=chunk_schedule if chunk_schedule else None,
                prefer_schedule=_is_chunking_scheduled_enabled(data),
            )
        if selected_group is not None:
            chunk_size = _group_chunk_size(selected_group)
            inflight = _group_inflight(selected_group)
            chunk_schedule = _group_chunk_schedule(selected_group)

    if chunk_size <= 0:
        chunk_size = _safe_int(data.get("prompt_len", data.get("npu_dim", 0)), 0)
    if chunk_size <= 0 and chunk_schedule:
        chunk_size = _schedule_fallback_chunk_size(chunk_schedule)

    return chunk_size, inflight

def _get_active_chunking_schedule(data):
    chunk_size, inflight = _get_active_chunking_settings(data)
    current_prompt_len = _safe_int(data.get("prompt_len", 0), 0)
    schedule = []

    if str(data.get("heterogeneity", "")).lower() == "gpu":
        gpu_cfg = data.get("gpu_chunking")
        if isinstance(gpu_cfg, dict):
            schedule = _extract_chunk_schedule_from_map(gpu_cfg)

    kernels_grouped = data.get("kernels_gemm_chunked")
    if _is_grouped_kernels_gemm_chunked(kernels_grouped):
        selected_group = _select_grouped_chunked_entry(
            kernels_grouped,
            desired_chunk_size=chunk_size,
            desired_inflight=inflight,
            desired_prompt_len=current_prompt_len,
            desired_schedule=schedule if schedule else None,
            prefer_schedule=_is_chunking_scheduled_enabled(data),
        )
        if selected_group is not None:
            schedule = _group_chunk_schedule(selected_group)

    return _normalize_chunk_schedule(schedule)

def _get_chunked_kernels_ref(
    data,
    create=False,
    reset=False,
    chunk_size=None,
    inflight=None,
    prompt_len=None,
    chunk_schedule=None,
):
    group = _get_chunked_group_ref(
        data,
        create=create,
        reset=reset,
        chunk_size=chunk_size,
        inflight=inflight,
        prompt_len=prompt_len,
        chunk_schedule=chunk_schedule,
    )
    if group is None:
        kernels = data.get("kernels_gemm_chunked", [])
        if _is_grouped_kernels_gemm_chunked(kernels):
            return []
        return kernels if isinstance(kernels, list) else []
    if not isinstance(group.get("kernels"), list):
        group["kernels"] = []
    return group["kernels"]


def _get_chunked_stage_bubbles_ref(
    data,
    create=False,
    reset=False,
    chunk_size=None,
    inflight=None,
    prompt_len=None,
    chunk_schedule=None,
):
    group = _get_chunked_group_ref(
        data,
        create=create,
        reset=False,
        chunk_size=chunk_size,
        inflight=inflight,
        prompt_len=prompt_len,
        chunk_schedule=chunk_schedule,
    )
    if group is None:
        return []
    if reset or not isinstance(group.get("stage_bubbles"), list):
        group["stage_bubbles"] = []
    group["stage_bubbles"] = _normalize_stage_bubbles(group.get("stage_bubbles"))
    return group["stage_bubbles"]


def _set_chunked_stage_bubbles(
    data,
    stage_bubbles,
    chunk_size=None,
    inflight=None,
    prompt_len=None,
    chunk_schedule=None,
):
    group = _get_chunked_group_ref(
        data,
        create=True,
        reset=False,
        chunk_size=chunk_size,
        inflight=inflight,
        prompt_len=prompt_len,
        chunk_schedule=chunk_schedule,
    )
    if group is None:
        return
    group["stage_bubbles"] = _normalize_stage_bubbles(stage_bubbles)
    _normalize_chunked_group_meta(
        group,
        _group_chunk_size(group),
        _group_inflight(group),
        _group_prompt_len(group),
        chunk_schedule=_group_chunk_schedule(group),
    )

def _is_gemm_kernel_record(kernel):
    if not isinstance(kernel, dict):
        return False
    layer = str(kernel.get("layer", "")).strip()
    if not layer:
        return False
    for field in ("forM", "forK", "forN"):
        try:
            int(kernel.get(field))
        except Exception:
            return False
    return True

def _apply_chunking_format(data, chunk_size, inflight, prompt_len=None, chunk_schedule=None):
    data["chunking"] = True
    effective_prompt_len = _safe_int(prompt_len if prompt_len is not None else data.get("prompt_len", 0), 0)
    normalized_schedule = _normalize_chunk_schedule(chunk_schedule)
    if effective_prompt_len <= 0:
        if normalized_schedule:
            effective_prompt_len = int(sum(normalized_schedule))
        else:
            effective_prompt_len = int(chunk_size)
    data["prompt_len"] = int(effective_prompt_len)
    data.pop("npu_dim", None)
    data.pop("chunking_inflight", None)
    data["chunking_scheduled"] = bool(len(normalized_schedule) > 1)
    if str(data.get("heterogeneity", "")).lower() == "gpu":
        gpu_chunking = {
            "gpu_chunk_size": [int(chunk_size)],
            "gpu_chunking_inflight": int(inflight),
        }
        if len(normalized_schedule) > 1:
            gpu_chunking["gpu_chunk_size_schedule"] = list(normalized_schedule)
            gpu_chunking["gpu_chunk_shedule"] = list(normalized_schedule)
        data["gpu_chunking"] = gpu_chunking

def sort_kernels_gemv(data, layer_order=None):
    if not isinstance(data, dict):
        return
    kernels = data.get("kernels_gemv")
    if not isinstance(kernels, list):
        return
    order = layer_order or GEMV_LAYER_ORDER
    order_idx = {name: idx for idx, name in enumerate(order)}

    def _key(k):
        layer = str(k.get("layer", ""))
        return (
            order_idx.get(layer, len(order_idx) + 1000),
            layer,
            int(k.get("forK", 0)),
            int(k.get("forN", 0)),
        )

    kernels.sort(key=_key)

def _reorder_kernel_fields(kernel, ordered_fields):
    if not isinstance(kernel, dict):
        return
    ordered = {}
    for key in ordered_fields:
        if key in kernel:
            ordered[key] = kernel[key]
    for key, value in kernel.items():
        if key not in ordered:
            ordered[key] = value
    kernel.clear()
    kernel.update(ordered)

def sort_kernels_gemm_entries(data, collection_name, layer_order=None, include_chunk_id=False):
    if not isinstance(data, dict):
        return
    kernels = data.get(collection_name)
    if not isinstance(kernels, list):
        return

    order = layer_order or GEMM_LAYER_ORDER
    order_idx = {name: idx for idx, name in enumerate(order)}

    def _key(k, chunk_first=False):
        layer = str(k.get("layer", ""))
        if include_chunk_id and chunk_first:
            key = [
                int(k.get("chunk_id", -1)),
                order_idx.get(layer, len(order_idx) + 1000),
                layer,
                int(k.get("forM", 0)),
                int(k.get("forK", 0)),
                int(k.get("forN", 0)),
            ]
        elif collection_name == "kernels_gemm":
            # Keep GEMM flat, but order by prompt length (forM) first, then layer order.
            key = [
                int(k.get("forM", 0)),
                order_idx.get(layer, len(order_idx) + 1000),
                layer,
                int(k.get("forK", 0)),
                int(k.get("forN", 0)),
            ]
        else:
            key = [
                order_idx.get(layer, len(order_idx) + 1000),
                layer,
                int(k.get("forM", 0)),
                int(k.get("forK", 0)),
                int(k.get("forN", 0)),
            ]
        if include_chunk_id and collection_name != "kernels_gemm_chunked":
            key.append(int(k.get("chunk_id", -1)))
        return tuple(key)

    if collection_name == "kernels_gemm_chunked" and _is_grouped_kernels_gemm_chunked(kernels):
        for group in kernels:
            group_kernels = group.get("kernels")
            if isinstance(group_kernels, list):
                for kernel in group_kernels:
                    _reorder_kernel_fields(kernel, GEMM_KERNEL_FIELD_ORDER)
                group_kernels.sort(key=lambda k: _key(k, chunk_first=True))
        kernels.sort(
            key=lambda g: (
                _group_prompt_len(g),
                _group_chunk_size(g),
                _group_inflight(g),
                _chunk_schedule_key(_group_chunk_schedule(g)),
            )
        )
        return

    if collection_name in ("kernels_gemm", "kernels_gemm_chunked"):
        for kernel in kernels:
            _reorder_kernel_fields(kernel, GEMM_KERNEL_FIELD_ORDER)
    kernels.sort(key=lambda k: _key(k, chunk_first=(collection_name == "kernels_gemm_chunked")))

def sort_all_kernels(data):
    sort_kernels_gemm_entries(data, "kernels_gemm")
    sort_kernels_gemm_entries(data, "kernels_gemm_chunked", include_chunk_id=True)
    sort_kernels_gemv(data)

def _strip_temp_trace_fields(data):
    if not isinstance(data, dict):
        return
    data.pop("trace_output_path", None)
    data.pop("trace_run_tag", None)
    data.pop("trace_sync_stages", None)

def save_config(data, model_conf):
    _strip_temp_trace_fields(data)
    sort_all_kernels(data)
    with open(model_conf["config"], 'w') as f:
        json.dump(data, f, indent=4)

def save_config_via_temp(data, model_conf, temp_suffix=".tmp.json5"):
    _strip_temp_trace_fields(data)
    sort_all_kernels(data)
    temp_path = model_conf["config"] + temp_suffix
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(temp_path, model_conf["config"])

def get_chunk_heuristics_path(model_conf):
    return model_conf["config"] + ".heuristics.json"

def load_chunk_heuristics(model_conf, log_handle=None):
    if not CHUNK_RECOVER:
        return None
    path = get_chunk_heuristics_path(model_conf)
    if not os.path.exists(path):
        return {"timings": {}, "layer_states": {}}
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("heuristics root is not a dict")
        timings = data.get("timings", {})
        if not isinstance(timings, dict):
            timings = {}
        layer_states = data.get("layer_states", {})
        if not isinstance(layer_states, dict):
            layer_states = {}
        cw_trace_runs = data.get("cw_trace_runs", {})
        if not isinstance(cw_trace_runs, dict):
            cw_trace_runs = {}
        data["timings"] = timings
        data["layer_states"] = layer_states
        data["cw_trace_runs"] = cw_trace_runs
        log_print(
            f"Loaded chunk recovery heuristics from {path} "
            f"({len(timings)} timing entries, {len(layer_states)} layer states, "
            f"{len(cw_trace_runs)} CW trace runs).",
            log_handle,
        )
        return data
    except Exception as e:
        log_print(f"Failed to load chunk recovery heuristics from {path}: {e}. Starting fresh.", log_handle)
        return {"timings": {}, "layer_states": {}, "cw_trace_runs": {}}

def save_chunk_heuristics(model_conf, heuristics_state):
    if not CHUNK_RECOVER or heuristics_state is None:
        return
    path = get_chunk_heuristics_path(model_conf)
    temp_path = path + ".tmp"
    with open(temp_path, 'w') as f:
        json.dump(heuristics_state, f, indent=4, sort_keys=True)
    os.replace(temp_path, path)

def make_chunk_heuristic_key(stage, **kwargs):
    payload = {"stage": stage}
    payload.update(kwargs)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))

def get_chunk_cached_time(heuristics_state, key):
    if heuristics_state is None:
        return None
    timings = heuristics_state.get("timings", {})
    if key not in timings:
        return None
    try:
        return float(timings[key])
    except Exception:
        return None

def set_chunk_cached_time(model_conf, heuristics_state, key, value):
    if heuristics_state is None or not math.isfinite(value):
        return
    heuristics_state.setdefault("timings", {})[key] = float(value)
    save_chunk_heuristics(model_conf, heuristics_state)

def get_chunk_layer_state(heuristics_state, key):
    if heuristics_state is None:
        return None
    states = heuristics_state.get("layer_states", {})
    state = states.get(key)
    if not isinstance(state, dict):
        return None
    return json.loads(json.dumps(state))

def set_chunk_layer_state(model_conf, heuristics_state, key, kernel_state):
    if heuristics_state is None or not isinstance(kernel_state, dict):
        return
    heuristics_state.setdefault("layer_states", {})[key] = json.loads(json.dumps(kernel_state))
    save_chunk_heuristics(model_conf, heuristics_state)


def get_chunk_artifact_state(heuristics_state, namespace, key):
    if heuristics_state is None:
        return None
    bucket = heuristics_state.get(str(namespace), {})
    if not isinstance(bucket, dict):
        return None
    if key not in bucket:
        return None
    return json.loads(json.dumps(bucket[key]))


def set_chunk_artifact_state(model_conf, heuristics_state, namespace, key, payload):
    if heuristics_state is None:
        return
    heuristics_state.setdefault(str(namespace), {})[key] = json.loads(json.dumps(payload))
    save_chunk_heuristics(model_conf, heuristics_state)

def setup_baseline_config(data, _current_M):
    """Sets up the global parameters for tuning."""
    data["dummy_weights"] = bool(DUMMY_WEIGHTS)
    data["warmup"] = bool(WARMUP)
    data["gemv_driven_split_K"] = False 
    data["heterogeneity"] = "hetero"
    data["debug_verbosity"] = VERBOSITY 

    # Keep existing GEMV kernel enablement untouched.
    # GEMM tuning runs with --max-new-tokens 0, so mutating kernels_gemv here only causes
    # unexpected config churn (especially in default mode=all).

    # We do NOT clear GEMM kernels blindly, as we might have tuned previous sizes.
    # But we should ensure *currently targeted* baseline kernels are reset? 
    # Or just let `get_or_create` handle it (it sets use=True).
    # To be safe and clean, we might want to disable conflicting kernels for this M?
    # For now, trusting `kernels_gemm` list structure.
        
    return data

def find_kernel_entry(data, collection_name, layer_name, M, K, N, chunk_id=None):
    if collection_name == "kernels_gemm_chunked":
        kernels = _get_chunked_kernels_ref(data, create=False)
    else:
        kernels = data.get(collection_name, [])
    for kernel in kernels:
        if kernel.get("layer") != layer_name:
            continue
        if int(kernel.get("forM", -1)) != int(M):
            continue
        if int(kernel.get("forK", -1)) != int(K):
            continue
        if int(kernel.get("forN", -1)) != int(N):
            continue
        if chunk_id is not None and int(kernel.get("chunk_id", -1)) != int(chunk_id):
            continue
        return kernel
    return None

def _make_kernel_template(layer_name, M, K, N, chunk_id=None):
    kernel = {
        "use": True,
        "layer": layer_name,
        "forM": M,
        "forK": K,
        "forN": N,
        "npuM": 0,
        "npuK": 0,
        "npuN": N,
        "config": -1,
        "num_tiles": 32,
        "fw_path": "",
        "tile_size": "64x128x64",
        "col": "8c",
        "dtype": DTYPE_BASE
    }
    if chunk_id is not None:
        kernel["chunk_id"] = int(chunk_id)
    return kernel

def get_or_create_kernel_entry(data, collection_name, layer_name, M, K, N, chunk_id=None):
    """Finds a GEMM kernel config or creates a new one in the requested collection."""
    found = find_kernel_entry(data, collection_name, layer_name, M, K, N, chunk_id=chunk_id)
    if not found:
        found = _make_kernel_template(layer_name, M, K, N, chunk_id=chunk_id)
        if collection_name == "kernels_gemm_chunked":
            _get_chunked_kernels_ref(data, create=True).append(found)
        else:
            if collection_name not in data:
                data[collection_name] = []
            data[collection_name].append(found)

    found["use"] = True
    if chunk_id is not None:
        found["chunk_id"] = int(chunk_id)
    return found

def get_or_create_kernel_gemm(data, layer_name, M, K, N):
    return get_or_create_kernel_entry(data, "kernels_gemm", layer_name, M, K, N)

def get_or_create_kernel_gemm_chunked(data, layer_name, M, K, N, chunk_id):
    found = find_kernel_entry(data, "kernels_gemm_chunked", layer_name, M, K, N, chunk_id=chunk_id)
    if not found:
        raise RuntimeError(
            f"Missing kernels_gemm_chunked entry for layer={layer_name} "
            f"M={M} K={K} N={N} chunk_id={chunk_id}. "
            f"Chunked kernels must be seeded before tuning."
        )
    found["use"] = True
    found["chunk_id"] = int(chunk_id)
    return found

def get_valid_chunk_sizes(prompt_len):
    return [
        chunk_size
        for chunk_size in CHUNK_SIZE_CANDIDATES
        if chunk_size < prompt_len
        and prompt_len % chunk_size == 0
        and (prompt_len // chunk_size) <= 8
    ]

def _resolve_chunk_plan(prompt_len, chunk_size, chunk_schedule=None, require_exact_schedule=False):
    prompt_len = int(prompt_len)
    schedule = _normalize_chunk_schedule(chunk_schedule)
    if schedule:
        if sum(schedule) != prompt_len:
            raise RuntimeError(
                f"Chunk schedule {_format_chunk_schedule(schedule)} must sum to prompt_len={prompt_len}."
            )
        return schedule

    chunk_size = _safe_int(chunk_size, 0)
    if chunk_size <= 0:
        raise RuntimeError(f"Invalid chunk_size={chunk_size} for prompt_len={prompt_len}.")
    if prompt_len % chunk_size != 0:
        if require_exact_schedule:
            raise RuntimeError(
                f"prompt_len={prompt_len} is not divisible by chunk_size={chunk_size}. "
                "Use an explicit chunk schedule for non-uniform chunking."
            )
        raise RuntimeError(
            f"Invalid scalar chunking: prompt_len={prompt_len} is not divisible by chunk_size={chunk_size}."
        )

    num_chunks = prompt_len // chunk_size
    if num_chunks <= 0:
        raise RuntimeError(f"No chunks generated for prompt_len={prompt_len}, chunk_size={chunk_size}.")
    return [int(chunk_size)] * int(num_chunks)

def _chunk_schedule_file_tag(schedule):
    key = _chunk_schedule_key(schedule)
    return key if key else "scalar"


def _resolve_group_chunk_plan(group):
    if not isinstance(group, dict):
        return []

    prompt_len = _group_prompt_len(group)
    schedule = _group_chunk_schedule(group)
    chunk_size = _group_chunk_size(group)

    if schedule:
        if prompt_len <= 0:
            prompt_len = int(sum(schedule))
        try:
            return _resolve_chunk_plan(
                prompt_len,
                _schedule_fallback_chunk_size(schedule, chunk_size),
                chunk_schedule=schedule,
                require_exact_schedule=True,
            )
        except Exception:
            return []

    if prompt_len > 0 and chunk_size > 0:
        try:
            return _resolve_chunk_plan(prompt_len, chunk_size)
        except Exception:
            return []

    return []


def _canonicalize_chunked_group(group):
    if not isinstance(group, dict):
        return {"changed": False, "dropped_kernels": 0}

    plan = _resolve_group_chunk_plan(group)
    prompt_len = _group_prompt_len(group)
    inflight = _group_inflight(group)
    schedule = _group_chunk_schedule(group)
    chunk_size = _group_chunk_size(group)
    changed = False
    dropped_kernels = 0

    if plan:
        canonical_prompt_len = int(prompt_len if prompt_len > 0 else sum(plan))
        canonical_chunk_size = int(_schedule_fallback_chunk_size(plan, chunk_size))
        before_meta = json.dumps(group, sort_keys=True)
        _normalize_chunked_group_meta(
            group,
            canonical_chunk_size,
            inflight,
            canonical_prompt_len,
            chunk_schedule=schedule if schedule else None,
        )
        changed = changed or (json.dumps(group, sort_keys=True) != before_meta)

    kernels = group.get("kernels")
    if not isinstance(kernels, list):
        group["kernels"] = []
        return {"changed": True, "dropped_kernels": dropped_kernels}

    before_stage_bubbles = json.dumps(group.get("stage_bubbles", []), sort_keys=True)
    group["stage_bubbles"] = _normalize_stage_bubbles(group.get("stage_bubbles", []))
    if json.dumps(group.get("stage_bubbles", []), sort_keys=True) != before_stage_bubbles:
        changed = True

    deduped = {}
    ordered_keys = []
    for kernel in kernels:
        if not _is_gemm_kernel_record(kernel):
            dropped_kernels += 1
            changed = True
            continue

        cloned = json.loads(json.dumps(kernel))
        chunk_id = _safe_int(cloned.get("chunk_id", -1), -1)
        if plan and 0 <= chunk_id < len(plan):
            canonical_m = int(plan[chunk_id])
            if int(cloned.get("forM", 0)) != canonical_m:
                cloned["forM"] = canonical_m
                changed = True

        dedupe_key = (
            str(cloned.get("layer", "")),
            _safe_int(cloned.get("chunk_id", -1), -1),
            _safe_int(cloned.get("forK", 0), 0),
            _safe_int(cloned.get("forN", 0), 0),
        )
        if dedupe_key not in deduped:
            ordered_keys.append(dedupe_key)
        else:
            changed = True
        deduped[dedupe_key] = cloned

    rebuilt = [deduped[key] for key in ordered_keys]
    if rebuilt != kernels:
        group["kernels"] = rebuilt
        changed = True

    return {"changed": changed, "dropped_kernels": dropped_kernels}


def _canonical_group_bucket_key(group):
    prompt_len = _group_prompt_len(group)
    inflight = _group_inflight(group)
    schedule = _group_chunk_schedule(group)
    if len(schedule) > 1:
        return ("schedule", int(prompt_len), int(inflight), _chunk_schedule_key(schedule))
    return ("scalar", int(prompt_len), int(inflight), int(_group_chunk_size(group)))


def canonicalize_chunked_config(data, prune_duplicates=False):
    if not isinstance(data, dict):
        return {"groups_changed": 0, "groups_dropped": 0, "kernels_dropped": 0}

    kernels_grouped = data.get("kernels_gemm_chunked")
    if not _is_grouped_kernels_gemm_chunked(kernels_grouped):
        return {"groups_changed": 0, "groups_dropped": 0, "kernels_dropped": 0}

    groups_changed = 0
    kernels_dropped = 0
    for group in kernels_grouped:
        result = _canonicalize_chunked_group(group)
        if result["changed"]:
            groups_changed += 1
        kernels_dropped += int(result["dropped_kernels"])

    groups_dropped = 0
    if prune_duplicates:
        kept_groups = []
        seen_buckets = set()
        for group in kernels_grouped:
            bucket = _canonical_group_bucket_key(group)
            if bucket in seen_buckets:
                groups_dropped += 1
                continue
            seen_buckets.add(bucket)
            kept_groups.append(group)

        data["kernels_gemm_chunked"] = kept_groups

    sort_kernels_gemm_entries(data, "kernels_gemm_chunked", include_chunk_id=True)
    return {
        "groups_changed": groups_changed,
        "groups_dropped": groups_dropped,
        "kernels_dropped": kernels_dropped,
    }

def is_true_k_split_baseline(kernel):
    if not kernel:
        return False
    npuK = int(kernel.get("npuK", 0))
    forK = int(kernel.get("forK", 0))
    return npuK > 0 and npuK < forK

def _seed_fraction_npuM_for_chunk(forM, rule=None):
    forM = max(0, _safe_int(forM, 0))
    if forM <= 0:
        return 0
    selected_rule = rule if isinstance(rule, dict) else _get_device_heuristic_rule()
    num = int(selected_rule.get("npuM_num", 1))
    den = max(1, int(selected_rule.get("npuM_den", 1)))
    raw = int((int(forM) * num) / den)

    if int(forM) <= 256:
        return int(256 if raw > (int(forM) / 2.0) else 0)

    seeded = int((int(raw) // 256) * 256)
    return int(max(0, min(int(forM), int(seeded))))


def _find_nearest_gemm_baseline(data, layer_name, M, K, N):
    kernels = data.get("kernels_gemm", [])
    if not isinstance(kernels, list):
        return None
    matches = []
    for kernel in kernels:
        if not isinstance(kernel, dict):
            continue
        if str(kernel.get("layer", "")) != str(layer_name):
            continue
        if _safe_int(kernel.get("forK", 0), 0) != int(K):
            continue
        if _safe_int(kernel.get("forN", 0), 0) != int(N):
            continue
        matches.append(kernel)
    if not matches:
        return None
    return min(
        matches,
        key=lambda kernel: (
            abs(_safe_int(kernel.get("forM", 0), 0) - int(M)),
            _safe_int(kernel.get("forM", 0), 0),
        ),
    )


def _synthesize_chunk_baseline_kernel(data, model_conf, layer_cfg, chunk_size):
    layer_name = str(layer_cfg.get("name", ""))
    K = int(layer_cfg.get("K", 0))
    N = int(layer_cfg.get("N", 0))
    if chunk_size <= 0 or K <= 0 or N <= 0:
        raise RuntimeError(f"Invalid synthetic chunk baseline request for layer={layer_name} M={chunk_size} K={K} N={N}.")

    exemplar = _find_nearest_gemm_baseline(data, layer_name, chunk_size, K, N)
    if exemplar is not None:
        kernel = clone_kernel_with_chunk_id(exemplar, -1)
        kernel.pop("chunk_id", None)
    else:
        kernel = _make_kernel_template(layer_name, chunk_size, K, N)
    kernel["forM"] = int(chunk_size)
    kernel["forK"] = int(K)
    kernel["forN"] = int(N)
    kernel["npuN"] = int(N)

    split_mode = resolve_gemm_split_mode(layer_cfg)
    prefer_k_seed = (
        exemplar is not None
        and split_mode in ("K", "MK")
        and bool(exemplar.get("use", False))
        and is_true_k_split_baseline(exemplar)
    )
    if prefer_k_seed:
        seeded_k = _cap_k_tuning_value(model_conf, K, _safe_int(exemplar.get("npuK", 0), 0))
        update_kernel_config_gemm(kernel, "K", int(seeded_k), int(chunk_size), int(K), int(N), layer_cfg)
    else:
        seeded_npuM = _seed_fraction_npuM_for_chunk(chunk_size)
        update_kernel_config_gemm(kernel, "M", int(seeded_npuM), int(chunk_size), int(K), int(N), layer_cfg)
    return kernel


def get_chunk_baselines_or_raise(data, model_conf, chunk_size):
    baselines = []
    missing = []
    chunk_size = int(chunk_size)
    layer_cfgs = [
        layer for layer in model_conf.get("layers", [])
        if isinstance(layer, dict)
    ]
    for layer in layer_cfgs:
        baseline = find_kernel_entry(data, "kernels_gemm", layer["name"], chunk_size, layer["K"], layer["N"])
        if baseline is None:
            missing.append(layer["name"])
        else:
            baselines.append(baseline)

    if missing and int(chunk_size) % 256 == 0:
        baselines = []
        for layer in layer_cfgs:
            baseline = find_kernel_entry(data, "kernels_gemm", layer["name"], chunk_size, layer["K"], layer["N"])
            if baseline is not None:
                baselines.append(baseline)
                continue
            baselines.append(_synthesize_chunk_baseline_kernel(data, model_conf, layer, chunk_size))
        missing = []

    if missing:
        raise RuntimeError(
            f"Missing kernels_gemm baseline for chunk size {chunk_size}; run --mode gemm first "
            f"(missing layers: {', '.join(missing)})"
        )

    return baselines

def clone_kernel_with_chunk_id(baseline, chunk_id):
    cloned = {}
    inserted_chunk_id = False
    for key, value in baseline.items():
        if key == "chunk_id":
            continue
        cloned[key] = json.loads(json.dumps(value))
        if key == "npuN":
            cloned["chunk_id"] = int(chunk_id)
            inserted_chunk_id = True
    if not inserted_chunk_id:
        cloned["chunk_id"] = int(chunk_id)
    return cloned

def _should_forbid_chunked_true_k(layer_cfg_or_name, inflight, chunking_split_k_enabled=False):
    if int(inflight) <= 1:
        return False
    if bool(chunking_split_k_enabled):
        return False
    if isinstance(layer_cfg_or_name, dict):
        layer_name = str(layer_cfg_or_name.get("name", ""))
    else:
        layer_name = str(layer_cfg_or_name)
    return layer_name == "down"


def _apply_chunked_seed_split_policy(kernel, inflight, layer_cfg=None, chunking_split_k_enabled=False):
    """
    During chunk-size/inflight search we seed from kernels_gemm. If baseline is true K-split
    and inflight > 1, convert seed to heuristic M-split to avoid unstable split-K overlap.
    """
    if int(inflight) <= 1:
        return
    if not is_true_k_split_baseline(kernel):
        return
    if bool(chunking_split_k_enabled) and resolve_gemm_split_mode(layer_cfg) in ("K", "MK"):
        return

    layer_name = str(kernel.get("layer", ""))
    forM = int(kernel.get("forM", 0))
    forK = int(kernel.get("forK", 0))
    forN = int(kernel.get("forN", 0))
    if forM <= 0 or forK <= 0 or forN <= 0:
        return

    heuristic_npuM, _ = _heuristic_npuM_for_split_k(forM)
    update_kernel_config_gemm(kernel, "M", heuristic_npuM, forM, forK, forN, {"name": layer_name})

def seed_kernels_gemm_chunked(data, model_conf, prompt_len, chunk_size, inflight, chunk_schedule=None):
    normalized_schedule = _normalize_chunk_schedule(chunk_schedule)
    explicit_schedule = normalized_schedule if len(normalized_schedule) > 1 else []
    chunk_plan = _resolve_chunk_plan(
        prompt_len,
        chunk_size,
        chunk_schedule=explicit_schedule if explicit_schedule else None,
    )
    effective_chunk_size = _schedule_fallback_chunk_size(chunk_plan, chunk_size)
    baseline_cache = {}
    chunking_split_k_enabled = resolve_chunking_split_k_enabled(model_conf)
    layer_cfg_by_name = {
        str(layer.get("name", "")): layer
        for layer in model_conf.get("layers", [])
        if isinstance(layer, dict)
    }

    def _baselines_for_chunk(m_dim):
        m_dim = int(m_dim)
        if m_dim not in baseline_cache:
            baseline_cache[m_dim] = get_chunk_baselines_or_raise(data, model_conf, m_dim)
        return baseline_cache[m_dim]

    _apply_chunking_format(
        data,
        effective_chunk_size,
        inflight,
        prompt_len=prompt_len,
        chunk_schedule=explicit_schedule if explicit_schedule else None,
    )
    kernels = _get_chunked_kernels_ref(
        data,
        create=True,
        reset=True,
        chunk_size=effective_chunk_size,
        inflight=inflight,
        prompt_len=prompt_len,
        chunk_schedule=explicit_schedule if explicit_schedule else None,
    )
    for chunk_id, chunk_m in enumerate(chunk_plan):
        baselines = _baselines_for_chunk(chunk_m)
        for baseline in baselines:
            cloned = clone_kernel_with_chunk_id(baseline, chunk_id)
            cloned["forM"] = int(chunk_m)
            _apply_chunked_seed_split_policy(
                cloned,
                inflight,
                layer_cfg=layer_cfg_by_name.get(str(cloned.get("layer", ""))),
                chunking_split_k_enabled=chunking_split_k_enabled,
            )
            kernels.append(cloned)
    sort_kernels_gemm_entries(data, "kernels_gemm_chunked", include_chunk_id=True)
    return len(chunk_plan), list(explicit_schedule)

def run_benchmark_gemm(log_handle, current_M, model_conf, config_override=None):
    """Run GEMM benchmark and return (time_sec, status)."""
    cmd_str = (
        f"pushd ../../ && source utils/setup.sh && popd && "
        f"python3 {model_conf['script']} --prompt-test {current_M} --max-new-tokens 0"
    )
    
    if config_override:
        cmd_str += f" --config-path {config_override}"
    
    total_time = 0.0
    valid_runs = 0
    parse_fail_runs = 0
    
    for i in range(RUN_AVERAGE):
        try:
            wall_start = time.time()
            result = subprocess.run(
                cmd_str,
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=BENCHMARK_TIMEOUT_SEC
            )
            wall_end = time.time()
            output = result.stdout
            
            if log_handle:
                log_handle.write(f"\n--- GEMM Benchmark Output (Iter {i+1}) ---\n")
                log_handle.write(output)
                log_handle.write(f"\n--- End Output ---\n")
                log_handle.flush()
            
            matches = re.findall(r"Prefill time:\s*([0-9]*\.?[0-9]+)\s*seconds", output)
            if matches:
                t = float(matches[-1])
                total_time += t
                valid_runs += 1
                if len(matches) > 1:
                    first_t = float(matches[0])
                    log_print(
                        f"GEMM benchmark note: found {len(matches)} prefill markers; "
                        f"using last={t:.6f}s (first={first_t:.6f}s).",
                        log_handle,
                    )
                if VERBOSITY >= 2:
                    wall_t = float(wall_end - wall_start)
                    log_print(
                        f"GEMM benchmark timing detail (iter {i+1}): parsed_prefill={t:.6f}s, "
                        f"subprocess_wall={wall_t:.6f}s",
                        log_handle,
                    )
            else:
                parse_fail_runs += 1
                log_print(
                    f"GEMM benchmark invalid output on iter {i+1}: "
                    "missing parseable 'Prefill time: ... seconds'.",
                    log_handle,
                )
                if VERBOSITY >= 2:
                    log_print(output, log_handle)
                    
            time.sleep(2)
        except subprocess.TimeoutExpired:
            log_print(f"GEMM Benchmark timed out after {BENCHMARK_TIMEOUT_SEC}s", log_handle)
            return float('inf'), "timeout"
        except Exception as e:
            log_print(f"Exception running benchmark: {e}", log_handle)
            return float('inf'), "exception"
        
    if valid_runs > 0:
        if parse_fail_runs > 0:
            log_print(
                f"GEMM benchmark warning: {parse_fail_runs}/{RUN_AVERAGE} runs were invalid and ignored.",
                log_handle,
            )
        return total_time / valid_runs, "ok"
    else:
        log_print("GEMM benchmark invalid result: no parseable prefill time in any run.", log_handle)
        return float('inf'), "invalid_parse"

def update_kernel_config_gemm(kernel, split_type, npu_val, M, K, N, layer = None, force_16k_m_path = False):
    if split_type == "M":
        kernel["npuM"] = npu_val
        kernel["npuK"] = K
        if bool(force_16k_m_path) and int(M) == SPECIAL_GEMM_16K_DIM:
            kernel["fw_path"] = f"hw_bins/npu2/{SPECIAL_GEMM_16K_DIM}x{K}x{N}/bf16_int4AWQ_bf16_M/"
        elif layer is not None and layer["name"] == "down":
            kernel["fw_path"] = f"hw_bins/npu2/{MAX_M_DIM}x{K}x{N}/bf16_int4AWQ_bf16_K/"
        else:
            kernel["fw_path"] = f"hw_bins/npu2/{MAX_M_DIM}x{K}x{N}/bf16_int4AWQ_bf16_M/"
    else:
        kernel["npuM"] = M
        kernel["npuK"] = npu_val
        kernel["fw_path"] = f"hw_bins/npu2/{MAX_M_DIM}x{K}x{N}/bf16_int4AWQ_bf16_K/"
    kernel["dtype"] = DTYPE_BASE

    if npu_val == 0:
        kernel["use"] = False
        if split_type == "M": kernel["npuM"] = 0
        else: kernel["npuK"] = 0
    else:
        kernel["use"] = True

def update_kernel_config_gemm_m_locked_k(kernel, npu_m_val, fixed_npuK, M, K, N):
    locked_npuK = int(max(0, min(int(fixed_npuK), int(K))))
    npu_m_val = int(max(0, int(npu_m_val)))
    kernel["npuM"] = npu_m_val
    kernel["npuK"] = locked_npuK
    kernel["npuN"] = int(N)
    kernel["fw_path"] = f"hw_bins/npu2/{MAX_M_DIM}x{K}x{N}/bf16_int4AWQ_bf16_K/"
    kernel["dtype"] = DTYPE_BASE
    kernel["use"] = bool(npu_m_val > 0 and locked_npuK > 0)
    if not kernel["use"]:
        kernel["npuM"] = 0

def tune_parameter_gemm_collection(data, layer, M, K, N, split_type, max_val, log_handle, model_conf,
                                   collection_name="kernels_gemm", chunk_id=None, benchmark_prompt_len=None,
                                   heuristics_state=None, abort_on_invalid=False,
                                   fixed_npuK=None,
                                   layer_baseline_ref_time=None,
                                   layer_baseline_slowdown_pct=GEMM_LAYER_BASELINE_DVFS_SLOWDOWN_PCT,
                                   layer_baseline_retry_wait_sec=LAYER_GUARD_DVFS_SETTLE_SEC):
    label = layer["name"] if chunk_id is None else f"{layer['name']} chunk_id={chunk_id}"
    max_val = int(max(0, _safe_int(max_val, 0)))
    if str(split_type).upper() == "K":
        capped_max = _k_tuning_upper_bound(model_conf, max_val)
        if capped_max < max_val:
            log_print(
                f"Applying MAX_K_SPLIT cap for {label}: limiting K-tuning max from {max_val} to {capped_max}.",
                log_handle,
            )
        max_val = int(capped_max)
    locked_npuK = None
    if str(split_type).upper() == "M" and fixed_npuK is not None:
        capped_locked = _cap_k_tuning_value(model_conf, K, fixed_npuK)
        if int(fixed_npuK) > 0 and int(capped_locked) != int(fixed_npuK):
            log_print(
                f"Applying MAX_K_SPLIT cap for {label}: fixed npuK {int(fixed_npuK)} -> {int(capped_locked)}.",
                log_handle,
            )
        locked_npuK = int(capped_locked)
    log_print(f"\n--- Tuning {label} {split_type}-split (Max {max_val}) ---", log_handle)
    prompt_len = int(benchmark_prompt_len) if benchmark_prompt_len is not None else int(M)
    
    history = {} 

    def measure(val):
        if val in history:
            return history[val]

        recovery_active = CHUNK_RECOVER and heuristics_state is not None
        recovery_key = None
        if recovery_active:
            if collection_name == "kernels_gemm_chunked":
                active_chunk_size, active_inflight = _get_active_chunking_settings(data)
                active_schedule = _get_active_chunking_schedule(data)
                active_schedule_key = _chunk_schedule_key(active_schedule)
                if active_chunk_size <= 0:
                    active_chunk_size = int(M)
                recovery_key = make_chunk_heuristic_key(
                    "inner",
                    prompt_len=int(prompt_len),
                    chunk_size=int(active_chunk_size),
                    chunk_schedule=active_schedule_key,
                    inflight=int(active_inflight),
                    layer=layer["name"],
                    chunk_id=int(chunk_id if chunk_id is not None else -1),
                    split_type=str(split_type),
                    value=int(val),
                    K=int(K),
                    N=int(N),
                    fixed_npuK=int(locked_npuK) if locked_npuK is not None else -1,
                )
            elif collection_name == "kernels_gemm":
                recovery_key = make_chunk_heuristic_key(
                    "gemm_inner",
                    prompt_len=int(prompt_len),
                    layer=layer["name"],
                    split_type=str(split_type),
                    value=int(val),
                    M=int(M),
                    K=int(K),
                    N=int(N),
                    fixed_npuK=int(locked_npuK) if locked_npuK is not None else -1,
                )

        if recovery_key is not None:
            cached_time = get_chunk_cached_time(heuristics_state, recovery_key)
            if cached_time is not None:
                history[val] = cached_time
                log_print(f"  Recovered Value {val}: {cached_time:.4f}s", log_handle)
                return cached_time

        if collection_name == "kernels_gemm_chunked":
            _, active_inflight = _get_active_chunking_settings(data)
            active_schedule = _get_active_chunking_schedule(data)
            active_schedule_tag = _chunk_schedule_file_tag(active_schedule)
            temp_config_path = (
                f"{model_conf['config']}.chunk_inner_"
                f"{int(prompt_len)}_{active_schedule_tag}_{int(M)}_{int(active_inflight)}_"
                f"{layer['name']}_{int(chunk_id if chunk_id is not None else -1)}_{split_type}_{int(val)}_tmp.json5"
            )
        else:
            temp_config_path = model_conf["config"] + ".tmp.json5"
        trial_data = json.loads(json.dumps(data))
        if collection_name == "kernels_gemm":
            # Baseline GEMM tuning must benchmark the non-chunked prefill path.
            trial_data["chunking"] = False
            trial_data["prompt_len"] = int(M)
            trial_data.pop("npu_dim", None)
        trial_data["dummy_weights"] = bool(DUMMY_WEIGHTS)
        
        temp_kernel_config = get_or_create_kernel_entry(
            trial_data, collection_name, layer['name'], M, K, N, chunk_id=chunk_id
        )
        force_special_16k_m_path = (
            collection_name == "kernels_gemm"
            and int(M) == SPECIAL_GEMM_16K_DIM
            and str(split_type).upper() == "M"
        )
        if str(split_type).upper() == "M" and locked_npuK is not None:
            update_kernel_config_gemm_m_locked_k(temp_kernel_config, val, locked_npuK, M, K, N)
        else:
            update_kernel_config_gemm(
                temp_kernel_config,
                split_type,
                val,
                M,
                K,
                N,
                layer,
                force_16k_m_path=force_special_16k_m_path,
            )
        
        sort_all_kernels(trial_data)
        with open(temp_config_path, 'w') as f:
            json.dump(trial_data, f, indent=4)
            
        t, reason = run_benchmark_gemm(log_handle, prompt_len, model_conf, config_override=temp_config_path)

        if not math.isfinite(t) and reason != "timeout":
            for retry_idx in range(1, max(0, int(RETRY_INVALID)) + 1):
                log_print(
                    f"  Value {val}: invalid benchmark result ({reason}); "
                    f"retrying {retry_idx}/{int(RETRY_INVALID)} for debug.",
                    log_handle,
                )
                retry_t, retry_reason = run_benchmark_gemm(
                    log_handle,
                    prompt_len,
                    model_conf,
                    config_override=temp_config_path,
                )
                if math.isfinite(retry_t):
                    t = retry_t
                    reason = f"ok_retry_{retry_idx}"
                    log_print(f"  Value {val}: recovered on retry ({t:.4f}s).", log_handle)
                    break
                reason = retry_reason
                if reason == "timeout":
                    break
            if not math.isfinite(t):
                log_print(
                    f"  Value {val}: invalid benchmark result after "
                    f"{int(RETRY_INVALID)} retries ({reason}).",
                    log_handle,
                )

        if not math.isfinite(t):
            if reason == "timeout":
                log_print(f"  Value {val}: benchmark timeout after {BENCHMARK_TIMEOUT_SEC}s.", log_handle)
            else:
                log_print(f"  Value {val}: invalid benchmark result ({reason}).", log_handle)
            if abort_on_invalid:
                raise TuningAbortError(
                    f"GEMM tuning aborted at layer={layer['name']} M={M} K={K} N={N} "
                    f"split={split_type} value={val}: benchmark {reason}."
                )
        elif (
            collection_name == "kernels_gemm"
            and str(split_type).upper() == "M"
            and int(val) == 0
            and layer_baseline_ref_time is not None
            and math.isfinite(float(layer_baseline_ref_time))
        ):
            ref_t = float(layer_baseline_ref_time)
            slowdown_pct = max(0.0, float(layer_baseline_slowdown_pct))
            threshold_t = ref_t * (1.0 + slowdown_pct / 100.0)
            if t > threshold_t:
                log_print(
                    f"  Value 0: {t:.4f}s is >{slowdown_pct:.1f}% slower than previous layer winner "
                    f"{ref_t:.4f}s (threshold {threshold_t:.4f}s).",
                    log_handle,
                )
                wait_sec = _layer_guard_wait_sec(layer_baseline_retry_wait_sec, recovery_active=recovery_active)
                if wait_sec > 0:
                    log_print(f"  DVFS guard: waiting {wait_sec}s before one retry.", log_handle)
                    time.sleep(wait_sec)

                retry_t, retry_reason = run_benchmark_gemm(
                    log_handle,
                    prompt_len,
                    model_conf,
                    config_override=temp_config_path,
                )
                if not math.isfinite(retry_t):
                    log_print(
                        f"  Value 0: DVFS guard retry failed ({retry_reason}).",
                        log_handle,
                    )
                    raise TuningAbortError("timeout")

                log_print(f"  Value 0: DVFS guard retry = {retry_t:.4f}s", log_handle)
                t = retry_t
                reason = "ok_dvfs_retry"

                if t > threshold_t:
                    log_print(
                        f"  Value 0: still >{slowdown_pct:.1f}% slower than previous layer "
                        f"winner after DVFS retry ({t:.4f}s > {threshold_t:.4f}s).",
                        log_handle,
                    )
                    raise TuningAbortError("timeout")
            log_print(f"  Value {val}: {t:.4f}s", log_handle)
        else:
            log_print(f"  Value {val}: {t:.4f}s", log_handle)
        history[val] = t

        if recovery_key is not None:
            set_chunk_cached_time(model_conf, heuristics_state, recovery_key, t)

        if not recovery_active and os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            
        return t

    # Search Algorithm: Recursive Sampling
    # Range [low, high]
    # We want to find min time.
    if max_val <= 0:
        t0 = measure(0)
        return 0, t0
    
    best_val = 0
    best_time = float('inf')

    step_size = 512 if split_type == "K" else 256

    def solve(low, high):
        nonlocal best_val, best_time
        
        # Base case: if range is small enough
        if high - low < step_size:
            return

        mid = (low + high) // 2
        # align to step_size
        mid = round(mid / step_size) * step_size
        
        # Ensure mid is strictly within (low, high) to avoid infinite recursion
        if mid <= low:
            mid = low + step_size
        if mid >= high:
            mid = high - step_size
            
        if mid <= low or mid >= high:
            return 
            
        vals_to_test = [low, mid, high]
        times = []
        
        for v in vals_to_test:
            if v > max_val: v = max_val 
            t = measure(v)
            times.append(t)
            
            if t < best_time:
                best_time = t
                best_val = v
                log_print(f"  * New Best for {label} {split_type}: {v} ({t:.4f}s)", log_handle)

        t_low, t_mid, t_high = times
        
        # Greedily search better half, or both if ambiguous
        if t_low < t_mid and t_low < t_high:
            solve(low, mid)
        elif t_high < t_mid and t_high < t_low:
            solve(mid, high)
        else:
            # Try both if valley or flat
            solve(low, mid)
            solve(mid, high)

    # Allow 0 and Max, solve covers these boundaries
    solve(0, max_val)
    
    # SAFETY: Ensure best_val is always 0 or a multiple of step_size
    if best_val != 0 and best_val % step_size != 0:
        log_print(f"WARNING: Invalid best_val {best_val}, rounding to nearest {step_size}", log_handle)
        best_val = round(best_val / step_size) * step_size
        if best_val > max_val:
            best_val = max_val
        # Re-validate
        if best_val != 0 and best_val % 256 != 0:
            best_val = 0  # Fallback to GPU-only
    
    return best_val, best_time

def tune_parameter_gemm(data, layer, M, K, N, split_type, max_val, log_handle, model_conf,
                        heuristics_state=None, abort_on_invalid=True,
                        layer_baseline_ref_time=None,
                        layer_baseline_slowdown_pct=GEMM_LAYER_BASELINE_DVFS_SLOWDOWN_PCT,
                        layer_baseline_retry_wait_sec=LAYER_GUARD_DVFS_SETTLE_SEC):
    return tune_parameter_gemm_collection(
        data,
        layer,
        M,
        K,
        N,
        split_type,
        max_val,
        log_handle,
        model_conf,
        collection_name="kernels_gemm",
        heuristics_state=heuristics_state,
        abort_on_invalid=abort_on_invalid,
        layer_baseline_ref_time=layer_baseline_ref_time,
        layer_baseline_slowdown_pct=layer_baseline_slowdown_pct,
        layer_baseline_retry_wait_sec=layer_baseline_retry_wait_sec,
    )
    
def tune_all_gemm(log_handle, model_conf):
    log_print(f"\n>>> Starting GEMM Tuning for {model_conf['name']}", log_handle)
    
    data = load_config(model_conf)
    if not data: return
    heuristics_state = load_chunk_heuristics(model_conf, log_handle) if CHUNK_RECOVER else None
    gemm_recovery_active = CHUNK_RECOVER and heuristics_state is not None

    def _gemm_layer_state_key(layer, M, K, N):
        return make_chunk_heuristic_key(
            "gemm_layer_state",
            prompt_len=int(M),
            layer=str(layer["name"]),
            M=int(M),
            K=int(K),
            N=int(N),
        )

    def _apply_recovered_gemm_layer_state(layer, M, K, N):
        if not gemm_recovery_active:
            return False
        key = _gemm_layer_state_key(layer, M, K, N)
        saved_kernel = get_chunk_layer_state(heuristics_state, key)
        if saved_kernel is None:
            return False
        target = get_or_create_kernel_gemm(data, layer["name"], M, K, N)
        target.clear()
        target.update(saved_kernel)
        target["layer"] = str(layer["name"])
        target["forM"] = int(M)
        target["forK"] = int(K)
        target["forN"] = int(N)
        target["use"] = bool(target.get("use", True))
        log_print(f"Recovered GEMM layer from CHUNK_RECOVER cache: {layer['name']} M={M}", log_handle)
        return True

    def _persist_gemm_layer_state(layer, M, K, N):
        if not gemm_recovery_active:
            return
        kernel = find_kernel_entry(data, "kernels_gemm", layer["name"], M, K, N)
        if kernel is None:
            return
        set_chunk_layer_state(model_conf, heuristics_state, _gemm_layer_state_key(layer, M, K, N), kernel)

    try:
        for size in _get_prompt_sizes_for_mode("gemm", model_conf):
            log_print(f"\n{'='*40}", log_handle)
            log_print(f"TUNING PROMPT SIZE: {size}", log_handle)
            log_print(f"{'='*40}", log_handle)
            
            data = setup_baseline_config(data, size)
            size_changed = False
            prev_layer_winner_time = None
            
            for layer in model_conf["layers"]:
                M = size 
                K = layer["K"]
                N = layer["N"]
                
                existing = find_kernel_entry(data, "kernels_gemm", layer["name"], M, K, N)
                if existing:
                    log_print(f"Skipping {layer['name']} (config already exists: npuM={existing.get('npuM')})", log_handle)
                    prev_layer_winner_time = None
                    continue

                if _apply_recovered_gemm_layer_state(layer, M, K, N):
                    size_changed = True
                    save_config_via_temp(data, model_conf, temp_suffix=".gemm_recover_commit_tmp.json5")
                    prev_layer_winner_time = None
                    continue
                
                split_mode = resolve_gemm_split_mode(layer)
                effective_split_mode = "M" if int(M) == SPECIAL_GEMM_16K_DIM else split_mode
                log_print(f"GEMM split policy for {layer['name']}: {effective_split_mode}", log_handle)

                best_m_val = 0
                best_m_time = float('inf')
                if effective_split_mode in ("M", "MK"):
                    best_m_val, best_m_time = tune_parameter_gemm(
                        data,
                        layer,
                        M,
                        K,
                        N,
                        "M",
                        M,
                        log_handle,
                        model_conf,
                        heuristics_state=heuristics_state,
                        abort_on_invalid=not gemm_recovery_active,
                        layer_baseline_ref_time=prev_layer_winner_time,
                        layer_baseline_slowdown_pct=GEMM_LAYER_BASELINE_DVFS_SLOWDOWN_PCT,
                        layer_baseline_retry_wait_sec=LAYER_GUARD_DVFS_SETTLE_SEC,
                    )

                best_k_val = 0
                best_k_time = float('inf')
                if effective_split_mode in ("K", "MK"):
                    best_k_val, best_k_time = tune_parameter_gemm(
                        data,
                        layer,
                        M,
                        K,
                        N,
                        "K",
                        K,
                        log_handle,
                        model_conf,
                        heuristics_state=heuristics_state,
                        abort_on_invalid=not gemm_recovery_active,
                    )

                final_kernel = get_or_create_kernel_gemm(data, layer["name"], M, K, N)

                use_special_16k_m_path = int(M) == SPECIAL_GEMM_16K_DIM

                if effective_split_mode == "K":
                    log_print(f"Winner for {layer['name']}: K-Split {best_k_val} ({best_k_time:.4f}s)", log_handle)
                    update_kernel_config_gemm(final_kernel, "K", best_k_val, M, K, N, layer)
                    layer_winner_time = best_k_time
                elif effective_split_mode == "M":
                    log_print(f"Winner for {layer['name']}: M-Split {best_m_val} ({best_m_time:.4f}s)", log_handle)
                    update_kernel_config_gemm(
                        final_kernel,
                        "M",
                        best_m_val,
                        M,
                        K,
                        N,
                        layer,
                        force_16k_m_path=use_special_16k_m_path,
                    )
                    layer_winner_time = best_m_time
                elif best_k_time < best_m_time:
                    log_print(f"Winner for {layer['name']}: K-Split {best_k_val} ({best_k_time:.4f}s)", log_handle)
                    update_kernel_config_gemm(final_kernel, "K", best_k_val, M, K, N, layer)
                    layer_winner_time = best_k_time
                else:
                    log_print(f"Winner for {layer['name']}: M-Split {best_m_val} ({best_m_time:.4f}s)", log_handle)
                    update_kernel_config_gemm(
                        final_kernel,
                        "M",
                        best_m_val,
                        M,
                        K,
                        N,
                        layer,
                        force_16k_m_path=use_special_16k_m_path,
                    )
                    layer_winner_time = best_m_time
                size_changed = True
                _persist_gemm_layer_state(layer, M, K, N)
                if gemm_recovery_active:
                    save_config_via_temp(data, model_conf, temp_suffix=".gemm_recover_commit_tmp.json5")
                prev_layer_winner_time = float(layer_winner_time) if math.isfinite(layer_winner_time) else None

            if size_changed:
                save_config(data, model_conf)
                log_print(f"Saved GEMM config updates for prompt size {size}", log_handle)
                
    except TuningAbortError as e:
        log_print(str(e), log_handle)
        raise SystemExit(1)
    except KeyboardInterrupt:
        log_print("GEMM Tuning Interrupted!", log_handle)

def upsert_chunked_kernels(data, chunked_kernels):
    target_kernels = _get_chunked_kernels_ref(data, create=True)
    if not isinstance(target_kernels, list):
        raise RuntimeError("Missing kernels_gemm_chunked destination table. Seed chunked kernels before updating them.")

    for kernel in chunked_kernels:
        existing = find_kernel_entry(
            data,
            "kernels_gemm_chunked",
            kernel["layer"],
            kernel["forM"],
            kernel["forK"],
            kernel["forN"],
            chunk_id=kernel.get("chunk_id"),
        )
        cloned = json.loads(json.dumps(kernel))
        if existing is None:
            raise RuntimeError(
                f"Missing seeded kernels_gemm_chunked slot for layer={kernel['layer']} "
                f"M={kernel['forM']} K={kernel['forK']} N={kernel['forN']} "
                f"chunk_id={kernel.get('chunk_id')}"
            )
        existing.clear()
        existing.update(cloned)

    sort_kernels_gemm_entries(data, "kernels_gemm_chunked", include_chunk_id=True)

def build_chunking_trial_data(data, model_conf, prompt_len, chunk_size, inflight, chunk_schedule=None):
    trial_data = json.loads(json.dumps(data))
    trial_data = setup_baseline_config(trial_data, prompt_len)
    baseline_gpu_chunking = json.loads(json.dumps(data.get("gpu_chunking"))) if isinstance(data.get("gpu_chunking"), dict) else None
    # Chunking search/tuning benchmarks should always run the hetero runtime path.
    trial_data["heterogeneity"] = "hetero"
    # Preserve baseline GPU chunking metadata in temp configs for traceability.
    if baseline_gpu_chunking is not None:
        trial_data["gpu_chunking"] = baseline_gpu_chunking
    # Keep trial configs focused on the candidate under test; do not carry stale groups.
    trial_data["kernels_gemm_chunked"] = []
    num_chunks, resolved_schedule = seed_kernels_gemm_chunked(
        trial_data,
        model_conf,
        prompt_len,
        chunk_size,
        inflight,
        chunk_schedule=chunk_schedule,
    )
    return trial_data, num_chunks, resolved_schedule

def benchmark_gemm_trial_data(log_handle, model_conf, prompt_len, trial_data, temp_suffix,
                              heuristics_state=None, recovery_key=None):
    temp_config_path = model_conf["config"] + temp_suffix
    trial_data["dummy_weights"] = bool(DUMMY_WEIGHTS)
    sort_all_kernels(trial_data)
    if CHUNK_RECOVER and heuristics_state is not None and recovery_key is not None:
        cached_time = get_chunk_cached_time(heuristics_state, recovery_key)
        if cached_time is not None:
            # Keep temp file content consistent with the current candidate even on cache hits.
            with open(temp_config_path, 'w') as f:
                json.dump(trial_data, f, indent=4)
            log_print(f"Recovered cached trial {recovery_key}: {cached_time:.4f}s", log_handle)
            return cached_time

    with open(temp_config_path, 'w') as f:
        json.dump(trial_data, f, indent=4)

    try:
        t, reason = run_benchmark_gemm(log_handle, prompt_len, model_conf, config_override=temp_config_path)
        if not math.isfinite(t) and reason != "timeout":
            for retry_idx in range(1, max(0, int(RETRY_INVALID)) + 1):
                log_print(
                    f"Invalid trial benchmark result ({reason}); "
                    f"retrying {retry_idx}/{int(RETRY_INVALID)} for debug.",
                    log_handle,
                )
                retry_t, retry_reason = run_benchmark_gemm(
                    log_handle,
                    prompt_len,
                    model_conf,
                    config_override=temp_config_path,
                )
                if math.isfinite(retry_t):
                    t = retry_t
                    break
                reason = retry_reason
                if reason == "timeout":
                    break
        if recovery_key is not None:
            set_chunk_cached_time(model_conf, heuristics_state, recovery_key, t)
        return t
    finally:
        if not (CHUNK_RECOVER and heuristics_state is not None) and os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def tune_chunked_use_toggle_for_locked_k(
    data,
    layer,
    M,
    K,
    N,
    chunk_id,
    prompt_len,
    fixed_npuK,
    log_handle,
    model_conf,
    heuristics_state=None,
):
    """For chunk_id>0 under locked K-split, only choose between disable and locked-enable."""
    best_val = 0
    best_time = float("inf")
    capped_fixed_npuK = _cap_k_tuning_value(model_conf, K, fixed_npuK)
    if int(fixed_npuK) > 0 and capped_fixed_npuK != int(fixed_npuK):
        log_print(
            f"Applying MAX_K_SPLIT cap for {layer['name']} chunk_id={chunk_id}: "
            f"locked npuK {int(fixed_npuK)} -> {int(capped_fixed_npuK)}.",
            log_handle,
        )
    candidate_vals = [0]
    if int(capped_fixed_npuK) > 0:
        candidate_vals.append(int(capped_fixed_npuK))

    _, active_inflight = _get_active_chunking_settings(data)
    active_schedule = _get_active_chunking_schedule(data)
    active_schedule_key = _chunk_schedule_key(active_schedule)
    active_schedule_tag = _chunk_schedule_file_tag(active_schedule)
    chunk_recovery_active = CHUNK_RECOVER and heuristics_state is not None

    for val in candidate_vals:
        recovery_key = None
        if chunk_recovery_active:
            recovery_key = make_chunk_heuristic_key(
                "toggle",
                prompt_len=int(prompt_len),
                chunk_size=int(M),
                chunk_schedule=active_schedule_key,
                inflight=int(active_inflight),
                layer=layer["name"],
                chunk_id=int(chunk_id),
                split_type="K",
                value=int(val),
                K=int(K),
                N=int(N),
            )
            cached_time = get_chunk_cached_time(heuristics_state, recovery_key)
            if cached_time is not None:
                log_print(f"  Recovered toggle Value {val}: {cached_time:.4f}s", log_handle)
                t = cached_time
                if t < best_time:
                    best_time = t
                    best_val = val
                continue

        temp_config_path = (
            f"{model_conf['config']}.chunk_toggle_"
            f"{int(prompt_len)}_{active_schedule_tag}_{int(M)}_{int(active_inflight)}_"
            f"{layer['name']}_{int(chunk_id)}_{int(val)}_tmp.json5"
        )
        trial_data = json.loads(json.dumps(data))
        trial_data["dummy_weights"] = bool(DUMMY_WEIGHTS)
        trial_kernel = get_or_create_kernel_entry(
            trial_data,
            "kernels_gemm_chunked",
            layer["name"],
            M,
            K,
            N,
            chunk_id=chunk_id,
        )
        update_kernel_config_gemm(trial_kernel, "K", int(val), M, K, N, layer)

        sort_all_kernels(trial_data)
        with open(temp_config_path, "w") as f:
            json.dump(trial_data, f, indent=4)

        t, reason = run_benchmark_gemm(log_handle, prompt_len, model_conf, config_override=temp_config_path)
        if not math.isfinite(t) and reason != "timeout":
            for retry_idx in range(1, max(0, int(RETRY_INVALID)) + 1):
                log_print(
                    f"Invalid toggle benchmark result ({reason}); "
                    f"retrying {retry_idx}/{int(RETRY_INVALID)} for debug.",
                    log_handle,
                )
                retry_t, retry_reason = run_benchmark_gemm(
                    log_handle,
                    prompt_len,
                    model_conf,
                    config_override=temp_config_path,
                )
                if math.isfinite(retry_t):
                    t = retry_t
                    break
                reason = retry_reason
                if reason == "timeout":
                    break
        log_print(f"  Toggle Value {val}: {t:.4f}s", log_handle)

        if recovery_key is not None:
            set_chunk_cached_time(model_conf, heuristics_state, recovery_key, t)

        if not chunk_recovery_active and os.path.exists(temp_config_path):
            os.remove(temp_config_path)

        if t < best_time:
            best_time = t
            best_val = val

    return best_val, best_time

def tune_chunked_candidate(
    log_handle,
    model_conf,
    base_data,
    prompt_len,
    chunk_size,
    inflight,
    heuristics_state=None,
    chunk_schedule=None,
):
    normalized_schedule = _normalize_chunk_schedule(chunk_schedule)
    effective_chunk_size = _schedule_fallback_chunk_size(normalized_schedule, chunk_size)
    working_data, num_chunks, resolved_schedule = build_chunking_trial_data(
        base_data,
        model_conf,
        prompt_len,
        effective_chunk_size,
        inflight,
        chunk_schedule=normalized_schedule,
    )
    chunk_plan = list(resolved_schedule) if resolved_schedule else _resolve_chunk_plan(prompt_len, effective_chunk_size)
    explicit_schedule = list(resolved_schedule) if resolved_schedule else list(normalized_schedule)
    chunk_schedule_key = _chunk_schedule_key(explicit_schedule)
    chunk_schedule_tag = _chunk_schedule_file_tag(explicit_schedule)
    chunk_id_sequence = list(range(num_chunks - 1, -1, -1)) if CHUNK_ID_EXPLORE_DESCEND else list(range(num_chunks))
    lock_source_chunk_id = chunk_id_sequence[0] if chunk_id_sequence else 0
    locked_k_by_layer = {}
    chunking_split_k_enabled = resolve_chunking_split_k_enabled(model_conf)
    chunk_recovery_active = CHUNK_RECOVER and heuristics_state is not None
    recovered_inner_timings = {}

    def _chunk_m(chunk_id):
        if 0 <= int(chunk_id) < len(chunk_plan):
            return int(chunk_plan[int(chunk_id)])
        return int(effective_chunk_size)

    def _is_split_k_only_layer(layer):
        return bool(chunking_split_k_enabled) and resolve_gemm_split_mode(layer) == "K"

    def _propagate_locked_k_to_future_chunks(layer, source_chunk_id, fixed_npuK):
        if not chunk_id_sequence:
            return
        try:
            source_idx = chunk_id_sequence.index(int(source_chunk_id))
        except ValueError:
            return
        propagated = 0
        for target_chunk_id in chunk_id_sequence[source_idx + 1:]:
            target_m = _chunk_m(target_chunk_id)
            kernel = get_or_create_kernel_gemm_chunked(
                working_data,
                layer["name"],
                target_m,
                int(layer["K"]),
                int(layer["N"]),
                int(target_chunk_id),
            )
            # Preserve current npuM/use state; only force the layer-wide locked npuK.
            kernel["npuK"] = int(max(0, min(int(fixed_npuK), int(layer["K"]))))
            kernel["fw_path"] = (
                f"hw_bins/npu2/{MAX_M_DIM}x{int(layer['K'])}x{int(layer['N'])}/bf16_int4AWQ_bf16_K/"
            )
            kernel["dtype"] = DTYPE_BASE
            if int(kernel.get("npuK", 0)) <= 0:
                kernel["use"] = False
            propagated += 1
        if propagated > 0:
            log_print(
                f"Propagated locked npuK={int(fixed_npuK)} for {layer['name']} to {propagated} remaining chunk(s).",
                log_handle,
            )

    if chunk_recovery_active:
        raw_timings = heuristics_state.get("timings", {})
        if isinstance(raw_timings, dict):
            for raw_key, raw_time in raw_timings.items():
                try:
                    payload = json.loads(raw_key)
                except Exception:
                    continue
                if not isinstance(payload, dict) or payload.get("stage") != "inner":
                    continue
                try:
                    if int(payload.get("prompt_len", -1)) != int(prompt_len):
                        continue
                    payload_schedule = str(payload.get("chunk_schedule", ""))
                    if chunk_schedule_key:
                        if payload_schedule and payload_schedule != chunk_schedule_key:
                            continue
                        if (not payload_schedule) and int(payload.get("chunk_size", -1)) != int(effective_chunk_size):
                            continue
                    else:
                        if payload_schedule:
                            continue
                        if int(payload.get("chunk_size", -1)) != int(effective_chunk_size):
                            continue
                    if int(payload.get("inflight", -1)) != int(inflight):
                        continue
                    key = (
                        int(payload.get("chunk_id", -1)),
                        str(payload.get("layer", "")),
                        str(payload.get("split_type", "")),
                    )
                    val = int(payload.get("value", -1))
                    t = float(raw_time)
                except Exception:
                    continue
                if val < 0:
                    continue
                recovered_inner_timings.setdefault(key, {})[val] = t

    def _measure_layer_guard_time(state_data, chunk_id, layer_name, tag):
        chunk_m = _chunk_m(chunk_id)
        suffix = (
            f".chunk_layer_guard_{int(prompt_len)}_{chunk_schedule_tag}_{int(chunk_m)}_{int(inflight)}_"
            f"{int(chunk_id)}_{layer_name}_{tag}_tmp.json5"
        )
        return benchmark_gemm_trial_data(
            log_handle,
            model_conf,
            prompt_len,
            state_data,
            temp_suffix=suffix,
            heuristics_state=None,
            recovery_key=None,
        )

    def _layer_state_key(layer, chunk_id):
        chunk_m = _chunk_m(chunk_id)
        return make_chunk_heuristic_key(
            "layer_state",
            prompt_len=int(prompt_len),
            chunk_size=int(effective_chunk_size),
            chunk_schedule=chunk_schedule_key,
            inflight=int(inflight),
            chunk_id=int(chunk_id),
            layer=str(layer["name"]),
            M=int(chunk_m),
            K=int(layer["K"]),
            N=int(layer["N"]),
        )

    def _get_recovered_inner_rows(layer, chunk_id, split_type):
        key = (int(chunk_id), str(layer["name"]), str(split_type))
        vals = recovered_inner_timings.get(key, {})
        rows = []
        for val, t in vals.items():
            try:
                rows.append((int(val), float(t)))
            except Exception:
                continue
        rows.sort(key=lambda x: x[0])
        return rows

    def _infer_recovered_winner(layer, kernel):
        K = int(layer["K"])
        npuK = int(kernel.get("npuK", 0))
        npuM = int(kernel.get("npuM", 0))
        use = bool(kernel.get("use", True))
        if npuK > 0 and npuK < K:
            return "K", int(npuK)
        if use and npuM > 0:
            return "M", int(npuM)
        return "M", 0

    def _lookup_recovered_time(layer, chunk_id, split_type, val):
        key = (int(chunk_id), str(layer["name"]), str(split_type))
        rows = recovered_inner_timings.get(key, {})
        if int(val) not in rows:
            return float("inf")
        try:
            return float(rows[int(val)])
        except Exception:
            return float("inf")

    def _log_recovered_layer_replay(layer, chunk_id, kernel):
        label = f"{layer['name']} chunk_id={chunk_id}"
        M = _chunk_m(chunk_id)

        def _replay_split(split_type, max_val):
            rows = _get_recovered_inner_rows(layer, chunk_id, split_type)
            if not rows:
                return
            best_val = 0
            best_time = float("inf")
            log_print(f"\n--- Tuning {label} {split_type}-split (Max {max_val}) ---", log_handle)
            for val, t in rows:
                if math.isfinite(t):
                    log_print(f"  Value {val}: {t:.4f}s (recovered)", log_handle)
                    if t < best_time:
                        best_time = t
                        best_val = val
                        log_print(
                            f"  * New Best for {label} {split_type}: {best_val} ({best_time:.4f}s)",
                            log_handle,
                        )
                else:
                    log_print("  Value {val}: timeout/invalid benchmark result. (recovered)".format(val=val), log_handle)

        _replay_split("M", int(M))

        baseline_kernel = find_kernel_entry(base_data, "kernels_gemm", layer["name"], int(M), int(layer["K"]), int(layer["N"]))
        forbid_true_k = _should_forbid_chunked_true_k(
            layer,
            inflight,
            chunking_split_k_enabled=chunking_split_k_enabled,
        )
        if forbid_true_k:
            log_print(
                f"Policy: forbidding true K-split for {layer['name']} when inflight={inflight} (>1).",
                log_handle,
            )
        elif is_true_k_split_baseline(baseline_kernel):
            _replay_split("K", int(layer["K"]))
        else:
            log_print(
                f"Skipping K-split for {layer['name']} chunk_id={chunk_id}: baseline is not a true K-split.",
                log_handle,
            )

        split_type, val = _infer_recovered_winner(layer, kernel)
        winner_time = _lookup_recovered_time(layer, chunk_id, split_type, val)
        if math.isfinite(winner_time):
            log_print(
                f"Winner for {layer['name']} chunk_id={chunk_id}: {split_type}-Split {val} ({winner_time:.4f}s) (recovered)",
                log_handle,
            )
        else:
            log_print(
                f"Winner for {layer['name']} chunk_id={chunk_id}: {split_type}-Split {val} (recovered state)",
                log_handle,
            )

    def _apply_recovered_kernel_state(layer, chunk_id):
        nonlocal locked_k_by_layer
        if not chunk_recovery_active:
            return False
        key = _layer_state_key(layer, chunk_id)
        saved_kernel = get_chunk_layer_state(heuristics_state, key)
        if saved_kernel is None:
            return False

        M = _chunk_m(chunk_id)
        K = int(layer["K"])
        N = int(layer["N"])
        target = get_or_create_kernel_gemm_chunked(working_data, layer["name"], M, K, N, chunk_id)
        target.clear()
        target.update(saved_kernel)
        # Enforce identity fields in case of stale/incompatible payload.
        target["layer"] = str(layer["name"])
        target["forM"] = int(M)
        target["forK"] = int(K)
        target["forN"] = int(N)
        target["chunk_id"] = int(chunk_id)
        target["use"] = bool(target.get("use", True))

        forbid_true_k = _should_forbid_chunked_true_k(
            layer,
            inflight,
            chunking_split_k_enabled=chunking_split_k_enabled,
        )
        recovered_npuK = int(target.get("npuK", 0))
        if chunk_id == lock_source_chunk_id and (not forbid_true_k):
            if _is_split_k_only_layer(layer):
                locked_val = int(_cap_k_tuning_value(model_conf, K, recovered_npuK))
                locked_k_by_layer[layer["name"]] = {"npuK": locked_val, "source_chunk": int(chunk_id)}
                _propagate_locked_k_to_future_chunks(layer, chunk_id, locked_val)
            elif recovered_npuK > 0 and recovered_npuK < K:
                locked_k_by_layer[layer["name"]] = {"npuK": recovered_npuK, "source_chunk": int(chunk_id)}

        log_print(f"Recovered completed layer from CHUNK_RECOVER cache: {layer['name']} chunk_id={chunk_id}", log_handle)
        _log_recovered_layer_replay(layer, chunk_id, target)
        return True

    def _persist_layer_state(layer, chunk_id):
        if not chunk_recovery_active:
            return
        M = _chunk_m(chunk_id)
        kernel = find_kernel_entry(
            working_data,
            "kernels_gemm_chunked",
            layer["name"],
            int(M),
            int(layer["K"]),
            int(layer["N"]),
            chunk_id=chunk_id,
        )
        if kernel is None:
            return
        set_chunk_layer_state(model_conf, heuristics_state, _layer_state_key(layer, chunk_id), kernel)

    def _tune_one_layer(layer, chunk_id):
        nonlocal working_data, locked_k_by_layer
        M = _chunk_m(chunk_id)
        K = int(layer["K"])
        N = int(layer["N"])
        split_k_only_layer = _is_split_k_only_layer(layer)
        forbid_true_k = _should_forbid_chunked_true_k(
            layer,
            inflight,
            chunking_split_k_enabled=chunking_split_k_enabled,
        )

        # Hardware constraint: once the source chunk picks a true K-split for a layer,
        # later chunks use that same npuK.
        if chunk_id != lock_source_chunk_id and layer["name"] in locked_k_by_layer:
            lock_state = locked_k_by_layer[layer["name"]]
            locked_npuK = int(lock_state["npuK"])
            source_chunk_id = int(lock_state["source_chunk"])
            if split_k_only_layer:
                log_print(
                    f"Locked K-split for {layer['name']} from chunk_id={source_chunk_id} (npuK={locked_npuK}); "
                    f"tuning chunk_id={chunk_id} as M-only with fixed npuK.",
                    log_handle,
                )
                m_tune_max = int(M) if int(locked_npuK) > 0 else 0
                best_m_val, best_m_time = tune_parameter_gemm_collection(
                    working_data,
                    layer,
                    M,
                    K,
                    N,
                    "M",
                    m_tune_max,
                    log_handle,
                    model_conf,
                    collection_name="kernels_gemm_chunked",
                    chunk_id=chunk_id,
                    benchmark_prompt_len=prompt_len,
                    heuristics_state=heuristics_state,
                    fixed_npuK=locked_npuK,
                )
                final_kernel = get_or_create_kernel_gemm_chunked(working_data, layer["name"], M, K, N, chunk_id)
                update_kernel_config_gemm_m_locked_k(final_kernel, best_m_val, locked_npuK, M, K, N)
                log_print(
                    f"Winner for {layer['name']} chunk_id={chunk_id}: M-Split {best_m_val} "
                    f"(fixed npuK={locked_npuK}, {best_m_time:.4f}s)",
                    log_handle,
                )
                return float(best_m_time)
            else:
                log_print(
                    f"Locked K-split for {layer['name']} from chunk_id={source_chunk_id} (npuK={locked_npuK}); "
                    f"tuning chunk_id={chunk_id} as use-toggle only.",
                    log_handle,
                )
                best_toggle_val, best_toggle_time = tune_chunked_use_toggle_for_locked_k(
                    working_data,
                    layer,
                    M,
                    K,
                    N,
                    chunk_id,
                    prompt_len,
                    locked_npuK,
                    log_handle,
                    model_conf,
                    heuristics_state=heuristics_state,
                )
                final_kernel = get_or_create_kernel_gemm_chunked(working_data, layer["name"], M, K, N, chunk_id)
                update_kernel_config_gemm(final_kernel, "K", best_toggle_val, M, K, N, layer)
                log_print(
                    f"Winner for {layer['name']} chunk_id={chunk_id}: "
                    f"{'use=true' if best_toggle_val > 0 else 'use=false'} "
                    f"(npuK={best_toggle_val}, {best_toggle_time:.4f}s)",
                    log_handle,
                )
                return float(best_toggle_time)

        if split_k_only_layer and chunk_id != lock_source_chunk_id and layer["name"] not in locked_k_by_layer:
            seeded_kernel = find_kernel_entry(
                working_data,
                "kernels_gemm_chunked",
                layer["name"],
                M,
                K,
                N,
                chunk_id=chunk_id,
            )
            seeded_npuK = int(seeded_kernel.get("npuK", 0)) if seeded_kernel else 0
            fallback_npuK = int(_cap_k_tuning_value(model_conf, K, seeded_npuK))
            locked_k_by_layer[layer["name"]] = {"npuK": fallback_npuK, "source_chunk": int(lock_source_chunk_id)}
            log_print(
                f"Split-K lock missing for {layer['name']} chunk_id={chunk_id}; "
                f"using seeded npuK={fallback_npuK} and continuing with M-only tuning.",
                log_handle,
            )
            return _tune_one_layer(layer, chunk_id)

        baseline_kernel = find_kernel_entry(base_data, "kernels_gemm", layer["name"], M, K, N)

        if split_k_only_layer and chunk_id == lock_source_chunk_id:
            best_k_val, best_k_time = tune_parameter_gemm_collection(
                working_data,
                layer,
                M,
                K,
                N,
                "K",
                K,
                log_handle,
                model_conf,
                collection_name="kernels_gemm_chunked",
                chunk_id=chunk_id,
                benchmark_prompt_len=prompt_len,
                heuristics_state=heuristics_state,
            )
            final_kernel = get_or_create_kernel_gemm_chunked(working_data, layer["name"], M, K, N, chunk_id)
            update_kernel_config_gemm(final_kernel, "K", best_k_val, M, K, N, layer)
            locked_k_by_layer[layer["name"]] = {"npuK": int(best_k_val), "source_chunk": int(chunk_id)}
            _propagate_locked_k_to_future_chunks(layer, chunk_id, int(best_k_val))
            log_print(
                f"Winner for {layer['name']} chunk_id={chunk_id}: K-Split {best_k_val} ({best_k_time:.4f}s)",
                log_handle,
            )
            return float(best_k_time)

        best_m_val, best_m_time = tune_parameter_gemm_collection(
            working_data,
            layer,
            M,
            K,
            N,
            "M",
            M,
            log_handle,
            model_conf,
            collection_name="kernels_gemm_chunked",
            chunk_id=chunk_id,
            benchmark_prompt_len=prompt_len,
            heuristics_state=heuristics_state,
        )

        best_split = "M"
        best_val = best_m_val
        best_time = best_m_time

        if forbid_true_k:
            log_print(
                f"Policy: forbidding true K-split for {layer['name']} when inflight={inflight} (>1).",
                log_handle,
            )
        elif is_true_k_split_baseline(baseline_kernel):
            best_k_val, best_k_time = tune_parameter_gemm_collection(
                working_data,
                layer,
                M,
                K,
                N,
                "K",
                K,
                log_handle,
                model_conf,
                collection_name="kernels_gemm_chunked",
                chunk_id=chunk_id,
                benchmark_prompt_len=prompt_len,
                heuristics_state=heuristics_state,
            )
            if best_k_time < best_time:
                best_split = "K"
                best_val = best_k_val
                best_time = best_k_time
        else:
            log_print(
                f"Skipping K-split for {layer['name']} chunk_id={chunk_id}: baseline is not a true K-split.",
                log_handle,
            )

        final_kernel = get_or_create_kernel_gemm_chunked(working_data, layer["name"], M, K, N, chunk_id)
        update_kernel_config_gemm(final_kernel, best_split, best_val, M, K, N, layer)
        log_print(
            f"Winner for {layer['name']} chunk_id={chunk_id}: {best_split}-Split {best_val} ({best_time:.4f}s)",
            log_handle,
        )
        if chunk_id == lock_source_chunk_id and (not forbid_true_k) and best_split == "K" and int(best_val) > 0 and int(best_val) < int(K):
            locked_k_by_layer[layer["name"]] = {"npuK": int(best_val), "source_chunk": int(chunk_id)}
            log_print(
                f"Locking {layer['name']} K-split across chunks from chunk_id={chunk_id} at npuK={int(best_val)}.",
                log_handle,
            )
            if split_k_only_layer:
                _propagate_locked_k_to_future_chunks(layer, chunk_id, int(best_val))

        return float(best_time)

    prev_chunk_tail_winner_time = None
    for chunk_id in chunk_id_sequence:
        chunk_tail_winner_time = None
        current_chunk_m = _chunk_m(chunk_id)
        log_print(
            f"\n--- Inner tuning for chunk_size={current_chunk_m}, inflight={inflight}, "
            f"chunk_id={chunk_id} / {num_chunks - 1} ---",
            log_handle,
        )
        for layer_idx, layer in enumerate(model_conf["layers"]):
            layer_name = layer["name"]
            if _apply_recovered_kernel_state(layer, chunk_id):
                recovered_kernel = find_kernel_entry(
                    working_data,
                    "kernels_gemm_chunked",
                    layer["name"],
                    int(_chunk_m(chunk_id)),
                    int(layer["K"]),
                    int(layer["N"]),
                    chunk_id=chunk_id,
                )
                if recovered_kernel is not None:
                    recovered_split, recovered_val = _infer_recovered_winner(layer, recovered_kernel)
                    recovered_time = _lookup_recovered_time(layer, chunk_id, recovered_split, recovered_val)
                    if math.isfinite(recovered_time):
                        chunk_tail_winner_time = float(recovered_time)
                continue
            pre_layer_data = json.loads(json.dumps(working_data))
            pre_layer_locked = dict(locked_k_by_layer)
            enforce_regression_guard = True
            settle_wait_sec = _layer_guard_wait_sec(
                LAYER_GUARD_DVFS_SETTLE_SEC,
                recovery_active=chunk_recovery_active,
            )
            if settle_wait_sec > 0:
                log_print(
                    f"Sleeping {settle_wait_sec}s for DVFS settle before "
                    f"baseline measurement for chunk_id={chunk_id}, layer={layer_name}.",
                    log_handle,
                )
                time.sleep(settle_wait_sec)
            baseline_time = _measure_layer_guard_time(pre_layer_data, chunk_id, layer_name, "before")
            if not math.isfinite(baseline_time) and chunk_recovery_active:
                log_print(
                    f"Chunked layer guard baseline timed out for chunk_id={chunk_id}, layer={layer_name}; "
                    "retrying once under CHUNK_RECOVER.",
                    log_handle,
                )
                baseline_time = _measure_layer_guard_time(pre_layer_data, chunk_id, layer_name, "before")
            if not math.isfinite(baseline_time):
                if chunk_recovery_active:
                    log_print(
                        f"Chunked layer guard baseline still invalid for chunk_id={chunk_id}, layer={layer_name}; "
                        "continuing without layer regression guard for this layer.",
                        log_handle,
                    )
                    enforce_regression_guard = False
                else:
                    raise TuningAbortError(
                        f"Chunked layer guard baseline failed for chunk_id={chunk_id}, layer={layer_name}."
                    )

            first_time = _tune_one_layer(layer, chunk_id)
            if not math.isfinite(first_time):
                raise TuningAbortError(
                    f"Chunked tuning produced invalid time for chunk_id={chunk_id}, layer={layer_name}."
                )

            if (
                layer_idx == 0
                and prev_chunk_tail_winner_time is not None
                and math.isfinite(float(prev_chunk_tail_winner_time))
            ):
                slowdown_pct = max(0.0, float(GEMM_LAYER_BASELINE_DVFS_SLOWDOWN_PCT))
                threshold_t = float(prev_chunk_tail_winner_time) * (1.0 + slowdown_pct / 100.0)
                if first_time > threshold_t:
                    wait_sec = _layer_guard_wait_sec(
                        LAYER_GUARD_DVFS_SETTLE_SEC,
                        recovery_active=chunk_recovery_active,
                    )
                    log_print(
                        f"Chunk-boundary guard: first layer ({layer_name}, chunk_id={chunk_id}) tuned={first_time:.4f}s "
                        f"is >{slowdown_pct:.1f}% slower than previous chunk tail winner "
                        f"{float(prev_chunk_tail_winner_time):.4f}s (threshold {threshold_t:.4f}s).",
                        log_handle,
                    )
                    if wait_sec > 0:
                        log_print(
                            f"Chunk-boundary guard: waiting {wait_sec}s then re-running {layer_name} once.",
                            log_handle,
                        )
                        time.sleep(wait_sec)

                    working_data = json.loads(json.dumps(pre_layer_data))
                    locked_k_by_layer = dict(pre_layer_locked)
                    retry_time = _tune_one_layer(layer, chunk_id)
                    if not math.isfinite(retry_time):
                        raise TuningAbortError(
                            f"Chunk-boundary guard rerun invalid for chunk_id={chunk_id}, layer={layer_name}."
                        )
                    if retry_time > threshold_t:
                        working_data = json.loads(json.dumps(pre_layer_data))
                        locked_k_by_layer = dict(pre_layer_locked)
                        log_print(
                            f"Chunk-boundary guard repeated for {layer_name}, chunk_id={chunk_id}: "
                            f"rerun={retry_time:.4f}s still above threshold {threshold_t:.4f}s. "
                            "Skipping this layer and keeping seeded dims.",
                            log_handle,
                        )
                        continue
                    first_time = retry_time

            if enforce_regression_guard and first_time > baseline_time + LAYER_REGRESSION_EPS_SEC:
                log_print(
                    f"Layer regression detected for chunk_id={chunk_id}, layer={layer_name}: "
                    f"baseline={baseline_time:.4f}s tuned={first_time:.4f}s. Re-running layer once.",
                    log_handle,
                )
                working_data = json.loads(json.dumps(pre_layer_data))
                locked_k_by_layer = dict(pre_layer_locked)
                second_time = _tune_one_layer(layer, chunk_id)
                if not math.isfinite(second_time):
                    raise TuningAbortError(
                        f"Chunked layer rerun produced invalid time for chunk_id={chunk_id}, layer={layer_name}."
                    )
                if second_time > baseline_time + LAYER_REGRESSION_EPS_SEC:
                    # Keep progress only up to previous layer (before this layer changed).
                    working_data = json.loads(json.dumps(pre_layer_data))
                    locked_k_by_layer = dict(pre_layer_locked)
                    log_print(
                        f"Layer regression repeated for chunk_id={chunk_id}, layer={layer_name}: "
                        f"baseline={baseline_time:.4f}s rerun={second_time:.4f}s. "
                        "Skipping this layer and preserving the previous state.",
                        log_handle,
                    )
                    continue
                first_time = second_time
            _persist_layer_state(layer, chunk_id)
            chunk_tail_winner_time = float(first_time)

        if chunk_tail_winner_time is not None and math.isfinite(float(chunk_tail_winner_time)):
            prev_chunk_tail_winner_time = float(chunk_tail_winner_time)

    final_time = benchmark_gemm_trial_data(
        log_handle,
        model_conf,
        prompt_len,
        working_data,
        temp_suffix=f".chunk_final_{chunk_schedule_tag}_{int(effective_chunk_size)}_{int(inflight)}_tmp.json5",
        heuristics_state=heuristics_state,
        recovery_key=make_chunk_heuristic_key(
            "final",
            prompt_len=int(prompt_len),
            chunk_size=int(effective_chunk_size),
            chunk_schedule=chunk_schedule_key,
            inflight=int(inflight),
        ),
    )
    if len(explicit_schedule) > 1:
        log_print(
            f"Final tuned candidate schedule={_format_chunk_schedule(chunk_plan)}, inflight={inflight}: {final_time:.4f}s",
            log_handle,
        )
    else:
        log_print(
            f"Final tuned candidate chunking={effective_chunk_size}, inflight={inflight}: {final_time:.4f}s",
            log_handle,
        )
    return working_data, final_time

def tune_all_gemm_chunking(log_handle, model_conf, mode_variant="gemm_chunking"):
    scheduled_mode = (str(mode_variant) == "gemm_chunkingS")
    mode_label = "GEMM Chunking Schedule Tuning" if scheduled_mode else "GEMM Chunking Tuning"
    log_print(f"\n>>> Starting {mode_label} for {model_conf['name']}", log_handle)

    data = load_config(model_conf)
    if not data:
        return
    heuristics_state = load_chunk_heuristics(model_conf, log_handle) if CHUNK_RECOVER else None
    outer_candidate_cap, final_candidate_cap = _get_search_space_limits()

    try:
        for prompt_len in _get_prompt_sizes_for_mode(mode_variant, model_conf):
            log_print(f"\n{'='*40}", log_handle)
            log_print(f"CHUNKED GEMM TUNING PROMPT SIZE: {prompt_len}", log_handle)
            log_print(f"{'='*40}", log_handle)

            existing_chunked = data.get("kernels_gemm_chunked")
            if _is_grouped_kernels_gemm_chunked(existing_chunked):
                if scheduled_mode:
                    has_prompt_config = any(
                        _group_prompt_len(group) == int(prompt_len) and len(_group_chunk_schedule(group)) > 1
                        for group in existing_chunked
                    )
                else:
                    has_prompt_config = any(
                        _group_prompt_len(group) == int(prompt_len) and len(_group_chunk_schedule(group)) <= 1
                        for group in existing_chunked
                    )
                if has_prompt_config:
                    log_print(
                        f"Found existing kernels_gemm_chunked config for prompt_len={prompt_len}; skipping tuning.",
                        log_handle,
                    )
                    continue

            data = setup_baseline_config(data, prompt_len)
            candidate_specs = []
            forced_chunk_schedule = _get_forced_chunking_schedule()
            force_schedule_active = False

            if scheduled_mode:
                if forced_chunk_schedule:
                    try:
                        forced_plan = _resolve_chunk_plan(
                            prompt_len,
                            _schedule_fallback_chunk_size(forced_chunk_schedule),
                            chunk_schedule=forced_chunk_schedule,
                            require_exact_schedule=True,
                        )
                        for unique_chunk in sorted(set(forced_plan)):
                            get_chunk_baselines_or_raise(data, model_conf, unique_chunk)
                    except RuntimeError as e:
                        log_print(str(e), log_handle)
                        raise SystemExit(1)
                    candidate_specs = [
                        {
                            "chunk_size": int(_schedule_fallback_chunk_size(forced_plan)),
                            "chunk_schedule": list(forced_plan),
                        }
                    ]
                    force_schedule_active = True
                    log_print(
                        "FORCE_CHUNKING_SCHEDULE active: skipping schedule-space outer search and "
                        f"tuning inflight only for schedule={_format_chunk_schedule(forced_plan)}.",
                        log_handle,
                    )
                else:
                    candidate_schedules = get_valid_chunk_schedules(prompt_len)
                    if not candidate_schedules:
                        log_print(f"No valid chunk schedules for prompt size {prompt_len}; skipping.", log_handle)
                        continue
                    log_print(
                        f"Candidate chunk schedules ({len(candidate_schedules)}): "
                        + ", ".join(_format_chunk_schedule(s) for s in candidate_schedules),
                        log_handle,
                    )
                    for schedule in candidate_schedules:
                        try:
                            chunk_plan = _resolve_chunk_plan(
                                prompt_len,
                                _schedule_fallback_chunk_size(schedule),
                                chunk_schedule=schedule,
                                require_exact_schedule=True,
                            )
                            for unique_chunk in sorted(set(chunk_plan)):
                                get_chunk_baselines_or_raise(data, model_conf, unique_chunk)
                        except RuntimeError as e:
                            log_print(str(e), log_handle)
                            raise SystemExit(1)
                        candidate_specs.append(
                            {
                                "chunk_size": int(_schedule_fallback_chunk_size(chunk_plan)),
                                "chunk_schedule": list(chunk_plan),
                            }
                        )
            else:
                candidate_chunk_sizes = get_valid_chunk_sizes(prompt_len)
                if not candidate_chunk_sizes:
                    log_print(f"No valid chunk sizes for prompt size {prompt_len}; skipping.", log_handle)
                    continue
                log_print(f"Candidate chunk sizes: {candidate_chunk_sizes}", log_handle)
                for chunk_size in candidate_chunk_sizes:
                    try:
                        get_chunk_baselines_or_raise(data, model_conf, chunk_size)
                    except RuntimeError as e:
                        log_print(str(e), log_handle)
                        raise SystemExit(1)
                    candidate_specs.append({"chunk_size": int(chunk_size), "chunk_schedule": []})
                log_print(
                    f"SEARCH_SPACE outer_cap={int(outer_candidate_cap)} is ignored for gemm_chunking mode.",
                    log_handle,
                )
                if forced_chunk_schedule:
                    log_print(
                        "FORCE_CHUNKING_SCHEDULE is ignored for gemm_chunking mode (only used by gemm_chunkingS).",
                        log_handle,
                    )

            best_chunk_size = -1
            best_chunk_schedule = []
            best_inflight = -1
            best_outer_time = float("inf")
            outer_candidates = []
            forced_inflight = -1
            try:
                forced_inflight = int(FORCH_INFLIGHT)
            except Exception:
                forced_inflight = -1

            if scheduled_mode and (not force_schedule_active):
                if int(outer_candidate_cap) == -1:
                    log_print(
                        "SEARCH_SPACE outer_cap=-1: exhaustive schedule outer search enabled.",
                        log_handle,
                    )
                elif int(outer_candidate_cap) > 0 and len(candidate_specs) > int(outer_candidate_cap):
                    reduced_specs = _select_schedule_candidates_for_active_search(
                        candidate_specs,
                        max_pool=int(outer_candidate_cap),
                    )
                    log_print(
                        f"SEARCH_SPACE outer_cap={int(outer_candidate_cap)}: reduced "
                        f"{len(candidate_specs)} schedules -> {len(reduced_specs)} candidates for outer search.",
                        log_handle,
                    )
                    candidate_specs = [
                        {
                            "chunk_size": int(spec["chunk_size"]),
                            "chunk_schedule": list(spec["chunk_schedule"]),
                        }
                        for spec in reduced_specs
                    ]
                else:
                    log_print(
                        f"SEARCH_SPACE outer_cap={int(outer_candidate_cap)}: using all {len(candidate_specs)} "
                        "schedule candidates for outer search.",
                        log_handle,
                    )

            for candidate in candidate_specs:
                chunk_size = int(candidate["chunk_size"])
                chunk_schedule = _normalize_chunk_schedule(candidate.get("chunk_schedule"))
                chunk_plan = _resolve_chunk_plan(
                    prompt_len,
                    chunk_size,
                    chunk_schedule=chunk_schedule if chunk_schedule else None,
                    require_exact_schedule=scheduled_mode,
                )
                is_scheduled_candidate = len(chunk_schedule) > 1
                num_chunks = len(chunk_plan)
                candidate_name = (
                    f"schedule={_format_chunk_schedule(chunk_plan)}"
                    if is_scheduled_candidate
                    else f"chunk_size={chunk_size}"
                )
                capped_inflight_candidates = _get_inflight_candidates(num_chunks)
                max_allowed_inflight = capped_inflight_candidates[-1] if capped_inflight_candidates else 0

                if forced_inflight > 0:
                    if forced_inflight > num_chunks:
                        log_print(
                            f"\n--- Outer search for {candidate_name} ({num_chunks} chunks) ---",
                            log_handle,
                        )
                        log_print(
                            f"FORCH_INFLIGHT={forced_inflight} exceeds available chunks ({num_chunks}); "
                            f"skipping {candidate_name}.",
                            log_handle,
                        )
                        continue
                    if max_allowed_inflight <= 0 or forced_inflight > max_allowed_inflight:
                        log_print(
                            f"\n--- Outer search for {candidate_name} ({num_chunks} chunks) ---",
                            log_handle,
                        )
                        log_print(
                            f"FORCH_INFLIGHT={forced_inflight} exceeds MAX_INFLIGHT={MAX_INFLIGHT}; "
                            f"skipping {candidate_name}.",
                            log_handle,
                        )
                        continue
                    inflight_candidates = [forced_inflight]
                    log_print(
                        f"\n--- Outer search for {candidate_name} ({num_chunks} chunks, "
                        f"forced inflight={forced_inflight}) ---",
                        log_handle,
                    )
                    log_print(
                        f"FORCH_INFLIGHT={forced_inflight} active; skipping inflight tuning sweep.",
                        log_handle,
                    )
                else:
                    inflight_candidates = list(capped_inflight_candidates)
                    if not inflight_candidates:
                        log_print(
                            f"\n--- Outer search for {candidate_name} ({num_chunks} chunks) ---",
                            log_handle,
                        )
                        log_print(
                            f"No valid inflight candidates for {candidate_name} with MAX_INFLIGHT={MAX_INFLIGHT}; skipping.",
                            log_handle,
                        )
                        continue
                    max_swept_inflight = inflight_candidates[-1]
                    log_print(
                        f"\n--- Outer search for {candidate_name} ({num_chunks} chunks, "
                        f"inflight 1..{max_swept_inflight}) ---",
                        log_handle,
                    )

                candidate_best_inflight = inflight_candidates[0]
                candidate_best_time = float("inf")

                for inflight in inflight_candidates:
                    trial_data, _, resolved_schedule = build_chunking_trial_data(
                        data,
                        model_conf,
                        prompt_len,
                        chunk_size,
                        inflight,
                        chunk_schedule=chunk_schedule if chunk_schedule else None,
                    )
                    resolved_schedule_key = _chunk_schedule_key(resolved_schedule)
                    resolved_schedule_tag = _chunk_schedule_file_tag(resolved_schedule)
                    trial_time = benchmark_gemm_trial_data(
                        log_handle,
                        model_conf,
                        prompt_len,
                        trial_data,
                        temp_suffix=f".chunk_outer_{resolved_schedule_tag}_{chunk_size}_{inflight}_tmp.json5",
                        heuristics_state=heuristics_state,
                        recovery_key=make_chunk_heuristic_key(
                            "outer",
                            prompt_len=int(prompt_len),
                            chunk_size=int(chunk_size),
                            chunk_schedule=resolved_schedule_key,
                            inflight=int(inflight),
                        ),
                    )

                    if is_scheduled_candidate:
                        log_print(
                            f"  Outer candidate schedule={_format_chunk_schedule(chunk_plan)}, "
                            f"inflight={inflight}: {trial_time:.4f}s",
                            log_handle,
                        )
                    else:
                        log_print(
                            f"  Outer candidate chunking={chunk_size}, inflight={inflight}: {trial_time:.4f}s",
                            log_handle,
                        )
                    if not math.isfinite(trial_time):
                        log_print(
                            f"  Outer candidate {candidate_name}, inflight={inflight} failed "
                            f"(timeout/invalid benchmark result).",
                            log_handle,
                        )
                        continue
                    if trial_time < candidate_best_time:
                        candidate_best_time = trial_time
                        candidate_best_inflight = inflight
                        if is_scheduled_candidate:
                            log_print(
                                f"  * New best for schedule={_format_chunk_schedule(chunk_plan)}: "
                                f"inflight={inflight} ({trial_time:.4f}s)",
                                log_handle,
                            )
                        else:
                            log_print(
                                f"  * New best for chunk_size={chunk_size}: inflight={inflight} ({trial_time:.4f}s)",
                                log_handle,
                            )

                if not math.isfinite(candidate_best_time):
                    log_print(
                        f"All outer candidates failed for {candidate_name}; "
                        "aborting gemm_chunking with no config commit.",
                        log_handle,
                    )
                    raise SystemExit(1)

                outer_candidates.append(
                    {
                        "chunk_size": int(chunk_size),
                        "chunk_schedule": list(chunk_schedule),
                        "inflight": int(candidate_best_inflight),
                        "outer_time": float(candidate_best_time),
                    }
                )

                if candidate_best_time < best_outer_time:
                    best_outer_time = candidate_best_time
                    best_chunk_size = chunk_size
                    best_chunk_schedule = list(chunk_schedule)
                    best_inflight = candidate_best_inflight
                    if is_scheduled_candidate:
                        log_print(
                            f"* New global outer best: schedule={_format_chunk_schedule(chunk_plan)}, "
                            f"inflight={best_inflight} ({best_outer_time:.4f}s)",
                            log_handle,
                        )
                    else:
                        log_print(
                            f"* New global outer best: chunking={best_chunk_size}, inflight={best_inflight} "
                            f"({best_outer_time:.4f}s)",
                            log_handle,
                        )

            if best_chunk_size <= 0:
                log_print(f"No valid chunking winner found for prompt size {prompt_len}.", log_handle)
                continue

            if len(best_chunk_schedule) > 1:
                log_print(
                    f"\nWinner outer config for prompt {prompt_len}: "
                    f"schedule={_format_chunk_schedule(best_chunk_schedule)}, "
                    f"inflight={best_inflight} ({best_outer_time:.4f}s)",
                    log_handle,
                )
            else:
                log_print(
                    f"\nWinner outer config for prompt {prompt_len}: chunking={best_chunk_size}, "
                    f"inflight={best_inflight} ({best_outer_time:.4f}s)",
                    log_handle,
                )

            sorted_outer_candidates = sorted(
                outer_candidates,
                key=lambda c: float(c.get("outer_time", float("inf"))),
            )
            if int(final_candidate_cap) == -1:
                candidate_trials = list(sorted_outer_candidates)
                log_print(
                    "SEARCH_SPACE final_cap=-1: exhaustive final candidate tuning enabled.",
                    log_handle,
                )
            else:
                final_count = max(1, int(final_candidate_cap))
                candidate_trials = sorted_outer_candidates[:final_count]
                log_print(
                    f"SEARCH_SPACE final_cap={final_count}: tuning top {len(candidate_trials)} "
                    "outer winners in the inner stage.",
                    log_handle,
                )
            if candidate_trials:
                log_print("Selected outer winning configs:", log_handle)
                for rank, candidate in enumerate(candidate_trials, start=1):
                    candidate_schedule = _normalize_chunk_schedule(candidate.get("chunk_schedule"))
                    candidate_chunk_size = int(candidate.get("chunk_size", 0))
                    candidate_inflight = int(candidate.get("inflight", 1))
                    candidate_outer = float(candidate.get("outer_time", float("inf")))
                    candidate_outer_str = (
                        f"{candidate_outer:.4f}s" if math.isfinite(candidate_outer) else "inf"
                    )
                    if len(candidate_schedule) > 1:
                        candidate_label = f"schedule={_format_chunk_schedule(candidate_schedule)}"
                    else:
                        candidate_label = f"chunking={candidate_chunk_size}"
                    log_print(
                        f"  [{rank}] {candidate_label}, inflight={candidate_inflight}, outer={candidate_outer_str}",
                        log_handle,
                    )

            best_final_data = None
            best_final_time = float("inf")
            best_final_chunk_size = -1
            best_final_chunk_schedule = []
            best_final_inflight = -1

            for candidate in candidate_trials:
                chunk_size = int(candidate["chunk_size"])
                chunk_schedule = _normalize_chunk_schedule(candidate.get("chunk_schedule"))
                inflight = int(candidate["inflight"])
                if len(chunk_schedule) > 1:
                    log_print(
                        f"\n=== Full tuning candidate schedule={_format_chunk_schedule(chunk_schedule)}, "
                        f"inflight={inflight} (outer={candidate['outer_time']:.4f}s) ===",
                        log_handle,
                    )
                else:
                    log_print(
                        f"\n=== Full tuning candidate chunking={chunk_size}, inflight={inflight} "
                        f"(outer={candidate['outer_time']:.4f}s) ===",
                        log_handle,
                    )
                tuned_data, tuned_time = tune_chunked_candidate(
                    log_handle,
                    model_conf,
                    data,
                    prompt_len,
                    chunk_size,
                    inflight,
                    heuristics_state=heuristics_state,
                    chunk_schedule=chunk_schedule if chunk_schedule else None,
                )
                if tuned_time < best_final_time:
                    best_final_time = tuned_time
                    best_final_data = tuned_data
                    best_final_chunk_size = chunk_size
                    best_final_chunk_schedule = list(chunk_schedule)
                    best_final_inflight = inflight
                    if len(chunk_schedule) > 1:
                        log_print(
                            f"* New final chunking best: schedule={_format_chunk_schedule(chunk_schedule)}, "
                            f"inflight={best_final_inflight} ({best_final_time:.4f}s)",
                            log_handle,
                        )
                    else:
                        log_print(
                            f"* New final chunking best: chunking={best_final_chunk_size}, "
                            f"inflight={best_final_inflight} ({best_final_time:.4f}s)",
                            log_handle,
                        )

            if best_final_data is None:
                log_print(f"No tuned chunking winner found for prompt size {prompt_len}.", log_handle)
                continue

            _apply_chunking_format(
                data,
                best_final_chunk_size,
                best_final_inflight,
                prompt_len=prompt_len,
                chunk_schedule=best_final_chunk_schedule if best_final_chunk_schedule else None,
            )
            source_kernels = _get_chunked_kernels_ref(
                best_final_data,
                create=False,
                reset=False,
                chunk_size=best_final_chunk_size,
                inflight=best_final_inflight,
                prompt_len=prompt_len,
                chunk_schedule=best_final_chunk_schedule if best_final_chunk_schedule else None,
            )
            target_group = _get_chunked_group_ref(
                data,
                create=True,
                reset=True,
                chunk_size=best_final_chunk_size,
                inflight=best_final_inflight,
                prompt_len=prompt_len,
                chunk_schedule=best_final_chunk_schedule if best_final_chunk_schedule else None,
            )
            if not isinstance(target_group, dict):
                raise RuntimeError("Failed to resolve grouped chunked destination entry for commit.")
            _normalize_chunked_group_meta(
                target_group,
                best_final_chunk_size,
                best_final_inflight,
                int(prompt_len),
                chunk_schedule=best_final_chunk_schedule if len(best_final_chunk_schedule) > 1 else [],
            )
            target_kernels = target_group.get("kernels")
            if not isinstance(target_kernels, list):
                target_group["kernels"] = []
                target_kernels = target_group["kernels"]
            target_kernels.extend(json.loads(json.dumps(source_kernels)))
            canonicalize_chunked_config(data, prune_duplicates=True)
            sort_kernels_gemm_entries(data, "kernels_gemm_chunked", include_chunk_id=True)
            save_config_via_temp(data, model_conf, temp_suffix=".chunk_commit_tmp.json5")
            if len(best_final_chunk_schedule) > 1:
                log_print(
                    f"Committed chunked GEMM config for prompt {prompt_len} with "
                    f"schedule={_format_chunk_schedule(best_final_chunk_schedule)}, "
                    f"inflight={best_final_inflight} (final tuned prefill={best_final_time:.4f}s)",
                    log_handle,
                )
            else:
                log_print(
                    f"Committed chunked GEMM config for prompt {prompt_len} with "
                    f"chunking={best_final_chunk_size}, inflight={best_final_inflight} "
                    f"(final tuned prefill={best_final_time:.4f}s)",
                    log_handle,
                )

    except TuningAbortError as e:
        log_print(str(e), log_handle)
        raise SystemExit(1)
    except KeyboardInterrupt:
        log_print("GEMM Chunking Tuning Interrupted!", log_handle)

def _import_hetero_backend_module():
    global _HETERO_BACKEND_MODULE
    if _HETERO_BACKEND_MODULE is False:
        return None
    if _HETERO_BACKEND_MODULE is None:
        try:
            import unified_llm_w4a16_hetero_libtorch as backend_module
        except Exception:
            _HETERO_BACKEND_MODULE = False
            return None
        _HETERO_BACKEND_MODULE = backend_module
    return _HETERO_BACKEND_MODULE


def _ensure_cw_tracing_available(log_handle=None):
    build_dir = str(_BUILD_DIR)
    probe_cmd = f"""
pushd ../../ >/dev/null && source utils/setup.sh && popd >/dev/null
python3 - <<'PY'
import os
import sys
sys.path.insert(0, {build_dir!r})
import unified_llm_w4a16_hetero_libtorch as backend_module
sys.stdout.write("1\\n" if bool(getattr(backend_module, "GET_TRACES_ENABLED", False)) else "0\\n")
sys.stdout.flush()
os._exit(0)
PY
""".strip()
    try:
        result = subprocess.run(
            probe_cmd,
            shell=True,
            executable="/bin/bash",
            cwd=str(_SCRIPT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
        )
    except Exception as exc:
        raise RuntimeError(f"gemm_CW failed while probing GET_TRACES support: {exc}") from exc

    output = str(result.stdout or "").strip()
    if result.returncode != 0:
        raise RuntimeError(
            "gemm_CW requires the hetero pybind backend to be importable under utils/setup.sh. "
            f"Probe failed with code {result.returncode}: {output}"
        )
    enabled = output.splitlines()[-1].strip() in ("1", "true", "True")
    if not enabled:
        raise RuntimeError(
            "gemm_CW requires trace-enabled runtime support. Enable GET_TRACES in "
            "include/unified_llm_w4a16_hetero/unified_llm_w4a16.hpp and rebuild."
        )
    log_print("gemm_CW trace runtime check passed (GET_TRACES_ENABLED=1).", log_handle)


def _cw_clone_data(data):
    return json.loads(json.dumps(data))


def _cw_model_arch_key(model_conf):
    model_id = str(model_conf.get("id", "")).strip().lower()
    if model_id == "gemma":
        return "gemma"
    if model_id.startswith("llama3"):
        return "llama3"
    if model_id.startswith("phi"):
        return "phi"
    if model_id.startswith("qwen"):
        return "qwen"
    return model_id


def _cw_stage_layer_names(model_conf, stage):
    stage = str(stage).strip().upper()
    if stage == "A":
        return []
    if stage == "G2":
        return ["o", "gate", "up", "down"]
    if stage != "G1":
        return []
    if _cw_model_arch_key(model_conf) == "gemma":
        return ["qkv"]
    return ["q", "k", "v"]


def _cw_layer_cfg_by_name(model_conf):
    return {
        str(layer.get("name", "")): layer
        for layer in model_conf.get("layers", [])
        if isinstance(layer, dict) and str(layer.get("name", ""))
    }


def _cw_node_sort_key(node_id):
    stage_order = {name: idx for idx, name in enumerate(CW_STAGE_ORDER)}
    chunk_id, layer_id, stage = node_id
    return (
        int(chunk_id),
        int(layer_id),
        stage_order.get(str(stage), len(stage_order) + 1),
    )


def _cw_window_sort_key(window):
    return (
        -int(window.get("attention_active", 0)),
        -int(window.get("feeds_critical_attention_count", 0)),
        -int(window.get("critical_tunable_count", 0)),
        -float(window.get("duration_sec", 0.0)),
        int(window.get("start_ts_us", 0)),
    )


def _cw_window_center_us(window):
    start_us = int(window.get("start_ts_us", 0))
    end_us = int(window.get("end_ts_us", start_us))
    return int((start_us + end_us) // 2)


def _cw_noise_floor_sec(reference_latency_sec):
    ref = max(0.0, float(reference_latency_sec))
    return max(CW_ACCEPT_ABS_IMPROVEMENT_SEC, ref * CW_ACCEPT_REL_IMPROVEMENT_PCT)


def _cw_latency_improves_enough(best_latency_sec, candidate_latency_sec):
    best = float(best_latency_sec)
    candidate = float(candidate_latency_sec)
    margin = _cw_noise_floor_sec(best)
    return candidate < (best - margin)


def _cw_latency_delta_ms(reference_latency_sec, candidate_latency_sec):
    return (float(candidate_latency_sec) - float(reference_latency_sec)) * 1000.0


def _cw_stage_bubble_lookup(stage_bubbles):
    lookup = {}
    for spec in _normalize_stage_bubbles(stage_bubbles):
        key = (
            int(spec.get("chunk_id", -1)),
            int(spec.get("layer_id", -1)),
            str(spec.get("stage", "")),
        )
        lookup[key] = int(spec.get("delay_us", 0))
    return lookup


def _cw_get_selected_stage_bubbles(data):
    return _normalize_stage_bubbles(_get_chunked_stage_bubbles_ref(data, create=False, reset=False))


def _cw_find_chunk_kernel(data, layer_name, chunk_id):
    kernels = _get_chunked_kernels_ref(data, create=False, reset=False)
    for kernel in kernels:
        if str(kernel.get("layer", "")) != str(layer_name):
            continue
        if _safe_int(kernel.get("chunk_id", -1), -1) != int(chunk_id):
            continue
        return kernel
    return None


def _cw_chunk_id_sequence(num_chunks):
    if CHUNK_ID_EXPLORE_DESCEND:
        return list(range(int(num_chunks) - 1, -1, -1))
    return list(range(int(num_chunks)))


def _cw_lock_source_chunk_id(num_chunks):
    sequence = _cw_chunk_id_sequence(num_chunks)
    return int(sequence[0]) if sequence else 0


def _cw_future_locked_chunks(num_chunks, source_chunk_id):
    sequence = _cw_chunk_id_sequence(num_chunks)
    try:
        source_idx = sequence.index(int(source_chunk_id))
    except ValueError:
        return []
    return [int(chunk_id) for chunk_id in sequence[source_idx + 1:]]


def _cw_kernel_snapshot(kernel):
    if not isinstance(kernel, dict):
        return None
    keys = ("use", "npuM", "npuK", "npuN", "fw_path", "dtype")
    return {key: json.loads(json.dumps(kernel.get(key))) for key in keys}


def _cw_k_share_dim(model_conf, kernel):
    return max(0, _k_tuning_upper_bound(model_conf, _safe_int(kernel.get("forK", 0), 0)))


def _cw_effective_split_family(model_conf, layer_cfg, kernel, chunk_id, num_chunks):
    split_mode = resolve_gemm_split_mode(layer_cfg)
    if split_mode == "M":
        return {"family": "M", "locked_npuK": None, "is_source": False}

    source_chunk_id = _cw_lock_source_chunk_id(num_chunks)
    is_source = int(chunk_id) == int(source_chunk_id)
    forM = max(0, _safe_int(kernel.get("forM", 0), 0))
    npuM = max(0, _safe_int(kernel.get("npuM", 0), 0))
    npuK = max(0, _safe_int(kernel.get("npuK", 0), 0))
    k_dim = _cw_k_share_dim(model_conf, kernel)
    use = bool(kernel.get("use", True))
    active_true_k = k_dim > 0 and 0 < npuK < k_dim

    if split_mode == "K":
        if is_source:
            return {"family": "K", "locked_npuK": None, "is_source": True}
        return {"family": "M", "locked_npuK": npuK if npuK > 0 else 0, "is_source": False}

    # MK layers stay on their currently active family. If the source chunk selected K-split,
    # later chunks preserve that locked npuK and tune M around it.
    if is_source and active_true_k:
        return {"family": "K", "locked_npuK": None, "is_source": True}
    if (not is_source) and active_true_k:
        return {"family": "M", "locked_npuK": npuK, "is_source": False}
    if not use and active_true_k:
        return {"family": "K", "locked_npuK": None, "is_source": is_source}
    if npuM > 0 or forM > 0:
        locked_npuK = npuK if ((not is_source) and active_true_k) else None
        return {"family": "M", "locked_npuK": locked_npuK, "is_source": is_source}
    return {"family": "K", "locked_npuK": None, "is_source": is_source}


def _cw_share_index_for_kernel(model_conf, layer_cfg, kernel, chunk_id, num_chunks):
    family_info = _cw_effective_split_family(model_conf, layer_cfg, kernel, chunk_id, num_chunks)
    family = family_info["family"]
    if family == "K":
        dim = _cw_k_share_dim(model_conf, kernel)
        current_val = max(0, _safe_int(kernel.get("npuK", 0), 0)) if bool(kernel.get("use", True)) else 0
    else:
        dim = max(0, _safe_int(kernel.get("forM", 0), 0))
        current_val = max(0, _safe_int(kernel.get("npuM", 0), 0)) if bool(kernel.get("use", True)) else 0

    if dim <= 0 or current_val <= 0:
        current_ratio = 0.0
    else:
        current_ratio = max(0.0, min(1.0, float(current_val) / float(dim)))
    best_idx = min(
        range(len(CW_SHARE_RATIOS)),
        key=lambda idx: abs(float(CW_SHARE_RATIOS[idx]) - current_ratio),
    )
    family_info["share_index"] = int(best_idx)
    family_info["share_ratio"] = float(CW_SHARE_RATIOS[best_idx])
    return family_info


def _cw_share_value_from_index(model_conf, kernel, family, share_index):
    share_index = max(0, min(int(share_index), len(CW_SHARE_RATIOS) - 1))
    ratio = float(CW_SHARE_RATIOS[share_index])
    if family == "K":
        dim = _cw_k_share_dim(model_conf, kernel)
        raw = int(dim if share_index == len(CW_SHARE_RATIOS) - 1 else round(dim * ratio))
        target = _round_to_nearest_multiple(raw, 512)
        target = _cap_k_tuning_value(model_conf, _safe_int(kernel.get("forK", 0), 0), target)
        if share_index > 0 and target <= 0 and dim > 0:
            target = min(dim, 512)
        return int(max(0, min(target, dim)))
    dim = max(0, _safe_int(kernel.get("forM", 0), 0))
    raw = int(dim if share_index == len(CW_SHARE_RATIOS) - 1 else round(dim * ratio))
    target = _round_to_nearest_multiple(raw, 256)
    if share_index > 0 and target <= 0 and dim > 0:
        target = min(dim, 256)
    return int(max(0, min(target, dim)))


def _cw_propagate_locked_k(trial_data, model_conf, layer_cfg, source_chunk_id, locked_npuK, num_chunks):
    changed = False
    layer_name = str(layer_cfg.get("name", ""))
    for future_chunk_id in _cw_future_locked_chunks(num_chunks, source_chunk_id):
        kernel = _cw_find_chunk_kernel(trial_data, layer_name, future_chunk_id)
        if kernel is None:
            continue
        before = _cw_kernel_snapshot(kernel)
        current_npuM = max(0, _safe_int(kernel.get("npuM", 0), 0)) if bool(kernel.get("use", True)) else 0
        update_kernel_config_gemm_m_locked_k(
            kernel,
            current_npuM,
            int(locked_npuK),
            _safe_int(kernel.get("forM", 0), 0),
            _safe_int(kernel.get("forK", 0), 0),
            _safe_int(kernel.get("forN", 0), 0),
        )
        if before != _cw_kernel_snapshot(kernel):
            changed = True
    return changed


def _cw_make_kernel_action(data, model_conf, num_chunks, node_id, layer_name, direction):
    chunk_id, layer_id, stage = node_id
    layer_cfg = _cw_layer_cfg_by_name(model_conf).get(str(layer_name))
    if layer_cfg is None:
        return None
    kernel = _cw_find_chunk_kernel(data, layer_name, chunk_id)
    if kernel is None:
        return None

    family_info = _cw_share_index_for_kernel(model_conf, layer_cfg, kernel, chunk_id, num_chunks)
    family = family_info["family"]
    current_index = int(family_info["share_index"])
    target_index = current_index + 1 if str(direction) == "tighten" else current_index - 1
    if target_index < 0 or target_index >= len(CW_SHARE_RATIOS):
        return None
    if family == "K" and not bool(family_info.get("is_source", False)):
        return None

    current_val = (
        max(0, _safe_int(kernel.get("npuK", 0), 0))
        if family == "K"
        else max(0, _safe_int(kernel.get("npuM", 0), 0))
    )
    if not bool(kernel.get("use", True)):
        current_val = 0
    target_val = _cw_share_value_from_index(model_conf, kernel, family, target_index)
    if int(target_val) == int(current_val) and bool(target_val > 0) == bool(kernel.get("use", True)):
        return None

    return {
        "kind": "kernel",
        "chunk_id": int(chunk_id),
        "layer_id": int(layer_id),
        "stage": str(stage),
        "layer_name": str(layer_name),
        "direction": str(direction),
        "family": str(family),
        "share_index": int(target_index),
        "current_share_index": int(current_index),
        "target_value": int(target_val),
        "locked_npuK": family_info.get("locked_npuK"),
        "summary": (
            f"{direction} {layer_name} chunk_id={int(chunk_id)} "
            f"{family}-share {int(CW_SHARE_RATIOS[current_index] * 100)}% -> "
            f"{int(CW_SHARE_RATIOS[target_index] * 100)}% "
            f"(window node: chunk={int(chunk_id)} layer={int(layer_id)} stage={stage})"
        ),
    }


def _cw_make_bubble_action(data, node_record):
    node_id = tuple(node_record.get("node_id"))
    chunk_id, layer_id, stage = node_id
    if str(stage) not in ("G1", "G2"):
        return None
    bubble_lookup = _cw_stage_bubble_lookup(_cw_get_selected_stage_bubbles(data))
    current_delay = int(bubble_lookup.get(node_id, 0))
    slack_sec = max(0.0, float(node_record.get("slack_sec", 0.0)))
    safe_delay_us = int(max(0.0, slack_sec - CW_SLACK_TOLERANCE_SEC) * 1_000_000.0)
    safe_delay_us = min(CW_MAX_BUBBLE_US, (safe_delay_us // CW_BUBBLE_QUANTUM_US) * CW_BUBBLE_QUANTUM_US)
    if safe_delay_us <= int(current_delay):
        return None
    target_delay = int(min(safe_delay_us, int(current_delay) + CW_BUBBLE_QUANTUM_US))
    if target_delay <= int(current_delay):
        return None
    return {
        "kind": "bubble",
        "chunk_id": int(chunk_id),
        "layer_id": int(layer_id),
        "stage": str(stage),
        "delay_us": int(target_delay),
        "summary": (
            f"bubble chunk_id={int(chunk_id)} layer_id={int(layer_id)} "
            f"stage={stage} {int(current_delay)}us -> {int(target_delay)}us"
        ),
    }


def _cw_apply_kernel_action(trial_data, model_conf, num_chunks, action):
    layer_name = str(action.get("layer_name", ""))
    chunk_id = int(action.get("chunk_id", -1))
    kernel = _cw_find_chunk_kernel(trial_data, layer_name, chunk_id)
    layer_cfg = _cw_layer_cfg_by_name(model_conf).get(layer_name)
    if kernel is None or layer_cfg is None:
        return False

    before = _cw_kernel_snapshot(kernel)
    family = str(action.get("family", "M"))
    target_index = int(action.get("share_index", 0))
    target_value = _cw_share_value_from_index(model_conf, kernel, family, target_index)

    if family == "M" and action.get("locked_npuK") is not None:
        update_kernel_config_gemm_m_locked_k(
            kernel,
            int(target_value),
            int(action.get("locked_npuK", 0)),
            _safe_int(kernel.get("forM", 0), 0),
            _safe_int(kernel.get("forK", 0), 0),
            _safe_int(kernel.get("forN", 0), 0),
        )
    else:
        update_kernel_config_gemm(
            kernel,
            family,
            int(target_value),
            _safe_int(kernel.get("forM", 0), 0),
            _safe_int(kernel.get("forK", 0), 0),
            _safe_int(kernel.get("forN", 0), 0),
            layer_cfg,
        )

    changed = before != _cw_kernel_snapshot(kernel)
    if family == "K":
        changed = (
            _cw_propagate_locked_k(
                trial_data,
                model_conf,
                layer_cfg,
                chunk_id,
                max(0, _safe_int(kernel.get("npuK", 0), 0)),
                num_chunks,
            )
            or changed
        )
    return changed


def _cw_apply_bubble_action(trial_data, action):
    bubbles = _get_chunked_stage_bubbles_ref(trial_data, create=True, reset=False)
    key = (
        int(action.get("chunk_id", -1)),
        int(action.get("layer_id", -1)),
        str(action.get("stage", "")),
    )
    new_delay = int(action.get("delay_us", 0))
    normalized = []
    old_delay = 0
    replaced = False
    for spec in _normalize_stage_bubbles(bubbles):
        spec_key = (
            int(spec.get("chunk_id", -1)),
            int(spec.get("layer_id", -1)),
            str(spec.get("stage", "")),
        )
        if spec_key == key:
            old_delay = int(spec.get("delay_us", 0))
            if new_delay > 0:
                normalized.append(
                    {
                        "chunk_id": key[0],
                        "layer_id": key[1],
                        "stage": key[2],
                        "delay_us": int(new_delay),
                    }
                )
            replaced = True
        else:
            normalized.append(spec)
    if not replaced and new_delay > 0:
        normalized.append(
            {
                "chunk_id": key[0],
                "layer_id": key[1],
                "stage": key[2],
                "delay_us": int(new_delay),
            }
        )
    normalized = _normalize_stage_bubbles(normalized)
    bubbles[:] = normalized
    return int(old_delay) != int(new_delay)


def _cw_apply_edit_set(data, model_conf, num_chunks, edits):
    trial_data = _cw_clone_data(data)
    changed = False
    for action in edits:
        if str(action.get("kind", "")) == "kernel":
            changed = _cw_apply_kernel_action(trial_data, model_conf, num_chunks, action) or changed
        elif str(action.get("kind", "")) == "bubble":
            changed = _cw_apply_bubble_action(trial_data, action) or changed
    if not changed:
        return None
    sort_all_kernels(trial_data)
    return trial_data


def _cw_proposal_key(edits):
    payload = []
    for action in edits:
        kind = str(action.get("kind", ""))
        if kind == "kernel":
            payload.append(
                (
                    "kernel",
                    str(action.get("layer_name", "")),
                    int(action.get("chunk_id", -1)),
                    str(action.get("family", "")),
                    int(action.get("share_index", -1)),
                )
            )
        elif kind == "bubble":
            payload.append(
                (
                    "bubble",
                    int(action.get("chunk_id", -1)),
                    int(action.get("layer_id", -1)),
                    str(action.get("stage", "")),
                    int(action.get("delay_us", 0)),
                )
            )
    payload.sort()
    return json.dumps(payload, separators=(",", ":"), sort_keys=False)


def _cw_normalize_trace_row(raw):
    if not isinstance(raw, dict):
        return None
    stage = str(raw.get("stage", "")).strip().upper()
    if stage not in CW_STAGE_ORDER:
        return None
    try:
        row = {
            "run_id": str(raw.get("run_id", "")).strip(),
            "model": str(raw.get("model", "")).strip(),
            "arch": str(raw.get("arch", "")).strip(),
            "chunk_id": int(raw.get("chunk_id", -1)),
            "slot_id": int(raw.get("slot_id", -1)),
            "layer_id": int(raw.get("layer_id", -1)),
            "stage": stage,
            "host_ready_ts_us": int(raw.get("host_ready_ts_us", 0)),
            "start_ts_us": int(raw.get("start_ts_us", 0)),
            "end_ts_us": int(raw.get("end_ts_us", 0)),
            "start_pos": int(raw.get("start_pos", 0)),
            "seq_len": int(raw.get("seq_len", 0)),
        }
    except Exception:
        return None
    if row["chunk_id"] < 0 or row["layer_id"] < 0:
        return None
    if row["end_ts_us"] < row["start_ts_us"]:
        return None
    return row


def _cw_load_latest_trace_rows(trace_path, preferred_tag=""):
    if not trace_path or not os.path.exists(trace_path):
        return []
    grouped = defaultdict(list)
    run_order = []
    with open(trace_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except Exception:
                continue
            row = _cw_normalize_trace_row(raw)
            if row is None:
                continue
            run_id = str(row.get("run_id", ""))
            if not run_id:
                continue
            if run_id not in grouped:
                run_order.append(run_id)
            grouped[run_id].append(row)
    if not run_order:
        return []
    candidate_run_ids = list(run_order)
    if preferred_tag:
        tagged = [run_id for run_id in run_order if run_id.startswith(f"{preferred_tag}_")]
        if tagged:
            candidate_run_ids = tagged
    selected_run_id = candidate_run_ids[-1]
    rows = grouped.get(selected_run_id, [])
    rows.sort(
        key=lambda row: (
            int(row.get("start_ts_us", 0)),
            int(row.get("chunk_id", -1)),
            int(row.get("layer_id", -1)),
            CW_STAGE_ORDER.index(str(row.get("stage", "G1"))),
        )
    )
    return rows


def _cw_trial_signature(trial_data):
    payload = _cw_clone_data(trial_data)
    _strip_temp_trace_fields(payload)
    sort_all_kernels(payload)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _cw_make_recovery_key(stage, prompt_len, trial_data, **kwargs):
    active_inflight = -1
    try:
        _, active_inflight = _get_active_chunking_settings(trial_data)
    except Exception:
        active_inflight = -1
    active_schedule = _normalize_chunk_schedule(_get_active_chunking_schedule(trial_data))
    payload = {
        "prompt_len": int(prompt_len),
        "trial_signature": _cw_trial_signature(trial_data),
        "strategy_version": str(CW_STRATEGY_VERSION),
    }
    if int(active_inflight) > 0:
        payload["effective_inflight"] = int(active_inflight)
    if active_schedule:
        payload["effective_schedule"] = _chunk_schedule_key(active_schedule)
    payload.update(kwargs)
    return make_chunk_heuristic_key(str(stage), **payload)


def _cw_get_cached_trace_run(heuristics_state, recovery_key):
    cached = get_chunk_artifact_state(heuristics_state, "cw_trace_runs", recovery_key)
    if not isinstance(cached, dict):
        return None
    try:
        trace_latency_sec = float(cached.get("trace_latency_sec", float("inf")))
    except Exception:
        return None
    raw_rows = cached.get("trace_rows", [])
    if not isinstance(raw_rows, list):
        return None
    trace_rows = []
    for raw_row in raw_rows:
        row = _cw_normalize_trace_row(raw_row)
        if row is not None:
            trace_rows.append(row)
    if not math.isfinite(trace_latency_sec) or not trace_rows:
        return None
    return trace_latency_sec, trace_rows


def _cw_set_cached_trace_run(model_conf, heuristics_state, recovery_key, trace_latency_sec, trace_rows):
    if heuristics_state is None or not math.isfinite(float(trace_latency_sec)):
        return
    normalized_rows = []
    for raw_row in trace_rows:
        row = _cw_normalize_trace_row(raw_row)
        if row is not None:
            normalized_rows.append(row)
    if not normalized_rows:
        return
    set_chunk_artifact_state(
        model_conf,
        heuristics_state,
        "cw_trace_runs",
        recovery_key,
        {
            "trace_latency_sec": float(trace_latency_sec),
            "trace_rows": normalized_rows,
        },
    )


def _benchmark_gemm_trial_data_latency_only(
    log_handle,
    model_conf,
    prompt_len,
    trial_data,
    temp_suffix,
    heuristics_state=None,
    recovery_key=None,
):
    latency_trial_data = _cw_clone_data(trial_data)
    _strip_temp_trace_fields(latency_trial_data)
    return benchmark_gemm_trial_data(
        log_handle,
        model_conf,
        prompt_len,
        latency_trial_data,
        temp_suffix=temp_suffix,
        heuristics_state=heuristics_state,
        recovery_key=recovery_key,
    )


def _benchmark_gemm_trial_data_with_trace(
    log_handle,
    model_conf,
    prompt_len,
    trial_data,
    temp_suffix,
    trace_sync_stages=False,
    heuristics_state=None,
    recovery_key=None,
):
    if CHUNK_RECOVER and heuristics_state is not None and recovery_key is not None:
        cached = _cw_get_cached_trace_run(heuristics_state, recovery_key)
        if cached is not None:
            trace_latency_sec, trace_rows = cached
            log_print(
                f"Recovered cached CW trace trial {recovery_key}: "
                f"{trace_latency_sec:.4f}s ({len(trace_rows)} trace rows)",
                log_handle,
            )
            return trace_latency_sec, trace_rows

    trace_dir = tempfile.mkdtemp(prefix="gemm_cw_trace_")
    trace_path = os.path.join(trace_dir, "prefill_trace.jsonl")
    trace_tag = f"cw_{model_conf.get('id', 'model')}_{prompt_len}_{time.time_ns()}"
    traced_trial_data = _cw_clone_data(trial_data)
    traced_trial_data["trace_output_path"] = trace_path
    traced_trial_data["trace_run_tag"] = trace_tag
    traced_trial_data["trace_sync_stages"] = bool(trace_sync_stages)
    try:
        trace_latency_sec = benchmark_gemm_trial_data(
            log_handle,
            model_conf,
            prompt_len,
            traced_trial_data,
            temp_suffix=temp_suffix,
            heuristics_state=None,
            recovery_key=None,
        )
        trace_rows = _cw_load_latest_trace_rows(trace_path, preferred_tag=trace_tag)
        if recovery_key is not None:
            _cw_set_cached_trace_run(model_conf, heuristics_state, recovery_key, trace_latency_sec, trace_rows)
        return trace_latency_sec, trace_rows
    finally:
        shutil.rmtree(trace_dir, ignore_errors=True)


def _cw_build_trace_state(trace_rows, trace_latency_sec, model_conf):
    node_rows = {}
    for row in trace_rows:
        normalized = _cw_normalize_trace_row(row)
        if normalized is None:
            continue
        node_id = (normalized["chunk_id"], normalized["layer_id"], normalized["stage"])
        node_rows[node_id] = normalized

    if not node_rows:
        return None

    preds = {node_id: set() for node_id in node_rows}
    succs = {node_id: set() for node_id in node_rows}

    def _add_edge(src, dst):
        if src not in node_rows or dst not in node_rows:
            return
        succs[src].add(dst)
        preds[dst].add(src)

    stage_pairs = [("G1", "A"), ("A", "G2")]
    for chunk_id, layer_id, stage in list(node_rows.keys()):
        for src_stage, dst_stage in stage_pairs:
            if stage == src_stage:
                _add_edge((chunk_id, layer_id, src_stage), (chunk_id, layer_id, dst_stage))
        if stage == "G2":
            _add_edge((chunk_id, layer_id, "G2"), (chunk_id, layer_id + 1, "G1"))
        if stage == "A" and chunk_id > 0:
            _add_edge((chunk_id - 1, layer_id, "G1"), (chunk_id, layer_id, "A"))

    indegree = {node_id: len(pred_set) for node_id, pred_set in preds.items()}
    ready = sorted([node_id for node_id, deg in indegree.items() if deg == 0], key=_cw_node_sort_key)
    topo = []
    while ready:
        node_id = ready.pop(0)
        topo.append(node_id)
        for succ in sorted(succs[node_id], key=_cw_node_sort_key):
            indegree[succ] -= 1
            if indegree[succ] == 0:
                ready.append(succ)
                ready.sort(key=_cw_node_sort_key)
    if len(topo) != len(node_rows):
        topo = sorted(node_rows.keys(), key=_cw_node_sort_key)

    node_records = {}
    for node_id, row in node_rows.items():
        duration_sec = max(0.0, float(row["end_ts_us"] - row["start_ts_us"]) / 1_000_000.0)
        node_records[node_id] = {
            "node_id": node_id,
            "row": row,
            "duration_sec": duration_sec,
            "tunable": str(row["stage"]) in ("G1", "G2"),
        }

    earliest_start = {}
    earliest_end = {}
    for node_id in topo:
        max_pred_end = 0.0
        if preds[node_id]:
            max_pred_end = max(earliest_end[pred] for pred in preds[node_id])
        earliest_start[node_id] = max_pred_end
        earliest_end[node_id] = max_pred_end + float(node_records[node_id]["duration_sec"])

    longest_tail = {}
    for node_id in reversed(topo):
        best_succ = 0.0
        if succs[node_id]:
            best_succ = max(longest_tail[succ] for succ in succs[node_id])
        longest_tail[node_id] = float(node_records[node_id]["duration_sec"]) + best_succ

    total_path_sec = max(earliest_end.values()) if earliest_end else 0.0
    for node_id in topo:
        slack_sec = max(0.0, float(total_path_sec) - (float(earliest_start[node_id]) + float(longest_tail[node_id])))
        node_records[node_id]["earliest_start_sec"] = float(earliest_start[node_id])
        node_records[node_id]["earliest_end_sec"] = float(earliest_end[node_id])
        node_records[node_id]["slack_sec"] = float(slack_sec)
        node_records[node_id]["critical"] = bool(abs(slack_sec) <= CW_SLACK_TOLERANCE_SEC)

    boundaries = sorted(
        {
            int(record["row"]["start_ts_us"])
            for record in node_records.values()
        }.union(
            {
                int(record["row"]["end_ts_us"])
                for record in node_records.values()
            }
        )
    )
    windows = []
    previous_window = None
    for start_us, end_us in zip(boundaries, boundaries[1:]):
        if int(end_us) <= int(start_us):
            continue
        active_nodes = []
        for node_id, record in node_records.items():
            row = record["row"]
            if int(row["start_ts_us"]) < int(end_us) and int(row["end_ts_us"]) > int(start_us):
                active_nodes.append(node_id)
        if not active_nodes:
            continue
        active_nodes.sort(key=_cw_node_sort_key)
        critical_tunable_nodes = [
            node_id for node_id in active_nodes if node_records[node_id]["tunable"] and node_records[node_id]["critical"]
        ]
        noncritical_tunable_nodes = [
            node_id for node_id in active_nodes if node_records[node_id]["tunable"] and not node_records[node_id]["critical"]
        ]
        attention_nodes = [node_id for node_id in active_nodes if str(node_id[2]) == "A"]
        tunable_active_nodes = list(critical_tunable_nodes) + list(noncritical_tunable_nodes)
        signature = tuple(active_nodes)
        if (
            previous_window is not None
            and previous_window.get("active_signature") == signature
            and int(previous_window.get("end_ts_us", -1)) == int(start_us)
        ):
            previous_window["end_ts_us"] = int(end_us)
            previous_window["duration_sec"] = float(previous_window["end_ts_us"] - previous_window["start_ts_us"]) / 1_000_000.0
            continue
        feeds_critical_attention_nodes = []
        for node_id in active_nodes:
            if str(node_id[2]) != "G1":
                continue
            for succ in succs.get(node_id, set()):
                succ_record = node_records.get(succ)
                if succ_record is None or str(succ[2]) != "A":
                    continue
                if bool(succ_record.get("critical", False)):
                    feeds_critical_attention_nodes.append(node_id)
                    break
        previous_window = {
            "start_ts_us": int(start_us),
            "end_ts_us": int(end_us),
            "duration_sec": float(int(end_us) - int(start_us)) / 1_000_000.0,
            "active_nodes": active_nodes,
            "active_signature": signature,
            "critical_tunable_nodes": critical_tunable_nodes,
            "noncritical_tunable_nodes": noncritical_tunable_nodes,
            "tunable_active_nodes": tunable_active_nodes,
            "attention_nodes": attention_nodes,
            "feeds_critical_attention_nodes": feeds_critical_attention_nodes,
            "critical_tunable_count": len(critical_tunable_nodes),
            "noncritical_tunable_count": len(noncritical_tunable_nodes),
            "tunable_active_count": len(tunable_active_nodes),
            "attention_active": len(attention_nodes),
            "feeds_critical_attention_count": len(feeds_critical_attention_nodes),
            "attention_protected": bool(len(attention_nodes) > 0 or len(feeds_critical_attention_nodes) > 0),
        }
        windows.append(previous_window)

    base_pressure = 0.0
    attention_overlap_penalty = 0.0
    critical_attention_penalty = 0.0
    slot_attention_busy_sec = defaultdict(float)
    slot_projection_busy_sec = defaultdict(float)
    for window in windows:
        dt = float(window.get("duration_sec", 0.0))
        critical_tunable = int(window.get("critical_tunable_count", 0))
        noncritical_tunable = int(window.get("noncritical_tunable_count", 0))
        tunable_active = int(window.get("tunable_active_count", 0))
        attention_active = int(window.get("attention_active", 0))
        base_pressure += dt * ((2 * critical_tunable) + noncritical_tunable)
        attention_overlap_penalty += dt * attention_active * tunable_active
        critical_attention_penalty += dt * attention_active * critical_tunable

    observed_slots = set()
    for record in node_records.values():
        row = record.get("row", {})
        slot_id = max(0, _safe_int(row.get("slot_id", 0), 0))
        observed_slots.add(int(slot_id))
        duration_sec = float(record.get("duration_sec", 0.0))
        if str(row.get("stage", "")) == "A":
            slot_attention_busy_sec[int(slot_id)] += float(duration_sec)
        else:
            slot_projection_busy_sec[int(slot_id)] += float(duration_sec)
    if not observed_slots:
        observed_slots.add(0)
    attention_busy_values = [float(slot_attention_busy_sec.get(slot_id, 0.0)) for slot_id in sorted(observed_slots)]
    projection_busy_values = [float(slot_projection_busy_sec.get(slot_id, 0.0)) for slot_id in sorted(observed_slots)]
    attention_busy_spread_sec = max(attention_busy_values) - min(attention_busy_values) if attention_busy_values else 0.0
    projection_busy_spread_sec = max(projection_busy_values) - min(projection_busy_values) if projection_busy_values else 0.0

    protected_pressure = (
        float(base_pressure)
        + float(CW_ATTENTION_WEIGHT) * float(attention_overlap_penalty)
        + float(CW_CRITICAL_ATTENTION_WEIGHT) * float(critical_attention_penalty)
    )

    return {
        "rows": [node_rows[node_id] for node_id in topo],
        "node_records": node_records,
        "preds": preds,
        "succs": succs,
        "topo": topo,
        "windows": windows,
        "pressure": float(protected_pressure),
        "base_pressure": float(base_pressure),
        "attention_overlap_penalty": float(attention_overlap_penalty),
        "critical_attention_penalty": float(critical_attention_penalty),
        "protected_pressure": float(protected_pressure),
        "slot_attention_busy_sec": {int(slot_id): float(slot_attention_busy_sec.get(slot_id, 0.0)) for slot_id in sorted(observed_slots)},
        "slot_projection_busy_sec": {int(slot_id): float(slot_projection_busy_sec.get(slot_id, 0.0)) for slot_id in sorted(observed_slots)},
        "attention_busy_spread_sec": float(attention_busy_spread_sec),
        "projection_busy_spread_sec": float(projection_busy_spread_sec),
        "critical_path_sec": float(total_path_sec),
        "trace_latency_sec": float(trace_latency_sec),
        "trace_span_sec": float(max(int(record["row"]["end_ts_us"]) for record in node_records.values())) / 1_000_000.0,
        "arch": _cw_model_arch_key(model_conf),
    }


def _cw_windows_overlap(lhs, rhs):
    if not isinstance(lhs, dict) or not isinstance(rhs, dict):
        return False
    return not (
        int(lhs.get("end_ts_us", 0)) <= int(rhs.get("start_ts_us", 0))
        or int(rhs.get("end_ts_us", 0)) <= int(lhs.get("start_ts_us", 0))
    )


def _cw_bootstrap_rank_key(candidate):
    return (
        float(candidate.get("attention_busy_spread_sec", float("inf"))),
        float(candidate.get("projection_busy_spread_sec", float("inf"))),
        float(candidate.get("protected_pressure", float("inf"))),
        float(candidate.get("true_latency_sec", float("inf"))),
        int(candidate.get("inflight", 1)),
        _chunk_schedule_key(candidate.get("chunk_schedule", [])),
    )


def _cw_apply_latency_gate(
    candidates,
    gate_pct=CW_LATENCY_GATE_PCT,
    fallback_gate_pct=CW_LATENCY_GATE_FALLBACK_PCT,
    min_survivors=CW_MIN_GATE_SURVIVORS,
):
    valid = [
        candidate
        for candidate in candidates
        if math.isfinite(float(candidate.get("true_latency_sec", float("inf"))))
        and math.isfinite(float(candidate.get("protected_pressure", float("inf"))))
    ]
    if not valid:
        return [], float("inf"), None

    best_chunked_latency = min(float(candidate["true_latency_sec"]) for candidate in valid)
    gate_factor = 1.0 + max(0.0, float(gate_pct))
    survivors = [
        candidate
        for candidate in valid
        if float(candidate["true_latency_sec"]) <= float(best_chunked_latency) * gate_factor
    ]
    used_gate_pct = float(gate_pct)

    if len(survivors) < max(1, int(min_survivors)):
        fallback_factor = 1.0 + max(0.0, float(fallback_gate_pct))
        widened = [
            candidate
            for candidate in valid
            if float(candidate["true_latency_sec"]) <= float(best_chunked_latency) * fallback_factor
        ]
        if widened:
            survivors = widened
            used_gate_pct = float(fallback_gate_pct)

    if not survivors:
        survivors = list(valid)
        used_gate_pct = None

    survivors.sort(key=_cw_bootstrap_rank_key)
    return survivors, float(best_chunked_latency), used_gate_pct


def _cw_matching_scheduled_groups(data, prompt_len):
    kernels_grouped = data.get("kernels_gemm_chunked")
    if not _is_grouped_kernels_gemm_chunked(kernels_grouped):
        return []

    matches = []
    for group_idx, group in enumerate(kernels_grouped):
        schedule = _group_chunk_schedule(group)
        if int(_group_prompt_len(group)) != int(prompt_len):
            continue
        if len(schedule) <= 1:
            continue
        matches.append(
            {
                "group_index": int(group_idx),
                "group": json.loads(json.dumps(group)),
                "chunk_size": int(_group_chunk_size(group)),
                "inflight": int(_group_inflight(group)),
                "chunk_schedule": list(schedule),
            }
        )
    return matches


def _cw_drop_existing_scheduled_groups(data, prompt_len):
    kernels_grouped = data.get("kernels_gemm_chunked")
    if not _is_grouped_kernels_gemm_chunked(kernels_grouped):
        return 0

    kept_groups = []
    removed_count = 0
    for group in kernels_grouped:
        if int(_group_prompt_len(group)) == int(prompt_len) and len(_group_chunk_schedule(group)) > 1:
            removed_count += 1
            continue
        kept_groups.append(group)

    if removed_count > 0:
        data["kernels_gemm_chunked"] = kept_groups
    return int(removed_count)


def _cw_build_existing_group_trial_data(data, prompt_len, group_entry):
    trial_data = _cw_clone_data(data)
    trial_data = setup_baseline_config(trial_data, prompt_len)
    trial_data["heterogeneity"] = "hetero"
    trial_data["prompt_len"] = int(prompt_len)
    trial_data["kernels_gemm_chunked"] = []
    trial_data.pop("npu_dim", None)

    chunk_size = int(group_entry.get("chunk_size", 0))
    inflight = int(group_entry.get("inflight", 1))
    chunk_schedule = _normalize_chunk_schedule(group_entry.get("chunk_schedule"))
    _apply_chunking_format(
        trial_data,
        chunk_size,
        inflight,
        prompt_len=prompt_len,
        chunk_schedule=chunk_schedule,
    )

    group_copy = json.loads(json.dumps(group_entry.get("group", {})))
    _normalize_chunked_group_meta(
        group_copy,
        chunk_size,
        inflight,
        int(prompt_len),
        chunk_schedule=chunk_schedule,
    )
    trial_data["kernels_gemm_chunked"] = [group_copy]
    return trial_data


def _cw_benchmark_chunked_incumbent(log_handle, model_conf, prompt_len, data, heuristics_state=None):
    group_entries = _cw_matching_scheduled_groups(data, prompt_len)
    if not group_entries:
        return None

    best_incumbent = None
    for group_entry in group_entries:
        chunk_size = int(group_entry.get("chunk_size", 0))
        inflight = int(group_entry.get("inflight", 1))
        chunk_schedule = _normalize_chunk_schedule(group_entry.get("chunk_schedule"))
        schedule_tag = _chunk_schedule_file_tag(chunk_schedule)
        trial_data = _cw_build_existing_group_trial_data(data, prompt_len, group_entry)
        recovery_key = _cw_make_recovery_key(
            "cw_incumbent_latency",
            prompt_len,
            trial_data,
            schedule=_chunk_schedule_key(chunk_schedule),
            inflight=int(inflight),
        )
        latency_sec = _benchmark_gemm_trial_data_latency_only(
            log_handle,
            model_conf,
            prompt_len,
            trial_data,
            temp_suffix=(
                f".cw_incumbent_{schedule_tag}_{int(chunk_size)}_{int(inflight)}_"
                f"{int(group_entry.get('group_index', -1))}_tmp.json5"
            ),
            heuristics_state=heuristics_state,
            recovery_key=recovery_key,
        )
        if not math.isfinite(latency_sec):
            log_print(
                f"  Incumbent scheduled group reject schedule={_format_chunk_schedule(chunk_schedule)}, "
                f"inflight={inflight}: invalid/timeout benchmark result.",
                log_handle,
            )
            continue

        incumbent = {
            "group_index": int(group_entry.get("group_index", -1)),
            "chunk_size": int(chunk_size),
            "inflight": int(inflight),
            "chunk_schedule": list(chunk_schedule),
            "true_latency_sec": float(latency_sec),
        }
        log_print(
            f"  Incumbent scheduled group schedule={_format_chunk_schedule(chunk_schedule)}, inflight={inflight}: "
            f"true_latency={latency_sec:.4f}s",
            log_handle,
        )
        if best_incumbent is None or float(latency_sec) < float(best_incumbent["true_latency_sec"]):
            best_incumbent = incumbent

    return best_incumbent


def _cw_benchmark_unchunked_reference(log_handle, model_conf, prompt_len, data, heuristics_state=None):
    trial_data = _cw_clone_data(data)
    trial_data = setup_baseline_config(trial_data, prompt_len)
    trial_data["heterogeneity"] = "hetero"
    trial_data["prompt_len"] = int(prompt_len)
    trial_data["chunking"] = False
    trial_data["chunking_scheduled"] = False
    trial_data["chunking_inflight"] = 1
    trial_data.pop("npu_dim", None)
    return _benchmark_gemm_trial_data_latency_only(
        log_handle,
        model_conf,
        prompt_len,
        trial_data,
        temp_suffix=f".cw_unchunked_ref_{int(prompt_len)}_tmp.json5",
        heuristics_state=heuristics_state,
        recovery_key=_cw_make_recovery_key("cw_unchunked_latency", prompt_len, trial_data),
    )


def _cw_collect_noncritical_donor_nodes(trace_state, focus_window):
    node_records = trace_state.get("node_records", {})
    ranked_windows = []
    for window_idx, window in enumerate(trace_state.get("windows", [])):
        candidate_nodes = [
            node_id
            for node_id in window.get("noncritical_tunable_nodes", [])
            if float(node_records.get(node_id, {}).get("slack_sec", 0.0)) > float(CW_SLACK_TOLERANCE_SEC)
        ]
        if not candidate_nodes:
            continue
        ranked_windows.append(
            {
                "sort_key": (
                    0 if window is focus_window else 1,
                    0 if _cw_windows_overlap(window, focus_window) else 1,
                    0 if bool(window.get("attention_protected", False)) else 1,
                    abs(_cw_window_center_us(window) - _cw_window_center_us(focus_window)),
                    _cw_window_sort_key(window),
                    int(window_idx),
                ),
                "candidate_nodes": candidate_nodes,
            }
        )

    donors = []
    seen_nodes = set()
    for item in sorted(ranked_windows, key=lambda item: item["sort_key"]):
        candidate_nodes = item["candidate_nodes"]
        ranked_nodes = sorted(
            candidate_nodes,
            key=lambda node_id: (
                -float(node_records[node_id].get("slack_sec", 0.0)),
                -float(node_records[node_id].get("duration_sec", 0.0)),
                _cw_node_sort_key(node_id),
            ),
        )
        for node_id in ranked_nodes:
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)
            donors.append(node_id)
            if len(donors) >= max(CW_MAX_NONCRITICAL_PER_WINDOW * 2, CW_MAX_NONCRITICAL_PER_WINDOW):
                return donors
    return donors


def _cw_should_replace_incumbent(incumbent_latency_sec, candidate_latency_sec):
    if not math.isfinite(float(candidate_latency_sec)):
        return False
    if not math.isfinite(float(incumbent_latency_sec)):
        return True
    return _cw_latency_improves_enough(float(incumbent_latency_sec), float(candidate_latency_sec))


def _cw_generate_proposals(data, model_conf, trace_state, num_chunks):
    node_records = trace_state.get("node_records", {})
    candidate_windows = [
        window
        for window in trace_state.get("windows", [])
        if int(window.get("critical_tunable_count", 0)) > 0 or int(window.get("noncritical_tunable_count", 0)) > 0
    ]
    protected_windows = [window for window in candidate_windows if bool(window.get("attention_protected", False))]
    ranked_windows = protected_windows if protected_windows else candidate_windows
    ranked_windows = sorted(ranked_windows, key=_cw_window_sort_key)

    pair_proposals = []
    single_critical_proposals = []
    single_noncritical_proposals = []
    seen = set()
    for window in ranked_windows[:CW_MAX_WINDOWS]:
        critical_nodes = sorted(
            list(window.get("critical_tunable_nodes", [])),
            key=lambda node_id: (
                -float(node_records[node_id].get("duration_sec", 0.0)),
                _cw_node_sort_key(node_id),
            ),
        )[:CW_MAX_CRITICAL_PER_WINDOW]
        if protected_windows and not critical_nodes:
            continue
        noncritical_nodes = _cw_collect_noncritical_donor_nodes(trace_state, window)[:CW_MAX_NONCRITICAL_PER_WINDOW]

        critical_actions = []
        for node_id in critical_nodes:
            for layer_name in _cw_stage_layer_names(model_conf, node_id[2]):
                action = _cw_make_kernel_action(data, model_conf, num_chunks, node_id, layer_name, "tighten")
                if action is not None:
                    critical_actions.append(action)

        noncritical_actions = []
        for node_id in noncritical_nodes:
            for layer_name in _cw_stage_layer_names(model_conf, node_id[2]):
                action = _cw_make_kernel_action(data, model_conf, num_chunks, node_id, layer_name, "relax")
                if action is not None:
                    noncritical_actions.append(action)
            bubble_action = _cw_make_bubble_action(data, node_records[node_id])
            if bubble_action is not None:
                noncritical_actions.append(bubble_action)

        candidate_edit_sets = []
        for critical_action in critical_actions:
            for noncritical_action in noncritical_actions:
                candidate_edit_sets.append([critical_action, noncritical_action])
        for critical_action in critical_actions:
            candidate_edit_sets.append([critical_action])
        if not critical_actions:
            for noncritical_action in noncritical_actions:
                candidate_edit_sets.append([noncritical_action])

        for edits in candidate_edit_sets:
            key = _cw_proposal_key(edits)
            if key in seen:
                continue
            seen.add(key)
            proposal = {
                "window": window,
                "edits": edits,
                "summary": " + ".join(str(edit.get("summary", "")).strip() for edit in edits if edit.get("summary")),
            }
            if len(edits) >= 2:
                pair_proposals.append(proposal)
            elif edits and str(edits[0].get("kind", "")) == "kernel" and str(edits[0].get("direction", "")) == "tighten":
                single_critical_proposals.append(proposal)
            else:
                single_noncritical_proposals.append(proposal)

    reserved_single_tightens = min(
        max(0, int(CW_RESERVED_SINGLE_TIGHTEN_PROPOSALS)),
        len(single_critical_proposals),
        max(0, int(CW_MAX_PROPOSALS_PER_ITER)),
    )
    max_pair_count = max(0, int(CW_MAX_PROPOSALS_PER_ITER) - int(reserved_single_tightens))

    proposals = list(pair_proposals[:max_pair_count])
    proposals.extend(single_critical_proposals[:reserved_single_tightens])

    remaining = max(0, int(CW_MAX_PROPOSALS_PER_ITER) - len(proposals))
    if remaining > 0 and len(single_critical_proposals) > reserved_single_tightens:
        extra_single_critical = single_critical_proposals[
            reserved_single_tightens : reserved_single_tightens + remaining
        ]
        proposals.extend(extra_single_critical)
        remaining = max(0, int(CW_MAX_PROPOSALS_PER_ITER) - len(proposals))
    if remaining > 0:
        proposals.extend(single_noncritical_proposals[:remaining])

    return proposals[:CW_MAX_PROPOSALS_PER_ITER]


def _cw_tune_candidate(
    log_handle,
    model_conf,
    prompt_len,
    bootstrap_data,
    bootstrap_trace_state,
    bootstrap_latency_sec,
    chunk_size,
    inflight,
    chunk_schedule=None,
    heuristics_state=None,
):
    working_data = _cw_clone_data(bootstrap_data)
    best_trace_state = bootstrap_trace_state
    best_latency_sec = float(bootstrap_latency_sec)
    resolved_schedule = _get_active_chunking_schedule(working_data)
    num_chunks = len(_resolve_chunk_plan(prompt_len, chunk_size, chunk_schedule=chunk_schedule, require_exact_schedule=True))
    schedule_tag = _chunk_schedule_file_tag(resolved_schedule)

    for iteration_idx in range(1, CW_SEARCH_BUDGET + 1):
        proposals = _cw_generate_proposals(working_data, model_conf, best_trace_state, num_chunks)
        if not proposals:
            log_print(
                f"  CW iteration {iteration_idx}: no critical-interval proposals remained; stopping.",
                log_handle,
            )
            break

        improved = False
        log_print(
            f"  CW iteration {iteration_idx}: evaluating {len(proposals)} proposal(s) "
            f"(current best true_latency={best_latency_sec:.4f}s, "
            f"protected_pressure={best_trace_state['protected_pressure']:.6f}).",
            log_handle,
        )
        for proposal_idx, proposal in enumerate(proposals, start=1):
            trial_data = _cw_apply_edit_set(working_data, model_conf, num_chunks, proposal["edits"])
            if trial_data is None:
                continue
            temp_suffix = (
                f".cw_iter{int(iteration_idx)}_{schedule_tag}_{int(chunk_size)}_{int(inflight)}_"
                f"trial{int(proposal_idx)}_tmp.json5"
            )
            trace_recovery_key = _cw_make_recovery_key(
                "cw_trial_trace",
                prompt_len,
                trial_data,
                trace_sync_stages=False,
            )
            trial_trace_latency_sec, trial_trace_rows = _benchmark_gemm_trial_data_with_trace(
                log_handle,
                model_conf,
                prompt_len,
                trial_data,
                temp_suffix=temp_suffix,
                trace_sync_stages=False,
                heuristics_state=heuristics_state,
                recovery_key=trace_recovery_key,
            )
            if not math.isfinite(trial_trace_latency_sec):
                log_print(
                    f"    [trial {proposal_idx}] reject (invalid trace run): {proposal['summary']}",
                    log_handle,
                )
                continue
            trial_trace_state = _cw_build_trace_state(trial_trace_rows, trial_trace_latency_sec, model_conf)
            if trial_trace_state is None:
                log_print(
                    f"    [trial {proposal_idx}] reject (missing trace rows): {proposal['summary']}",
                    log_handle,
                )
                continue
            latency_recovery_key = _cw_make_recovery_key("cw_trial_latency", prompt_len, trial_data)
            trial_true_latency_sec = _benchmark_gemm_trial_data_latency_only(
                log_handle,
                model_conf,
                prompt_len,
                trial_data,
                temp_suffix=temp_suffix.replace("_tmp.json5", "_lat_tmp.json5"),
                heuristics_state=heuristics_state,
                recovery_key=latency_recovery_key,
            )
            if not math.isfinite(trial_true_latency_sec):
                log_print(
                    f"    [trial {proposal_idx}] reject (invalid latency run): {proposal['summary']}",
                    log_handle,
                )
                continue
            latency_delta_ms = _cw_latency_delta_ms(best_latency_sec, trial_true_latency_sec)
            if _cw_latency_improves_enough(best_latency_sec, trial_true_latency_sec):
                working_data = trial_data
                best_trace_state = trial_trace_state
                best_latency_sec = float(trial_true_latency_sec)
                improved = True
                log_print(
                    f"    [trial {proposal_idx}] ACCEPT {proposal['summary']} "
                    f"-> true_latency={best_latency_sec:.4f}s ({latency_delta_ms:+.2f} ms), "
                    f"trace_latency={trial_trace_state['trace_latency_sec']:.4f}s, "
                    f"protected_pressure={best_trace_state['protected_pressure']:.6f}",
                    log_handle,
                )
            else:
                log_print(
                    f"    [trial {proposal_idx}] reject {proposal['summary']} "
                    f"-> true_latency={trial_true_latency_sec:.4f}s ({latency_delta_ms:+.2f} ms), "
                    f"trace_latency={trial_trace_state['trace_latency_sec']:.4f}s, "
                    f"protected_pressure={trial_trace_state['protected_pressure']:.6f}",
                    log_handle,
                )

        if not improved:
            log_print(f"  CW iteration {iteration_idx}: no accepted edits; stopping.", log_handle)
            break

    return working_data, float(best_latency_sec), best_trace_state


def tune_all_gemm_cw(log_handle, model_conf):
    log_print(f"\n>>> Starting GEMM Critical-Interval Co-Optimization for {model_conf['name']}", log_handle)
    _ensure_cw_tracing_available(log_handle)

    data = load_config(model_conf)
    if not data:
        return
    heuristics_state = load_chunk_heuristics(model_conf, log_handle) if CHUNK_RECOVER else None

    outer_candidate_cap, final_candidate_cap = _get_search_space_limits()
    forced_chunk_schedule = _get_forced_chunking_schedule()
    forced_inflight = -1
    try:
        forced_inflight = int(FORCH_INFLIGHT)
    except Exception:
        forced_inflight = -1
    device_policy = _cw_get_device_policy()
    log_print(
        f"gemm_CW strategy={CW_STRATEGY_VERSION} device_policy={device_policy['rule_name']} "
        f"inflight_threads={device_policy['inflight_threads']}"
        + (f" forced_inflight={forced_inflight}" if forced_inflight > 0 else ""),
        log_handle,
    )

    try:
        for prompt_len in _get_prompt_sizes_for_mode("gemm_CW", model_conf):
            log_print(f"\n{'='*40}", log_handle)
            log_print(f"GEMM CW PROMPT SIZE: {prompt_len}", log_handle)
            log_print(f"{'='*40}", log_handle)

            data = setup_baseline_config(data, prompt_len)
            removed_scheduled_groups = _cw_drop_existing_scheduled_groups(data, prompt_len)
            if removed_scheduled_groups > 0:
                log_print(
                    f"Removed {removed_scheduled_groups} existing scheduled kernels_gemm_chunked group(s) "
                    f"for prompt_len={prompt_len} before gemm_CW search.",
                    log_handle,
                )
            incumbent_chunked = _cw_benchmark_chunked_incumbent(
                log_handle,
                model_conf,
                prompt_len,
                data,
                heuristics_state=heuristics_state,
            )
            if incumbent_chunked is not None:
                log_print(
                    f"Best incumbent scheduled chunked config: schedule={_format_chunk_schedule(incumbent_chunked['chunk_schedule'])}, "
                    f"inflight={incumbent_chunked['inflight']}, true_latency={incumbent_chunked['true_latency_sec']:.4f}s",
                    log_handle,
                )
            unchunked_reference_latency = _cw_benchmark_unchunked_reference(
                log_handle,
                model_conf,
                prompt_len,
                data,
                heuristics_state=heuristics_state,
            )
            if math.isfinite(unchunked_reference_latency):
                log_print(
                    f"Unchunked hetero reference (not eligible to win gemm_CW): "
                    f"true_latency={unchunked_reference_latency:.4f}s",
                    log_handle,
                )
            else:
                log_print(
                    "Unchunked hetero reference benchmark failed; continuing with chunked-only CW search.",
                    log_handle,
                )
            candidate_specs = []
            if forced_chunk_schedule:
                try:
                    forced_plan = _resolve_chunk_plan(
                        prompt_len,
                        _schedule_fallback_chunk_size(forced_chunk_schedule),
                        chunk_schedule=forced_chunk_schedule,
                        require_exact_schedule=True,
                    )
                except RuntimeError as exc:
                    log_print(str(exc), log_handle)
                    continue
                effective_inflight = _cw_resolve_effective_inflight(
                    len(forced_plan),
                    forced_inflight=forced_inflight if forced_inflight > 0 else None,
                )
                if effective_inflight is None:
                    log_print(
                        f"Skipping forced schedule={_format_chunk_schedule(forced_plan)} because num_chunks={len(forced_plan)} "
                        f"does not satisfy the required inflight={_cw_target_inflight_for_rule()} "
                        f"under the current CW lockstep policy.",
                        log_handle,
                    )
                    continue
                try:
                    for unique_chunk in sorted(set(forced_plan)):
                        get_chunk_baselines_or_raise(data, model_conf, unique_chunk)
                except RuntimeError as exc:
                    log_print(str(exc), log_handle)
                    raise SystemExit(1)
                forced_metrics = _cw_schedule_proxy_metrics(forced_plan, effective_inflight)
                candidate_specs = [
                    {
                        "chunk_size": int(_schedule_fallback_chunk_size(forced_plan)),
                        "chunk_schedule": list(forced_plan),
                        "effective_inflight": int(effective_inflight),
                        "num_chunks": int(len(forced_plan)),
                        **forced_metrics,
                    }
                ]
                log_print(
                    f"FORCE_CHUNKING_SCHEDULE active for gemm_CW: schedule={_format_chunk_schedule(forced_plan)}, "
                    f"effective_inflight={effective_inflight}",
                    log_handle,
                )
            else:
                candidate_specs = _cw_generate_lockstep_candidate_specs(
                    prompt_len,
                    forced_inflight=forced_inflight if forced_inflight > 0 else None,
                )
                if not candidate_specs:
                    log_print(
                        f"No valid lockstep CW schedules for prompt size {prompt_len} "
                        f"(quantum={CW_CHUNK_QUANTUM}, min_chunk={CW_MIN_CHUNK_SIZE}, "
                        f"required_inflight={device_policy['inflight_threads']}); skipping.",
                        log_handle,
                    )
                    continue
                filtered_specs = []
                for spec in candidate_specs:
                    chunk_plan = _normalize_chunk_schedule(spec.get("chunk_schedule"))
                    try:
                        for unique_chunk in sorted(set(chunk_plan)):
                            get_chunk_baselines_or_raise(data, model_conf, unique_chunk)
                    except RuntimeError as exc:
                        log_print(
                            f"Skipping lockstep schedule={_format_chunk_schedule(chunk_plan)}: {exc}",
                            log_handle,
                        )
                        continue
                    filtered_specs.append(spec)
                candidate_specs = filtered_specs
                if not candidate_specs:
                    log_print(
                        f"No CW schedules with available kernels_gemm baselines remained for prompt size {prompt_len}.",
                        log_handle,
                    )
                    continue
                if int(outer_candidate_cap) != -1 and len(candidate_specs) > int(outer_candidate_cap):
                    reduced_specs = _cw_select_schedule_candidates_for_active_search(
                        candidate_specs,
                        max_pool=int(outer_candidate_cap),
                    )
                    log_print(
                        f"SEARCH_SPACE outer_cap={int(outer_candidate_cap)}: reduced "
                        f"{len(candidate_specs)} schedules -> {len(reduced_specs)} traced bootstrap candidates.",
                        log_handle,
                    )
                    candidate_specs = [
                        {
                            "chunk_size": int(spec["chunk_size"]),
                            "chunk_schedule": list(spec["chunk_schedule"]),
                            "effective_inflight": int(spec["effective_inflight"]),
                            "num_chunks": int(spec.get("num_chunks", len(spec["chunk_schedule"]))),
                            "slot_projection_proxy": list(spec.get("slot_projection_proxy", [])),
                            "slot_attention_proxy": list(spec.get("slot_attention_proxy", [])),
                            "projection_proxy_spread": float(spec.get("projection_proxy_spread", float("inf"))),
                            "attention_proxy_spread": float(spec.get("attention_proxy_spread", float("inf"))),
                            "weighted_proxy_total": float(spec.get("weighted_proxy_total", float("inf"))),
                            "distinct_chunk_sizes": int(spec.get("distinct_chunk_sizes", len(set(spec["chunk_schedule"])))),
                            "chunk_size_transitions": int(spec.get("chunk_size_transitions", 0)),
                        }
                        for spec in reduced_specs
                    ]
                else:
                    log_print(
                        f"SEARCH_SPACE outer_cap={int(outer_candidate_cap)}: using all "
                        f"{len(candidate_specs)} schedule candidates for traced bootstrap.",
                        log_handle,
                    )

            bootstrap_candidates = []
            for candidate in candidate_specs:
                chunk_size = int(candidate["chunk_size"])
                chunk_schedule = _normalize_chunk_schedule(candidate.get("chunk_schedule"))
                chunk_plan = _resolve_chunk_plan(
                    prompt_len,
                    chunk_size,
                    chunk_schedule=chunk_schedule,
                    require_exact_schedule=True,
                )
                inflight = int(candidate.get("effective_inflight", candidate.get("inflight", 1)))
                if inflight <= 0:
                    log_print(
                        f"Skipping schedule={_format_chunk_schedule(chunk_plan)} because it resolved to an invalid inflight={inflight}.",
                        log_handle,
                    )
                    continue

                trial_data, _, resolved_schedule = build_chunking_trial_data(
                    data,
                    model_conf,
                    prompt_len,
                    chunk_size,
                    inflight,
                    chunk_schedule=chunk_schedule,
                )
                schedule_tag = _chunk_schedule_file_tag(resolved_schedule)
                latency_recovery_key = _cw_make_recovery_key(
                    "cw_bootstrap_latency",
                    prompt_len,
                    trial_data,
                    schedule=_chunk_schedule_key(resolved_schedule),
                    inflight=int(inflight),
                )
                true_latency_sec = _benchmark_gemm_trial_data_latency_only(
                    log_handle,
                    model_conf,
                    prompt_len,
                    trial_data,
                    temp_suffix=f".cw_bootstrap_latency_{schedule_tag}_{int(chunk_size)}_{int(inflight)}_tmp.json5",
                    heuristics_state=heuristics_state,
                    recovery_key=latency_recovery_key,
                )
                if not math.isfinite(true_latency_sec):
                    log_print(
                        f"  Bootstrap reject schedule={_format_chunk_schedule(chunk_plan)}, inflight={inflight}: "
                        "invalid/timeout latency benchmark result.",
                        log_handle,
                    )
                    continue
                trace_recovery_key = _cw_make_recovery_key(
                    "cw_bootstrap_trace",
                    prompt_len,
                    trial_data,
                    schedule=_chunk_schedule_key(resolved_schedule),
                    inflight=int(inflight),
                    trace_sync_stages=False,
                )
                trace_latency_sec, trace_rows = _benchmark_gemm_trial_data_with_trace(
                    log_handle,
                    model_conf,
                    prompt_len,
                    trial_data,
                    temp_suffix=f".cw_bootstrap_trace_{schedule_tag}_{int(chunk_size)}_{int(inflight)}_tmp.json5",
                    trace_sync_stages=False,
                    heuristics_state=heuristics_state,
                    recovery_key=trace_recovery_key,
                )
                if not math.isfinite(trace_latency_sec):
                    log_print(
                        f"  Bootstrap reject schedule={_format_chunk_schedule(chunk_plan)}, inflight={inflight}: "
                        "invalid/timeout trace benchmark result.",
                        log_handle,
                    )
                    continue
                trace_state = _cw_build_trace_state(trace_rows, trace_latency_sec, model_conf)
                if trace_state is None:
                    log_print(
                        f"  Bootstrap reject schedule={_format_chunk_schedule(chunk_plan)}, inflight={inflight}: "
                        "missing trace rows.",
                        log_handle,
                    )
                    continue
                bootstrap_candidates.append(
                    {
                        "chunk_size": int(chunk_size),
                        "chunk_schedule": list(resolved_schedule),
                        "inflight": int(inflight),
                        "true_latency_sec": float(true_latency_sec),
                        "trace_latency_sec": float(trace_latency_sec),
                        "pressure": float(trace_state["pressure"]),
                        "base_pressure": float(trace_state["base_pressure"]),
                        "attention_overlap_penalty": float(trace_state["attention_overlap_penalty"]),
                        "critical_attention_penalty": float(trace_state["critical_attention_penalty"]),
                        "protected_pressure": float(trace_state["protected_pressure"]),
                        "attention_busy_spread_sec": float(trace_state["attention_busy_spread_sec"]),
                        "projection_busy_spread_sec": float(trace_state["projection_busy_spread_sec"]),
                        "projection_proxy_spread": float(candidate.get("projection_proxy_spread", float("inf"))),
                        "attention_proxy_spread": float(candidate.get("attention_proxy_spread", float("inf"))),
                        "trial_data": trial_data,
                        "trace_state": trace_state,
                    }
                )
                log_print(
                    f"  Bootstrap candidate schedule={_format_chunk_schedule(chunk_plan)}, inflight={inflight}: "
                    f"true_latency={true_latency_sec:.4f}s trace_latency={trace_latency_sec:.4f}s "
                    f"attention_busy_spread={trace_state['attention_busy_spread_sec']:.6f}s "
                    f"projection_busy_spread={trace_state['projection_busy_spread_sec']:.6f}s "
                    f"base_pressure={trace_state['base_pressure']:.6f} "
                    f"attention_overlap={trace_state['attention_overlap_penalty']:.6f} "
                    f"critical_attention={trace_state['critical_attention_penalty']:.6f} "
                    f"protected_pressure={trace_state['protected_pressure']:.6f}",
                    log_handle,
                )

            if not bootstrap_candidates:
                log_print(f"No valid bootstrap candidates for prompt size {prompt_len}.", log_handle)
                continue

            gated_candidates, best_chunked_latency, used_gate_pct = _cw_apply_latency_gate(bootstrap_candidates)
            if used_gate_pct is None:
                log_print(
                    "Latency gate fallback: retaining all valid chunked candidates because no gated survivors remained.",
                    log_handle,
                )
            else:
                log_print(
                    f"Latency gate kept {len(gated_candidates)}/{len(bootstrap_candidates)} chunked candidates: "
                    f"true_latency <= {best_chunked_latency * (1.0 + used_gate_pct):.4f}s "
                    f"({int(round(used_gate_pct * 100.0))}% over best chunked latency {best_chunked_latency:.4f}s).",
                    log_handle,
                )

            bootstrap_candidates.sort(key=_cw_bootstrap_rank_key)
            log_print("Bootstrap ranking by lockstep attention/projection balance:", log_handle)
            for rank, candidate in enumerate(bootstrap_candidates, start=1):
                log_print(
                    f"  [{rank}] schedule={_format_chunk_schedule(candidate['chunk_schedule'])}, "
                    f"inflight={candidate['inflight']}, true_latency={candidate['true_latency_sec']:.4f}s, "
                    f"trace_latency={candidate['trace_latency_sec']:.4f}s, "
                    f"attention_busy_spread={candidate['attention_busy_spread_sec']:.6f}s, "
                    f"projection_busy_spread={candidate['projection_busy_spread_sec']:.6f}s, "
                    f"base_pressure={candidate['base_pressure']:.6f}, "
                    f"attention_overlap={candidate['attention_overlap_penalty']:.6f}, "
                    f"critical_attention={candidate['critical_attention_penalty']:.6f}, "
                    f"protected_pressure={candidate['protected_pressure']:.6f}",
                    log_handle,
                )

            if int(final_candidate_cap) == -1:
                candidate_trials = gated_candidates
                log_print("SEARCH_SPACE final_cap=-1: tuning every latency-gated bootstrap candidate.", log_handle)
            else:
                final_count = max(1, int(final_candidate_cap))
                candidate_trials = gated_candidates[:final_count]
                log_print(
                    f"SEARCH_SPACE final_cap={final_count}: tuning top {len(candidate_trials)} latency-gated "
                    "lockstep-balanced candidates.",
                    log_handle,
                )

            best_final_data = None
            best_final_trace_state = None
            best_final_latency = float("inf")
            best_final_schedule = []
            best_final_chunk_size = -1
            best_final_inflight = -1

            for candidate in candidate_trials:
                log_print(
                    f"\n=== gemm_CW candidate schedule={_format_chunk_schedule(candidate['chunk_schedule'])}, "
                    f"inflight={candidate['inflight']} "
                    f"(bootstrap true_latency={candidate['true_latency_sec']:.4f}s, "
                    f"trace_latency={candidate['trace_latency_sec']:.4f}s, "
                    f"protected_pressure={candidate['protected_pressure']:.6f}) ===",
                    log_handle,
                )
                tuned_data, tuned_latency, tuned_trace_state = _cw_tune_candidate(
                    log_handle,
                    model_conf,
                    prompt_len,
                    candidate["trial_data"],
                    candidate["trace_state"],
                    candidate["true_latency_sec"],
                    candidate["chunk_size"],
                    candidate["inflight"],
                    chunk_schedule=candidate["chunk_schedule"],
                    heuristics_state=heuristics_state,
                )
                if (
                    best_final_data is None
                    or float(tuned_latency) < float(best_final_latency)
                    or (
                        abs(float(tuned_latency) - float(best_final_latency))
                        <= max(_cw_noise_floor_sec(best_final_latency), _cw_noise_floor_sec(tuned_latency))
                        and float(tuned_trace_state["protected_pressure"]) < float(best_final_trace_state["protected_pressure"])
                    )
                ):
                    best_final_data = tuned_data
                    best_final_latency = float(tuned_latency)
                    best_final_trace_state = tuned_trace_state
                    best_final_schedule = list(candidate["chunk_schedule"])
                    best_final_chunk_size = int(candidate["chunk_size"])
                    best_final_inflight = int(candidate["inflight"])
                    log_print(
                        f"* New gemm_CW best: schedule={_format_chunk_schedule(best_final_schedule)}, "
                        f"inflight={best_final_inflight}, true_latency={best_final_latency:.4f}s, "
                        f"protected_pressure={best_final_trace_state['protected_pressure']:.6f}",
                        log_handle,
                    )

            if best_final_data is None:
                log_print(f"No gemm_CW winner found for prompt size {prompt_len}.", log_handle)
                continue

            if incumbent_chunked is not None and not _cw_should_replace_incumbent(
                incumbent_chunked["true_latency_sec"],
                best_final_latency,
            ):
                log_print(
                    f"Preserving incumbent scheduled chunked config for prompt {prompt_len}: "
                    f"incumbent true_latency={incumbent_chunked['true_latency_sec']:.4f}s "
                    f"(schedule={_format_chunk_schedule(incumbent_chunked['chunk_schedule'])}, "
                    f"inflight={incumbent_chunked['inflight']}) remains better than or within noise of "
                    f"best gemm_CW result {best_final_latency:.4f}s.",
                    log_handle,
                )
                if math.isfinite(unchunked_reference_latency):
                    log_print(
                        f"Reference gap vs unchunked hetero baseline: "
                        f"{(best_final_latency - unchunked_reference_latency) * 1000.0:+.2f} ms",
                        log_handle,
                    )
                continue

            _apply_chunking_format(
                data,
                best_final_chunk_size,
                best_final_inflight,
                prompt_len=prompt_len,
                chunk_schedule=best_final_schedule,
            )
            source_kernels = _get_chunked_kernels_ref(
                best_final_data,
                create=False,
                reset=False,
                chunk_size=best_final_chunk_size,
                inflight=best_final_inflight,
                prompt_len=prompt_len,
                chunk_schedule=best_final_schedule,
            )
            target_kernels = _get_chunked_kernels_ref(
                data,
                create=True,
                reset=True,
                chunk_size=best_final_chunk_size,
                inflight=best_final_inflight,
                prompt_len=prompt_len,
                chunk_schedule=best_final_schedule,
            )
            target_kernels[:] = json.loads(json.dumps(source_kernels))
            _set_chunked_stage_bubbles(
                data,
                _cw_get_selected_stage_bubbles(best_final_data),
                chunk_size=best_final_chunk_size,
                inflight=best_final_inflight,
                prompt_len=prompt_len,
                chunk_schedule=best_final_schedule,
            )
            canonicalize_chunked_config(data, prune_duplicates=True)
            sort_kernels_gemm_entries(data, "kernels_gemm_chunked", include_chunk_id=True)
            save_config_via_temp(data, model_conf, temp_suffix=".cw_commit_tmp.json5")
            incumbent_delta_ms = None
            if incumbent_chunked is not None and math.isfinite(float(incumbent_chunked["true_latency_sec"])):
                incumbent_delta_ms = (float(incumbent_chunked["true_latency_sec"]) - float(best_final_latency)) * 1000.0
            unchunked_gap_ms = None
            if math.isfinite(unchunked_reference_latency):
                unchunked_gap_ms = (float(best_final_latency) - float(unchunked_reference_latency)) * 1000.0
            log_print(
                f"Committed gemm_CW config for prompt {prompt_len}: "
                f"schedule={_format_chunk_schedule(best_final_schedule)}, inflight={best_final_inflight}, "
                f"true_latency={best_final_latency:.4f}s, protected_pressure={best_final_trace_state['protected_pressure']:.6f}, "
                f"stage_bubbles={len(_cw_get_selected_stage_bubbles(best_final_data))}, "
                f"improvement_vs_incumbent_ms="
                f"{(f'{incumbent_delta_ms:+.2f}' if incumbent_delta_ms is not None else 'n/a')}, "
                f"gap_vs_unchunked_ms={f'{unchunked_gap_ms:+.2f}' if unchunked_gap_ms is not None else 'n/a'}",
                log_handle,
            )

    except KeyboardInterrupt:
        log_print("GEMM Critical-Interval Co-Optimization Interrupted!", log_handle)

def resolve_gemv_split_mode(layer_name, K, N, layer_cfg=None):
    if layer_cfg and "gemv_split" in layer_cfg:
        mode = str(layer_cfg["gemv_split"]).strip().upper()
        if mode in ("M", "K"):
            return mode
    if K == 14336 or N == 14336 or layer_name == "down":
        return "K"
    return "M"


def get_or_create_kernel_gemv(data, layer_name, M, K, N, split_mode=None):
    found = None
    if "kernels_gemv" not in data:
        data["kernels_gemv"] = []
        
    for k in data["kernels_gemv"]:
        if k["layer"] == layer_name and k["forK"] == K and k["forN"] == N:
            found = k
            break
            
    split_mode = (split_mode or "").upper()
    if split_mode not in ("M", "K"):
        split_mode = resolve_gemv_split_mode(layer_name, K, N)
    suffix = "_K" if split_mode == "K" else "_M"
    fw_path = f"hw_bins/npu2/1x{K}x{N}/bf16_int4AWQ_bf16{suffix}/"

    if not found:
        
        found = {
            "use": True,
            "layer": layer_name,
            "forM": 1,
            "forK": K,
            "forN": N,
            "cpuN": 0,
            "cpuThreads": 4,
            "npuM": 1, 
            "npuK": K, 
            "npuN": N,
            "config": -1,
            "num_tiles": 32,
            "fw_path": fw_path, 
            "tile_size": "128x64",
            "col": "8c",
            "dtype": DTYPE_BASE
        }
        data["kernels_gemv"].append(found)
    
    # Always align fw_path with selected split policy.
    found["fw_path"] = fw_path
    found["dtype"] = DTYPE_BASE
    found["use"] = True
    return found

def run_benchmark_gemv(log_handle, model_conf, config_override=None):
    cmd_str = f"pushd ../../ && source utils/setup.sh && popd && python3 {model_conf['script']}" 
    
    if config_override:
        cmd_str += f" --config-path {config_override}"
        
    total_time = 0.0
    valid_runs = 0
    
    for i in range(RUN_AVERAGE):
        proc = None
        try:
            proc = subprocess.Popen(
                cmd_str,
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            output, _ = proc.communicate(timeout=BENCHMARK_TIMEOUT_SEC)
            if log_handle:
                log_handle.write(f"\n--- GEMV Benchmark Output (Iter {i+1}) ---\n")
                log_handle.write(output)
                log_handle.write(f"\n--- End Output ---\n")
                log_handle.flush()
            
            m = re.search(r"Total Generation Time:\s*([\d\.]+)\s*seconds", output)
            
            if proc.returncode != 0:
                log_print(f"GEMV Benchmark Failed: {proc.returncode}", log_handle)
                sys.exit(1)
                
            if m:
                t = float(m.group(1))
                total_time += t
                valid_runs += 1
            else:
                if "fail" in output.lower():
                     log_print(f"GEMV Benchmark Failed (detected 'fail' in output):", log_handle)
                     sys.exit(1)
                else:
                     log_print("Could not parse 'Total Generation Time'.", log_handle)
                     sys.exit(1)
            
            time.sleep(2)

        except subprocess.TimeoutExpired:
            if proc is not None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                try:
                    proc.communicate(timeout=5)
                except Exception:
                    pass
            log_print(f"timeout: GEMV benchmark timed out after {BENCHMARK_TIMEOUT_SEC}s", log_handle)
            return float('inf')
        except Exception as e:
            if proc is not None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            log_print(f"Exception running benchmark: {e}", log_handle)
            return float('inf')
            
    if valid_runs == 0:
        return float('inf')
        
    return total_time / valid_runs

def tune_layer_gemv(data, layer, K, N, log_handle, model_conf, improvement_threshold, skip_cpu_tuning=False):
    log_print(f"\n--- Tuning GEMV for {layer['name']} ---", log_handle)
    
    max_cpu_threads = get_max_cpu_threads()
    log_print(f"  Max CPU Threads detected: {max_cpu_threads}", log_handle)
    
    history = {}
    threshold = improvement_threshold
    cpu_step = 128
    npu_step = 2048
    use_packed_weights = bool(data.get("usePackedWeights", False))
    split_mode = resolve_gemv_split_mode(layer["name"], K, N, layer)
    k_tuning_upper = _k_tuning_upper_bound(model_conf, K)
    disable_npu_tuning_by_policy = bool(layer.get("gemv_disable_npu_tuning", False))
    disable_npu_tuning = disable_npu_tuning_by_policy
    # Baseline must reflect pre-existing config state for this layer.
    # Do not create a new kernel here, otherwise baseline gets biased by template defaults.
    base_kernel = None
    for k in data.get("kernels_gemv", []):
        if k.get("layer") == layer["name"] and int(k.get("forK", -1)) == int(K) and int(k.get("forN", -1)) == int(N):
            base_kernel = k
            break
    if base_kernel is None:
        base_kernel = {
            "use": False,
            "cpuN": 0,
            "cpuThreads": 1,
            "npuK": 0,
            "npuN": 0,
        }
    base_npuK = int(base_kernel.get("npuK", 0))
    # In gemv_split=K mode, cpuN applies to the GPU-weight branch independently of npuN.
    independent_cpu_npu = (split_mode == "K")

    log_print(f"  GEMV split policy: {split_mode}", log_handle)
    if split_mode == "K":
        log_print(f"  GEMV base npuK: {base_npuK} (forK={K})", log_handle)
        if k_tuning_upper < int(K):
            log_print(
                f"  Applying MAX_K_SPLIT cap for {layer['name']}: npuK tuning upper bound {k_tuning_upper}/{K}",
                log_handle,
            )
    if disable_npu_tuning_by_policy:
        log_print(f"  NPU tuning disabled by layer policy for {layer['name']}", log_handle)
    if split_mode == "K" and not use_packed_weights:
        disable_npu_tuning = True
        log_print("  NPU K tuning disabled: usePackedWeights=false.", log_handle)

    def get_thread_candidates():
        vals = []
        curr = 1
        while curr <= max_cpu_threads:
            vals.append(curr)
            curr *= 2
        if not vals:
            vals = [1]
        if vals[-1] != max_cpu_threads:
            vals.append(max_cpu_threads)
        return vals

    threads_candidates = get_thread_candidates()

    def measure_point(cpuN, npuN, threads, npuK_override=None):
        cpuN = int(max(0, min(N, cpuN)))
        if independent_cpu_npu:
            # For K-split (npuK < forK), NPU branch always spans full N when enabled.
            npuN = N if int(npuN) > 0 else 0
        else:
            npuN = int(max(0, min(N - cpuN, npuN)))
        threads = int(max(1, min(max_cpu_threads, threads)))

        key = (cpuN, npuN, threads, int(npuK_override) if npuK_override is not None else -1)
        if key in history:
            return history[key]
        
        temp_config_path = model_conf["config"] + ".gemv_tmp.json5"
        trial_data = json.loads(json.dumps(data))
        trial_kernel = get_or_create_kernel_gemv(trial_data, layer['name'], 1, K, N, split_mode=split_mode)

        if npuN > 0:
            if npuK_override is not None:
                npuK = int(npuK_override)
            else:
                npuK = (base_npuK if independent_cpu_npu else K)
            npuK = _cap_k_tuning_value(model_conf, K, npuK)
            if npuK <= 0:
                npuN = 0
        else:
            npuK = 0

        if cpuN == 0 and npuN == 0:
            trial_kernel["use"] = False
            trial_kernel["npuK"] = 0
            trial_kernel["cpuN"] = 0
            trial_kernel["npuN"] = 0
            trial_kernel["cpuThreads"] = threads
        else:
            trial_kernel["use"] = True
            trial_kernel["cpuN"] = cpuN
            trial_kernel["npuN"] = npuN
            trial_kernel["cpuThreads"] = threads
            trial_kernel["npuK"] = npuK
        
        with open(temp_config_path, 'w') as f:
            json.dump(trial_data, f, indent=4)
        
        t = run_benchmark_gemv(log_handle, model_conf, config_override=temp_config_path)
        
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            
        history[key] = t
        return t

    def solve_1d(low, high, step_size, measure_func, max_val_constraint, log_prefix=""):
        low = max(0, min(int(low), int(max_val_constraint)))
        high = max(0, min(int(high), int(max_val_constraint)))
        if high < low:
            low, high = high, low

        best_v = low
        best_t = measure_func(low)
        log_print(f"      {log_prefix}D&C Probe: v={low} -> {best_t:.4f}s", log_handle)
        if high != low:
            t_high = measure_func(high)
            log_print(f"      {log_prefix}D&C Probe: v={high} -> {t_high:.4f}s", log_handle)
            if t_high < best_t:
                best_t = t_high
                best_v = high
        
        def _recurse(l, h):
            nonlocal best_v, best_t
            if h - l < step_size:
                return

            mid = (l + h) // 2
            mid = round(mid / step_size) * step_size
            
            if mid <= l: mid = l + step_size
            if mid >= h: mid = h - step_size
            if mid <= l or mid >= h: return
            
            vals = [l, mid, h]
            curr_times = []
            
            for v in vals:
                if v > max_val_constraint:
                    v = max_val_constraint
                t = measure_func(v)
                curr_times.append(t)
                log_print(f"      {log_prefix}D&C Probe: v={v} -> {t:.4f}s", log_handle)
                
                if t < best_t:
                    best_t = t
                    best_v = v
            
            t_low, t_mid, t_high = curr_times
            if t_low < t_mid and t_low < t_high:
                _recurse(l, mid)
            elif t_high < t_mid and t_high < t_low:
                _recurse(mid, h)
            else:
                _recurse(l, mid)
                _recurse(mid, h)
        
        # Initial call
        _recurse(low, high)
        return best_v, best_t

    def solve_2d(x_values, y_low, y_high, y_step, measure_func, max_y_constraint):
        """
        Solves for best (x, y) where x is discrete (from x_values) and y is continuous-ish (1D search).
        Uses divide-and-conquer on x_values indices.
        measure_func(x, y) -> time
        Returns (best_x, best_y, best_time)
        """
        global_best_time = float('inf')
        global_best_x = -1
        global_best_y = -1
        
        # Helper to solve Y for a specific X
        def solve_y_for_x(x):
             def measure_y(y_val):
                return measure_func(x, y_val)
             by, bt = solve_1d(y_low, y_high, y_step, measure_y, max_y_constraint, log_prefix=f"[X={x}] ")
             return by, bt

        # Recursive search on X indices
        def _recurse_x(idx_low, idx_high):
            nonlocal global_best_time, global_best_x, global_best_y
            
            if idx_high < idx_low: return
            
            # If range is small, just linear scan (or base case)
            if idx_high == idx_low:
                # Single point
                x = x_values[idx_low]
                by, bt = solve_y_for_x(x)
                log_print(f"    [X={x}] Best Y={by} -> {bt:.4f}s", log_handle)
                if bt < global_best_time:
                    global_best_time = bt
                    global_best_x = x
                    global_best_y = by
                return

            mid_idx = (idx_low + idx_high) // 2
            
            # We probe low, mid, high indices
            idxs_to_test = {idx_low, mid_idx, idx_high} # Set to avoid duplicates
            
            # Store results to decide direction
            results = {} 
            
            for idx in idxs_to_test:
                x = x_values[idx]
                by, bt = solve_y_for_x(x)
                results[idx] = bt
                log_print(f"    [X={x}] Best Y={by} -> {bt:.4f}s", log_handle)
                
                if bt < global_best_time:
                    global_best_time = bt
                    global_best_x = x
                    global_best_y = by
            
            t_low = results[idx_low]
            t_mid = results[mid_idx]
            t_high = results[idx_high]
            
            # Standard D&C decision logic matching solve_1d style
            if t_low < t_mid and t_low < t_high:
                _recurse_x(idx_low, mid_idx - 1)
            elif t_high < t_mid and t_high < t_low:
                _recurse_x(mid_idx + 1, idx_high)
            else:
                 # Valley or flat, try both sides (excluding mid itself)
                _recurse_x(idx_low, mid_idx - 1)
                _recurse_x(mid_idx + 1, idx_high)

        _recurse_x(0, len(x_values) - 1)
            
        return global_best_x, global_best_y, global_best_time

    def align_cpuN(val, max_cpuN):
        max_cpuN = int(max(0, max_cpuN))
        if max_cpuN == 0:
            return 0
        v = int(round(float(val) / float(cpu_step)) * cpu_step)
        v = max(0, min(v, max_cpuN))
        if v != 0 and v % cpu_step != 0:
            v = (v // cpu_step) * cpu_step
        return max(0, min(v, max_cpuN))

    def fast_cpu_search(max_cpuN, fixed_npuN, reference_time, phase_label, fixed_npuK=None):
        """
        Faster CPU GEMV tuning strategy:
        1) Optimize cpuN with cpuThreads fixed to 8 (or 1 if only one thread exists).
        2) If improved vs reference_time, optimize cpuThreads for that fixed cpuN.
        3) Re-check cpuN neighbors with fixed best cpuThreads using a +/- (2 * cpu_step) jump.
        Returns (best_threads, best_cpuN, best_time).
        """
        if independent_cpu_npu:
            max_cpuN = int(max(0, min(max_cpuN, N)))
        else:
            max_cpuN = int(max(0, min(max_cpuN, N - fixed_npuN)))
        fixed_threads = 8 if max_cpu_threads >= 8 else 1

        if max_cpuN <= 0:
            t0 = measure_point(0, fixed_npuN, fixed_threads, npuK_override=fixed_npuK)
            return fixed_threads, 0, t0

        log_print(
            f"    {phase_label}: Step 1/3 optimize cpuN with cpuThreads={fixed_threads}",
            log_handle
        )
        best_cpuN, best_time = solve_1d(
            0,
            max_cpuN,
            cpu_step,
            lambda cn: measure_point(align_cpuN(cn, max_cpuN), fixed_npuN, fixed_threads, npuK_override=fixed_npuK),
            max_cpuN,
            log_prefix=f"[{phase_label} cpuN@T{fixed_threads}] "
        )
        best_cpuN = align_cpuN(best_cpuN, max_cpuN)
        best_threads = fixed_threads
        best_time = measure_point(best_cpuN, fixed_npuN, best_threads, npuK_override=fixed_npuK)

        if best_time < reference_time:
            log_print(
                f"    {phase_label}: Step 2/3 optimize cpuThreads at fixed cpuN={best_cpuN}",
                log_handle
            )
            for thr in threads_candidates:
                t = measure_point(best_cpuN, fixed_npuN, thr, npuK_override=fixed_npuK)
                log_print(
                    f"      [{phase_label} threads] cpuN={best_cpuN}, threads={thr} -> {t:.4f}s",
                    log_handle
                )
                if t < best_time:
                    best_time = t
                    best_threads = thr

            log_print(
                f"    {phase_label}: Step 3/3 neighbor cpuN check at fixed threads={best_threads} (delta={cpu_step * 2})",
                log_handle
            )
            delta = cpu_step * 2
            neighbor_vals = {
                align_cpuN(best_cpuN - delta, max_cpuN),
                best_cpuN,
                align_cpuN(best_cpuN + delta, max_cpuN)
            }
            for cn in sorted(neighbor_vals):
                t = measure_point(cn, fixed_npuN, best_threads, npuK_override=fixed_npuK)
                log_print(
                    f"      [{phase_label} neighbors] cpuN={cn}, threads={best_threads} -> {t:.4f}s",
                    log_handle
                )
                if t < best_time:
                    best_time = t
                    best_cpuN = cn
        else:
            log_print(
                f"    {phase_label}: No cpuN candidate improved vs reference ({reference_time:.4f}s). Skipping thread sweep.",
                log_handle
            )

        return best_threads, best_cpuN, best_time

    # Phase 0: Baseline from current config state for this layer.
    baseline_use = bool(base_kernel.get("use", True))
    raw_baseline_cpuN = int(base_kernel.get("cpuN", 0))
    raw_baseline_threads = int(base_kernel.get("cpuThreads", 1))
    raw_baseline_npuN = int(base_kernel.get("npuN", 0))
    raw_baseline_npuK = int(base_kernel.get("npuK", base_npuK))

    if not baseline_use:
        raw_baseline_cpuN = 0
        raw_baseline_npuN = 0
        raw_baseline_npuK = 0

    baseline_cpuN = int(max(0, min(N, raw_baseline_cpuN)))
    baseline_threads = int(max(1, min(max_cpu_threads, raw_baseline_threads)))
    if independent_cpu_npu:
        baseline_npuN = N if raw_baseline_npuN > 0 else 0
    else:
        baseline_npuN = int(max(0, min(N - baseline_cpuN, raw_baseline_npuN)))

    if baseline_npuN > 0:
        baseline_npuK = raw_baseline_npuK if raw_baseline_npuK > 0 else int(K)
        baseline_npuK = _cap_k_tuning_value(model_conf, K, baseline_npuK)
        if baseline_npuK <= 0:
            baseline_npuN = 0
    else:
        baseline_npuK = 0

    baseline_conf = {
        "npuK": int(baseline_npuK),
        "cpuN": int(baseline_cpuN),
        "cpuThreads": int(baseline_threads),
        "npuN": int(baseline_npuN),
    }
    baseline_time = measure_point(
        baseline_conf["cpuN"],
        baseline_conf["npuN"],
        baseline_conf["cpuThreads"],
        npuK_override=(baseline_conf["npuK"] if baseline_conf["npuN"] > 0 else None),
    )
    log_print(f"  > Baseline (Current Config): {baseline_conf} -> {baseline_time:.4f}s", log_handle)

    # Mode policy:
    # - gemv_split=M: CPU-only tuning (skip NPU).
    # - gemv_split=K: tune npuK first; if improved, tune cpuN/threads with fixed npuK and npuN=forN.
    candidates = [("Baseline", baseline_conf, baseline_time)]

    if split_mode == "M":
        log_print("  > gemv_split=M: skipping NPU tuning by policy.", log_handle)
        if skip_cpu_tuning:
            log_print("  > Phase 1: Skipped by flag (--gemv-skip-cpu-tuning).", log_handle)
        else:
            log_print("  > Phase 1: CPU-only search (npuN=0)", log_handle)
            best_cpu_thr, best_cpuN, best_cpu_time = fast_cpu_search(
                N,
                0,
                baseline_time,
                "CPU-only",
                fixed_npuK=0
            )
            cpu_only_conf = {"npuK": 0, "cpuN": best_cpuN, "cpuThreads": best_cpu_thr, "npuN": 0}
            candidates.append(("CPU-only", cpu_only_conf, best_cpu_time))
            log_print(f"  > CPU-only best: {cpu_only_conf} -> {best_cpu_time:.4f}s", log_handle)
    else:
        if disable_npu_tuning:
            log_print("  > gemv_split=K: NPU K tuning skipped (policy/runtime constraints).", log_handle)
        else:
            k_step = 256
            best_k = 0
            best_k_time = baseline_time
            npu_k_abort_ratio = 2.5
            npu_k_abort_time = baseline_time * npu_k_abort_ratio
            npu_k_aborted = False
            if k_tuning_upper <= 0:
                log_print(
                    f"  > gemv_split=K: no valid npuK candidates after MAX_K_SPLIT cap (upper={k_tuning_upper}, forK={K}).",
                    log_handle,
                )
            else:
                k_low = k_step if k_tuning_upper >= k_step else k_tuning_upper
                log_print(
                    f"  > Phase 2: NPU-K D&C search in [{k_low}, {k_tuning_upper}] step {k_step}",
                    log_handle
                )

                def measure_npu_k(kk):
                    nonlocal npu_k_aborted
                    if npu_k_aborted:
                        return float("inf")
                    t = measure_point(0, N, 1, npuK_override=int(kk))
                    if baseline_time > 0 and t > npu_k_abort_time:
                        npu_k_aborted = True
                        log_print(
                            f"      [NPU-K] Early stop: npuK={int(kk)} -> {t:.4f}s exceeds {npu_k_abort_ratio:.2f}x baseline ({baseline_time:.4f}s).",
                            log_handle
                        )
                    return t

                best_k, best_k_time = solve_1d(
                    k_low,
                    k_tuning_upper,
                    k_step,
                    measure_npu_k,
                    k_tuning_upper,
                    log_prefix="[NPU-K] "
                )

                if npu_k_aborted:
                    log_print(
                        f"  > Phase 2: Aborted NPU-K search (>{npu_k_abort_ratio:.2f}x baseline). Skipping NPU-K tuning.",
                        log_handle
                    )
                    best_k = 0
                    best_k_time = baseline_time
                # Ensure base npuK is considered even if D&C path does not naturally probe it.
                elif 0 < base_npuK <= k_tuning_upper:
                    base_k = int(_cap_k_tuning_value(model_conf, K, base_npuK))
                    t_base = measure_point(0, N, 1, npuK_override=base_k)
                    log_print(f"      [NPU-K] Seed Probe: npuK={base_k}, npuN={N} -> {t_base:.4f}s", log_handle)
                    if t_base < best_k_time:
                        best_k = base_k
                        best_k_time = t_base

            if best_k > 0:
                npu_only_conf = {"npuK": best_k, "cpuN": 0, "cpuThreads": 1, "npuN": N}
                candidates.append(("NPU-K-only", npu_only_conf, best_k_time))
            k_improved = (best_k > 0 and best_k_time < baseline_time * (1.0 - threshold))

            if k_improved and not skip_cpu_tuning:
                if int(best_k) >= int(K):
                    log_print(
                        f"  > Phase 3: skipping CPU tuning for {layer['name']} (npuK == forK == {K}).",
                        log_handle
                    )
                else:
                    log_print(
                        f"  > Phase 3: npuK improved ({best_k}, {best_k_time:.4f}s). Tuning CPU on GPU branch with fixed npuN={N}.",
                        log_handle
                    )
                    best_thr_k, best_cpuN_k, best_cpu_time_k = fast_cpu_search(
                        N,
                        N,
                        best_k_time,
                        "Ksplit-CPU",
                        fixed_npuK=best_k
                    )
                    cpu_npu_conf = {"npuK": best_k, "cpuN": best_cpuN_k, "cpuThreads": best_thr_k, "npuN": N}
                    candidates.append(("NPU-K+CPU", cpu_npu_conf, best_cpu_time_k))
                    log_print(f"  > NPU-K+CPU best: {cpu_npu_conf} -> {best_cpu_time_k:.4f}s", log_handle)
            elif best_k > 0 and not k_improved:
                log_print(
                    f"  > npuK best ({best_k_time:.4f}s) did not beat baseline by threshold; falling back to CPU-only tuning.",
                    log_handle
                )
                if not skip_cpu_tuning:
                    log_print("  > Phase 3b: CPU-only search (npu disabled)", log_handle)
                    best_cpu_thr, best_cpuN, best_cpu_time = fast_cpu_search(
                        N,
                        0,
                        baseline_time,
                        "CPU-only-fallback",
                        fixed_npuK=0
                    )
                    cpu_only_conf = {"npuK": 0, "cpuN": best_cpuN, "cpuThreads": best_cpu_thr, "npuN": 0}
                    candidates.append(("CPU-only", cpu_only_conf, best_cpu_time))
                    log_print(f"  > CPU-only best: {cpu_only_conf} -> {best_cpu_time:.4f}s", log_handle)
                else:
                    log_print("  > CPU-only fallback skipped by flag (--gemv-skip-cpu-tuning).", log_handle)
            elif k_improved and skip_cpu_tuning:
                log_print("  > Phase 3 skipped by flag (--gemv-skip-cpu-tuning).", log_handle)
            elif best_k == 0 and not skip_cpu_tuning:
                log_print("  > No positive npuK winner; running CPU-only tuning.", log_handle)
                best_cpu_thr, best_cpuN, best_cpu_time = fast_cpu_search(
                    N,
                    0,
                    baseline_time,
                    "CPU-only-fallback",
                    fixed_npuK=0
                )
                cpu_only_conf = {"npuK": 0, "cpuN": best_cpuN, "cpuThreads": best_cpu_thr, "npuN": 0}
                candidates.append(("CPU-only", cpu_only_conf, best_cpu_time))
                log_print(f"  > CPU-only best: {cpu_only_conf} -> {best_cpu_time:.4f}s", log_handle)
        if disable_npu_tuning and not skip_cpu_tuning:
            log_print("  > Phase 1: CPU-only search (NPU disabled)", log_handle)
            best_cpu_thr, best_cpuN, best_cpu_time = fast_cpu_search(
                N,
                0,
                baseline_time,
                "CPU-only",
                fixed_npuK=0
            )
            cpu_only_conf = {"npuK": 0, "cpuN": best_cpuN, "cpuThreads": best_cpu_thr, "npuN": 0}
            candidates.append(("CPU-only", cpu_only_conf, best_cpu_time))
            log_print(f"  > CPU-only best: {cpu_only_conf} -> {best_cpu_time:.4f}s", log_handle)

    if len(candidates) <= 1:
        winner_name, final_best_config, final_best_time = ("Baseline", baseline_conf, baseline_time)
    else:
        best_non_baseline_name, best_non_baseline_conf, best_non_baseline_time = min(candidates[1:], key=lambda x: x[2])
        if best_non_baseline_time < baseline_time * (1.0 - threshold):
            winner_name, final_best_config, final_best_time = (
                best_non_baseline_name,
                best_non_baseline_conf,
                best_non_baseline_time
            )
        else:
            log_print(
                f"  > Constraint: Best non-baseline ({best_non_baseline_time:.4f}s) is not >{threshold * 100:.2f}% better than baseline ({baseline_time:.4f}s). Using baseline.",
                log_handle
            )
            winner_name, final_best_config, final_best_time = ("Baseline", baseline_conf, baseline_time)

    log_print(f"  * Winner ({winner_name}): {final_best_config} -> {final_best_time:.4f}s", log_handle)
    return final_best_config, final_best_time

def tune_all_gemv(log_handle, model_conf, improvement_threshold, skip_cpu_tuning=False, use_current_npuk=False):
    log_print(f"\n>>> Starting GEMV Tuning for {model_conf['name']}", log_handle)
    log_print(f"\n{'='*40}", log_handle)
    log_print(f"TUNING GEMV (Generation, M=1)", log_handle)
    log_print(f"{'='*40}", log_handle)
    
    data = load_config(model_conf)
    if not data: return
    
    data["dummy_weights"] = bool(DUMMY_WEIGHTS)
    data["gemv_driven_split_K"] = True

    target_order = list(GEMV_LAYER_ORDER)
    order_idx = {name: idx for idx, name in enumerate(target_order)}
    npu_target_layers = []
    cpu_tuning_layers = []
    by_name = {l["name"]: l for l in model_conf["layers"]}
    for nm in target_order:
        if nm not in by_name:
            continue
        layer_cfg = by_name[nm]
        cpu_tuning_layers.append(layer_cfg)
        layer_K = int(layer_cfg["K"])
        layer_N = int(layer_cfg["N"])
        layer_split = resolve_gemv_split_mode(nm, layer_K, layer_N, layer_cfg)
        if bool(layer_cfg.get("gemv_disable_npu_tuning", False)):
            log_print(f"Skipping layer {nm} in unified npuK flow (gemv_disable_npu_tuning=True).", log_handle)
            continue
        if layer_split != "K":
            log_print(f"Skipping layer {nm} in unified npuK flow (gemv_split={layer_split}).", log_handle)
            continue
        npu_target_layers.append(layer_cfg)
    cpu_tuning_layers.sort(key=lambda l: order_idx.get(l["name"], 10**9))
    npu_target_layers.sort(key=lambda l: order_idx.get(l["name"], 10**9))
    if not npu_target_layers:
        log_print("No target GEMV layers found for unified npuK search; proceeding with CPU tuning only.", log_handle)

    max_cpu_threads = get_max_cpu_threads()
    k_step = 256
    cpu_step = 128
    threshold = improvement_threshold
    temp_config_path = model_conf["config"] + ".gemv_tmp.json5"
    bench_cache = {}
    heuristics_state = load_chunk_heuristics(model_conf, log_handle) if CHUNK_RECOVER else None
    gemv_recovery_active = CHUNK_RECOVER and heuristics_state is not None

    def get_threads_candidates():
        vals = []
        curr = 1
        while curr <= max_cpu_threads:
            vals.append(curr)
            curr *= 2
        if not vals:
            vals = [1]
        if vals[-1] != max_cpu_threads:
            vals.append(max_cpu_threads)
        return vals

    def find_kernel(trial_data, layer_name, K, N):
        for kk in trial_data.get("kernels_gemv", []):
            if kk.get("layer") == layer_name and int(kk.get("forK", -1)) == int(K) and int(kk.get("forN", -1)) == int(N):
                return kk
        return None

    # Stage A placeholder seeding:
    # Ensure all GEMV layers have explicit config entries (including k/v),
    # so later stages don't rely on implicit get_or_create side-effects.
    seeded = 0
    for layer in cpu_tuning_layers:
        name = layer["name"]
        K = int(layer["K"])
        N = int(layer["N"])
        if find_kernel(data, name, K, N) is not None:
            continue
        split_mode = resolve_gemv_split_mode(name, K, N, layer)
        kernel = get_or_create_kernel_gemv(data, name, 1, K, N, split_mode=split_mode)
        kernel["use"] = False
        kernel["cpuN"] = 0
        kernel["cpuThreads"] = 1
        kernel["npuK"] = 0
        kernel["npuN"] = 0
        seeded += 1
        log_print(
            f"Seed Stage A placeholder: layer={name} split={split_mode} forK={K} forN={N}",
            log_handle
        )
    if seeded > 0:
        log_print(f"Seeded {seeded} Stage A GEMV placeholder kernel entries.", log_handle)

    def set_layer_state(trial_data, layer, use, npuK=0, cpuN=0, cpuThreads=1):
        K = int(layer["K"])
        N = int(layer["N"])
        split_mode = resolve_gemv_split_mode(layer["name"], K, N, layer)
        kernel = get_or_create_kernel_gemv(trial_data, layer["name"], 1, K, N, split_mode=split_mode)
        if not use:
            kernel["use"] = False
            kernel["cpuN"] = 0
            kernel["cpuThreads"] = 1
            kernel["npuK"] = 0
            kernel["npuN"] = 0
            return

        kk = int(npuK)
        if kk <= 0:
            kk = K
        if K >= k_step:
            kk = int(round(float(kk) / float(k_step)) * k_step)
            kk = max(k_step, min(kk, K))
        else:
            kk = K
        kk = _cap_k_tuning_value(model_conf, K, kk)
        if kk <= 0:
            kernel["use"] = False
            kernel["cpuN"] = 0
            kernel["cpuThreads"] = 1
            kernel["npuK"] = 0
            kernel["npuN"] = 0
            return

        kernel["use"] = True
        kernel["cpuN"] = int(max(0, min(N, cpuN)))
        kernel["cpuThreads"] = int(max(1, min(max_cpu_threads, cpuThreads)))
        kernel["npuK"] = kk
        kernel["npuN"] = N

    def signature(trial_data):
        sig = []
        for layer in cpu_tuning_layers:
            K = int(layer["K"])
            N = int(layer["N"])
            k = find_kernel(trial_data, layer["name"], K, N)
            if k is None:
                sig.append((layer["name"], 0, 0, 0, 0, 1))
            else:
                sig.append((
                    layer["name"],
                    int(bool(k.get("use", False))),
                    int(k.get("npuK", 0)),
                    int(k.get("npuN", 0)),
                    int(k.get("cpuN", 0)),
                    int(k.get("cpuThreads", 1)),
                ))
        return tuple(sig)

    def gemv_recovery_key(sig, include_timeout=True):
        payload = {
            "model": str(model_conf.get("id", model_conf.get("name", ""))),
            "run_average": int(RUN_AVERAGE),
            "dummy_weights": bool(DUMMY_WEIGHTS),
            "signature": sig,
        }
        if include_timeout:
            payload["timeout_sec"] = int(BENCHMARK_TIMEOUT_SEC)
        return make_chunk_heuristic_key("gemv_trial", **payload)

    def is_timeout_penalty_time(t):
        if t is None or not math.isfinite(t):
            return False
        penalty = float(max(1, int(BENCHMARK_TIMEOUT_SEC)))
        return abs(float(t) - penalty) <= 1e-12

    def lookup_gemv_cached_time(sig):
        if not gemv_recovery_active:
            return None, None

        # Preferred: exact key for current settings.
        primary_key = gemv_recovery_key(sig, include_timeout=True)
        t = get_chunk_cached_time(heuristics_state, primary_key)
        if t is not None and not is_timeout_penalty_time(t):
            return t, primary_key

        # Backward-compatible: key variant without timeout_sec.
        compat_key = gemv_recovery_key(sig, include_timeout=False)
        t = get_chunk_cached_time(heuristics_state, compat_key)
        if t is not None and not is_timeout_penalty_time(t):
            return t, compat_key

        # Legacy fallback: tolerate timeout changes by matching same trial signature.
        timings = heuristics_state.get("timings", {})
        model_id = str(model_conf.get("id", model_conf.get("name", "")))
        sig_json = [list(x) for x in sig]
        for k, _ in timings.items():
            try:
                obj = json.loads(k)
            except Exception:
                continue
            if obj.get("stage") != "gemv_trial":
                continue
            if str(obj.get("model", "")) != model_id:
                continue
            if int(obj.get("run_average", -1)) != int(RUN_AVERAGE):
                continue
            if bool(obj.get("dummy_weights", False)) != bool(DUMMY_WEIGHTS):
                continue
            if obj.get("signature") != sig_json:
                continue
            t = get_chunk_cached_time(heuristics_state, k)
            if t is not None and not is_timeout_penalty_time(t):
                return t, k
        return None, None

    def measure_trial(trial_data, tag):
        key = signature(trial_data)
        if key in bench_cache:
            return bench_cache[key]
        recovery_key = None
        if gemv_recovery_active:
            cached_time, recovery_key = lookup_gemv_cached_time(key)
            if cached_time is not None:
                bench_cache[key] = cached_time
                log_print(f"    {tag}: recovered cached trial -> {cached_time:.4f}s", log_handle)
                # Normalize onto the current key format as we go.
                set_chunk_cached_time(
                    model_conf,
                    heuristics_state,
                    gemv_recovery_key(key, include_timeout=True),
                    cached_time,
                )
                set_chunk_cached_time(
                    model_conf,
                    heuristics_state,
                    gemv_recovery_key(key, include_timeout=False),
                    cached_time,
                )
                return cached_time

        sort_kernels_gemv(trial_data)
        with open(temp_config_path, "w") as f:
            json.dump(trial_data, f, indent=4)
        try:
            t = run_benchmark_gemv(log_handle, model_conf, config_override=temp_config_path)
        finally:
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
        if not math.isfinite(t):
            # No timeout-penalty substitution: return timeout/invalid as non-finite.
            # Keep only in-memory to avoid immediate re-benchmarking in this run.
            bench_cache[key] = t
            log_print(
                f"    {tag}: timeout/invalid benchmark result; returning timeout (inf).",
                log_handle,
            )
            return t
        bench_cache[key] = t
        if gemv_recovery_active:
            set_chunk_cached_time(model_conf, heuristics_state, gemv_recovery_key(key, include_timeout=True), t)
            set_chunk_cached_time(model_conf, heuristics_state, gemv_recovery_key(key, include_timeout=False), t)
        log_print(f"    {tag}: {t:.4f}s", log_handle)
        return t

    def solve_1d(low, high, step_size, measure_func, max_val_constraint, log_prefix=""):
        low = max(0, min(int(low), int(max_val_constraint)))
        high = max(0, min(int(high), int(max_val_constraint)))
        if high < low:
            low, high = high, low

        best_v = low
        best_t = measure_func(low)
        log_print(f"      {log_prefix}D&C Probe: v={low} -> {best_t:.4f}s", log_handle)
        if high != low:
            t_high = measure_func(high)
            log_print(f"      {log_prefix}D&C Probe: v={high} -> {t_high:.4f}s", log_handle)
            if t_high < best_t:
                best_t = t_high
                best_v = high

        def _recurse(l, h):
            nonlocal best_v, best_t
            if h - l < step_size:
                return
            mid = (l + h) // 2
            mid = round(mid / step_size) * step_size
            if mid <= l:
                mid = l + step_size
            if mid >= h:
                mid = h - step_size
            if mid <= l or mid >= h:
                return

            vals = [l, mid, h]
            times = []
            for v in vals:
                if v > max_val_constraint:
                    v = max_val_constraint
                t = measure_func(v)
                times.append(t)
                log_print(f"      {log_prefix}D&C Probe: v={v} -> {t:.4f}s", log_handle)
                if t < best_t:
                    best_t = t
                    best_v = v

            t_low, t_mid, t_high = times
            if t_low < t_mid and t_low < t_high:
                _recurse(l, mid)
            elif t_high < t_mid and t_high < t_low:
                _recurse(mid, h)
            else:
                _recurse(l, mid)
                _recurse(mid, h)

        _recurse(low, high)
        return int(best_v), float(best_t)

    def align_cpuN(v, N):
        v = int(round(float(v) / float(cpu_step)) * cpu_step)
        v = max(0, min(v, int(N)))
        if v != 0 and v % cpu_step != 0:
            v = (v // cpu_step) * cpu_step
        return v

    def align_npuk(forK, factor):
        forK = int(forK)
        upper = _k_tuning_upper_bound(model_conf, forK)
        if upper <= 0:
            return 0
        if upper < k_step:
            return upper
        raw = int(round((forK * factor) / float(k_step)) * k_step)
        raw = max(k_step, min(raw, upper))
        return raw

    try:
        selected_trial = None
        selected_time = None
        enabled = []
        unified_npuk = {}

        if use_current_npuk:
            log_print("\n  > Stage A/B/C bypass: using current per-layer npuK config, skipping npuK search.", log_handle)
            selected_trial = json.loads(json.dumps(data))
            for layer in npu_target_layers:
                name = layer["name"]
                K = int(layer["K"])
                N = int(layer["N"])
                existing = find_kernel(selected_trial, name, K, N)
                if existing is None:
                    log_print(f"    Skip {name}: no existing GEMV kernel entry.", log_handle)
                    continue
                if not bool(existing.get("use", False)):
                    log_print(f"    Skip {name}: existing kernel use=false.", log_handle)
                    continue
                curr_k = int(existing.get("npuK", 0))
                curr_n = int(existing.get("npuN", 0))
                if curr_k <= 0 or curr_n <= 0:
                    log_print(f"    Skip {name}: invalid current NPU config npuK={curr_k}, npuN={curr_n}.", log_handle)
                    continue
                aligned_k = align_npuk(K, float(curr_k) / float(max(1, K)))
                curr_cpuN = int(existing.get("cpuN", 0))
                curr_threads = int(existing.get("cpuThreads", 1))
                set_layer_state(selected_trial, layer, True, npuK=aligned_k, cpuN=curr_cpuN, cpuThreads=curr_threads)
                unified_npuk[name] = aligned_k
                enabled.append(name)
                log_print(f"    Use {name}: npuK={aligned_k}, cpuN={curr_cpuN}, cpuThreads={curr_threads}", log_handle)

            if not enabled:
                log_print("No enabled layers with current npuK config; using current config baseline for CPU tuning.", log_handle)
                selected_time = measure_trial(selected_trial, "[Bypass] Baseline (current config)")
            else:
                selected_time = measure_trial(selected_trial, "[Bypass] Baseline (current npuK config)")
            log_print(f"  > Bypass baseline: {selected_time:.4f}s", log_handle)
        else:
            if not npu_target_layers:
                log_print("\n  > Stage A/B/C: skipped (no NPU-target layers).", log_handle)
                selected_trial = json.loads(json.dumps(data))
                selected_time = measure_trial(selected_trial, "[Stage A/B/C] Baseline (current config)")
            else:
                # Stage A: per-layer independent npuK search with all target layers disabled by default.
                base_trial = json.loads(json.dumps(data))
                for layer in npu_target_layers:
                    set_layer_state(base_trial, layer, False)

                log_print("\n  > Stage A: Independent npuK search per layer (all target layers off by default)", log_handle)
                isolated_baseline = measure_trial(base_trial, "[Stage A] Baseline (all target layers off)")
                per_layer_results = {}

                for layer in npu_target_layers:
                    name = layer["name"]
                    K = int(layer["K"])
                    N = int(layer["N"])
                    k_upper = _k_tuning_upper_bound(model_conf, K)
                    if k_upper <= 0:
                        log_print(
                            f"    Layer {name}: skipping npuK search (MAX_K_SPLIT cap gives upper bound {k_upper}).",
                            log_handle
                        )
                        continue
                    k_low = k_step if k_upper >= k_step else k_upper
                    log_print(f"    Layer {name}: searching npuK in [{k_low}, {k_upper}] step {k_step}", log_handle)

                    def measure_k(v):
                        trial = json.loads(json.dumps(base_trial))
                        set_layer_state(trial, layer, True, npuK=int(v), cpuN=0, cpuThreads=1)
                        return measure_trial(trial, f"[Stage A][{name}] npuK={int(v)}")

                    best_k, best_t = solve_1d(k_low, k_upper, k_step, measure_k, k_upper, log_prefix=f"[{name}] ")
                    speedup = 0.0
                    if isolated_baseline > 0:
                        speedup = (isolated_baseline - best_t) / isolated_baseline
                    per_layer_results[name] = {"best_k": best_k, "best_t": best_t, "K": K, "N": N, "speedup": speedup}
                    log_print(
                        f"    Layer {name}: best npuK={best_k} -> {best_t:.4f}s (speedup {(speedup * 100.0):.2f}%)",
                        log_handle
                    )

                if not per_layer_results:
                    log_print("No per-layer npuK results produced; aborting GEMV unified tuning.", log_handle)
                    return

                winner_name = max(per_layer_results.keys(), key=lambda nm: per_layer_results[nm]["speedup"])
                winner = per_layer_results[winner_name]
                winner_layer_k = int(winner["K"])
                winner_k_upper = _k_tuning_upper_bound(model_conf, winner_layer_k)
                winner_abs_k = int(winner["best_k"]) if int(winner["best_k"]) > 0 else int(winner_k_upper)
                log_print(
                    f"  > Stage B: Unified absolute npuK from layer {winner_name}: npuK={winner_abs_k}",
                    log_handle
                )

                for layer in npu_target_layers:
                    layer_k = int(layer["K"])
                    layer_upper = _k_tuning_upper_bound(model_conf, layer_k)
                    uk = int(min(layer_k, winner_abs_k))
                    if layer_k >= k_step:
                        uk = int(round(float(uk) / float(k_step)) * k_step)
                        uk = max(k_step, min(uk, layer_k))
                    else:
                        uk = layer_k
                    uk = _cap_k_tuning_value(model_conf, layer_k, uk)
                    if uk <= 0:
                        log_print(
                            f"    Unified npuK[{layer['name']}] capped to 0 (upper bound {layer_upper}); layer will remain disabled.",
                            log_handle
                        )
                    unified_npuk[layer["name"]] = uk
                    log_print(f"    Unified npuK[{layer['name']}] = {uk}/{layer_k}", log_handle)

                # Stage C: greedy enablement one-by-one with unified npuK.
                log_print("\n  > Stage C: Greedy layer enablement with unified npuK", log_handle)
                selected_trial = json.loads(json.dumps(base_trial))
                selected_time = isolated_baseline
                enabled = []

                for layer in npu_target_layers:
                    name = layer["name"]
                    cand = json.loads(json.dumps(selected_trial))
                    set_layer_state(cand, layer, True, npuK=unified_npuk[name], cpuN=0, cpuThreads=1)
                    t = measure_trial(cand, f"[Stage C] Enable {name}")
                    if t < selected_time * (1.0 - threshold):
                        selected_trial = cand
                        selected_time = t
                        enabled.append(name)
                        log_print(f"    + Keep {name}: {t:.4f}s", log_handle)
                    else:
                        log_print(f"    - Skip {name}: {t:.4f}s (no >{threshold*100.0:.2f}% gain)", log_handle)

                # Stage C-2: subtractive pass over enabled layers.
                log_print("\n  > Stage C-2: Greedy subtractive pass (disable enabled layers one-by-one)", log_handle)
                if not enabled:
                    log_print("    No enabled layers to test in Stage C-2.", log_handle)
                else:
                    for name in list(enabled):
                        layer_cfg = by_name[name]
                        cand = json.loads(json.dumps(selected_trial))
                        set_layer_state(cand, layer_cfg, False)
                        t = measure_trial(cand, f"[Stage C-2] Disable {name}")
                        if t < selected_time * (1.0 - threshold):
                            selected_trial = cand
                            selected_time = t
                            enabled.remove(name)
                            log_print(f"    + Disable {name}: {t:.4f}s", log_handle)
                        else:
                            log_print(f"    - Keep {name}: {t:.4f}s (disable no >{threshold*100.0:.2f}% gain)", log_handle)

        # Stage D: tune cpuN and threads per enabled layer, with unified npuK fixed.
        if skip_cpu_tuning:
            log_print("\n  > Stage D: CPU tuning skipped by flag (--gemv-skip-cpu-tuning).", log_handle)
        else:
            log_print("\n  > Stage D: Per-layer cpuN/cpuThreads tuning with unified npuK fixed", log_handle)
            threads_candidates = get_threads_candidates()
            for layer in cpu_tuning_layers:
                name = layer["name"]
                K = int(layer["K"])
                N = int(layer["N"])
                split_mode = resolve_gemv_split_mode(name, K, N, layer)
                base_kernel = find_kernel(selected_trial, name, int(layer["K"]), N)
                base_cpuN = int(base_kernel.get("cpuN", 0)) if base_kernel else 0
                base_threads = int(base_kernel.get("cpuThreads", 1)) if base_kernel else 1
                base_threads = max(1, min(max_cpu_threads, base_threads))
                is_unified_npu_layer = name in unified_npuk
                npu_enabled_for_layer = bool(
                    is_unified_npu_layer and base_kernel and base_kernel.get("use", False) and int(base_kernel.get("npuN", 0)) > 0
                )
                current_layer_k = int(base_kernel.get("npuK", 0)) if base_kernel else 0
                if is_unified_npu_layer:
                    fixed_k = int(unified_npuk.get(name, current_layer_k if current_layer_k > 0 else 0))
                else:
                    fixed_k = 0
                if npu_enabled_for_layer and int(fixed_k) >= int(K):
                    log_print(
                        f"    Layer {name}: CPU tuning skipped (npuK == forK == {K}).",
                        log_handle
                    )
                    continue

                def eval_cpu(cn, thr):
                    trial = json.loads(json.dumps(selected_trial))
                    if npu_enabled_for_layer:
                        set_layer_state(trial, layer, True, npuK=fixed_k, cpuN=cn, cpuThreads=thr)
                    else:
                        # CPU-only exploration for layers currently disabled from NPU.
                        kernel = get_or_create_kernel_gemv(trial, layer["name"], 1, K, N, split_mode=split_mode)
                        cn = int(max(0, min(N, cn)))
                        thr = int(max(1, min(max_cpu_threads, thr)))
                        if cn == 0:
                            kernel["use"] = False
                            kernel["cpuN"] = 0
                            kernel["cpuThreads"] = 1
                            kernel["npuK"] = 0
                            kernel["npuN"] = 0
                        else:
                            kernel["use"] = True
                            kernel["cpuN"] = cn
                            kernel["cpuThreads"] = thr
                            kernel["npuK"] = 0
                            kernel["npuN"] = 0
                    return measure_trial(trial, f"[Stage D][{name}] cpuN={cn}, threads={thr}")

                layer_ref_time = eval_cpu(base_cpuN, base_threads)
                fixed_threads = 8 if max_cpu_threads >= 8 else 1
                best_cpuN, best_time = solve_1d(
                    0,
                    N,
                    cpu_step,
                    lambda v: eval_cpu(align_cpuN(v, N), fixed_threads),
                    N,
                    log_prefix=f"[{name} cpuN@T{fixed_threads}] "
                )
                best_cpuN = align_cpuN(best_cpuN, N)
                best_threads = fixed_threads
                best_time = eval_cpu(best_cpuN, best_threads)

                if best_time < layer_ref_time * (1.0 - threshold):
                    for thr in threads_candidates:
                        t = eval_cpu(best_cpuN, thr)
                        if t < best_time:
                            best_time = t
                            best_threads = thr

                    delta = cpu_step * 2
                    for cn in sorted({align_cpuN(best_cpuN - delta, N), best_cpuN, align_cpuN(best_cpuN + delta, N)}):
                        t = eval_cpu(cn, best_threads)
                        if t < best_time:
                            best_time = t
                            best_cpuN = cn

                    if best_time < layer_ref_time * (1.0 - threshold):
                        if npu_enabled_for_layer:
                            set_layer_state(selected_trial, layer, True, npuK=fixed_k, cpuN=best_cpuN, cpuThreads=best_threads)
                        else:
                            kernel = get_or_create_kernel_gemv(selected_trial, layer["name"], 1, K, N, split_mode=split_mode)
                            if best_cpuN == 0:
                                kernel["use"] = False
                                kernel["cpuN"] = 0
                                kernel["cpuThreads"] = 1
                                kernel["npuK"] = 0
                                kernel["npuN"] = 0
                            else:
                                kernel["use"] = True
                                kernel["cpuN"] = int(best_cpuN)
                                kernel["cpuThreads"] = int(best_threads)
                                kernel["npuK"] = 0
                                kernel["npuN"] = 0
                        selected_time = best_time
                        log_print(
                            f"    Layer {name}: keep cpu split cpuN={best_cpuN}, threads={best_threads} -> {best_time:.4f}s",
                            log_handle
                        )
                    else:
                        log_print(f"    Layer {name}: CPU tuning not beneficial after thread/neighbor checks.", log_handle)
                else:
                    log_print(f"    Layer {name}: CPU tuning skipped (no gain over reference).", log_handle)

        data["kernels_gemv"] = selected_trial.get("kernels_gemv", [])
        save_config(data, model_conf)
        log_print(f"\nSaved unified GEMV config. Enabled layers: {enabled}", log_handle)
        log_print(f"Final measured time: {selected_time:.4f}s", log_handle)

    except KeyboardInterrupt:
        log_print("GEMV Tuning Interrupted!", log_handle)
    except TuningAbortError as e:
        log_print(str(e), log_handle)
        raise SystemExit(1)
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    model_choices = [model["id"] for model in models_config]
    default_model = MODEL if MODEL in model_choices else model_choices[0]
    parser.add_argument(
        "--mode",
        choices=["gemm", "gemm_chunking", "gemm_chunkingS", "gemm_CW", "gemv", "all"],
        default="all",
        help="Tuning mode: gemm, gemm_chunking, gemm_chunkingS, gemm_CW, gemv, or all"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.002,
        help="Minimum relative improvement required to accept a non-baseline GEMV config (e.g. 0.02 = 2%%).",
    )
    parser.add_argument(
        "--gemv-skip-cpu-tuning",
        action="store_true",
        help="Skip GEMV CPU-only tuning phase and directly test NPU-only behavior.",
    )
    parser.add_argument(
        "--gemv-use-current-npuk",
        action="store_true",
        help="Skip unified npuK search and use current per-layer npuK from config, then run CPU tuning.",
    )
    parser.add_argument(
        "--clean-temp",
        action="store_true",
        help="Delete config temp files and .log files, then exit.",
    )
    parser.add_argument(
        "--canonicalize",
        action="store_true",
        help="Canonicalize grouped chunked GEMM configs: realign forM to the chunk plan and prune stale duplicate groups.",
    )
    parser.add_argument(
        "--model",
        choices=model_choices + ["all"],
        default=default_model,
        help="Model to tune. Available targets come from models_config, including gemma, llama3_8b, llama3_70b, qwen14b, and phi3.5_3.8b.",
    )
    args = parser.parse_args()
    if args.threshold < 0.0 or args.threshold >= 1.0:
        raise ValueError("--threshold must be in [0.0, 1.0).")

    if args.clean_temp:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        removed = clear_temp_artifacts(script_dir)
        print(f"Removed {len(removed)} files.")
        for path in removed:
            print(f"  {path}")
        return
    
    log_handle = open(LOG_FILE, 'w')
    log_print(f"Starting Tuning run (Mode: {args.mode})...", log_handle)
    log_print(f"Selected model target: {args.model}", log_handle)
    log_print(f"GEMV improvement threshold: {args.threshold:.4f}", log_handle)
    
    for model_conf in models_config:
        if args.model != "all" and model_conf.get("id") != args.model:
            continue

        resolved_config, cpu_model_name, selected_profile = _resolve_model_config_path_for_platform(model_conf)
        model_conf["config"] = resolved_config
        if cpu_model_name:
            log_print(
                f"Resolved config profile '{selected_profile}' from lscpu model '{cpu_model_name}': {resolved_config}",
                log_handle,
            )
        else:
            log_print(
                f"Resolved fallback config profile '{selected_profile}' (lscpu model not detected): {resolved_config}",
                log_handle,
            )
        _bootstrap_missing_config_with_submodel(model_conf, log_handle)

        log_print(f"\n{'#'*60}", log_handle)
        log_print(f"PROCESSING MODEL: {model_conf['name']}", log_handle)
        log_print(f"{'#'*60}", log_handle)

        if args.canonicalize:
            data = load_config(model_conf)
            if not data:
                continue
            result = canonicalize_chunked_config(data, prune_duplicates=True)
            save_config_via_temp(data, model_conf, temp_suffix=".canonicalize_tmp.json5")
            log_print(
                f"Canonicalized {model_conf['config']}: "
                f"groups_changed={result['groups_changed']} "
                f"groups_dropped={result['groups_dropped']} "
                f"kernels_dropped={result['kernels_dropped']}",
                log_handle,
            )
            continue

        if args.mode == "gemm" or args.mode == "all":
            tune_all_gemm(log_handle, model_conf)

        if args.mode == "gemm_chunking" or args.mode == "all":
            tune_all_gemm_chunking(log_handle, model_conf, mode_variant="gemm_chunking")

        if args.mode == "gemm_chunkingS":
            tune_all_gemm_chunking(log_handle, model_conf, mode_variant="gemm_chunkingS")

        if args.mode == "gemm_CW" or args.mode == "all":
            tune_all_gemm_cw(log_handle, model_conf)
            
        if args.mode == "all":
            log_print("Sleeping 4s before GEMV tuning...", log_handle)
            time.sleep(4)
            
        if args.mode == "gemv" or args.mode == "all":
            tune_all_gemv(
                log_handle,
                model_conf,
                args.threshold,
                args.gemv_skip_cpu_tuning,
                args.gemv_use_current_npuk,
            )
    
    log_handle.close()
                                    
if __name__ == "__main__":
    main()
