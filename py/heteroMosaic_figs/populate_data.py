#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_INPUT_PATH = Path(__file__).resolve().parent / "results"
RESULTS_DIR = DEFAULT_INPUT_PATH
REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_RESULTS_DIR = REPO_ROOT / "build" / "results"

# Default model target for results population.
# Available values:
# MODEL = "gemma"
# MODEL = "llama3_8b"
# MODEL = "llama3_70b"
MODEL = "llama3_8b"
# MODEL = "qwen14b"

MODEL_CONFIGS = {
    "gemma": {
        "output_json": RESULTS_DIR / "gemma_results.json",
        "summary_aliases": ("gemma", "gemma1-2b_q4_k_s"),
        "heteromosaic_script": "gemma_w4a16_model.py",
    },
    "llama3_8b": {
        "output_json": RESULTS_DIR / "llama3-8b_results.json",
        "summary_aliases": ("llama3-8b", "llama3_8b", "llama3", "llama3-8b_q4_k_s"),
        "heteromosaic_script": "llama3_8b_w4a16_model.py",
    },
    "llama3_70b": {
        "output_json": RESULTS_DIR / "llama3-70b_results.json",
        "summary_aliases": ("llama3-70b", "llama3_70b", "llama3 70b", "llama3-70b_q4_k_s"),
        "heteromosaic_script": "llama3_70b_w4a16_model.py",
    },
    "qwen14b": {
        "output_json": RESULTS_DIR / "qwen14b_results.json",
        "summary_aliases": ("qwen14b", "qwen2.5-14b", "qwen25-14b", "qwen2.5-14b_q4_k_s"),
        "heteromosaic_script": "qwen25_14b_w4a16_model.py",
    },
    "phi35_3.8b": {
        "output_json": RESULTS_DIR / "phi35_3.8b_results.json",
        "summary_aliases": (
            "phi35_3.8b",
            "phi3.5-3.8b",
            "phi3.5_3.8b",
            "phi35-3.8b",
            "phi3.5-3.8b_q4_k_s",
        ),
        "heteromosaic_script": "phi35_3.8b_w4a16_model.py",
    },
}

PLATFORM_NAME_MAP = {
    "350": "Ryzen AI 7 350",
    "370": "Ryzen AI 9 HX 370",
    "395": "Ryzen AI MAX+ 395",
}
BENCHMARK_FILE_RE = re.compile(r"benchmark_(\d+)\.txt$")

BENCHMARK_ROW_RE = re.compile(
    r"^\s*\d+\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([0-9.]+)(?:\s*\|.*)?\s*$"
)
SUMMARY_ROW_RE = re.compile(r"^\s*([A-Za-z0-9._+-]+)\s*\|\s*(.+?)\|\s*$")
SUMMARY_CELL_RE = re.compile(r"([0-9.]+)\s*/\s*([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse a benchmark .txt file and populate a results JSON file using "
            "prefill-only selection rules for iGPU, npu, HeteroInfer, and HeteroMosaic."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to a benchmark text file or a directory containing benchmark_*.txt files (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="Top-level JSON key for single-file mode; ignored when auto-loading benchmark_*.txt files",
    )
    parser.add_argument(
        "--machine",
        choices=sorted(PLATFORM_NAME_MAP),
        default=None,
        help="Machine id for machine-scoped output and platform naming (350, 370, or 395).",
    )
    parser.add_argument(
        "--heteromosaic-input",
        type=Path,
        default=None,
        help="HeteroMosaic benchmark.json state or captured benchmark stdout.",
    )
    parser.add_argument(
        "--llamacpp-input",
        type=Path,
        default=None,
        help="llama.cpp benchmark.json state or captured benchmark stdout.",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help=(
            "Model id used for the output filename and summary matching "
            f"(default: {MODEL}; available: {', '.join(MODEL_CONFIGS)})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path; defaults to the selected model's results JSON",
    )
    return parser.parse_args()


def round4(value: float) -> float:
    return float(f"{value:.4f}")


def normalize_model_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def get_model_config(model_id: str | None) -> dict:
    selected_model = (model_id or MODEL).strip()
    if selected_model not in MODEL_CONFIGS:
        available = ", ".join(sorted(MODEL_CONFIGS))
        raise SystemExit(f"Unsupported model '{selected_model}'. Available values: {available}")
    return MODEL_CONFIGS[selected_model]


def resolve_output_path(requested_output: Path | None, model_config: dict, machine_id: str | None = None) -> Path:
    if requested_output is not None:
        return requested_output.resolve()
    if machine_id is not None:
        return (BUILD_RESULTS_DIR / machine_id / Path(model_config["output_json"]).name).resolve()
    return Path(model_config["output_json"]).resolve()


def infer_platform_name(input_txt: Path, explicit_platform: str | None) -> str:
    if explicit_platform:
        return explicit_platform

    match = BENCHMARK_FILE_RE.match(input_txt.name)
    if match:
        platform_id = match.group(1)
        if platform_id in PLATFORM_NAME_MAP:
            return PLATFORM_NAME_MAP[platform_id]

    return input_txt.stem


def discover_input_files(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        files = sorted(path.resolve() for path in input_path.glob("benchmark_*.txt"))
        if files:
            return files
        raise SystemExit(f"No benchmark_*.txt files found in directory: {input_path}")

    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")
    if not input_path.is_file():
        raise SystemExit(f"Input path is not a file: {input_path}")

    match = BENCHMARK_FILE_RE.match(input_path.name)
    if match:
        sibling_files = sorted(path.resolve() for path in input_path.parent.glob("benchmark_*.txt"))
        if sibling_files:
            return sibling_files

    return [input_path.resolve()]


def classify_scenario(scenario: str) -> str | None:
    normalized = scenario.strip().lower()
    if normalized.startswith("gpu/") or normalized.startswith("backend=gpu,"):
        return "iGPU"
    if normalized.startswith("npu/") or normalized.startswith("backend=npu,"):
        return "npu"
    if normalized.startswith("hetero/c=0/") or (
        normalized.startswith("backend=hetero,") and "chunking=false" in normalized
    ):
        return "HeteroInfer"
    if normalized.startswith("hetero/c=1/") or (
        normalized.startswith("backend=hetero,") and "chunking=true" in normalized
    ):
        return "HeteroMosaic"
    return None


def classify_case(case: dict) -> str | None:
    backend = str(case.get("backend", "")).strip().lower()
    if backend == "gpu":
        return "iGPU"
    if backend == "npu":
        return "npu"
    if backend == "hetero":
        return "HeteroMosaic" if bool(case.get("chunking")) else "HeteroInfer"
    return None


def update_prefill_category(
    results: dict[str, dict[str, float]],
    size: str,
    category: str | None,
    prefill: float,
) -> None:
    if category is None:
        return

    bucket = results.setdefault(size, {})
    if category in {"iGPU", "npu", "HeteroMosaic"}:
        current = bucket.get(category)
        if current is None or prefill < current:
            bucket[category] = prefill
        return

    if category == "HeteroInfer":
        current = bucket.get(category)
        if current is None or prefill > current:
            bucket[category] = prefill


def update_prefill_best(results: dict[str, dict[str, float]], size: str, scenario: str, prefill: float) -> None:
    update_prefill_category(results, size, classify_scenario(scenario), prefill)


def parse_benchmark_rows(text: str) -> tuple[dict[str, dict[str, float]], list[str]]:
    prefill_by_size: dict[str, dict[str, float]] = {}
    seen_sizes: list[str] = []

    for line in text.splitlines():
        match = BENCHMARK_ROW_RE.match(line)
        if not match:
            continue

        size = match.group(1)
        scenario = match.group(2).strip()
        status = match.group(3).strip().lower()
        prefill = float(match.group(4))

        if status != "done":
            continue

        if size not in seen_sizes:
            seen_sizes.append(size)

        update_prefill_best(prefill_by_size, size, scenario, prefill)

    return prefill_by_size, seen_sizes


def load_json_state(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read benchmark state JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Benchmark state must contain a JSON object: {path}")
    return payload


def parse_heteromosaic_state(
    state: dict,
    model_config: dict,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    expected_script = model_config.get("heteromosaic_script")
    cached_script = state.get("script")
    if expected_script and cached_script != expected_script:
        raise SystemExit(
            f"HeteroMosaic cache contains script '{cached_script}', expected '{expected_script}'. "
            "Run the requested model benchmark first."
        )

    prefill_by_size: dict[str, dict[str, float]] = {}
    seen_sizes: list[str] = []
    for case in state.get("cases", []):
        if not isinstance(case, dict) or case.get("status") != "done":
            continue
        metrics = case.get("metrics")
        if not isinstance(metrics, dict):
            continue
        try:
            size = str(int(case["prompt_size"]))
            prefill = float(metrics["prefill_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if prefill <= 0:
            continue
        if size not in seen_sizes:
            seen_sizes.append(size)
        update_prefill_category(
            prefill_by_size,
            size,
            classify_case(case),
            prefill,
        )

    if not seen_sizes:
        raise SystemExit("No completed HeteroMosaic benchmark cases were found in the cache.")
    return prefill_by_size, seen_sizes


def parse_llamacpp_state(
    state: dict,
    benchmark_sizes: list[str],
    requested_model: str,
) -> dict[str, float]:
    model_config = get_model_config(requested_model)
    aliases = {
        normalize_model_name(requested_model),
        *(normalize_model_name(alias) for alias in model_config.get("summary_aliases", ())),
    }
    requested_sizes = set(benchmark_sizes)
    results: dict[str, float] = {}
    for case in state.get("cases", []):
        if not isinstance(case, dict) or case.get("status") != "done":
            continue
        if normalize_model_name(str(case.get("model", ""))) not in aliases:
            continue
        result = case.get("result")
        if not isinstance(result, dict):
            continue
        try:
            size = str(int(case["prompt_size"]))
            ttft = float(result["ttft"])
        except (KeyError, TypeError, ValueError):
            continue
        if size in requested_sizes and ttft > 0:
            results[size] = ttft

    missing_sizes = sorted(requested_sizes - set(results), key=int)
    if missing_sizes:
        raise SystemExit(
            "llama.cpp cache is missing completed TTFT results for prompt sizes: "
            + ", ".join(missing_sizes)
        )
    return results


def infer_model_and_llamacpp(text: str, benchmark_sizes: list[str], requested_model: str | None) -> tuple[str, dict[str, float]]:
    summary_entries: list[tuple[str, list[float]]] = []

    for line in text.splitlines():
        match = SUMMARY_ROW_RE.match(line)
        if not match:
            continue

        model_name = match.group(1).strip()
        tail = match.group(2)
        cells = [float(cell_match.group(1)) for cell_match in SUMMARY_CELL_RE.finditer(tail)]
        if cells:
            summary_entries.append((model_name, cells))

    if requested_model is not None:
        model_config = get_model_config(requested_model)
        aliases = {
            normalize_model_name(requested_model),
            *(normalize_model_name(alias) for alias in model_config.get("summary_aliases", ())),
        }
        for model_name, cells in summary_entries:
            if normalize_model_name(model_name) in aliases:
                return requested_model, map_llamacpp_sizes(cells, benchmark_sizes)

        if len(summary_entries) == 1:
            _, cells = summary_entries[0]
            return requested_model, map_llamacpp_sizes(cells, benchmark_sizes)

        raise SystemExit(
            f"Could not find summary row for requested model '{requested_model}'. "
            f"Found: {', '.join(model_name for model_name, _ in summary_entries)}"
        )

    if not summary_entries:
        raise SystemExit("Could not find any summary row like 'model | ttft / tok | ...'")

    model_name, cells = summary_entries[0]
    return model_name, map_llamacpp_sizes(cells, benchmark_sizes)


def map_llamacpp_sizes(ttft_values: list[float], benchmark_sizes: list[str]) -> dict[str, float]:
    if not benchmark_sizes:
        raise SystemExit("No benchmark size rows were found in the input text")

    ordered_sizes = sorted(benchmark_sizes, key=int)
    if len(ttft_values) < len(ordered_sizes):
        raise SystemExit(
            f"Summary row has only {len(ttft_values)} TTFT values, but benchmark rows contain {len(ordered_sizes)} prompt sizes"
        )

    return {size: ttft_values[idx] for idx, size in enumerate(ordered_sizes)}


def build_output(
    prefill_by_size: dict[str, dict[str, float]],
    llamacpp_by_size: dict[str, float],
) -> dict[str, dict[str, list[float] | list[str]]]:
    all_sizes = sorted(set(prefill_by_size) | set(llamacpp_by_size), key=int)
    platform_data: dict[str, dict[str, list[float] | list[str]]] = {}

    for size in all_sizes:
        categories: dict[str, list[float] | list[str]] = {}
        if size in llamacpp_by_size:
            categories["llama.cpp"] = [llamacpp_by_size[size]]
        else:
            categories["llama.cpp"] = ["NA"]

        prefill_bucket = prefill_by_size.get(size, {})
        for label in ("iGPU", "npu", "HeteroInfer", "HeteroMosaic"):
            if label in prefill_bucket:
                categories[label] = [round4(prefill_bucket[label])]
            else:
                categories[label] = ["NA"]

        platform_data[size] = categories

    return platform_data


def main() -> int:
    args = parse_args()
    model_config = get_model_config(args.model)
    if args.machine and args.platform:
        raise SystemExit("Use either --machine or --platform, not both.")
    if (args.heteromosaic_input is None) != (args.llamacpp_input is None):
        raise SystemExit("--heteromosaic-input and --llamacpp-input must be provided together.")

    output: dict[str, dict[str, dict[str, list[float] | list[str]]]] = {}
    model_name: str | None = args.model

    if args.heteromosaic_input is not None and args.llamacpp_input is not None:
        heteromosaic_input = args.heteromosaic_input.resolve()
        llamacpp_input = args.llamacpp_input.resolve()
        for input_path in (heteromosaic_input, llamacpp_input):
            if not input_path.is_file():
                raise SystemExit(f"Input file does not exist: {input_path}")

        if heteromosaic_input.suffix.lower() == ".json":
            prefill_by_size, benchmark_sizes = parse_heteromosaic_state(
                load_json_state(heteromosaic_input),
                model_config,
            )
        else:
            heteromosaic_text = heteromosaic_input.read_text(encoding="utf-8")
            prefill_by_size, benchmark_sizes = parse_benchmark_rows(heteromosaic_text)

        if llamacpp_input.suffix.lower() == ".json":
            current_model_name = args.model
            llamacpp_by_size = parse_llamacpp_state(
                load_json_state(llamacpp_input),
                benchmark_sizes,
                args.model,
            )
        else:
            llamacpp_text = llamacpp_input.read_text(encoding="utf-8")
            current_model_name, llamacpp_by_size = infer_model_and_llamacpp(
                llamacpp_text,
                benchmark_sizes,
                args.model,
            )
        if model_name is None:
            model_name = current_model_name
        platform_name = (
            PLATFORM_NAME_MAP[args.machine]
            if args.machine is not None
            else (args.platform or heteromosaic_input.stem)
        )
        output[platform_name] = build_output(prefill_by_size, llamacpp_by_size)
    else:
        input_path = args.input_path.resolve()
        input_files = discover_input_files(input_path)
        for input_txt in input_files:
            text = input_txt.read_text(encoding="utf-8")
            prefill_by_size, benchmark_sizes = parse_benchmark_rows(text)
            current_model_name, llamacpp_by_size = infer_model_and_llamacpp(text, benchmark_sizes, args.model)
            if model_name is None:
                model_name = current_model_name

            explicit_platform = args.platform if len(input_files) == 1 else None
            if args.machine is not None and len(input_files) == 1:
                explicit_platform = PLATFORM_NAME_MAP[args.machine]
            platform_name = infer_platform_name(input_txt, explicit_platform)
            output[platform_name] = build_output(prefill_by_size, llamacpp_by_size)

    output_path = resolve_output_path(args.output, model_config, args.machine)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=4) + "\n", encoding="utf-8")

    print(f"Model: {model_name}")
    print(f"Platforms: {', '.join(output.keys())}")
    for platform_name, platform_data in output.items():
        print(f"{platform_name} prompt sizes: {', '.join(sorted(platform_data.keys(), key=int))}")
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
