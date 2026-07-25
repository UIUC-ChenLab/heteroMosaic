#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_INPUT_PATH = Path(__file__).resolve().parent / "results"
RESULTS_DIR = DEFAULT_INPUT_PATH

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
        "summary_aliases": ("gemma",),
    },
    "llama3_8b": {
        "output_json": RESULTS_DIR / "llama3-8b_results.json",
        "summary_aliases": ("llama3-8b", "llama3_8b", "llama3"),
    },
    "llama3_70b": {
        "output_json": RESULTS_DIR / "llama3-70b_results.json",
        "summary_aliases": ("llama3-70b", "llama3_70b", "llama3 70b"),
    },
    "qwen14b": {
        "output_json": RESULTS_DIR / "qwen14b_results.json",
        "summary_aliases": ("qwen14b", "qwen2.5-14b", "qwen25-14b"),
    },
    "phi35_3.8b": {
        "output_json": RESULTS_DIR / "phi35_3.8b_results.json",
        "summary_aliases": ("phi35_3.8b", "phi3.5-3.8b", "phi3.5_3.8b", "phi35-3.8b"),
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


def resolve_output_path(requested_output: Path | None, model_config: dict) -> Path:
    if requested_output is not None:
        return requested_output.resolve()
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
    if scenario.startswith("gpu/"):
        return "iGPU"
    if scenario.startswith("npu/"):
        return "npu"
    if scenario.startswith("hetero/c=0/"):
        return "HeteroInfer"
    if scenario.startswith("hetero/c=1/"):
        return "HeteroMosaic"
    return None


def update_prefill_best(results: dict[str, dict[str, float]], size: str, scenario: str, prefill: float) -> None:
    bucket = results.setdefault(size, {})
    category = classify_scenario(scenario)
    if category is None:
        return

    if category in {"iGPU", "npu", "HeteroMosaic"}:
        current = bucket.get(category)
        if current is None or prefill < current:
            bucket[category] = prefill
        return

    if category == "HeteroInfer":
        current = bucket.get(category)
        if current is None or prefill > current:
            bucket[category] = prefill


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
    input_path = args.input_path.resolve()
    input_files = discover_input_files(input_path)
    model_config = get_model_config(args.model)

    output: dict[str, dict[str, dict[str, list[float] | list[str]]]] = {}
    model_name: str | None = args.model

    for input_txt in input_files:
        text = input_txt.read_text(encoding="utf-8")
        prefill_by_size, benchmark_sizes = parse_benchmark_rows(text)
        current_model_name, llamacpp_by_size = infer_model_and_llamacpp(text, benchmark_sizes, args.model)
        if model_name is None:
            model_name = current_model_name

        platform_name = infer_platform_name(input_txt, args.platform if len(input_files) == 1 else None)
        output[platform_name] = build_output(prefill_by_size, llamacpp_by_size)

    output_path = resolve_output_path(args.output, model_config)
    output_path.write_text(json.dumps(output, indent=4) + "\n", encoding="utf-8")

    print(f"Model: {model_name}")
    print(f"Platforms: {', '.join(output.keys())}")
    for platform_name, platform_data in output.items():
        print(f"{platform_name} prompt sizes: {', '.join(sorted(platform_data.keys(), key=int))}")
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
