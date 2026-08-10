#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = REPO_ROOT / "utils" / "setup.sh"
HETEROMOSAIC_DIR = REPO_ROOT / "py" / "unified_llm_w4a16"
HETEROMOSAIC_BENCHMARK = HETEROMOSAIC_DIR / "_benchmark_heteroMosaic.py"
LLAMACPP_DIR = REPO_ROOT / "utils" / "llamacpp_scripts"
LLAMACPP_BENCHMARK = LLAMACPP_DIR / "benchmark.py"
POPULATE_SCRIPT = REPO_ROOT / "py" / "heteroMosaic_figs" / "populate_data.py"
GRAPH_SCRIPT = REPO_ROOT / "py" / "heteroMosaic_figs" / "graph_e2e_data.py"
BUILD_RESULTS_DIR = REPO_ROOT / "build" / "results"
FIGURES_DIR = BUILD_RESULTS_DIR / "fig"

MACHINE_CONFIGS = {
    "350": {
        "platform": "Ryzen AI 7 350",
        "tokens": ("RYZEN AI 7 350", "RADEON 860M"),
    },
    "370": {
        "platform": "Ryzen AI 9 HX 370",
        "tokens": ("RYZEN AI 9 HX 370", "RADEON 890M"),
    },
    "395": {
        "platform": "Ryzen AI MAX+ 395",
        "tokens": ("RYZEN AI MAX+ 395", "RADEON 8060S"),
    },
}

MODEL_CONFIGS = {
    "llama3_8b": {
        "heteromosaic_model": "llama3_8b",
        "llamacpp_model": "llama3-8b_q4_k_s",
        "result_filename": "llama3-8b_results.json",
        "prompt_sizes": (1024, 2048, 4096, 8192, 16384),
    },
    "gemma": {
        "heteromosaic_model": "gemma",
        "llamacpp_model": "gemma1-2b_q4_k_s",
        "result_filename": "gemma_results.json",
        "prompt_sizes": (1024, 2048, 4096, 8192),
    },
    "llama3_70b": {
        "heteromosaic_model": "llama3_70b",
        "llamacpp_model": "llama3-70b_q4_k_s",
        "result_filename": "llama3-70b_results.json",
        "prompt_sizes": (1024, 2048, 4096, 8192, 16384),
    },
    "qwen14b": {
        "heteromosaic_model": "qwen14b",
        "llamacpp_model": "qwen2.5-14b_q4_k_s",
        "result_filename": "qwen14b_results.json",
        "prompt_sizes": (1024, 2048, 4096, 8192, 16384),
    },
    "phi35_3.8b": {
        "heteromosaic_model": "phi3.5_3.8b",
        "llamacpp_model": "phi3.5-3.8b_q4_k_s",
        "result_filename": "phi35_3.8b_results.json",
        "prompt_sizes": (1024, 2048, 4096, 8192, 16384),
    },
}

DEFAULT_MODELS = ("llama3_8b", "phi35_3.8b", "qwen14b")
MODEL_ALIASES = {
    "llama": "llama3_8b",
    "llama3": "llama3_8b",
    "llama3-8b": "llama3_8b",
    "phi": "phi35_3.8b",
    "phi3.5": "phi35_3.8b",
    "phi3.5-3.8b": "phi35_3.8b",
    "phi3.5_3.8b": "phi35_3.8b",
    "qwen": "qwen14b",
    "qwen14": "qwen14b",
    "qwen2.5-14b": "qwen14b",
    "qwen25_14b": "qwen14b",
}


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("prompt sizes must be positive integers")
    return parsed


def model_id(value: str) -> str:
    requested = value.strip().lower()
    selected = MODEL_ALIASES.get(requested, requested)
    if selected not in MODEL_CONFIGS:
        available = ", ".join(sorted(MODEL_CONFIGS))
        raise argparse.ArgumentTypeError(
            f"unsupported model '{value}'; available model ids: {available}; "
            "short aliases: llama, phi, qwen"
        )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run cached HeteroMosaic and llama.cpp benchmarks, populate machine-specific "
            "result JSON, and generate end-to-end figures."
        )
    )
    parser.add_argument(
        "--models",
        "--model",
        dest="models",
        nargs="+",
        type=model_id,
        default=list(DEFAULT_MODELS),
        metavar="MODEL",
        help=(
            f"Models to process (default: {' '.join(DEFAULT_MODELS)}). "
            "Short aliases: llama, phi, qwen."
        ),
    )
    parser.add_argument(
        "--machines",
        "--machine",
        dest="machines",
        nargs="+",
        choices=sorted(MACHINE_CONFIGS),
        default=None,
        help="Machine ids to process; defaults to local hardware detection.",
    )
    parser.add_argument(
        "--prompt-sizes",
        nargs="+",
        type=positive_int,
        default=None,
        metavar="TOKENS",
        help="Override the configured prompt sizes for every selected model.",
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Reuse previously captured stdout logs and only populate/plot results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and output paths without running subprocesses or writing files.",
    )
    return parser.parse_args()


def detect_local_machine() -> str | None:
    try:
        output = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError):
        return None
    normalized = " ".join(output.upper().split())
    for machine_id, config in MACHINE_CONFIGS.items():
        if all(token in normalized for token in config["tokens"]):
            return machine_id
    return None


def shell_command(argv: list[str]) -> list[str]:
    command = f"source {shlex.quote(str(SETUP_SCRIPT))} && exec {shlex.join(argv)}"
    return ["/bin/bash", "-c", command]


def display_command(argv: list[str], cwd: Path) -> str:
    return f"(cd {shlex.quote(str(cwd))} && source {shlex.quote(str(SETUP_SCRIPT))} && {shlex.join(argv)})"


def run_and_capture(label: str, argv: list[str], cwd: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print(f"\n[{label}] {display_command(argv, cwd)}", flush=True)
    with temp_output.open("w", encoding="utf-8") as output_file:
        process = subprocess.Popen(
            shell_command(argv),
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            output_file.write(line)
            output_file.flush()
        returncode = process.wait()

    if returncode == 0:
        os.replace(temp_output, output_path)
        print(f"[{label}] Captured stdout: {output_path}")
        return

    failed_output = output_path.with_name(
        f"{output_path.stem}.failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}{output_path.suffix}"
    )
    os.replace(temp_output, failed_output)
    raise RuntimeError(
        f"{label} failed with exit code {returncode}. Partial stdout: {failed_output}"
    )


def require_cached_inputs(paths: tuple[Path, Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            "--skip-benchmarks requires existing benchmark snapshots or captured logs. Missing: "
            + ", ".join(missing)
        )


def require_benchmark_state(label: str, state_path: Path) -> None:
    if not state_path.is_file():
        raise RuntimeError(
            f"{label} completed without writing its model-specific state: {state_path}"
        )


def pipeline_paths(machine_id: str, model_id: str, model_config: dict) -> dict[str, Path]:
    machine_dir = BUILD_RESULTS_DIR / machine_id
    log_dir = machine_dir / "logs" / model_id
    return {
        "heteromosaic_stdout": log_dir / "heteromosaic.stdout.txt",
        "llamacpp_stdout": log_dir / "llamacpp.stdout.txt",
        "heteromosaic_state": log_dir / "heteromosaic.benchmark.json",
        "llamacpp_state": log_dir / "llamacpp.benchmark.json",
        "heteromosaic_input": log_dir / "heteromosaic.benchmark.json",
        "llamacpp_input": log_dir / "llamacpp.benchmark.json",
        "populate_stdout": log_dir / "populate_data.stdout.txt",
        "graph_stdout": log_dir / "graph_e2e_data.stdout.txt",
        "result_json": machine_dir / model_config["result_filename"],
        "figure_dir": FIGURES_DIR / machine_id,
    }


def build_pipeline_commands(
    machine_id: str,
    model_id: str,
    model_config: dict,
    prompt_sizes: list[int],
    paths: dict[str, Path],
) -> list[tuple[str, list[str], Path, Path]]:
    size_args = [str(size) for size in prompt_sizes]
    return [
        (
            "HeteroMosaic benchmark",
            [
                "python3",
                HETEROMOSAIC_BENCHMARK.name,
                "--model",
                model_config["heteromosaic_model"],
                "--prompt-sizes",
                *size_args,
                "--state-file",
                str(paths["heteromosaic_state"]),
            ],
            HETEROMOSAIC_DIR,
            paths["heteromosaic_stdout"],
        ),
        (
            "llama.cpp benchmark",
            [
                "python3",
                LLAMACPP_BENCHMARK.name,
                "--models",
                model_config["llamacpp_model"],
                "--prompt-sizes",
                *size_args,
                "--state-file",
                str(paths["llamacpp_state"]),
            ],
            LLAMACPP_DIR,
            paths["llamacpp_stdout"],
        ),
        (
            "Populate data",
            [
                "python3",
                str(POPULATE_SCRIPT),
                "--model",
                model_id,
                "--machine",
                machine_id,
                "--heteromosaic-input",
                str(paths["heteromosaic_input"]),
                "--llamacpp-input",
                str(paths["llamacpp_input"]),
                "--output",
                str(paths["result_json"]),
            ],
            REPO_ROOT,
            paths["populate_stdout"],
        ),
        (
            "Generate figures",
            [
                "python3",
                str(GRAPH_SCRIPT),
                "--model",
                model_id,
                "--machine",
                machine_id,
                "--output-dir",
                str(paths["figure_dir"]),
                str(paths["result_json"]),
            ],
            REPO_ROOT,
            paths["graph_stdout"],
        ),
    ]


def validate_repository_paths() -> None:
    required = [
        SETUP_SCRIPT,
        HETEROMOSAIC_BENCHMARK,
        LLAMACPP_BENCHMARK,
        POPULATE_SCRIPT,
        GRAPH_SCRIPT,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Required pipeline files are missing: " + ", ".join(missing))


def main() -> int:
    args = parse_args()
    validate_repository_paths()
    detected_machine = detect_local_machine()
    machines = args.machines or ([detected_machine] if detected_machine else [])
    if not machines:
        raise RuntimeError("Could not detect this machine; pass --machines 350, 370, or 395.")

    if not args.skip_benchmarks and not args.dry_run:
        nonlocal_machines = [machine for machine in machines if machine != detected_machine]
        if nonlocal_machines:
            raise RuntimeError(
                "Hardware benchmarks can only run for the locally detected machine "
                f"({detected_machine or 'unknown'}). Use --skip-benchmarks for captured data from: "
                + ", ".join(nonlocal_machines)
            )

    for machine_id in machines:
        for model_id in args.models:
            model_config = MODEL_CONFIGS[model_id]
            prompt_sizes = list(args.prompt_sizes or model_config["prompt_sizes"])
            paths = pipeline_paths(machine_id, model_id, model_config)

            if args.skip_benchmarks and not args.dry_run:
                state_snapshots = (paths["heteromosaic_state"], paths["llamacpp_state"])
                captured_logs = (paths["heteromosaic_stdout"], paths["llamacpp_stdout"])
                if all(path.is_file() for path in state_snapshots):
                    paths["heteromosaic_input"], paths["llamacpp_input"] = state_snapshots
                elif all(path.is_file() for path in captured_logs):
                    # Backward compatibility for runs captured before state snapshots were added.
                    paths["heteromosaic_input"], paths["llamacpp_input"] = captured_logs
                else:
                    require_cached_inputs(state_snapshots)

            commands = build_pipeline_commands(machine_id, model_id, model_config, prompt_sizes, paths)

            print(
                f"\nPipeline: machine={machine_id} ({MACHINE_CONFIGS[machine_id]['platform']}), "
                f"model={model_id}, prompt_sizes={prompt_sizes}"
            )
            print(f"Result JSON: {paths['result_json']}")
            print(f"Figure directory: {paths['figure_dir']}")

            if args.dry_run:
                for label, argv, cwd, output_path in commands:
                    if args.skip_benchmarks and label in {"HeteroMosaic benchmark", "llama.cpp benchmark"}:
                        print(f"[skip] {label}; reuse {output_path}")
                    else:
                        print(f"[{label}] {display_command(argv, cwd)}")
                        print(f"  stdout -> {output_path}")
                continue

            benchmark_commands = commands[:2]
            processing_commands = commands[2:]
            if not args.skip_benchmarks:
                for label, argv, cwd, output_path in benchmark_commands:
                    run_and_capture(label, argv, cwd, output_path)
                    state_path = (
                        paths["heteromosaic_state"]
                        if label == "HeteroMosaic benchmark"
                        else paths["llamacpp_state"]
                    )
                    require_benchmark_state(label, state_path)

            for label, argv, cwd, output_path in processing_commands:
                run_and_capture(label, argv, cwd, output_path)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
