#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = REPO_ROOT / "utils" / "setup.sh"
BUILD_BIN_DIR = REPO_ROOT / "build" / "bin"
BUILD_BENCHMARKS_DIR = REPO_ROOT / "build" / "benchmarks"
GRAPH_SCRIPT = REPO_ROOT / "py" / "heteroMosaic_figs" / "graph_microbench_data.py"
FIGURES_ROOT = REPO_ROOT / "build" / "results" / "microbenchmark_figs"

VALID_SPLIT_DIMS = ("M", "K", "N")
DEVICE_TOKENS = {
    "KRKN": ("RYZEN AI 7 350", "RADEON 860M"),
    "STXP": ("RYZEN AI 9 HX 370", "RADEON 890M"),
    "STXH": ("RYZEN AI MAX+ 395", "RADEON 8060S"),
}


def split_dim(value: str) -> str:
    selected = value.strip().upper()
    if selected not in VALID_SPLIT_DIMS:
        raise argparse.ArgumentTypeError(
            f"unsupported split dimension '{value}'; choose M, K, or N"
        )
    return selected


def device_id(value: str) -> str:
    selected = value.strip().upper()
    if selected not in DEVICE_TOKENS:
        available = ", ".join(DEVICE_TOKENS)
        raise argparse.ArgumentTypeError(
            f"unsupported device '{value}'; choose one of: {available}"
        )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one tensor-parallel NPU/GPU GEMM sweep and generate its "
            "microbenchmark figures. The M-split sweep is used by default."
        )
    )
    parser.add_argument(
        "split_dim",
        nargs="?",
        type=split_dim,
        default=None,
        metavar="{M,K,N}",
        help="Dimension to split (default: M).",
    )
    parser.add_argument(
        "--split-dim",
        dest="split_dim_option",
        type=split_dim,
        default=None,
        metavar="{M,K,N}",
        help="Dimension to split; equivalent to the optional positional argument.",
    )
    parser.add_argument(
        "--device",
        type=device_id,
        default=None,
        metavar="{KRKN,STXP,STXH}",
        help="Device label for result filenames; defaults to local hardware detection.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Rerun points already marked completed in the result JSON.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Reuse the selected result JSON and only regenerate figures.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected paths and commands without running them.",
    )
    args = parser.parse_args()

    if args.split_dim is not None and args.split_dim_option is not None:
        if args.split_dim != args.split_dim_option:
            parser.error("the positional split dimension and --split-dim disagree")
    args.split_dim = args.split_dim_option or args.split_dim or "M"
    del args.split_dim_option

    if args.skip_benchmark and args.force_rerun:
        parser.error("--skip-benchmark and --force-rerun cannot be used together")
    return args


def detect_local_device() -> str | None:
    try:
        output = subprocess.check_output(
            ["lscpu"], text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    normalized = " ".join(output.upper().split())
    for selected_device, tokens in DEVICE_TOKENS.items():
        if all(token in normalized for token in tokens):
            return selected_device
    return None


def shell_command(argv: list[str]) -> list[str]:
    command = f"source {shlex.quote(str(SETUP_SCRIPT))} && exec {shlex.join(argv)}"
    return ["/bin/bash", "-c", command]


def display_command(argv: list[str]) -> str:
    return (
        f"(cd {shlex.quote(str(REPO_ROOT))} && "
        f"source {shlex.quote(str(SETUP_SCRIPT))} && {shlex.join(argv)})"
    )


def run_command(label: str, argv: list[str]) -> None:
    print(f"\n[{label}] {display_command(argv)}", flush=True)
    subprocess.run(shell_command(argv), cwd=REPO_ROOT, check=True)


def validate_paths(executable: Path) -> None:
    required_files = (SETUP_SCRIPT, executable, GRAPH_SCRIPT)
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise RuntimeError("Required pipeline files are missing: " + ", ".join(missing))
    if not executable.stat().st_mode & 0o111:
        raise RuntimeError(f"Microbenchmark is not executable: {executable}")


def main() -> int:
    args = parse_args()
    selected_device = args.device or detect_local_device()
    if selected_device is None:
        raise RuntimeError(
            "Could not detect the local device; pass --device KRKN, STXP, or STXH."
        )

    benchmark_name = f"tensor_parallel_gemm_npu_gpu_sweep{args.split_dim}"
    executable = BUILD_BIN_DIR / benchmark_name
    result_json = BUILD_BENCHMARKS_DIR / f"{benchmark_name}_results_{selected_device}.json"
    figure_dir = FIGURES_ROOT / f"microbench_{selected_device}_{args.split_dim}"
    validate_paths(executable)

    benchmark_argv = [
        str(executable),
        f"--results_json={result_json}",
        f"--force_rerun={int(args.force_rerun)}",
    ]
    graph_argv = [
        "python3",
        str(GRAPH_SCRIPT),
        "--device",
        selected_device,
        "--split-dim",
        args.split_dim,
        "--json",
        str(result_json),
        "--fig-dir",
        str(figure_dir),
        "--aggregate-only",
    ]

    print(
        f"Pipeline: device={selected_device}, split_dim={args.split_dim}\n"
        f"Result JSON: {result_json}\n"
        f"Figure directory: {figure_dir}"
    )

    if args.dry_run:
        if args.skip_benchmark:
            print(f"[skip] Microbenchmark; reuse {result_json}")
        else:
            print(f"[Microbenchmark] {display_command(benchmark_argv)}")
        print(f"[Generate figures] {display_command(graph_argv)}")
        return 0

    if not args.skip_benchmark:
        run_command("Microbenchmark", benchmark_argv)

    if not result_json.is_file():
        raise RuntimeError(f"Microbenchmark result JSON does not exist: {result_json}")

    run_command("Generate figures", graph_argv)
    print(f"\nCompleted. Figures are in: {figure_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"Error: command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode or 1)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
