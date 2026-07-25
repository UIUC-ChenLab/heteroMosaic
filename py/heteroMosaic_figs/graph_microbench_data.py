#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as plticker
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "graph_microbench_data.py requires matplotlib and numpy. "
        "Install them in your Python environment, for example with: pip install matplotlib numpy"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "py" / "heteroMosaic_figs" / "results"
PLOTS_DIR = REPO_ROOT / "py" / "heteroMosaic_figs" / "plots"
ALL_PLATFORMS_FIG_DIRECTORY = PLOTS_DIR / "microbench_allplatforms"

# Default device target for plotting.
# Available values:
# DEVICE = "KRKN"
DEVICE = "STXP"
# DEVICE = "STXH"

SPLIT_DIM = "M"
# SPLIT_DIM = "K"
# SPLIT_DIM = "N"

GENERATE_ALL_PLOTS = True

INTERPOLATE_SPLIT_N = 512

DEVICE_RUN_ORDER = ("KRKN", "STXP", "STXH")
SPLIT_DIM_RUN_ORDER = ("M", "K", "N")
VALID_SPLIT_DIMS = set(SPLIT_DIM_RUN_ORDER)
VALID_DEVICES = set(DEVICE_RUN_ORDER)

DEFAULT_JSON_PATH = RESULTS_DIR / f"tensor_parallel_gemm_npu_gpu_sweep{SPLIT_DIM}_results_{DEVICE}.json"
DEFAULT_FIG_DIRECTORY = PLOTS_DIR / f"microbench_{DEVICE}_{SPLIT_DIM}"
PRINT_RAW_DATA = False
ERROR_BARS_IN_SINGLE_RUN_LATENCY_FIG = False
DEFAULT_FIGURE_WIDTH = 6.0
DEFAULT_ASPECT_RATIO = 5.0 / 3.0
FORCE_LEGEND_ON_BOTTOM_RIGHT = False

def normalize_split_dim(split_dim: str) -> str:
    selected_split_dim = split_dim.strip().upper()
    if selected_split_dim not in VALID_SPLIT_DIMS:
        available = ", ".join(sorted(VALID_SPLIT_DIMS))
        raise SystemExit(f"Unsupported split dimension '{selected_split_dim}'. Available values: {available}")
    return selected_split_dim


def split_keys(split_dim: str) -> tuple[str, str]:
    suffix = normalize_split_dim(split_dim).lower()
    return (f"gpu_{suffix}", f"npu_{suffix}")


def split_total_dim(shape: tuple[int, int, int], split_dim: str) -> int:
    dim_index = {"M": 0, "K": 1, "N": 2}[normalize_split_dim(split_dim)]
    return shape[dim_index]


def default_json_path(device_id: str, split_dim: str) -> Path:
    return RESULTS_DIR / f"tensor_parallel_gemm_npu_gpu_sweep{normalize_split_dim(split_dim)}_results_{device_id}.json"


def default_fig_dir(device_id: str, split_dim: str) -> Path:
    return PLOTS_DIR / f"microbench_{device_id}_{normalize_split_dim(split_dim)}"


def all_platforms_speedup_stem(device_id: str, split_dim: str) -> Path:
    suffix = f"{device_id.lower()}{normalize_split_dim(split_dim)}"
    return ALL_PLATFORMS_FIG_DIRECTORY / f"speedup_over_igpu_plot_{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot microbenchmark split-dimension sweep data normalized to the iGPU baseline."
    )
    parser.add_argument(
        "--device",
        default=DEVICE,
        help=f"Device id used for the default JSON and plot directory. Available: {', '.join(sorted(VALID_DEVICES))}",
    )
    parser.add_argument(
        "--split-dim",
        default=SPLIT_DIM,
        help="Sweep split dimension used for default JSON path and result fields.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Path to the microbenchmark JSON file. Defaults to the selected device's results JSON.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Directory where output figures will be written. Defaults to the selected device's plot directory.",
    )
    parser.add_argument(
        "--print-raw-data",
        action="store_true",
        help="Print normalized datapoints for each shape.",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=float,
        default=DEFAULT_ASPECT_RATIO,
        help="Figure aspect ratio as width / height.",
    )
    return parser.parse_args()


def get_device_config(device_id: str) -> dict:
    selected_device = device_id.strip().upper()
    if selected_device not in VALID_DEVICES:
        available = ", ".join(sorted(VALID_DEVICES))
        raise SystemExit(f"Unsupported device '{selected_device}'. Available values: {available}")
    return {"device": selected_device}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


def figure_size(aspect_ratio: float) -> tuple[float, float]:
    if aspect_ratio <= 0.0:
        raise SystemExit("--aspect-ratio must be greater than 0.")
    return (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_WIDTH / aspect_ratio)


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    fig.savefig(output_stem.with_suffix(".png"), dpi=200)
    fig.savefig(output_stem.with_suffix(".pdf"))


def load_results(json_path: Path) -> dict:
    if not json_path.exists():
        raise SystemExit(f"JSON file does not exist: {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict) or not isinstance(data.get("results"), dict):
        raise SystemExit(f"Expected top-level 'results' object in: {json_path}")

    return data


def gops_samples(entry: dict) -> list[float]:
    latencies = entry.get("latencies_us", [])
    if not latencies:
        return []

    m = float(entry["M"])
    k = float(entry["K"])
    n = float(entry["N"])
    ops = 2.0 * m * k * n
    return [ops / (float(latency) * 1000.0) for latency in latencies if float(latency) > 0.0]


def build_shape_entries(results: dict, split_dim: str) -> dict[tuple[int, int, int], list[dict]]:
    grouped: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    gpu_key, npu_key = split_keys(split_dim)

    for entry in results.values():
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "completed":
            continue
        if any(key not in entry for key in ("M", "K", "N", gpu_key, npu_key, "throughput_gops")):
            continue

        shape = (int(entry["M"]), int(entry["K"]), int(entry["N"]))
        grouped[shape].append(entry)

    for shape_entries in grouped.values():
        shape_entries.sort(key=lambda item: int(item[npu_key]))

    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def baseline_entry(entries: list[dict], split_dim: str) -> dict:
    _, npu_key = split_keys(split_dim)
    for entry in entries:
        if int(entry[npu_key]) == 0:
            return entry
    raise SystemExit(f"Each shape must include an {npu_key}=0 iGPU baseline entry.")


def throughput_error(gops_values: list[float], avg_gops: float) -> tuple[float, float]:
    if not gops_values:
        return (0.0, 0.0)
    return (abs(min(gops_values) - avg_gops), abs(max(gops_values) - avg_gops))


def normalized_error(gops_values: list[float], baseline_gops: float, avg_normalized: float) -> tuple[float, float]:
    if not gops_values or baseline_gops <= 0.0:
        return (0.0, 0.0)
    normalized_samples = [value / baseline_gops for value in gops_values]
    return (abs(min(normalized_samples) - avg_normalized), abs(max(normalized_samples) - avg_normalized))


def interpolate_split_n_avg_gops(npu_n: int, avg_gops: float, avg_gops_by_npu_n: dict[int, float]) -> float:
    if INTERPOLATE_SPLIT_N <= 0 or npu_n % INTERPOLATE_SPLIT_N == 0:
        return avg_gops

    lower_n = (npu_n // INTERPOLATE_SPLIT_N) * INTERPOLATE_SPLIT_N
    upper_n = lower_n + INTERPOLATE_SPLIT_N
    if lower_n not in avg_gops_by_npu_n or upper_n not in avg_gops_by_npu_n:
        return avg_gops

    mix = (npu_n - lower_n) / INTERPOLATE_SPLIT_N
    return avg_gops_by_npu_n[lower_n] + mix * (avg_gops_by_npu_n[upper_n] - avg_gops_by_npu_n[lower_n])


def warn_invalid_points(shape: tuple[int, int, int], entries: list[dict], split_dim: str) -> None:
    _, npu_key = split_keys(split_dim)
    invalid_points = sorted(int(entry[npu_key]) for entry in entries if not entry.get("allclose", True))
    if invalid_points:
        print(f"Warning: {shape[0]}x{shape[1]}x{shape[2]} has non-allclose points at {npu_key}={invalid_points}")


def plot_single_run_latencies(
    fig_dir: Path, shape: tuple[int, int, int], entries: list[dict], split_dim: str, aspect_ratio: float
) -> None:
    fig = plt.figure(figsize=figure_size(aspect_ratio))
    ax = fig.add_subplot()
    _, npu_key = split_keys(split_dim)

    for entry in entries:
        latencies = [float(value) for value in entry.get("latencies_us", [])]
        if not latencies:
            continue
        npu_split = int(entry[npu_key])
        x_values = range(len(latencies))
        label = f"{npu_key}={npu_split}"
        if ERROR_BARS_IN_SINGLE_RUN_LATENCY_FIG:
            avg_latency = float(np.mean(latencies))
            lower = [abs(min(latencies) - avg_latency)] * len(latencies)
            upper = [abs(max(latencies) - avg_latency)] * len(latencies)
            ax.errorbar(x_values, latencies, yerr=[lower, upper], marker="o", linewidth=1.2, markersize=3, label=label)
        else:
            ax.plot(x_values, latencies, marker="o", linewidth=1.2, markersize=3, label=label)

    ax.set_ylabel("Latency (us)")
    ax.set_xlabel("Run Iteration")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.xaxis.set_major_locator(plticker.MaxNLocator(integer=True))
    if FORCE_LEGEND_ON_BOTTOM_RIGHT:
        ax.legend(loc="lower right", bbox_to_anchor=(0.98, 0.02), bbox_transform=ax.transAxes)
        fig.tight_layout()
    else:
        fig.legend(loc="center left", bbox_to_anchor=(0.92, 0.5))
        fig.tight_layout(rect=(0.0, 0.0, 0.88, 1.0))
    save_figure(fig, fig_dir / f"{shape[0]}x{shape[1]}x{shape[2]}_single_run_latency")
    plt.close(fig)


def generate_plots(
    args: argparse.Namespace,
    device_id: str,
    split_dim: str,
    json_path_override: Path | None = None,
    fig_dir_override: Path | None = None,
) -> None:
    device_config = get_device_config(device_id)
    split_dim = normalize_split_dim(split_dim)
    device_id = device_config["device"]
    json_path = (json_path_override or default_json_path(device_id, split_dim)).resolve()
    fig_dir = (fig_dir_override or default_fig_dir(device_id, split_dim)).resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    ALL_PLATFORMS_FIG_DIRECTORY.mkdir(parents=True, exist_ok=True)

    data = load_results(json_path)
    grouped = build_shape_entries(data["results"], split_dim)
    if not grouped:
        raise SystemExit("No completed microbenchmark entries were found in the JSON file.")

    total_speedup_fig = plt.figure(figsize=figure_size(args.aspect_ratio))
    total_speedup_ax = total_speedup_fig.add_subplot()

    total_gops_fig = plt.figure(figsize=figure_size(args.aspect_ratio))
    total_gops_ax = total_gops_fig.add_subplot()
    gpu_key, npu_key = split_keys(split_dim)

    for shape, entries in grouped.items():
        warn_invalid_points(shape, entries, split_dim)

        baseline = baseline_entry(entries, split_dim)
        baseline_gops = float(baseline["throughput_gops"])
        if baseline_gops <= 0.0:
            raise SystemExit(f"Invalid iGPU baseline throughput for {shape[0]}x{shape[1]}x{shape[2]}")
        total_split_dim = split_total_dim(shape, split_dim)

        avg_gops_dict: dict[float, float] = {}
        min_gops_dict: dict[float, float] = {}
        max_gops_dict: dict[float, float] = {}
        avg_speedup_dict: dict[float, float] = {}
        min_speedup_dict: dict[float, float] = {}
        max_speedup_dict: dict[float, float] = {}
        avg_gops_by_npu_n = {}
        if split_dim == "N" and INTERPOLATE_SPLIT_N > 0:
            avg_gops_by_npu_n = {int(entry[npu_key]): float(entry["throughput_gops"]) for entry in entries}

        for entry in entries:
            npu_split = int(entry[npu_key])
            gpu_fraction = int(entry[gpu_key]) / total_split_dim
            raw_avg_gops = float(entry["throughput_gops"])
            avg_gops = raw_avg_gops
            if split_dim == "N":
                avg_gops = interpolate_split_n_avg_gops(npu_split, avg_gops, avg_gops_by_npu_n)
            gops_values = gops_samples(entry)
            avg_speedup = avg_gops / baseline_gops

            gops_lo, gops_hi = throughput_error(gops_values, raw_avg_gops)
            speedup_lo, speedup_hi = normalized_error(gops_values, baseline_gops, raw_avg_gops / baseline_gops)

            avg_gops_dict[gpu_fraction] = avg_gops
            min_gops_dict[gpu_fraction] = gops_lo
            max_gops_dict[gpu_fraction] = gops_hi
            avg_speedup_dict[gpu_fraction] = avg_speedup
            min_speedup_dict[gpu_fraction] = speedup_lo
            max_speedup_dict[gpu_fraction] = speedup_hi

        avg_gops_dict = dict(sorted(avg_gops_dict.items()))
        min_gops_dict = dict(sorted(min_gops_dict.items()))
        max_gops_dict = dict(sorted(max_gops_dict.items()))
        avg_speedup_dict = dict(sorted(avg_speedup_dict.items()))
        min_speedup_dict = dict(sorted(min_speedup_dict.items()))
        max_speedup_dict = dict(sorted(max_speedup_dict.items()))

        yerr_speedup = [list(min_speedup_dict.values()), list(max_speedup_dict.values())]
        yerr_gops = [list(min_gops_dict.values()), list(max_gops_dict.values())]

        label = f"{shape[0]}_{shape[1]}_{shape[2]}"
        total_speedup_ax.errorbar(avg_speedup_dict.keys(), avg_speedup_dict.values(), yerr=yerr_speedup, marker="o", label=label)
        total_gops_ax.errorbar(avg_gops_dict.keys(), avg_gops_dict.values(), yerr=yerr_gops, marker="o", label=label)

        if PRINT_RAW_DATA or args.print_raw_data:
            print(f"Matrix size {shape[0]}x{shape[1]}x{shape[2]}:")
            for fraction, speedup in avg_speedup_dict.items():
                print(f"  {fraction:.4f}, {speedup:.6f}")
            print()

        speedup_fig = plt.figure(figsize=figure_size(args.aspect_ratio))
        speedup_ax = speedup_fig.add_subplot()
        speedup_ax.errorbar(avg_speedup_dict.keys(), avg_speedup_dict.values(), yerr=yerr_speedup, marker="o")
        speedup_ax.set_ylabel("Speedup over iGPU")
        speedup_ax.set_xlabel(f"Fraction of GEMM {split_dim} on GPU")
        speedup_ax.set_xlim(0.0, 1.0)
        speedup_ax.set_xticks(np.arange(0.0, 1.05, 0.1))
        speedup_ax.grid(True, which="both", ls="--", alpha=0.5)
        speedup_fig.tight_layout()
        save_figure(speedup_fig, fig_dir / f"{shape[0]}x{shape[1]}x{shape[2]}_speedup_over_igpu")
        plt.close(speedup_fig)

        gop_fig = plt.figure(figsize=figure_size(args.aspect_ratio))
        gop_ax = gop_fig.add_subplot()
        gop_ax.errorbar(avg_gops_dict.keys(), avg_gops_dict.values(), yerr=yerr_gops, marker="o")
        gop_ax.set_ylabel("Measured GOPS")
        gop_ax.set_xlabel(f"Fraction of GEMM {split_dim} on GPU")
        gop_ax.set_xlim(0.0, 1.0)
        gop_ax.set_xticks(np.arange(0.0, 1.05, 0.1))
        gop_ax.grid(True, which="both", ls="--", alpha=0.5)
        gop_fig.tight_layout()
        save_figure(gop_fig, fig_dir / f"{shape[0]}x{shape[1]}x{shape[2]}_gops")
        plt.close(gop_fig)

        plot_single_run_latencies(fig_dir, shape, entries, split_dim, args.aspect_ratio)

    total_speedup_ax.set_ylabel("Speedup over iGPU")
    total_speedup_ax.set_xlabel(f"Fraction of GEMM {split_dim} on GPU")
    total_speedup_ax.set_xlim(0.0, 1.0)
    total_speedup_ax.set_xticks(np.arange(0.0, 1.05, 0.1))
    total_speedup_ax.grid(True, which="both", ls="--", alpha=0.5)
    if FORCE_LEGEND_ON_BOTTOM_RIGHT:
        total_speedup_ax.legend(loc="lower right", bbox_to_anchor=(0.98, 0.02), bbox_transform=total_speedup_ax.transAxes)
    else:
        total_speedup_ax.legend()
    total_speedup_fig.tight_layout()
    save_figure(total_speedup_fig, fig_dir / "speedup_over_igpu_plot")
    save_figure(total_speedup_fig, all_platforms_speedup_stem(device_id, split_dim))
    plt.close(total_speedup_fig)

    total_gops_ax.set_ylabel("Measured GOPS")
    total_gops_ax.set_xlabel(f"Fraction of GEMM {split_dim} on GPU")
    total_gops_ax.set_xlim(0.0, 1.0)
    total_gops_ax.set_xticks(np.arange(0.0, 1.05, 0.1))
    total_gops_ax.grid(True, which="both", ls="--", alpha=0.5)
    if FORCE_LEGEND_ON_BOTTOM_RIGHT:
        total_gops_ax.legend(loc="lower right", bbox_to_anchor=(0.98, 0.02), bbox_transform=total_gops_ax.transAxes)
    else:
        total_gops_ax.legend()
    total_gops_fig.tight_layout()
    save_figure(total_gops_fig, fig_dir / "gops_plot")
    plt.close(total_gops_fig)

    print(f"Wrote figures to {fig_dir}")


def main() -> None:
    args = parse_args()
    setup_style()

    if GENERATE_ALL_PLOTS:
        for device_id in DEVICE_RUN_ORDER:
            for split_dim in SPLIT_DIM_RUN_ORDER:
                generate_plots(args, device_id, split_dim)
        return

    generate_plots(args, args.device, args.split_dim, args.json, args.fig_dir)


if __name__ == "__main__":
    main()
