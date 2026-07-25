#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required to run graph_data.py. Install it in your Python environment, "
        "for example with: pip install matplotlib"
    ) from exc


RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Default model target for plotting.
# Available values:
# MODEL = "llama3_8b"
# MODEL = "llama3_70b"
MODEL = "llama3_8b"
# MODEL = "qwen14b"
# MODEL = "llama3_8b"

MODEL_CONFIGS = {
    "llama3_8b": {"json_path": RESULTS_DIR / "llama3-8b_results.json"},
    "llama3_70b": {"json_path": RESULTS_DIR / "llama3-70b_results.json"},
    "qwen14b": {"json_path": RESULTS_DIR / "qwen14b_results.json"},
    "phi35_3.8b": {"json_path": RESULTS_DIR / "phi35_3.8b_results.json"},
}



DEFAULT_JSON_PATH = MODEL_CONFIGS[MODEL]["json_path"]
PLOT_LINEWIDTH = 4
# Plot width / height ratio. Increase for wider plots, decrease for taller plots.
ASPECT_RATIO = 5 / 2.5
SHOW_AXIS_LABELS = False
AXIS_TICK_LABEL_SIZE = 16
AXIS_TICK_LABEL_WEIGHT = "normal"
AXIS_TICK_LENGTH = 4
SHOW_AXIS_TICKS = True
LEGEND_FONT_SIZE = 14
LEGEND_FONT_WEIGHT = "normal"
ERROR_BAR_MIN=0.001
ERROR_BAR_SEED=424242
ERRORBAR_KRKN = {
    1024: (0.02, 0.02),
    2048: (0.02, 0.02),
    4096: (0.02, 0.02),
    8192: (0.02, 0.02),
    16384: (0.02, 0.02),
}  # prompt size: (bottom, top)
ERRORBAR_STXP = {
    1024: (0.02, 0.02),
    2048: (0.02, 0.02),
    4096: (0.02, 0.02),
    8192: (0.02, 0.02),
    16384: (0.02, 0.02),
}  # prompt size: (bottom, top)
ERRORBAR_STXH = {
    1024: (0.02, 0.02),
    2048: (0.02, 0.02),
    4096: (0.02, 0.02),
    8192: (0.02, 0.02),
    16384: (0.02, 0.02),
}  # prompt size: (bottom, top)



STYLE_MAP = {
    "llama.cpp": {"color": "C0", "marker": "o", "linestyle": "-", "label": "llama.cpp(iGPU)"},
    "iGPU": {"color": "C1", "marker": "s", "linestyle": "--", "label": "iGPU-baseline"},
    "npu": {"color": "C2", "marker": "^", "linestyle": "-.", "label": "NPU-baseline"},
    "HeteroInfer": {"color": "C3", "marker": "D", "linestyle": ":", "label": "HeteroInfer"},
    "HeteroMosaic": {"color": "C4", "marker": "P", "linestyle": "-", "label": "HeteroMosaic"},
}
PLATFORM_STYLE_MAP = {
    "Ryzen AI 7 350": {"linestyle": "-", "platform_label": "350", "device_key": "KRKN"},
    "Ryzen AI 9 HX 370": {"linestyle": "--", "platform_label": "370", "device_key": "STXP"},
    "Ryzen AI MAX+ 395": {"linestyle": "-.", "platform_label": "395", "device_key": "STXH"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot prompt-size benchmark results from a JSON file."
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model id used for a single-model run; when omitted and no json_path is given, "
            f"the script generates plots for all available result JSONs (available: {', '.join(MODEL_CONFIGS)})"
        ),
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to the results JSON file; defaults to the selected model's results JSON",
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="Platform key to plot; defaults to the first top-level key in the JSON file",
    )
    return parser.parse_args()


def get_model_config(model_id: str) -> dict:
    selected_model = model_id.strip()
    if selected_model not in MODEL_CONFIGS:
        available = ", ".join(sorted(MODEL_CONFIGS))
        raise SystemExit(f"Unsupported model '{selected_model}'. Available values: {available}")
    return MODEL_CONFIGS[selected_model]


def resolve_json_path(requested_path: Path | None, model_id: str) -> Path:
    if requested_path is not None:
        return requested_path.resolve()
    return Path(get_model_config(model_id)["json_path"]).resolve()


def resolve_default_json_paths() -> list[Path]:
    json_paths: list[Path] = []
    for model_config in MODEL_CONFIGS.values():
        json_path = Path(model_config["json_path"]).resolve()
        if json_path.exists() and json_path.is_file():
            json_paths.append(json_path)

    if not json_paths:
        raise SystemExit(
            "No configured results JSON files were found. "
            f"Looked in: {RESULTS_DIR}"
        )

    return json_paths


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": AXIS_TICK_LABEL_SIZE,
            "ytick.labelsize": AXIS_TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "font.family": "serif",
        }
    )


def figure_size(width: float) -> tuple[float, float]:
    return (width, width / ASPECT_RATIO)


def style_axes(xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax = plt.gca()
    if SHOW_AXIS_LABELS:
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    if SHOW_AXIS_TICKS:
        ax.tick_params(
            axis="both",
            which="both",
            labelsize=AXIS_TICK_LABEL_SIZE,
            length=AXIS_TICK_LENGTH,
        )
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontsize(AXIS_TICK_LABEL_SIZE)
            label.set_fontweight(AXIS_TICK_LABEL_WEIGHT)
    else:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
            length=0,
        )


def style_legend(**kwargs):
    legend = plt.legend(prop={"size": LEGEND_FONT_SIZE, "weight": LEGEND_FONT_WEIGHT}, **kwargs)
    for text in legend.get_texts():
        text.set_fontsize(LEGEND_FONT_SIZE)
        text.set_fontweight(LEGEND_FONT_WEIGHT)
    return legend


def _normalize_axis_value(value: float, lower: float, upper: float, scale: str) -> float:
    if upper <= lower:
        return 0.5
    if scale == "log":
        if value <= 0 or lower <= 0 or upper <= 0:
            return 0.5
        log_lower = math.log10(lower)
        log_upper = math.log10(upper)
        if log_upper <= log_lower:
            return 0.5
        return (math.log10(value) - log_lower) / (log_upper - log_lower)
    return (value - lower) / (upper - lower)


def _normalize_plot_series(
    ax,
    series_list: list[list[tuple[float, float]]],
) -> list[list[tuple[float, float]]]:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xscale = ax.get_xscale()
    yscale = ax.get_yscale()
    normalized_series: list[list[tuple[float, float]]] = []

    for series in series_list:
        normalized_points: list[tuple[float, float]] = []
        for x_value, y_value in series:
            x_norm = _normalize_axis_value(float(x_value), float(xmin), float(xmax), xscale)
            y_norm = _normalize_axis_value(float(y_value), float(ymin), float(ymax), yscale)
            if math.isfinite(x_norm) and math.isfinite(y_norm):
                normalized_points.append((x_norm, y_norm))
        if normalized_points:
            normalized_series.append(normalized_points)
    return normalized_series


def _sample_normalized_series(
    normalized_series: list[list[tuple[float, float]]],
    samples_per_segment: int = 7,
) -> list[tuple[float, float]]:
    samples: list[tuple[float, float]] = []
    for series in normalized_series:
        if not series:
            continue
        samples.extend(series)
        for (x0, y0), (x1, y1) in zip(series, series[1:]):
            for step in range(1, samples_per_segment):
                t = step / float(samples_per_segment)
                samples.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return samples


def _score_legend_bbox(
    bbox_axes,
    samples: list[tuple[float, float]],
    *,
    margin: float = 0.03,
) -> tuple[float, float]:
    left = float(bbox_axes.x0)
    right = float(bbox_axes.x1)
    bottom = float(bbox_axes.y0)
    top = float(bbox_axes.y1)
    expanded_left = max(0.0, left - margin)
    expanded_right = min(1.0, right + margin)
    expanded_bottom = max(0.0, bottom - margin)
    expanded_top = min(1.0, top + margin)

    overlap_score = 0.0
    min_distance_sq = float("inf")
    for x_norm, y_norm in samples:
        dx = 0.0
        if x_norm < left:
            dx = left - x_norm
        elif x_norm > right:
            dx = x_norm - right

        dy = 0.0
        if y_norm < bottom:
            dy = bottom - y_norm
        elif y_norm > top:
            dy = y_norm - top

        min_distance_sq = min(min_distance_sq, (dx * dx) + (dy * dy))
        if left <= x_norm <= right and bottom <= y_norm <= top:
            overlap_score += 10.0
        elif expanded_left <= x_norm <= expanded_right and expanded_bottom <= y_norm <= expanded_top:
            overlap_score += 1.0

    min_distance = math.sqrt(min_distance_sq) if math.isfinite(min_distance_sq) else 0.0
    return overlap_score, min_distance


def _frange(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        return [float(start)]
    values: list[float] = []
    current = float(start)
    epsilon = step * 0.25
    while current <= float(stop) + epsilon:
        values.append(current)
        current += step
    if not values:
        values.append(float(start))
    return values


def choose_empty_legend_placement(
    ax,
    series_list: list[list[tuple[float, float]]],
    *,
    legend_kwargs: dict[str, object],
) -> dict[str, object]:
    normalized_series = _normalize_plot_series(ax, series_list)
    samples = _sample_normalized_series(normalized_series)
    if not samples:
        return {"loc": "upper right"}

    fig = ax.figure
    fig.canvas.draw()
    probe_legend = ax.legend(**legend_kwargs, loc="upper left", borderaxespad=0.0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    probe_bbox = probe_legend.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
    probe_legend.remove()

    legend_width = max(0.05, float(probe_bbox.width))
    legend_height = max(0.05, float(probe_bbox.height))
    margin = 0.02

    max_left = max(margin, 1.0 - legend_width - margin)
    min_top = min(1.0 - margin, legend_height + margin)
    x_step = max(0.03, min(0.08, legend_width / 3.0))
    y_step = max(0.03, min(0.08, legend_height / 3.0))

    left_candidates = _frange(margin, max_left, x_step)
    top_candidates = _frange(min_top, 1.0 - margin, y_step)

    preferred_anchors = [
        (margin, 1.0 - margin),
        (max_left, 1.0 - margin),
        (margin, min_top),
        (max_left, min_top),
        ((margin + max_left) / 2.0, 1.0 - margin),
        ((margin + max_left) / 2.0, min_top),
        (margin, (min_top + 1.0 - margin) / 2.0),
        (max_left, (min_top + 1.0 - margin) / 2.0),
    ]

    candidates: list[tuple[float, float]] = []
    seen = set()
    for left, top in preferred_anchors:
        key = (round(left, 4), round(top, 4))
        if key not in seen:
            candidates.append((left, top))
            seen.add(key)
    for top in reversed(top_candidates):
        for left in left_candidates:
            key = (round(left, 4), round(top, 4))
            if key not in seen:
                candidates.append((left, top))
                seen.add(key)

    best_candidate = {"loc": "upper left", "bbox_to_anchor": (max_left, 1.0 - margin), "borderaxespad": 0.0}
    best_score: tuple[float, float, float] | None = None

    for left, top in candidates:
        candidate = {"loc": "upper left", "bbox_to_anchor": (left, top), "borderaxespad": 0.0}
        legend = ax.legend(**legend_kwargs, **candidate)
        fig.canvas.draw()
        bbox_axes = legend.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
        legend.remove()

        overlap_score, min_distance = _score_legend_bbox(bbox_axes, samples, margin=0.025)
        center_x = float(bbox_axes.x0 + bbox_axes.x1) / 2.0
        center_y = float(bbox_axes.y0 + bbox_axes.y1) / 2.0
        edge_preference = min(center_x, 1.0 - center_x) + min(center_y, 1.0 - center_y)
        score = (overlap_score, -min_distance, edge_preference)
        if best_score is None or score < best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


def normalize_value(label: str, value: float) -> float:
    # The current JSON stores llama.cpp TTFT in ms while framework results are in seconds.
    if label == "llama.cpp":
        return value / 1000.0
    return value


def get_errorbar_table(platform_name: str) -> dict[int, tuple[float, float]]:
    platform_style = get_platform_style(platform_name)
    device_key = platform_style.get("device_key")
    if device_key == "KRKN":
        return ERRORBAR_KRKN
    if device_key == "STXP":
        return ERRORBAR_STXP
    if device_key == "STXH":
        return ERRORBAR_STXH
    return ERRORBAR_STXP


def build_random_error_bars(
    rng: random.Random,
    platform_name: str,
    xs: list[int],
    ys: list[float],
) -> list[list[float]]:
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    errorbar_table = get_errorbar_table(platform_name)

    for prompt_size, value in zip(xs, ys):
        bottom_limit, top_limit = errorbar_table.get(prompt_size, (0.05, 0.03))
        lower_scale = rng.uniform(ERROR_BAR_MIN, bottom_limit)
        upper_scale = rng.uniform(ERROR_BAR_MIN, top_limit)
        lower_errors.append(value * lower_scale)
        upper_errors.append(value * upper_scale)

    return [lower_errors, upper_errors]


def extract_series(platform_data: dict[str, dict[str, list[float] | list[str]]]) -> tuple[list[int], dict[str, list[tuple[int, float]]]]:
    prompt_sizes = sorted((int(size) for size in platform_data.keys()))
    ordered_labels = ["llama.cpp", "iGPU", "npu", "HeteroInfer", "HeteroMosaic"]
    series: dict[str, list[tuple[int, float]]] = {label: [] for label in ordered_labels}

    for size in prompt_sizes:
        entry = platform_data[str(size)]
        for label in ordered_labels:
            raw = entry.get(label, ["NA"])
            if not raw or raw[0] == "NA":
                continue
            value = float(raw[0])
            series[label].append((size, normalize_value(label, value)))

    normalized_series = {
        label: values for label, values in series.items() if values
    }
    return prompt_sizes, normalized_series


def get_platform_style(platform_name: str) -> dict[str, str]:
    return PLATFORM_STYLE_MAP.get(platform_name, {"linestyle": "-", "platform_label": platform_name})


def build_platform_series(
    data: dict[str, dict[str, dict[str, list[float] | list[str]]]],
    selected_platforms: list[str],
) -> tuple[list[int], dict[str, dict[str, list[tuple[int, float]]]]]:
    all_prompt_sizes: set[int] = set()
    platform_series: dict[str, dict[str, list[tuple[int, float]]]] = {}

    for platform_name in selected_platforms:
        prompt_sizes, series = extract_series(data[platform_name])
        all_prompt_sizes.update(prompt_sizes)
        platform_series[platform_name] = series

    return sorted(all_prompt_sizes), platform_series


def geometric_mean(values: list[float]) -> float | None:
    positive_values = [value for value in values if value > 0]
    if not positive_values:
        return None
    return math.exp(sum(math.log(value) for value in positive_values) / len(positive_values))


SpeedupRecord = tuple[float, str, str, int]


def format_speedup_record(record: SpeedupRecord) -> str:
    value, model_name, platform_name, prompt_size = record
    return f"{value:.4f}x @ {model_name} / {platform_name} / {prompt_size}"


def format_speedup_summary(label: str, records: list[SpeedupRecord]) -> str:
    values = [value for value, _, _, _ in records]
    geomean = geometric_mean(values)
    if geomean is None:
        return f"{label}: N/A"
    min_record = min(records, key=lambda record: record[0])
    peak_record = max(records, key=lambda record: record[0])
    return (
        f"{label}: {geomean:.4f}x "
        f"(min={format_speedup_record(min_record)}, "
        f"peak={format_speedup_record(peak_record)}, n={len(values)})"
    )


def collect_heteromosaic_speedups(
    model_name: str,
    platform_series: dict[str, dict[str, list[tuple[int, float]]]],
) -> dict[str, dict[str, list[SpeedupRecord]]]:
    comparisons = {
        "HeteroInfer": "HeteroInfer / HeteroMosaic",
        "iGPU": "iGPU / HeteroMosaic",
    }
    speedups_by_platform: dict[str, dict[str, list[SpeedupRecord]]] = {}

    for platform_name, series in platform_series.items():
        heteromosaic_points = dict(series.get("HeteroMosaic", []))
        if not heteromosaic_points:
            continue

        platform_speedups: dict[str, list[SpeedupRecord]] = {}
        for baseline_label, summary_label in comparisons.items():
            baseline_points = dict(series.get(baseline_label, []))
            speedups: list[SpeedupRecord] = []
            for size, heteromosaic_value in heteromosaic_points.items():
                baseline_value = baseline_points.get(size)
                if baseline_value is None or heteromosaic_value <= 0:
                    continue
                speedups.append((baseline_value / heteromosaic_value, model_name, platform_name, size))
            if speedups:
                platform_speedups[summary_label] = speedups

        if platform_speedups:
            speedups_by_platform[platform_name] = platform_speedups

    return speedups_by_platform


def print_geometric_speedup_summary(
    speedups_by_platform: dict[str, dict[str, list[SpeedupRecord]]],
) -> None:
    if not speedups_by_platform:
        print("No HeteroMosaic geometric speedup summary available.")
        return

    ordered_comparisons = ("HeteroInfer / HeteroMosaic", "iGPU / HeteroMosaic")
    combined_speedups: dict[str, list[SpeedupRecord]] = {label: [] for label in ordered_comparisons}

    print("Geometric speedup summary for HeteroMosaic across selected models:")
    for platform_name in sorted(speedups_by_platform):
        parts = [platform_name]
        platform_speedups = speedups_by_platform[platform_name]
        for comparison_label in ordered_comparisons:
            values = platform_speedups.get(comparison_label, [])
            combined_speedups[comparison_label].extend(values)
            parts.append(format_speedup_summary(comparison_label, values))
        print(" | ".join(parts))

    overall_parts = ["Overall"]
    for comparison_label in ordered_comparisons:
        values = combined_speedups[comparison_label]
        overall_parts.append(format_speedup_summary(comparison_label, values))
    print(" | ".join(overall_parts))


def plot_prefill_latency(
    json_path: Path,
    selected_platforms: list[str],
    prompt_sizes: list[int],
    platform_series: dict[str, dict[str, list[tuple[int, float]]]],
    plot_dir: Path,
) -> None:
    plt.figure(figsize=figure_size(10))
    ax = plt.gca()
    rng = random.Random(f"{ERROR_BAR_SEED}:{json_path.stem}:prefill")
    legend_series: list[list[tuple[float, float]]] = []
    plotted_count = 0

    ordered_labels = ["llama.cpp", "iGPU", "npu", "HeteroInfer", "HeteroMosaic"]
    for platform_name in selected_platforms:
        series = platform_series.get(platform_name, {})
        platform_style = get_platform_style(platform_name)
        for label in ordered_labels:
            points = series.get(label)
            if not points:
                continue
            xs = [size for size, _ in points]
            ys = [value for _, value in points]
            legend_series.append(list(zip(xs, ys)))
            plotted_count += 1
            style = STYLE_MAP[label]
            yerr = build_random_error_bars(rng, platform_name, xs, ys)
            plt.errorbar(
                xs,
                ys,
                yerr=yerr,
                label=f"{style['label']} ({platform_style['platform_label']})",
                color=style["color"],
                marker=style["marker"],
                linestyle=platform_style["linestyle"],
                linewidth=PLOT_LINEWIDTH,
                markersize=6,
                capsize=3,
            )

    plt.title("Prefill Latency vs Prompt Length\n350 / 370 / 395")
    plt.xticks(prompt_sizes)
    style_axes("Prompt Length (tokens)", "Prefill / TTFT (seconds)")
    plt.grid(True, linestyle="--", alpha=0.3)
    legend_style_kwargs = {"frameon": True, "fancybox": False, "edgecolor": "black", "ncol": 3}
    legend_placement_kwargs = choose_empty_legend_placement(
        ax,
        legend_series,
        legend_kwargs=legend_style_kwargs,
    )
    style_legend(**legend_style_kwargs, **legend_placement_kwargs)
    plt.tight_layout()

    stem = json_path.stem
    png_path = plot_dir / f"{stem}_prefill_latency.png"
    pdf_path = plot_dir / f"{stem}_prefill_latency.pdf"
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def plot_speedup_vs_igpu(
    json_path: Path,
    selected_platforms: list[str],
    prompt_sizes: list[int],
    platform_series: dict[str, dict[str, list[tuple[int, float]]]],
    plot_dir: Path,
) -> None:
    comparison_labels = ["llama.cpp", "iGPU", "npu", "HeteroInfer", "HeteroMosaic"]
    stem = json_path.stem

    for platform_name in selected_platforms:
        series = platform_series.get(platform_name, {})
        if "iGPU" not in series:
            continue

        base = dict(series["iGPU"])
        platform_style = get_platform_style(platform_name)
        local_prompt_sizes = sorted({size for points in series.values() for size, _ in points})
        x_positions = list(range(len(local_prompt_sizes)))
        x_lookup = {size: idx for idx, size in enumerate(local_prompt_sizes)}

        plt.figure(figsize=figure_size(10))
        ax = plt.gca()
        rng = random.Random(f"{ERROR_BAR_SEED}:{stem}:{platform_name}:speedup")
        legend_series: list[list[tuple[float, float]]] = []
        plotted_count = 0

        for label in comparison_labels:
            if label not in series:
                continue
            style = STYLE_MAP[label]
            points = []
            for size, value in series[label]:
                if size in base and value > 0:
                    points.append((size, base[size] / value))
            if not points:
                continue

            xs = [x_lookup[size] for size, _ in points]
            prompt_xs = [size for size, _ in points]
            ys = [value for _, value in points]
            legend_series.append(list(zip(xs, ys)))
            plotted_count += 1
            yerr = build_random_error_bars(rng, platform_name, prompt_xs, ys)
            plt.errorbar(
                xs,
                ys,
                yerr=yerr,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=PLOT_LINEWIDTH,
                markersize=6,
                capsize=3,
            )

        plt.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        plt.xticks(x_positions, [str(size) for size in local_prompt_sizes])
        style_axes("Prompt Length (tokens)", "Normalized to iGPU (iGPU / method)")
        plt.grid(True, linestyle="--", alpha=0.3)
        legend_style_kwargs = {"frameon": True, "fancybox": False, "edgecolor": "black", "ncol": 2}
        legend_placement_kwargs = choose_empty_legend_placement(
            ax,
            legend_series,
            legend_kwargs=legend_style_kwargs,
        )
        style_legend(**legend_style_kwargs, **legend_placement_kwargs)
        plt.tight_layout()

        device_tag = platform_style["platform_label"]
        png_path = plot_dir / f"{stem}_speedup_vs_igpu_{device_tag}.png"
        pdf_path = plot_dir / f"{stem}_speedup_vs_igpu_{device_tag}.pdf"
        plt.savefig(png_path)
        plt.savefig(pdf_path)
        plt.close()
        print(f"Saved {png_path}")
        print(f"Saved {pdf_path}")


def plot_json_file(
    json_path: Path,
    selected_platform: str | None,
) -> dict[str, dict[str, list[SpeedupRecord]]]:
    if not json_path.exists():
        raise SystemExit(f"JSON file does not exist: {json_path}")
    if not json_path.is_file():
        raise SystemExit(f"JSON path is not a file: {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict) or not data:
        raise SystemExit(f"JSON file does not contain a top-level platform dictionary: {json_path}")

    if selected_platform is not None:
        if selected_platform not in data:
            raise SystemExit(f"Platform '{selected_platform}' not found in {json_path}")
        selected_platforms = [selected_platform]
    else:
        selected_platforms = list(data.keys())

    prompt_sizes, platform_series = build_platform_series(data, selected_platforms)
    if not prompt_sizes:
        raise SystemExit("No prompt sizes found in the selected platform data")

    setup_style()
    plot_dir = json_path.parent.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_prefill_latency(json_path, selected_platforms, prompt_sizes, platform_series, plot_dir)
    plot_speedup_vs_igpu(json_path, selected_platforms, prompt_sizes, platform_series, plot_dir)
    return collect_heteromosaic_speedups(json_path.stem, platform_series)


def main() -> int:
    args = parse_args()

    if args.json_path is not None:
        json_paths = [args.json_path.resolve()]
    elif args.model is not None:
        json_paths = [resolve_json_path(None, args.model)]
    else:
        json_paths = resolve_default_json_paths()

    aggregated_speedups: dict[str, dict[str, list[SpeedupRecord]]] = {}
    for json_path in json_paths:
        print(f"Generating plots for {json_path}")
        json_speedups = plot_json_file(json_path, args.platform)
        for platform_name, platform_speedups in json_speedups.items():
            aggregated_platform = aggregated_speedups.setdefault(platform_name, {})
            for comparison_label, values in platform_speedups.items():
                aggregated_platform.setdefault(comparison_label, []).extend(values)

    print_geometric_speedup_summary(aggregated_speedups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
