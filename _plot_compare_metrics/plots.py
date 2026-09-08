import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.colors as mcolors

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "matplotlib is required. Install with: pip install matplotlib==3.7.5"
    ) from exc

from .data import get_common_epoch_metrics, is_numeric_sequence, moving_average, sanitize_filename, to_epoch_axis
from .labels import compress_labels_for_axis, display_metric_name


BACKEND_BASE_COLORS = {
    "ref": "#1f77b4",
    "gas": "#ff7f0e",
    "gar": "#2ca02c",
    "geo": "#9467bd",
    "unknown": "#7f7f7f",
}

BACKEND_LINESTYLES = {
    "ref": "-",
    "gas": "--",
    "gar": "-.",
    "geo": ":",
    "unknown": "-",
}


def _is_memory_metric(metric: str) -> bool:
    return metric in {"peak_gpu_mem_mb", "peak_cpu_mem_mb"}


def _mean_std(values: np.ndarray) -> Tuple[float, float]:
    mean = float(np.mean(values)) if values.size else 0.0
    if values.size > 1:
        std = float(np.std(values, ddof=1))
    else:
        std = 0.0
    return mean, std


def _summary_value_and_std(log: Dict[str, Any], key: str) -> Tuple[Optional[float], Optional[float]]:
    if _is_memory_metric(key):
        values = log.get(key)
        if not is_numeric_sequence(values):
            return None, None
        mean, std = _mean_std(np.array(values, dtype=float))
        return mean, std

    summary = log.get("summary", {})
    if not isinstance(summary, dict):
        return None, None
    val = summary.get(key, None)
    if isinstance(val, (int, float)) and not (
        isinstance(val, float) and (math.isnan(val) or math.isinf(val))
    ):
        return float(val), None
    return None, None


def _extract_backend(log: Dict[str, Any]) -> str:
    model = str(log.get("model", "")).lower()
    # Expected values look like: GCN_ref, GCN_gas, GCN_gar, GCN_geo
    if "_" in model:
        candidate = model.split("_")[-1]
        if candidate in BACKEND_BASE_COLORS:
            return candidate
    for candidate in ["ref", "gas", "gar", "geo"]:
        if candidate in model:
            return candidate
    return "unknown"


def _shade_color(base_color: str, position: int, count: int) -> str:
    if count <= 1:
        return base_color

    rgb = np.array(mcolors.to_rgb(base_color), dtype=float)
    # Spread shades from darker to lighter while preserving backend identity.
    offsets = np.linspace(-0.24, 0.24, count)
    offset = float(offsets[position])
    if offset >= 0.0:
        rgb = rgb * (1.0 - offset) + np.ones(3, dtype=float) * offset
    else:
        rgb = rgb * (1.0 + offset)
    rgb = np.clip(rgb, 0.0, 1.0)
    return mcolors.to_hex(rgb)


def _build_run_styles(logs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    backends = [_extract_backend(log) for log in logs]

    backend_to_indices: Dict[str, List[int]] = {}
    for idx, backend in enumerate(backends):
        if backend not in backend_to_indices:
            backend_to_indices[backend] = []
        backend_to_indices[backend].append(idx)

    styles: List[Dict[str, str]] = [
        {"backend": "unknown", "color": BACKEND_BASE_COLORS["unknown"], "linestyle": BACKEND_LINESTYLES["unknown"]}
        for _ in logs
    ]

    for backend, indices in backend_to_indices.items():
        base_color = BACKEND_BASE_COLORS.get(backend, BACKEND_BASE_COLORS["unknown"])
        line_style = BACKEND_LINESTYLES.get(backend, BACKEND_LINESTYLES["unknown"])
        for local_pos, idx in enumerate(indices):
            styles[idx] = {
                "backend": backend,
                "color": _shade_color(base_color, local_pos, len(indices)),
                "linestyle": line_style,
            }

    return styles


def get_summary_value(log: Dict[str, Any], key: str) -> Optional[float]:
    value, _ = _summary_value_and_std(log, key)
    return value


def summary_keys_available(logs: List[Dict[str, Any]], preferred_summary_keys: List[str]) -> List[str]:
    available = []
    for key in preferred_summary_keys:
        values = [get_summary_value(log, key) for log in logs]
        if sum(v is not None for v in values) >= 2:
            available.append(key)
    return available


def save_epoch_plot(
    metric: str,
    logs: List[Dict[str, Any]],
    labels: List[str],
    out_dir: str,
    dpi: int,
    ma_window: int,
    ma_min_points: int,
) -> str:
    plt.figure(figsize=(9, 5))
    metric_name = display_metric_name(metric)

    run_styles = _build_run_styles(logs)

    for log, label, style in zip(logs, labels, run_styles):
        y = np.array(log[metric], dtype=float)
        x = to_epoch_axis(log, len(y))
        if len(y) >= ma_min_points:
            y_ma = moving_average(y, min(ma_window, len(y)))
            plt.plot(
                x,
                y_ma,
                linewidth=1.8,
                linestyle=style["linestyle"],
                color=style["color"],
                label=label + " (MA)",
            )
        else:
            plt.plot(
                x,
                y,
                linewidth=1.6,
                linestyle=style["linestyle"],
                color=style["color"],
                label=label,
            )

    plt.title("Epoch Metric: " + metric_name)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_name = "epoch_" + sanitize_filename(metric) + ".png"
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    return out_path


def save_summary_bar_plot(
    key: str,
    logs: List[Dict[str, Any]],
    labels: List[str],
    out_dir: str,
    dpi: int,
) -> Optional[str]:
    stats = [_summary_value_and_std(log, key) for log in logs]
    values = [val for val, _ in stats]
    stds = [std for _, std in stats]
    idx = [i for i, v in enumerate(values) if v is not None]
    if len(idx) < 2:
        return None

    x_labels = [labels[i] for i in idx]
    common_context, compact_x_labels = compress_labels_for_axis(x_labels, idx)
    y_vals = [values[i] for i in idx]  # type: ignore
    y_errs = [stds[i] for i in idx]
    key_label = display_metric_name(key)

    bar_colors = [BACKEND_BASE_COLORS.get(_extract_backend(logs[i]), BACKEND_BASE_COLORS["unknown"]) for i in idx]

    plt.figure(figsize=(10, 5))
    yerr = None
    if _is_memory_metric(key):
        yerr = y_errs
    bars = plt.bar(
        np.arange(len(y_vals)),
        y_vals,
        yerr=yerr,
        color=bar_colors,
        edgecolor="#1f1f1f",
        linewidth=0.4,
        capsize=4,
    )
    for b, y in zip(bars, y_vals):
        plt.text(
            b.get_x() + b.get_width() * 0.5,
            y,
            "{:.2f}".format(y) if _is_memory_metric(key) else "{:.4f}".format(y),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    plt.xticks(np.arange(len(y_vals)), compact_x_labels, rotation=20, ha="right", fontsize=8)
    plt.ylabel(key_label)
    if common_context:
        plt.title("Summary Comparison: " + key_label + "\nContext: " + common_context)
    else:
        plt.title("Summary Comparison: " + key_label)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_name = "summary_" + sanitize_filename(key) + ".png"
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    return out_path


def speedup_vs_first(logs: List[Dict[str, Any]]) -> List[Optional[float]]:
    first = get_summary_value(logs[0], "avg_epoch_time_sec")
    out: List[Optional[float]] = []
    for log in logs:
        cur = get_summary_value(log, "avg_epoch_time_sec")
        if first is None or cur is None or cur <= 0.0:
            out.append(None)
        else:
            out.append(first / cur)
    return out


def sum_metric_series(log: Dict[str, Any], key: str) -> Optional[float]:
    values = log.get(key)
    if not is_numeric_sequence(values):
        return None
    return float(sum(values))


def save_total_training_breakdown_plot(
    logs: List[Dict[str, Any]],
    labels: List[str],
    out_dir: str,
    dpi: int,
) -> Optional[str]:
    preprocess_vals: List[float] = []
    train_vals: List[float] = []
    eval_vals: List[float] = []
    overhead_vals: List[float] = []

    for log in logs:
        preprocess = get_summary_value(log, "preprocess_time_sec")
        preprocess_sec = 0.0 if preprocess is None else preprocess

        train_total = sum_metric_series(log, "train_time_sec")
        eval_total = sum_metric_series(log, "eval_time_sec")
        epoch_total = sum_metric_series(log, "epoch_time_sec")

        train_sec = 0.0 if train_total is None else train_total
        eval_sec = 0.0 if eval_total is None else eval_total

        if epoch_total is not None:
            overhead_sec = max(epoch_total - train_sec - eval_sec, 0.0)
        else:
            overhead_sec = 0.0

        preprocess_vals.append(preprocess_sec)
        train_vals.append(train_sec)
        eval_vals.append(eval_sec)
        overhead_vals.append(overhead_sec)

    totals = [
        preprocess_vals[i] + train_vals[i] + eval_vals[i] + overhead_vals[i]
        for i in range(len(labels))
    ]
    if sum(1 for v in totals if v > 0.0) < 2:
        return None

    common_context, compact_x_labels = compress_labels_for_axis(labels)
    x = np.arange(len(labels))

    # Backend-aware stacked colors: each backend keeps its own color family,
    # and each component gets a consistent shade offset.
    backends = [_extract_backend(log) for log in logs]
    component_offsets = {
        "preprocess": -0.20,
        "train": 0.05,
        "eval": 0.20,
        "overhead": 0.35,
    }
    preprocess_colors = [
        _shade_color(BACKEND_BASE_COLORS.get(backend, BACKEND_BASE_COLORS["unknown"]), 0, 1)
        if component_offsets["preprocess"] == 0.0
        else _shade_color(
            mcolors.to_hex(
                np.clip(
                    np.array(mcolors.to_rgb(BACKEND_BASE_COLORS.get(backend, BACKEND_BASE_COLORS["unknown"])), dtype=float)
                    * (1.0 + component_offsets["preprocess"]),
                    0.0,
                    1.0,
                )
            ),
            0,
            1,
        )
        for backend in backends
    ]
    train_colors = [
        _shade_color(
            BACKEND_BASE_COLORS.get(backend, BACKEND_BASE_COLORS["unknown"]),
            0,
            1,
        )
        for backend in backends
    ]
    eval_colors = [
        _shade_color(
            mcolors.to_hex(
                np.clip(
                    np.array(mcolors.to_rgb(BACKEND_BASE_COLORS.get(backend, BACKEND_BASE_COLORS["unknown"])), dtype=float)
                    * (1.0 - component_offsets["eval"]) + np.ones(3, dtype=float) * component_offsets["eval"],
                    0.0,
                    1.0,
                )
            ),
            0,
            1,
        )
        for backend in backends
    ]
    overhead_colors = [
        _shade_color(
            mcolors.to_hex(
                np.clip(
                    np.array(mcolors.to_rgb(BACKEND_BASE_COLORS.get(backend, BACKEND_BASE_COLORS["unknown"])), dtype=float)
                    * (1.0 - component_offsets["overhead"]) + np.ones(3, dtype=float) * component_offsets["overhead"],
                    0.0,
                    1.0,
                )
            ),
            0,
            1,
        )
        for backend in backends
    ]

    plt.figure(figsize=(11, 6))
    b0 = np.array(preprocess_vals, dtype=float)
    b1 = b0 + np.array(train_vals, dtype=float)
    b2 = b1 + np.array(eval_vals, dtype=float)

    plt.bar(
        x,
        preprocess_vals,
        label="Preprocessing",
        color=preprocess_colors,
        edgecolor="#1f1f1f",
        linewidth=0.4,
        hatch="//",
    )
    plt.bar(
        x,
        train_vals,
        bottom=b0,
        label="Training",
        color=train_colors,
        edgecolor="#1f1f1f",
        linewidth=0.4,
    )
    plt.bar(
        x,
        eval_vals,
        bottom=b1,
        label="Evaluation",
        color=eval_colors,
        edgecolor="#1f1f1f",
        linewidth=0.4,
        hatch="..",
    )
    plt.bar(
        x,
        overhead_vals,
        bottom=b2,
        label="Other Epoch Overhead",
        color=overhead_colors,
        edgecolor="#1f1f1f",
        linewidth=0.4,
        hatch="xx",
    )

    for i, total in enumerate(totals):
        plt.text(x[i], total, "{:.4f}".format(total), ha="center", va="bottom", fontsize=8)

    plt.xticks(x, compact_x_labels, rotation=20, ha="right", fontsize=8)
    plt.ylabel("Time [s]")
    if common_context:
        plt.title("Total Training Breakdown\nContext: " + common_context)
    else:
        plt.title("Total Training Breakdown")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "summary_total_training_breakdown.png")
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    return out_path
