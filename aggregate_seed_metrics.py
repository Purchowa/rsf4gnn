#!/usr/bin/env python3

"""Aggregate seed-repeated experiment logs into a CSV summary.

The script searches for JSON logs whose file name contains ``_seed<digits>``,
groups runs by stripping that suffix, and computes mean/std across the
repeated seeds for the most relevant thesis metrics.

Example:

    python aggregate_seed_metrics.py log_a100 log_energy --out_csv seed_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from _plot_compare_metrics.io import load_json


SEED_SUFFIX_RE = re.compile(r"_seed(?P<seed>\d+)(?=\.json$)", re.IGNORECASE)

DEFAULT_SUMMARY_KEYS = [
    "avg_epoch_time_sec",
    "time_to_target_sec",
    "best_test_acc",
    "best_test_f1_micro",
    "best_test_f1_macro",
    "peak_gpu_mem_mb",
    "peak_cpu_mem_mb",
    "reordering_time_sec",
    "baseline_epoch_time_sec",
    "amortization_epoch",
    "total_energy_j",
    "total_energy_wh",
    "avg_power_w",
    "mean_power_from_energy_w",
    "avg_power_above_idle_w",
    "max_power_w",
    "min_power_w",
    "measured_duration_sec",
    "num_power_samples",
    "idle_power_w",
]


@dataclass(frozen=True)
class LogEntry:
    path: str
    group_id: str
    seed: int
    data: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate repeated seed logs into mean/std CSV summaries."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input files or directories to scan recursively for *_seed<id>.json logs.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="seed_summary.csv",
        help="Path of the aggregated CSV output.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help=(
            "Optional explicit list of summary metrics to aggregate. "
            "Defaults to the thesis-relevant metrics used by this repository."
        ),
    )
    return parser.parse_args()


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not (
        isinstance(value, float) and (math.isnan(value) or math.isinf(value))
    )


def _is_numeric_sequence(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for item in value:
        if not _is_numeric(item):
            return False
    return True


def _extract_seed(path: Path) -> Optional[int]:
    match = SEED_SUFFIX_RE.search(path.name)
    if match is None:
        return None
    return int(match.group("seed"))


def _group_id_for_path(path: Path) -> str:
    return SEED_SUFFIX_RE.sub("", path.name)


def _collect_json_files(inputs: Sequence[str]) -> List[Path]:
    collected: List[Path] = []
    for raw_input in inputs:
        path = Path(raw_input)
        if path.is_file():
            if path.suffix.lower() == ".json" and _extract_seed(path) is not None:
                collected.append(path)
            continue
        if path.is_dir():
            for candidate in sorted(path.rglob("*.json")):
                if _extract_seed(candidate) is not None:
                    collected.append(candidate)
    return collected


def _load_entries(paths: Iterable[Path]) -> List[LogEntry]:
    entries: List[LogEntry] = []
    for path in paths:
        seed = _extract_seed(path)
        if seed is None:
            continue
        entries.append(
            LogEntry(
                path=str(path),
                group_id=_group_id_for_path(path),
                seed=seed,
                data=load_json(str(path)),
            )
        )
    return entries


def _get_summary_value(log: Dict[str, Any], key: str) -> Optional[float]:
    summary = log.get("summary")
    if not isinstance(summary, dict):
        return None
    value = summary.get(key)
    if _is_numeric(value):
        return float(value)
    return None


def _reduce_peak_metric(log: Dict[str, Any], key: str) -> Optional[float]:
    summary_value = _get_summary_value(log, key)
    if summary_value is not None:
        return summary_value

    values = log.get(key)
    if _is_numeric_sequence(values):
        return float(max(values))
    return None


def _derived_amortization_epoch(log: Dict[str, Any]) -> Optional[float]:
    value = _get_summary_value(log, "amortization_epoch")
    if value is not None:
        return value

    baseline = _get_summary_value(log, "baseline_epoch_time_sec")
    avg_epoch = _get_summary_value(log, "avg_epoch_time_sec")
    reorder = _get_summary_value(log, "reordering_time_sec")
    if baseline is None or avg_epoch is None or reorder is None:
        return None
    if baseline <= avg_epoch or reorder <= 0.0:
        return None
    return reorder / (baseline - avg_epoch)


def _metric_value(log: Dict[str, Any], key: str) -> Optional[float]:
    if key in {"peak_gpu_mem_mb", "peak_cpu_mem_mb"}:
        return _reduce_peak_metric(log, key)
    if key == "amortization_epoch":
        return _derived_amortization_epoch(log)
    return _get_summary_value(log, key)


def _mean_and_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), None
    return float(mean(values)), float(stdev(values))


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return format(value, ".10g")


def _build_rows(entries: Sequence[LogEntry], metric_keys: Sequence[str]) -> List[Dict[str, str]]:
    grouped: DefaultDict[str, List[LogEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.group_id].append(entry)

    rows: List[Dict[str, str]] = []
    for group_id in sorted(grouped):
        group_entries = sorted(grouped[group_id], key=lambda item: item.seed)
        row: Dict[str, str] = {
            "group_id": group_id,
            "n_runs": str(len(group_entries)),
            "seeds": ",".join(str(entry.seed) for entry in group_entries),
            "example_file": group_entries[0].path,
        }

        for metric_key in metric_keys:
            values = []
            for entry in group_entries:
                metric_value = _metric_value(entry.data, metric_key)
                if metric_value is not None:
                    values.append(metric_value)
            mean_value, std_value = _mean_and_std(values)
            row[f"{metric_key}_mean"] = _fmt_float(mean_value)
            row[f"{metric_key}_std"] = _fmt_float(std_value)
            row[f"{metric_key}_n"] = str(len(values))

        rows.append(row)

    return rows


def _write_csv(rows: Sequence[Dict[str, str]], out_csv: str, metric_keys: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)

    fieldnames = ["group_id", "n_runs", "seeds", "example_file"]
    for metric_key in metric_keys:
        fieldnames.extend([
            f"{metric_key}_mean",
            f"{metric_key}_std",
            f"{metric_key}_n",
        ])

    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_summary(rows: Sequence[Dict[str, str]], metric_keys: Sequence[str]) -> None:
    print("Aggregated seed groups:", len(rows))
    if not rows:
        return

    for row in rows:
        print("- {} | seeds={} | n={}".format(row["group_id"], row["seeds"], row["n_runs"]))
        for metric_key in metric_keys:
            mean_value = row[f"{metric_key}_mean"]
            std_value = row[f"{metric_key}_std"]
            if mean_value or std_value:
                print("  - {}: {} ± {} (n={})".format(metric_key, mean_value, std_value, row[f"{metric_key}_n"]))


def main() -> None:
    args = parse_args()
    metric_keys = args.metrics if args.metrics is not None and len(args.metrics) > 0 else DEFAULT_SUMMARY_KEYS

    candidate_paths = _collect_json_files(args.inputs)
    entries = _load_entries(candidate_paths)

    if not entries:
        raise SystemExit("No seed-repeated JSON logs were found. Expected file names containing _seed<digits>.json")

    rows = _build_rows(entries, metric_keys)
    _write_csv(rows, args.out_csv, metric_keys)
    _print_summary(rows, metric_keys)
    print("CSV written to {}".format(args.out_csv))


if __name__ == "__main__":
    main()