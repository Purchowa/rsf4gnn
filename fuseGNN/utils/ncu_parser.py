"""
Nsight Compute CSV parsing helpers for GPU cache metrics.

This module parses NCU CSV outputs into a normalized counters dictionary
that can be stored under the diagnostics block in JSON logs.
"""

from __future__ import annotations

import csv
from typing import Dict, List, Tuple, Iterable


def _iter_csv_rows(csv_path: str) -> Iterable[dict]:
    """Yield CSV rows, skipping banner/comment lines.

    Nsight Compute CSV output sometimes prefixes lines with "==" or "#".
    This helper filters those while preserving the header row.
    """
    with open(csv_path, "r", encoding="utf-8") as handle:
        lines: List[str] = []
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("==") or stripped.startswith("#"):
                continue
            lines.append(line)
        if not lines:
            return
        reader = csv.DictReader(lines)
        for row in reader:
            yield row


def _parse_metric_value(value: str) -> Tuple[float, str]:
    """Parse a metric value from CSV and return value + unit hint."""
    raw = value.strip()
    if not raw:
        return 0.0, ""
    unit = ""
    if raw.endswith("%"):
        unit = "%"
        raw = raw[:-1]
    raw = raw.replace(",", "")
    try:
        return float(raw), unit
    except ValueError:
        return 0.0, unit


def parse_ncu_csv(csv_path: str) -> Tuple[Dict[str, float], Dict[str, object]]:
    """
    Parse NCU CSV output into aggregated metric values.

    Returns:
        Tuple of (metrics, metadata)
    """
    metrics: Dict[str, List[float]] = {}
    metric_units: Dict[str, str] = {}

    for row in _iter_csv_rows(csv_path):
        metric_name = (row.get("Metric Name") or row.get("Metric Name ") or "").strip()
        metric_value = (row.get("Metric Value") or "").strip()
        metric_unit = (row.get("Metric Unit") or "").strip()
        if not metric_name:
            continue
        value, unit_hint = _parse_metric_value(metric_value)
        unit = metric_unit or unit_hint
        metrics.setdefault(metric_name, []).append(value)
        if unit:
            metric_units[metric_name] = unit

    aggregated: Dict[str, float] = {}
    aggregation: Dict[str, str] = {}
    sample_counts: Dict[str, int] = {}

    for name, values in metrics.items():
        if not values:
            continue
        unit = metric_units.get(name, "")
        is_rate = unit == "%" or name.endswith(".pct") or "rate" in name.lower()
        sample_counts[name] = len(values)
        if is_rate:
            aggregated[name] = sum(values) / float(len(values))
            aggregation[name] = "avg"
        else:
            aggregated[name] = sum(values)
            aggregation[name] = "sum"

    metadata = {
        "metric_units": metric_units,
        "aggregation": aggregation,
        "sample_counts": sample_counts,
    }

    return aggregated, metadata


def extract_gpu_l2_counters(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize GPU L2 cache metrics into diagnostics counters.

    Expected inputs can include NCU metric names like:
    - lts__t_sectors_hit_rate.pct
    - lts__t_sectors_hit.sum
    - lts__t_sectors_miss.sum
    """
    counters: Dict[str, float] = {}

    if "lts__t_sectors_hit_rate.pct" in metrics:
        counters["gpu_l2_hit_rate"] = metrics["lts__t_sectors_hit_rate.pct"]
    if "lts__t_sectors_hit.sum" in metrics:
        counters["gpu_l2_hits"] = metrics["lts__t_sectors_hit.sum"]
    if "lts__t_sectors_miss.sum" in metrics:
        counters["gpu_l2_misses"] = metrics["lts__t_sectors_miss.sum"]

    return counters
