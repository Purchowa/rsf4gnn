import os

from typing import Any, Dict, List, Optional

from .plots import get_summary_value, speedup_vs_first


def write_report(
    out_dir: str,
    file_paths: List[str],
    labels: List[str],
    logs: List[Dict[str, Any]],
    used_epoch_metrics: List[str],
    used_summary_keys: List[str],
) -> str:
    report_path = os.path.join(out_dir, "report.md")
    sp = speedup_vs_first(logs)

    lines: List[str] = []
    lines.append("# JSON Comparison Report")
    lines.append("")
    lines.append("## Inputs")
    for p in file_paths:
        lines.append("- " + p)
    lines.append("")
    lines.append("## Runs")
    for i, lab in enumerate(labels):
        lines.append("- [{}] {}".format(i, lab))
    lines.append("")
    lines.append("## Plotted Epoch Metrics")
    if used_epoch_metrics:
        for m in used_epoch_metrics:
            lines.append("- " + m)
    else:
        lines.append("- None (no common numeric epoch metrics).")
    lines.append("")
    lines.append("## Summary Table")
    header = ["run", "label"] + used_summary_keys + ["speedup_vs_run0_by_avg_epoch_time"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for i, (lab, log) in enumerate(zip(labels, logs)):
        row: List[str] = [str(i), lab]
        for k in used_summary_keys:
            v = get_summary_value(log, k)
            row.append("null" if v is None else "{:.6f}".format(v))
        row.append("null" if sp[i] is None else "{:.6f}".format(sp[i]))  # type: ignore
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Notes")
    lines.append("- speedup_vs_run0_by_avg_epoch_time is higher-is-better.")
    lines.append("- speedup is computed only if avg_epoch_time_sec is available and positive.")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path
