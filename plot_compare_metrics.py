import argparse
import os
from typing import List

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "matplotlib is required. Install with: pip install matplotlib==3.7.5"
    ) from exc

from _plot_compare_metrics import (
    build_run_label,
    get_common_epoch_metrics,
    load_json,
    save_epoch_plot,
    save_summary_bar_plot,
    save_total_training_breakdown_plot,
    summary_keys_available,
    write_report,
    extract_diagnostics_summary,
    format_diagnostics_report,
)


PREFERRED_EPOCH_METRICS = [
    "train_loss",
    "train_acc",
    "val_acc",
    "test_acc",
    "train_f1_micro",
    "val_f1_micro",
    "test_f1_micro",
    "train_f1_macro",
    "val_f1_macro",
    "test_f1_macro",
    "epoch_time_sec",
    "train_time_sec",
    "eval_time_sec",
    "nodes_per_sec",
    "edges_per_sec",
]

PREFERRED_SUMMARY_KEYS = [
    "best_test_acc",
    "best_test_f1_micro",
    "best_test_f1_macro",
    "avg_epoch_time_sec",
    "preprocess_time_sec",
    "reordering_time_sec",
    "time_to_target_sec",
    "amortization_epoch",
    "peak_gpu_mem_mb",
    "peak_cpu_mem_mb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare >=2 training JSON logs and generate plots."
    )
    parser.add_argument(
        "log_files",
        nargs="+",
        help="Paths to JSON logs (at least two).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./plots_compare",
        help="Directory where plots and report will be saved.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="DPI for saved PNG files.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plots.",
    )
    parser.add_argument(
        "--ma_window",
        type=int,
        default=30,
        help="Moving-average window size for smoothing noisy epoch curves.",
    )
    parser.add_argument(
        "--ma_min_points",
        type=int,
        default=80,
        help="Apply moving average only when a series has at least this many points.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help=(
            "Optional custom labels for plots, in the same order as log_files. "
            "Example: --labels plot_1 plot_2"
        ),
    )
    args = parser.parse_args()

    if len(args.log_files) < 2:
        parser.error("Provide at least two JSON files.")
    if args.ma_window < 2:
        parser.error("--ma_window must be >= 2.")
    if args.ma_min_points < 2:
        parser.error("--ma_min_points must be >= 2.")
    if args.labels is not None and len(args.labels) != len(args.log_files):
        parser.error("--labels must have the same length as log_files.")
    return args


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    logs = [load_json(p) for p in args.log_files]
    if args.labels is not None:
        labels = args.labels
    else:
        labels = [build_run_label(log, p) for log, p in zip(logs, args.log_files)]

    epoch_metrics = get_common_epoch_metrics(logs, PREFERRED_EPOCH_METRICS)
    summary_keys = summary_keys_available(logs, PREFERRED_SUMMARY_KEYS)

    created_files: List[str] = []

    for metric in epoch_metrics:
        out = save_epoch_plot(
            metric,
            logs,
            labels,
            args.out_dir,
            args.dpi,
            args.ma_window,
            args.ma_min_points,
        )
        created_files.append(out)

    for key in summary_keys:
        out = save_summary_bar_plot(key, logs, labels, args.out_dir, args.dpi)
        if out is not None:
            created_files.append(out)

    total_breakdown = save_total_training_breakdown_plot(
        logs=logs,
        labels=labels,
        out_dir=args.out_dir,
        dpi=args.dpi,
    )
    if total_breakdown is not None:
        created_files.append(total_breakdown)

    report = write_report(
        args.out_dir,
        args.log_files,
        labels,
        logs,
        epoch_metrics,
        summary_keys,
    )
    created_files.append(report)

    # Extract and report hardware counter diagnostics if available
    diagnostics_summary = extract_diagnostics_summary(logs)
    if diagnostics_summary.get('available'):
        diagnostics_report = format_diagnostics_report(diagnostics_summary)
        print("\n" + diagnostics_report)

    print("Saved files:")
    for p in created_files:
        print("- " + p)

    if args.show:
        # No persistent figures kept open (saved and closed above), so this
        # just keeps behavior explicit for future extension.
        plt.show()


if __name__ == "__main__":
    main()
    
