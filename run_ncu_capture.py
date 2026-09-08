#!/usr/bin/env python3
"""
Run a short training session under Nsight Compute and merge GPU L2 counters
into the JSON diagnostics log.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict

from fuseGNN.utils.ncu_parser import parse_ncu_csv, extract_gpu_l2_counters


DEFAULT_NCU_METRICS = [
    "lts__t_sectors_hit_rate.pct",
    "lts__t_sectors_hit.sum",
    "lts__t_sectors_miss.sum",
]


def _update_diagnostics(log_path: str, counters: Dict[str, float], metadata: Dict[str, object]) -> None:
    with open(log_path, "r", encoding="utf-8") as handle:
        log = json.load(handle)

    diagnostics = log.setdefault("diagnostics", {})
    diagnostics.setdefault("counters", {})
    diagnostics.setdefault("metadata", {})

    diagnostics["counters"].update(counters)
    diagnostics["metadata"].update(metadata)
    diagnostics["status"] = "success" if counters else "partial"

    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(log, handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Nsight Compute GPU L2 cache capture")
    parser.add_argument("--gcn_lib", required=True, help="Path to gcnLib directory")
    parser.add_argument("--dataset", default="Cora", help="Dataset to use")
    parser.add_argument("--max_iter", type=int, default=2, help="Number of epochs to run")
    parser.add_argument("--model", default="GCN", help="Model name")
    parser.add_argument("--mode", default="gar", help="Convolution backend mode")
    parser.add_argument("--log_dir", default="./log", help="Log directory")
    parser.add_argument("--log_filename", default="logs.json", help="Log filename")
    parser.add_argument("--ncu_metrics", default=None, help="Comma-separated NCU metrics")
    parser.add_argument("--ncu_path", default="ncu", help="Path to Nsight Compute CLI")
    args = parser.parse_args()

    gcn_lib_path = os.path.abspath(args.gcn_lib)
    if not os.path.exists(gcn_lib_path):
        print(f"Error: gcnLib path not found: {gcn_lib_path}")
        return 1

    metrics = DEFAULT_NCU_METRICS
    if args.ncu_metrics:
        metrics = [m.strip() for m in args.ncu_metrics.split(",") if m.strip()]

    exp_name = f"{args.model}_{args.mode}_{args.dataset}"
    log_dir = os.path.abspath(args.log_dir)
    log_path = os.path.join(log_dir, exp_name, args.log_filename)

    with tempfile.TemporaryDirectory(prefix="ncu_capture_") as tmp_dir:
        csv_path = os.path.join(tmp_dir, "ncu_output.csv")
        cmd = [
            args.ncu_path,
            "--csv",
            "--metrics",
            ",".join(metrics),
            "--target-processes",
            "all",
            "--profile-from-start",
            "on",
            "--log-file",
            csv_path,
            sys.executable,
            os.path.join(gcn_lib_path, "training_main.py"),
            "--data",
            args.dataset,
            "--max_iter",
            str(args.max_iter),
            "--log_dir",
            log_dir,
            "--log_filename",
            args.log_filename,
        ]

        result = subprocess.run(cmd, cwd=gcn_lib_path, capture_output=True, text=True)
        if result.returncode != 0:
            print("NCU run failed:")
            print(result.stderr[:500])
            return result.returncode

        if not os.path.exists(csv_path):
            print(f"NCU CSV output not found at {csv_path}")
            return 1
        if not os.path.exists(log_path):
            print(f"Training log not found at {log_path}")
            return 1

        metrics_data, meta = parse_ncu_csv(csv_path)
        counters = extract_gpu_l2_counters(metrics_data)
        metadata = {
            "capture_enabled": True,
            "capture_source": "ncu",
            "ncu_metrics": metrics,
            "sample_epochs": args.max_iter,
        }
        metadata.update(meta)

        _update_diagnostics(log_path, counters, metadata)

    print(f"NCU counters merged into {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
