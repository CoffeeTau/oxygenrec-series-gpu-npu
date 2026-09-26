#!/usr/bin/env python3
"""Compare matched NPU runs with proprietary internal formats disabled/enabled."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


FACTOR = "npu_internal_format"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("treatment", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _without_factor(workload: dict) -> dict:
    matched = dict(workload)
    matched.pop(FACTOR, None)
    return matched


def _loss_medians(runs: list[dict], batch_size: int) -> dict[str, float]:
    selected = [row for row in runs if int(row["batch_size"]) == batch_size]
    if not selected or not all(row.get("all_losses_finite") is True for row in selected):
        raise ValueError(f"batch {batch_size} has missing or non-finite loss evidence")
    return {
        key: statistics.median(float(row[key]) for row in selected)
        for key in ("first_loss", "last_loss", "mean_loss")
    }


def compare(control: dict, treatment: dict) -> dict:
    for label, baseline in (("control", control), ("treatment", treatment)):
        if baseline.get("status") != "passed":
            raise ValueError(f"{label} status is not passed")
        if baseline.get("protocol") != "oxygenrec_v2_single_device_training_performance_v1":
            raise ValueError(f"{label} is not a performance baseline")

    for key in (
        "platform",
        "device",
        "device_name",
        "precision",
        "inputs",
        "source_files_sha256",
    ):
        if control[key] != treatment[key]:
            raise ValueError(f"control/treatment differ in {key}")
    if control["platform"] != "npu":
        raise ValueError("internal-format A/B requires the NPU platform")
    if control["source"]["commit"] != treatment["source"]["commit"]:
        raise ValueError("control/treatment source commits differ")
    if control["source"].get("tracked_dirty") or treatment["source"].get("tracked_dirty"):
        raise ValueError("control/treatment require clean tracked worktrees")

    control_workload = control["workload"]
    treatment_workload = treatment["workload"]
    if _without_factor(control_workload) != _without_factor(treatment_workload):
        raise ValueError("control/treatment workload differs beyond npu_internal_format")
    if control_workload.get(FACTOR) != "disable":
        raise ValueError("control npu_internal_format must be disable")
    if treatment_workload.get(FACTOR) != "enable":
        raise ValueError("treatment npu_internal_format must be enable")

    control_by_batch = {int(row["batch_size"]): row for row in control["aggregates"]}
    treatment_by_batch = {
        int(row["batch_size"]): row for row in treatment["aggregates"]
    }
    if control_by_batch.keys() != treatment_by_batch.keys():
        raise ValueError("control/treatment batch-size sets differ")

    rows = []
    for batch_size in sorted(control_by_batch):
        left = control_by_batch[batch_size]
        right = treatment_by_batch[batch_size]
        control_throughput = float(left["samples_per_second_median"])
        treatment_throughput = float(right["samples_per_second_median"])
        control_losses = _loss_medians(control["runs"], batch_size)
        treatment_losses = _loss_medians(treatment["runs"], batch_size)
        rows.append({
            "batch_size": batch_size,
            "control_samples_per_second_median": control_throughput,
            "treatment_samples_per_second_median": treatment_throughput,
            "treatment_to_control_throughput_ratio": (
                treatment_throughput / control_throughput
            ),
            "throughput_change_percent": (
                (treatment_throughput / control_throughput - 1.0) * 100.0
            ),
            "control_mean_step_ms_median": (
                1000.0 * float(left["mean_step_seconds_median"])
            ),
            "treatment_mean_step_ms_median": (
                1000.0 * float(right["mean_step_seconds_median"])
            ),
            "control_throughput_cv": float(left["samples_per_second_cv"]),
            "treatment_throughput_cv": float(right["samples_per_second_cv"]),
            "control_peak_memory_allocated_bytes": int(
                left["peak_memory_allocated_bytes_max"]
            ),
            "treatment_peak_memory_allocated_bytes": int(
                right["peak_memory_allocated_bytes_max"]
            ),
            "control_loss_medians": control_losses,
            "treatment_loss_medians": treatment_losses,
            "first_measured_loss_absolute_difference": abs(
                treatment_losses["first_loss"] - control_losses["first_loss"]
            ),
        })

    return {
        "schema_version": 1,
        "protocol": "oxygenrec_v2_npu_internal_format_performance_ab_v1",
        "scope": "runtime_format_performance_ab_not_quality_training",
        "platform": control["platform"],
        "device_name": control["device_name"],
        "precision": control["precision"],
        "factor": FACTOR,
        "control_value": control_workload[FACTOR],
        "treatment_value": treatment_workload[FACTOR],
        "optimizer": control_workload["optimizer"],
        "loss_semantics": (
            "first_loss is the first measured loss after all configured warmup steps; "
            "this performance A/B does not establish recommendation quality"
        ),
        "source_commit": control["source"]["commit"],
        "rows": rows,
        "status": "passed",
    }


def main() -> int:
    args = parse_args()
    missing = [
        f"{label}={path}"
        for label, path in (("control", args.control), ("treatment", args.treatment))
        if not path.is_file()
    ]
    if missing:
        print(
            "cannot compare because result file is missing; the corresponding "
            f"experiment likely failed before writing JSON: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 2
    result = compare(
        json.loads(args.control.read_text(encoding="utf-8")),
        json.loads(args.treatment.read_text(encoding="utf-8")),
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
