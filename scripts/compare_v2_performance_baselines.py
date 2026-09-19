#!/usr/bin/env python3
"""Compare matched GPU and NPU v2 training-performance baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpu", type=Path)
    parser.add_argument("npu", type=Path)
    return parser.parse_args()


def compare(gpu: dict, npu: dict) -> dict:
    if gpu["platform"] != "gpu" or npu["platform"] != "npu":
        raise ValueError("expected GPU baseline first and NPU baseline second")
    for key in ("protocol", "precision", "inputs", "workload", "source_files_sha256"):
        if gpu[key] != npu[key]:
            raise ValueError(f"GPU/NPU baselines differ in {key}")
    if gpu["source"]["commit"] != npu["source"]["commit"]:
        raise ValueError("GPU/NPU source commits differ")
    gpu_by_batch = {row["batch_size"]: row for row in gpu["aggregates"]}
    npu_by_batch = {row["batch_size"]: row for row in npu["aggregates"]}
    if gpu_by_batch.keys() != npu_by_batch.keys():
        raise ValueError("GPU/NPU batch-size sets differ")
    rows = []
    for batch_size in sorted(gpu_by_batch):
        left = gpu_by_batch[batch_size]
        right = npu_by_batch[batch_size]
        gpu_throughput = left["samples_per_second_median"]
        npu_throughput = right["samples_per_second_median"]
        gpu_memory = left["peak_memory_allocated_bytes_max"]
        npu_memory = right["peak_memory_allocated_bytes_max"]
        rows.append({
            "batch_size": batch_size,
            "gpu_samples_per_second_median": gpu_throughput,
            "npu_samples_per_second_median": npu_throughput,
            "npu_to_gpu_throughput_ratio": npu_throughput / gpu_throughput,
            "gpu_mean_step_ms_median": 1000 * left["mean_step_seconds_median"],
            "npu_mean_step_ms_median": 1000 * right["mean_step_seconds_median"],
            "gpu_peak_memory_allocated_bytes": gpu_memory,
            "npu_peak_memory_allocated_bytes": npu_memory,
            "npu_to_gpu_peak_memory_ratio": (
                npu_memory / gpu_memory if gpu_memory else None
            ),
            "gpu_throughput_cv": left["samples_per_second_cv"],
            "npu_throughput_cv": right["samples_per_second_cv"],
        })
    return {
        "protocol": "oxygenrec_v2_gpu_npu_performance_comparison_v1",
        "precision": gpu["precision"],
        "gpu_device_name": gpu["device_name"],
        "npu_device_name": npu["device_name"],
        "scope": "hardware_and_software_stack_observation_not_equivalent_hardware_claim",
        "rows": rows,
    }


def main() -> int:
    args = parse_args()
    result = compare(
        json.loads(args.gpu.read_text(encoding="utf-8")),
        json.loads(args.npu.read_text(encoding="utf-8")),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
