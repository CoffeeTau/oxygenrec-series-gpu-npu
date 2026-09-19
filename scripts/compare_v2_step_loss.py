#!/usr/bin/env python3
"""Compare paired GPU/NPU step diagnostics; check each step, not epoch means."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpu", type=Path)
    parser.add_argument("npu", type=Path)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-3)
    parser.add_argument("--relative-tolerance", type=float, default=1e-3)
    return parser.parse_args()


def compare(gpu: dict, npu: dict, absolute_tolerance: float, relative_tolerance: float) -> dict:
    if gpu["platform"] != "gpu" or npu["platform"] != "npu":
        raise ValueError("expected GPU result first and NPU result second")
    for key in ("protocol", "precision", "dropout_mode", "configured_dropout", "inputs", "seed", "epoch", "train_samples", "batch_size", "learning_rate", "optimizer_state_loaded", "source_files_sha256"):
        if gpu[key] != npu[key]:
            raise ValueError(f"GPU/NPU differ in {key}")
    if gpu["source"]["commit"] != npu["source"]["commit"]:
        raise ValueError("GPU/NPU source commits differ")
    if len(gpu["steps"]) != len(npu["steps"]):
        raise ValueError("GPU/NPU step counts differ")
    differences = []
    for left, right in zip(gpu["steps"], npu["steps"]):
        if left["step"] != right["step"] or left["batch_sha256"] != right["batch_sha256"]:
            raise ValueError(f"step or actual batch differs at step {left['step']}")
        absolute = abs(left["loss_before_update"] - right["loss_before_update"])
        relative = absolute / max(abs(left["loss_before_update"]), 1e-12)
        gradient_l2_absolute = abs(left["gradient_l2"] - right["gradient_l2"])
        gradient_l2_relative = gradient_l2_absolute / max(abs(left["gradient_l2"]), 1e-12)
        differences.append({
            "step": left["step"],
            "gpu_loss": left["loss_before_update"],
            "npu_loss": right["loss_before_update"],
            "absolute_difference": absolute,
            "relative_difference": relative,
            "gpu_gradient_l2": left["gradient_l2"],
            "npu_gradient_l2": right["gradient_l2"],
            "gradient_l2_relative_difference": gradient_l2_relative,
        })
    if not differences:
        raise ValueError("no steps to compare")
    return {
        "steps": len(differences),
        "mean_absolute_difference": sum(row["absolute_difference"] for row in differences) / len(differences),
        "max_absolute_difference": max(row["absolute_difference"] for row in differences),
        "max_relative_difference": max(row["relative_difference"] for row in differences),
        "max_gradient_l2_relative_difference": max(row["gradient_l2_relative_difference"] for row in differences),
        "first_above_absolute_tolerance": next((row for row in differences if row["absolute_difference"] > absolute_tolerance), None),
        "first_above_relative_tolerance": next((row for row in differences if row["relative_difference"] > relative_tolerance), None),
        "per_step": differences,
    }


def main() -> int:
    args = parse_args()
    if args.absolute_tolerance < 0 or args.relative_tolerance < 0:
        raise ValueError("tolerances must be nonnegative")
    result = compare(
        json.loads(args.gpu.read_text(encoding="utf-8")),
        json.loads(args.npu.read_text(encoding="utf-8")),
        args.absolute_tolerance,
        args.relative_tolerance,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
