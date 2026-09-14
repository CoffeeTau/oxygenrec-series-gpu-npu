#!/usr/bin/env python3
"""Compare independently generated GPU and NPU compact probe JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--npu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    return parser.parse_args()


def flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, list):
        result: list[float] = []
        for child in value:
            result.extend(flatten_numbers(child))
        return result
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    raise TypeError(f"expected nested numeric lists, got {type(value).__name__}")


def compare_numbers(
    gpu_value: Any, npu_value: Any, *, atol: float, rtol: float
) -> dict[str, Any]:
    gpu = flatten_numbers(gpu_value)
    npu = flatten_numbers(npu_value)
    if len(gpu) != len(npu):
        return {
            "all_close": False,
            "gpu_count": len(gpu),
            "npu_count": len(npu),
            "max_abs": None,
            "mean_abs": None,
        }
    deltas = [abs(left - right) for left, right in zip(gpu, npu)]
    close = all(
        math.isfinite(left)
        and math.isfinite(right)
        and abs(left - right) <= atol + rtol * abs(left)
        for left, right in zip(gpu, npu)
    )
    return {
        "all_close": close,
        "count": len(gpu),
        "max_abs": max(deltas, default=0.0),
        "mean_abs": sum(deltas) / len(deltas) if deltas else 0.0,
    }


def compare_summary_maps(
    gpu_map: dict[str, Any],
    npu_map: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    gpu_names = set(gpu_map)
    npu_names = set(npu_map)
    missing = sorted(gpu_names - npu_names)
    unexpected = sorted(npu_names - gpu_names)
    tensors: dict[str, Any] = {}
    all_close = not missing and not unexpected
    for name in sorted(gpu_names & npu_names):
        gpu = gpu_map[name]
        npu = npu_map[name]
        structure_match = (
            gpu["shape"] == npu["shape"]
            and gpu["numel"] == npu["numel"]
            and gpu["finite"]
            and npu["finite"]
            and [row["index"] for row in gpu["samples"]]
            == [row["index"] for row in npu["samples"]]
        )
        gpu_values = [gpu[key] for key in ("min", "max", "mean", "std", "l1", "l2")]
        npu_values = [npu[key] for key in ("min", "max", "mean", "std", "l1", "l2")]
        stats = compare_numbers(gpu_values, npu_values, atol=atol, rtol=rtol)
        samples = compare_numbers(
            [row["value"] for row in gpu["samples"]],
            [row["value"] for row in npu["samples"]],
            atol=atol,
            rtol=rtol,
        )
        close = structure_match and stats["all_close"] and samples["all_close"]
        all_close = all_close and close
        tensors[name] = {
            "all_close": close,
            "structure_match": structure_match,
            "statistics": stats,
            "fixed_samples": samples,
        }
    return {
        "all_close": all_close,
        "scope": "per_tensor_statistics_and_fixed_samples",
        "missing": missing,
        "unexpected": unexpected,
        "tensors": tensors,
    }


def main() -> int:
    args = parse_args()
    gpu = json.loads(args.gpu.read_text(encoding="utf-8"))
    npu = json.loads(args.npu.read_text(encoding="utf-8"))
    if gpu.get("platform") != "gpu" or npu.get("platform") != "npu":
        raise ValueError("--gpu and --npu must point to their matching platform probes")

    exact_fields = {
        "schema_version": gpu.get("schema_version") == npu.get("schema_version"),
        "protocol": gpu.get("protocol") == npu.get("protocol"),
        "source_commit": gpu.get("source", {}).get("commit")
        == npu.get("source", {}).get("commit"),
        "source_files": gpu.get("source_file_sha256")
        == npu.get("source_file_sha256"),
        "inputs": gpu.get("inputs") == npu.get("inputs"),
        "model_config": gpu.get("model_config") == npu.get("model_config"),
        "batch_metadata": gpu.get("batch_metadata") == npu.get("batch_metadata"),
        "batch": gpu.get("batch") == npu.get("batch"),
        "beam_width": gpu.get("beam_width") == npu.get("beam_width"),
        "missing_gradients": gpu["train_step"]["missing_gradients"]
        == npu["train_step"]["missing_gradients"],
    }
    continuous = {
        "loss": compare_numbers(
            gpu["inference"]["loss"], npu["inference"]["loss"],
            atol=args.atol, rtol=args.rtol,
        ),
        "ntp_loss": compare_numbers(
            gpu["inference"]["ntp_loss"], npu["inference"]["ntp_loss"],
            atol=args.atol, rtol=args.rtol,
        ),
        "level_losses": compare_numbers(
            gpu["inference"]["level_losses"], npu["inference"]["level_losses"],
            atol=args.atol, rtol=args.rtol,
        ),
        "logits": compare_numbers(
            gpu["inference"]["logits"], npu["inference"]["logits"],
            atol=args.atol, rtol=args.rtol,
        ),
        "beam_scores": compare_numbers(
            gpu["inference"]["beam_scores"], npu["inference"]["beam_scores"],
            atol=args.atol, rtol=args.rtol,
        ),
        "train_loss": compare_numbers(
            gpu["train_step"]["loss"], npu["train_step"]["loss"],
            atol=args.atol, rtol=args.rtol,
        ),
        "max_parameter_delta": compare_numbers(
            gpu["train_step"]["max_parameter_delta"],
            npu["train_step"]["max_parameter_delta"],
            atol=args.atol,
            rtol=args.rtol,
        ),
    }
    discrete = {
        "greedy_sids_match": gpu["inference"]["greedy_sids"]
        == npu["inference"]["greedy_sids"],
        "beam_sids_match": gpu["inference"]["beam_sids"]
        == npu["inference"]["beam_sids"],
    }
    gradients = compare_summary_maps(
        gpu["train_step"]["gradients"],
        npu["train_step"]["gradients"],
        atol=args.atol,
        rtol=args.rtol,
    )
    parameter_deltas = compare_summary_maps(
        gpu["train_step"]["parameter_deltas"],
        npu["train_step"]["parameter_deltas"],
        atol=args.atol,
        rtol=args.rtol,
    )
    passed = (
        all(exact_fields.values())
        and all(row["all_close"] for row in continuous.values())
        and all(discrete.values())
        and gradients["all_close"]
        and parameter_deltas["all_close"]
    )
    report = {
        "schema_version": 1,
        "comparison": "gpu_npu_compact_independent_probes",
        "passed": passed,
        "atol": args.atol,
        "rtol": args.rtol,
        "exact_prerequisites": exact_fields,
        "continuous": continuous,
        "discrete": discrete,
        "gradients": gradients,
        "parameter_deltas": parameter_deltas,
        "performance": {"gpu": gpu["performance"], "npu": npu["performance"]},
        "runtime": {"gpu": gpu.get("runtime"), "npu": npu.get("runtime")},
        "scope_note": (
            "Forward logits and generation are compared directly; gradients and "
            "parameter deltas use per-tensor statistics plus fixed samples. Run the "
            "strict artifact workflow only when deeper tensor diagnosis is needed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "OK" if passed else "FAIL",
        "stage=v2_device_probe_comparison",
        f"output={args.output}",
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
