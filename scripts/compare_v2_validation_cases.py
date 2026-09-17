#!/usr/bin/env python3
"""Compare compact GPU/NPU fixed-validation case files and emit changed cases only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--npu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "oxygenrec_v2_full_fixed_validation_v1_cases":
        raise ValueError(f"unexpected case protocol in {path}")
    return payload


def compare_payloads(
    gpu: dict[str, Any], npu: dict[str, Any]
) -> dict[str, Any]:
    if gpu.get("platform") != "gpu" or npu.get("platform") != "npu":
        raise ValueError("--gpu and --npu must contain GPU and NPU payloads")
    for field in ("precision", "source_commit", "inputs"):
        if gpu.get(field) != npu.get(field):
            raise ValueError(f"GPU/NPU case payloads differ in {field}")

    gpu_cases = gpu.get("cases")
    npu_cases = npu.get("cases")
    if not isinstance(gpu_cases, list) or not isinstance(npu_cases, list):
        raise ValueError("case payloads must contain a cases list")
    if len(gpu_cases) != len(npu_cases):
        raise ValueError("GPU/NPU case counts differ")

    changed = []
    greedy_changed = 0
    beam_changed = 0
    for gpu_case, npu_case in zip(gpu_cases, npu_cases, strict=True):
        case_index = gpu_case.get("case_index")
        if case_index != npu_case.get("case_index"):
            raise ValueError("GPU/NPU case ordering differs")
        if gpu_case.get("target") != npu_case.get("target"):
            raise ValueError(f"target differs at case {case_index}")
        greedy_differs = gpu_case.get("greedy") != npu_case.get("greedy")
        beam_differs = gpu_case.get("beam") != npu_case.get("beam")
        greedy_changed += int(greedy_differs)
        beam_changed += int(beam_differs)
        if greedy_differs or beam_differs:
            changed.append(
                {
                    "case_index": case_index,
                    "target": gpu_case["target"],
                    "greedy_changed": greedy_differs,
                    "beam_changed": beam_differs,
                    "gpu": {
                        "greedy": gpu_case["greedy"],
                        "beam": gpu_case["beam"],
                        "beam_scores": gpu_case["beam_scores"],
                    },
                    "npu": {
                        "greedy": npu_case["greedy"],
                        "beam": npu_case["beam"],
                        "beam_scores": npu_case["beam_scores"],
                    },
                }
            )

    return {
        "schema_version": 1,
        "protocol": "oxygenrec_v2_full_fixed_validation_changed_cases_v1",
        "precision": gpu["precision"],
        "source_commit": gpu["source_commit"],
        "inputs": gpu["inputs"],
        "case_count": len(gpu_cases),
        "greedy_changed_cases": greedy_changed,
        "beam_changed_cases": beam_changed,
        "changed_cases": changed,
    }


def main() -> int:
    args = parse_args()
    comparison = compare_payloads(load_payload(args.gpu), load_payload(args.npu))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "OK stage=v2_fixed_validation_changed_cases "
        f"precision={comparison['precision']} cases={comparison['case_count']} "
        f"greedy_changed={comparison['greedy_changed_cases']} "
        f"beam_changed={comparison['beam_changed_cases']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
