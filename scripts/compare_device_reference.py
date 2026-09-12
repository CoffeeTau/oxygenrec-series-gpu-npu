#!/usr/bin/env python3
"""Compare OxygenREC-v2 Full on one accelerator with a frozen GPU artifact."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.device import device_name, resolve_device
from oxygenrec.migration_alignment import (
    REFERENCE_PROTOCOL,
    REFERENCE_SCHEMA_VERSION,
    compare_tensor_maps,
    deterministic_train_step,
    file_sha256,
    inference_outputs,
    load_model,
    measure_forward,
    move_batch,
    restored_config,
    source_file_hashes,
)
from oxygenrec.sid import PrefixTrie, SIDRegistry
from oxygenrec.model import OxygenRECConfig, OxygenRECModel


def git_state(project_root: Path) -> dict[str, object]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments], cwd=project_root, check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()

    try:
        revision = run("rev-parse", "HEAD")
        tracked_status = run(
            "status", "--porcelain", "--untracked-files=no"
        )
        untracked = run("ls-files", "--others", "--exclude-standard")
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": None,
            "tracked_dirty": None,
            "untracked_file_count": None,
        }
    return {
        "commit": revision,
        "tracked_dirty": bool(tracked_status),
        "untracked_file_count": len(untracked.splitlines()) if untracked else 0,
    }


def scalar_check(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    expected_value = float(expected)
    actual_value = float(actual)
    return {
        "expected": expected_value,
        "actual": actual_value,
        "abs_delta": abs(actual_value - expected_value),
        "close": bool(
            torch.allclose(expected, actual.cpu(), atol=atol, rtol=rtol)
        ),
    }


def sequence_check(
    expected: tuple[torch.Tensor, ...],
    actual: tuple[torch.Tensor, ...],
    *,
    prefix: str,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    expected_map = {
        f"{prefix}_{index}": tensor for index, tensor in enumerate(expected)
    }
    actual_map = {
        f"{prefix}_{index}": tensor for index, tensor in enumerate(actual)
    }
    return compare_tensor_maps(
        expected_map, actual_map, atol=atol, rtol=rtol
    )


def compare_legacy_v1(
    args: argparse.Namespace,
    reference: dict[str, Any],
    device: torch.device,
) -> int:
    """Keep the frozen v1 single-item reference protocol runnable."""

    checkpoint_hash = file_sha256(args.checkpoint)
    if checkpoint_hash != reference["checkpoint_sha256"]:
        raise RuntimeError("checkpoint SHA-256 does not match GPU reference")
    registry = SIDRegistry.from_json(args.sid_registry)
    if registry.version != reference["sid_registry_version"]:
        raise RuntimeError("SID registry version does not match GPU reference")
    known = {field.name for field in fields(OxygenRECConfig)}
    config = OxygenRECConfig(**{
        key: value
        for key, value in reference["model_config"].items()
        if key in known
    })
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model = OxygenRECModel(config).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    history = reference["history_sids"].to(device)
    padding = reference["history_padding_mask"].to(device)
    targets = reference["target_sids"].to(device)
    trie = PrefixTrie.from_registry(registry)
    with torch.inference_mode():
        output = model(history, padding, target_sids=targets)
        beam = model.beam_search(
            history, padding, trie, beam_width=int(reference["beam_width"])
        )
    logits_close = all(
        torch.allclose(actual.cpu(), expected, atol=args.atol, rtol=args.rtol)
        for actual, expected in zip(
            output.logits, reference["logits"], strict=True
        )
    )
    max_abs = max(
        float((actual.cpu() - expected).abs().max())
        for actual, expected in zip(
            output.logits, reference["logits"], strict=True
        )
    )
    loss_delta = abs(float(output.loss.cpu()) - float(reference["loss"]))
    beam_match = torch.equal(beam.semantic_ids.cpu(), reference["beam_sids"])
    if not logits_close or not beam_match:
        raise RuntimeError(
            "legacy v1 device alignment failed: "
            f"logits_close={logits_close} beam_match={beam_match}"
        )
    print(
        f"OK protocol=legacy_v1 device={device} logits_close={logits_close} "
        f"max_abs={max_abs:.6e} loss_delta={loss_delta:.6e} "
        f"beam_match={beam_match}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timed-steps", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    reference = torch.load(
        args.reference, map_location="cpu", weights_only=True
    )
    if reference.get("protocol") is None:
        return compare_legacy_v1(args, reference, device)
    if reference.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise RuntimeError("unsupported v2 reference schema")
    if reference.get("protocol") != REFERENCE_PROTOCOL:
        raise RuntimeError("reference is not an OxygenREC-v2 Full artifact")
    if args.output is None:
        raise ValueError("--output is required for the v2 comparison protocol")

    checkpoint_hash = file_sha256(args.checkpoint)
    registry_hash = file_sha256(args.sid_registry)
    if checkpoint_hash != reference["checkpoint_sha256"]:
        raise RuntimeError("checkpoint SHA-256 does not match GPU reference")
    if registry_hash != reference["sid_registry_sha256"]:
        raise RuntimeError("SID registry SHA-256 does not match GPU reference")

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("variant") != "full":
        raise RuntimeError("device comparison requires the Full checkpoint")
    registry = SIDRegistry.from_json(args.sid_registry)
    if registry.version != reference["sid_registry_version"]:
        raise RuntimeError("SID registry version does not match GPU reference")
    config = restored_config(checkpoint)
    batch = move_batch(reference["batch"], device)
    trie = PrefixTrie.from_registry(registry)

    inference_model = load_model(checkpoint, config, device)
    actual = inference_outputs(
        inference_model,
        batch,
        trie,
        beam_width=int(reference["beam_width"]),
        output_items=config.max_target_items,
    )
    performance = measure_forward(
        inference_model,
        batch,
        device,
        warmup_steps=args.warmup_steps,
        timed_steps=args.timed_steps,
    )
    train_model = load_model(checkpoint, config, device)
    train_step = deterministic_train_step(
        train_model,
        batch,
        device,
        learning_rate=float(reference["train_step"]["learning_rate"]),
    )

    expected = reference["inference"]
    checks = {
        "logits": sequence_check(
            expected["logits"], actual["logits"], prefix="logit",
            atol=args.atol, rtol=args.rtol,
        ),
        "loss": scalar_check(
            expected["loss"], actual["loss"], atol=args.atol, rtol=args.rtol
        ),
        "ntp_loss": scalar_check(
            expected["ntp_loss"], actual["ntp_loss"],
            atol=args.atol, rtol=args.rtol,
        ),
        "level_losses": sequence_check(
            expected["level_losses"], actual["level_losses"],
            prefix="level_loss", atol=args.atol, rtol=args.rtol,
        ),
        "greedy_exact": bool(
            torch.equal(expected["greedy_sids"], actual["greedy_sids"])
        ),
        "beam_sids_exact": bool(
            torch.equal(expected["beam_sids"], actual["beam_sids"])
        ),
        "beam_scores": compare_tensor_maps(
            {"beam_scores": expected["beam_scores"]},
            {"beam_scores": actual["beam_scores"]},
            atol=args.atol,
            rtol=args.rtol,
        ),
        "train_loss": scalar_check(
            reference["train_step"]["loss"], train_step["loss"],
            atol=args.atol, rtol=args.rtol,
        ),
        "gradients": compare_tensor_maps(
            reference["train_step"]["gradients"], train_step["gradients"],
            atol=args.atol, rtol=args.rtol,
        ),
        "parameter_deltas": compare_tensor_maps(
            reference["train_step"]["parameter_deltas"],
            train_step["parameter_deltas"], atol=args.atol, rtol=args.rtol,
        ),
        "missing_gradients_exact": (
            reference["train_step"]["missing_gradients"]
            == train_step["missing_gradients"]
        ),
    }
    numerical_passed = bool(
        checks["logits"]["all_close"]
        and checks["loss"]["close"]
        and checks["ntp_loss"]["close"]
        and checks["level_losses"]["all_close"]
        and checks["greedy_exact"]
        and checks["beam_sids_exact"]
        and checks["beam_scores"]["all_close"]
        and checks["train_loss"]["close"]
        and checks["gradients"]["all_close"]
        and checks["parameter_deltas"]["all_close"]
        and checks["missing_gradients_exact"]
    )
    project_root = Path(__file__).resolve().parents[1]
    current_source = git_state(project_root)
    current_source_hashes = source_file_hashes(project_root)
    source_match = bool(
        reference["source"].get("commit")
        and reference["source"].get("commit") == current_source.get("commit")
        and reference["source"].get("tracked_dirty") is False
        and current_source.get("tracked_dirty") is False
    )
    source_files_match = (
        reference.get("source_file_sha256") == current_source_hashes
    )
    passed = numerical_passed and source_match and source_files_match
    report = {
        "schema_version": 1,
        "protocol": "oxygenrec_v2_full_gpu_device_comparison_fp32",
        "passed": passed,
        "device_type": device.type,
        "device_name": device_name(device),
        "dtype": reference["dtype"],
        "atol": args.atol,
        "rtol": args.rtol,
        "reference_source": reference["source"],
        "current_source": current_source,
        "source_commit_match": source_match,
        "source_files_match": source_files_match,
        "source_file_sha256": current_source_hashes,
        "numerical_passed": numerical_passed,
        "checkpoint_sha256": checkpoint_hash,
        "reference_artifact_sha256": file_sha256(args.reference),
        "sid_registry_sha256": registry_hash,
        "batch_metadata": reference["batch_metadata"],
        "checks": checks,
        "performance": performance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError(f"v2 device alignment failed; report={args.output}")
    print(
        "OK stage=v2_device_reference_compare "
        f"device={device} logits_max_abs={checks['logits']['max_abs']:.6e} "
        f"gradient_max_abs={checks['gradients']['max_abs']:.6e} "
        f"delta_max_abs={checks['parameter_deltas']['max_abs']:.6e} "
        f"greedy_exact={checks['greedy_exact']} "
        f"beam_exact={checks['beam_sids_exact']} report={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
