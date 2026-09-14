#!/usr/bin/env python3
"""Run the same compact OxygenREC-v2 alignment probe on GPU or NPU."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.device import device_name, resolve_device
from oxygenrec.migration_alignment import (
    REFERENCE_PROTOCOL,
    REFERENCE_SCHEMA_VERSION,
    build_fixed_batch,
    deterministic_train_step,
    file_sha256,
    git_state,
    inference_outputs,
    load_model,
    measure_forward,
    move_batch,
    restored_config,
    source_file_hashes,
)
from oxygenrec.sid import PrefixTrie, SIDRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("gpu", "npu"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timed-steps", type=int, default=10)
    return parser.parse_args()


def tensor_values(tensor: torch.Tensor) -> Any:
    return tensor.detach().cpu().tolist()


def tensor_summary(tensor: torch.Tensor, sample_count: int = 16) -> dict[str, Any]:
    value = tensor.detach().cpu().to(torch.float64).reshape(-1)
    numel = value.numel()
    if numel == 0:
        samples: list[dict[str, float | int]] = []
        minimum = maximum = mean = std = l1 = l2 = 0.0
    else:
        indexes = sorted(
            {round(position * (numel - 1) / max(sample_count - 1, 1))
             for position in range(min(sample_count, numel))}
        )
        samples = [
            {"index": index, "value": float(value[index])} for index in indexes
        ]
        minimum = float(value.min())
        maximum = float(value.max())
        mean = float(value.mean())
        std = float(value.std(unbiased=False))
        l1 = float(value.abs().sum())
        l2 = float(torch.linalg.vector_norm(value))
    return {
        "shape": list(tensor.shape),
        "numel": numel,
        "finite": bool(torch.isfinite(value).all()),
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": std,
        "l1": l1,
        "l2": l2,
        "samples": samples,
    }


def main() -> int:
    args = parse_args()
    if args.samples < 1 or args.beam_width < 1:
        raise ValueError("samples and beam-width must be positive")
    device = resolve_device(args.device)
    expected_type = "cuda" if args.platform == "gpu" else "npu"
    if device.type != expected_type:
        raise ValueError(
            f"platform {args.platform!r} requires a {expected_type!r} device"
        )

    project_root = Path(__file__).resolve().parents[1]
    source = git_state(project_root)
    if source["commit"] is None or source["tracked_dirty"]:
        raise RuntimeError(
            "device probe requires a known commit and no tracked-file changes"
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != "full":
        raise ValueError("v2 device probe requires the Full checkpoint")
    registry = SIDRegistry.from_json(args.sid_registry)
    if checkpoint.get("sid_registry_version") != registry.version:
        raise RuntimeError("checkpoint and SID registry versions differ")
    config = restored_config(checkpoint)
    cpu_batch, batch_metadata = build_fixed_batch(
        args.events, checkpoint, registry, samples=args.samples
    )
    batch = move_batch(cpu_batch, device)
    trie = PrefixTrie.from_registry(registry)
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = float(checkpoint.get("args", {}).get("learning_rate", 3e-4))

    inference_model = load_model(checkpoint, config, device)
    inference = inference_outputs(
        inference_model,
        batch,
        trie,
        beam_width=args.beam_width,
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
        train_model, batch, device, learning_rate=learning_rate
    )

    payload = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "protocol": REFERENCE_PROTOCOL,
        "probe_kind": "compact_independent_device_probe",
        "platform": args.platform,
        "device_type": device.type,
        "device_name": device_name(device),
        "dtype": "float32",
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_npu": (
                getattr(importlib.import_module("torch_npu"), "__version__", None)
                if device.type == "npu"
                else None
            ),
        },
        "source": source,
        "source_file_sha256": source_file_hashes(project_root),
        "inputs": {
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "events_sha256": file_sha256(args.events),
            "sid_registry_sha256": file_sha256(args.sid_registry),
            "sid_registry_version": registry.version,
        },
        "model_config": checkpoint["model_config"],
        "batch_metadata": batch_metadata,
        "batch": {name: tensor_values(value) for name, value in cpu_batch.items()},
        "beam_width": args.beam_width,
        "inference": {
            "loss": float(inference["loss"]),
            "ntp_loss": float(inference["ntp_loss"]),
            "level_losses": [float(value) for value in inference["level_losses"]],
            "logits": [tensor_values(value) for value in inference["logits"]],
            "greedy_sids": tensor_values(inference["greedy_sids"]),
            "beam_sids": tensor_values(inference["beam_sids"]),
            "beam_scores": tensor_values(inference["beam_scores"]),
        },
        "train_step": {
            "learning_rate": train_step["learning_rate"],
            "loss": float(train_step["loss"]),
            "gradients": {
                name: tensor_summary(value)
                for name, value in train_step["gradients"].items()
            },
            "missing_gradients": train_step["missing_gradients"],
            "parameter_deltas": {
                name: tensor_summary(value)
                for name, value in train_step["parameter_deltas"].items()
            },
            "max_parameter_delta": train_step["max_parameter_delta"],
            "comparison_scope": "per_tensor_statistics_and_fixed_samples",
        },
        "performance": performance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "OK stage=v2_device_probe "
        f"platform={args.platform} device={device} samples={args.samples} "
        f"loss={float(inference['loss']):.6f} "
        f"commit={source['commit']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
