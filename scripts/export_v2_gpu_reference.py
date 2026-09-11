#!/usr/bin/env python3
"""Export a frozen OxygenREC-v2 Full FP32 GPU reference artifact."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.data import (
    Split,
    TemporalBoundaries,
    build_daily_listwise_samples,
    build_listwise_sid_model_batch,
    load_retailrocket_events,
)
from oxygenrec.device import device_name, resolve_device
from oxygenrec.migration_alignment import (
    REFERENCE_PROTOCOL,
    REFERENCE_SCHEMA_VERSION,
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


BEHAVIOR_WEIGHTS = (1.2, 1.5, 2.0)


def git_state(project_root: Path) -> dict[str, object]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments], cwd=project_root, check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()

    try:
        revision = run("rev-parse", "HEAD")
        status = run("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": revision, "dirty": bool(status)}


def build_fixed_batch(
    events_path: Path,
    checkpoint: dict,
    registry: SIDRegistry,
    *,
    samples: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    config = restored_config(checkpoint)
    checkpoint_args = checkpoint.get("args", {})
    seed = int(checkpoint_args.get("seed", 17))
    train_limit = int(checkpoint_args.get("max_train_samples", 5_000))
    boundaries = TemporalBoundaries(**checkpoint["boundaries"])
    events = [
        event for event in load_retailrocket_events(events_path)
        if event.item_id in registry.item_to_sid
    ]
    rows = build_daily_listwise_samples(
        events,
        boundaries,
        list_size=config.max_target_items,
        max_history=config.max_history_items,
        max_samples_per_split={
            Split.TRAIN: train_limit,
            Split.VALIDATION: samples,
            Split.TEST: 1,
        },
        sample_seed=seed,
    )
    by_split = defaultdict(list)
    for row in rows:
        by_split[row.split].append(row)
    validation = by_split[Split.VALIDATION]
    if len(validation) != samples:
        raise RuntimeError(f"expected {samples} validation lists, got {len(validation)}")
    raw = build_listwise_sid_model_batch(
        validation, registry, max_history_items=config.max_history_items
    )
    behavior_ids = torch.tensor(raw.target_behavior_ids, dtype=torch.long)
    sid_tokens = config.max_target_items * config.sid_levels
    behavior_weights = torch.tensor(BEHAVIOR_WEIGHTS, dtype=torch.float32)
    batch = {
        "history_sids": torch.tensor(raw.history_sids, dtype=torch.long),
        "history_padding_mask": torch.tensor(
            raw.history_padding_mask, dtype=torch.bool
        ),
        "history_behavior_ids": torch.tensor(
            raw.history_behavior_ids, dtype=torch.long
        ),
        "target_sids": torch.tensor(raw.target_sids, dtype=torch.long),
        "behavior_instruction_ids": behavior_ids,
        "token_weights": behavior_weights[behavior_ids].unsqueeze(1).expand(
            -1, sid_tokens
        ).clone(),
    }
    metadata = {
        "sample_seed": seed,
        "validation_samples": samples,
        "list_size": config.max_target_items,
        "max_history_items": config.max_history_items,
        "boundaries": asdict(boundaries),
    }
    return batch, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timed-steps", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples < 1 or args.beam_width < 1:
        raise ValueError("samples and beam-width must be positive")
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise ValueError("the reference must be exported on a CUDA device")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != "full":
        raise ValueError("v2 migration reference requires the Full checkpoint")
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

    project_root = Path(__file__).resolve().parents[1]
    source = git_state(project_root)
    if source["commit"] is None or source["dirty"]:
        raise RuntimeError(
            "GPU reference export requires a clean Git worktree with a known commit"
        )
    payload = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "protocol": REFERENCE_PROTOCOL,
        "device_type": device.type,
        "device_name": device_name(device),
        "dtype": "float32",
        "source": source,
        "source_file_sha256": source_file_hashes(project_root),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "events_sha256": file_sha256(args.events),
        "sid_registry_sha256": file_sha256(args.sid_registry),
        "sid_registry_version": registry.version,
        "model_config": checkpoint["model_config"],
        "batch_metadata": batch_metadata,
        "batch": cpu_batch,
        "beam_width": args.beam_width,
        "inference": inference,
        "train_step": train_step,
        "performance": performance,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / "v2_full_gpu_reference.pt"
    summary_path = args.output_dir / "v2_full_gpu_reference.json"
    torch.save(payload, artifact_path)
    artifact_hash = file_sha256(artifact_path)
    summary = {
        key: value
        for key, value in payload.items()
        if key not in {"batch", "inference", "train_step"}
    }
    summary.update({
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_hash,
        "loss": float(inference["loss"]),
        "logit_shapes": [list(tensor.shape) for tensor in inference["logits"]],
        "greedy_shape": list(inference["greedy_sids"].shape),
        "beam_shape": list(inference["beam_sids"].shape),
        "gradient_tensors": len(train_step["gradients"]),
        "missing_gradients": train_step["missing_gradients"],
        "max_parameter_delta": train_step["max_parameter_delta"],
    })
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "OK stage=v2_gpu_reference "
        f"device={device} samples={args.samples} loss={float(inference['loss']):.6f} "
        f"logit_steps={len(inference['logits'])} "
        f"greedy_shape={tuple(inference['greedy_sids'].shape)} "
        f"beam_shape={tuple(inference['beam_sids'].shape)} "
        f"checkpoint_sha256={payload['checkpoint_sha256']} "
        f"artifact_sha256={artifact_hash} output={artifact_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
