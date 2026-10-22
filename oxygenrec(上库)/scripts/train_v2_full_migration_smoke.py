#!/usr/bin/env python3
"""Run a short v2 Full training/save/reload migration check on one device."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.data import (
    Split,
    TemporalBoundaries,
    build_daily_listwise_samples,
    build_listwise_sid_model_batch,
    load_retailrocket_events,
)
from oxygenrec.device import (
    device_name,
    max_memory_allocated,
    reset_peak_memory_stats,
    resolve_device,
    seed_torch,
    synchronize,
)
from oxygenrec.migration_alignment import (
    file_sha256,
    git_state,
    load_model,
    restored_config,
)
from oxygenrec.model import OxygenRECModel
from oxygenrec.sid import SIDRegistry


BEHAVIOR_WEIGHTS = (1.2, 1.5, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("gpu", "npu"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate", type=float)
    return parser.parse_args()


def require_inputs(args: argparse.Namespace) -> None:
    missing = [
        str(path)
        for path in (args.events, args.checkpoint, args.sid_registry)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing migration training inputs: " + ", ".join(missing)
        )


def make_full_batch(samples, registry, config, device) -> dict[str, torch.Tensor]:
    raw = build_listwise_sid_model_batch(
        samples, registry, max_history_items=config.max_history_items
    )
    behavior_ids = torch.tensor(
        raw.target_behavior_ids, dtype=torch.long, device=device
    )
    weights = torch.tensor(BEHAVIOR_WEIGHTS, dtype=torch.float32, device=device)
    sid_tokens = config.max_target_items * config.sid_levels
    return {
        "history_sids": torch.tensor(raw.history_sids, dtype=torch.long, device=device),
        "history_padding_mask": torch.tensor(
            raw.history_padding_mask, dtype=torch.bool, device=device
        ),
        "history_behavior_ids": torch.tensor(
            raw.history_behavior_ids, dtype=torch.long, device=device
        ),
        "target_sids": torch.tensor(raw.target_sids, dtype=torch.long, device=device),
        "behavior_instruction_ids": behavior_ids,
        "token_weights": weights[behavior_ids]
        .unsqueeze(1)
        .expand(-1, sid_tokens),
    }


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [cpu_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(child) for child in value)
    return value


def gradient_metrics(model: OxygenRECModel) -> tuple[int, float, float]:
    tensors = 0
    squared_l2 = 0.0
    maximum = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("training produced non-finite gradients")
        tensors += 1
        squared_l2 += float((gradient.float() ** 2).sum())
        maximum = max(maximum, float(gradient.abs().max()))
    if not tensors:
        raise RuntimeError("training produced no gradients")
    return tensors, math.sqrt(squared_l2), maximum


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")
    require_inputs(args)
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
            "migration training requires a known commit and no tracked-file changes"
        )

    original = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if original.get("variant") != "full":
        raise ValueError("migration training requires a v2 Full checkpoint")
    registry = SIDRegistry.from_json(args.sid_registry)
    if original.get("sid_registry_version") != registry.version:
        raise RuntimeError("checkpoint and SID registry versions differ")
    config = restored_config(original)
    checkpoint_args = original.get("args", {})
    seed = args.seed if args.seed is not None else int(checkpoint_args.get("seed", 17))
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else float(checkpoint_args.get("learning_rate", 3e-4))
    )
    max_train_samples = (
        args.max_train_samples
        if args.max_train_samples is not None
        else int(checkpoint_args.get("max_train_samples", 5_000))
    )
    if learning_rate <= 0 or max_train_samples < 1:
        raise ValueError("learning-rate and max-train-samples must be positive")

    print("stage=build_v2_full_migration_training_cohort")
    boundaries = TemporalBoundaries(**original["boundaries"])
    events = [
        event
        for event in load_retailrocket_events(args.events)
        if event.item_id in registry.item_to_sid
    ]
    rows = build_daily_listwise_samples(
        events,
        boundaries,
        list_size=config.max_target_items,
        max_history=config.max_history_items,
        max_samples_per_split={
            Split.TRAIN: max_train_samples,
            Split.VALIDATION: min(8, max_train_samples),
            Split.TEST: 1,
        },
        sample_seed=seed,
    )
    by_split = defaultdict(list)
    for row in rows:
        by_split[row.split].append(row)
    train_samples = by_split[Split.TRAIN]
    if not train_samples:
        raise RuntimeError("migration training cohort is empty")
    order = list(range(len(train_samples)))
    random.Random(seed).shuffle(order)
    ordered = [train_samples[index] for index in order]

    seed_torch(seed, device)
    model = load_model(original, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer_state_loaded = False
    if original.get("optimizer_state"):
        optimizer.load_state_dict(original["optimizer_state"])
        optimizer_state_loaded = True

    print(
        "stage=train_v2_full_migration_smoke "
        f"platform={args.platform} device={device} precision={args.precision} "
        f"steps={args.steps}"
    )
    model.train()
    reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    losses: list[float] = []
    gradient_l2: list[float] = []
    gradient_max_abs: list[float] = []
    gradient_tensors = 0
    cursor = 0
    for _ in range(args.steps):
        if cursor >= len(ordered):
            random.Random(seed + len(losses) + 1).shuffle(ordered)
            cursor = 0
        sample_batch = ordered[cursor : cursor + args.batch_size]
        cursor += len(sample_batch)
        batch = make_full_batch(sample_batch, registry, config, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.precision):
            output = model(**batch)
        if not bool(torch.isfinite(output.loss)):
            raise RuntimeError("training produced a non-finite loss")
        output.loss.backward()
        gradient_tensors, l2, maximum = gradient_metrics(model)
        optimizer.step()
        losses.append(float(output.loss.detach()))
        gradient_l2.append(l2)
        gradient_max_abs.append(maximum)
    synchronize(device)
    elapsed = time.perf_counter() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / (
        f"v2_full_{args.platform}_{args.precision}_step{args.steps}.pt"
    )
    trained_state = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    saved = {
        "variant": "full",
        "migration_training": True,
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "steps": args.steps,
        "precision": args.precision,
        "model_config": asdict(config),
        "model_state": trained_state,
        "optimizer_state": cpu_tree(optimizer.state_dict()),
        "sid_registry_version": registry.version,
        "boundaries": asdict(boundaries),
        "args": {
            "seed": seed,
            "learning_rate": learning_rate,
            "batch_size": args.batch_size,
            "max_train_samples": max_train_samples,
        },
    }
    torch.save(saved, checkpoint_path)

    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reload_model = OxygenRECModel(config)
    reload_model.load_state_dict(reloaded["model_state"], strict=True)
    checkpoint_reload_match = all(
        torch.equal(trained_state[name], value)
        for name, value in reload_model.state_dict().items()
    )
    reload_model = reload_model.to(device)
    reload_optimizer = torch.optim.AdamW(reload_model.parameters(), lr=learning_rate)
    reload_optimizer.load_state_dict(reloaded["optimizer_state"])
    # Keep train mode so this reload check exercises the migrated training path rather
    # than PyTorch's eval-only Transformer fast path, which may use a different kernel.
    reload_model.train()
    reload_batch = make_full_batch(
        ordered[: min(args.batch_size, len(ordered))], registry, config, device
    )
    with torch.inference_mode(), autocast_context(device, args.precision):
        reload_loss = reload_model(**reload_batch).loss
    reload_loss_finite = bool(torch.isfinite(reload_loss))
    if not checkpoint_reload_match or not reload_loss_finite:
        raise RuntimeError("saved checkpoint did not reload into a usable model")

    summary = {
        "schema_version": 1,
        "protocol": "oxygenrec_v2_full_migration_training_smoke",
        "platform": args.platform,
        "device": str(device),
        "device_name": device_name(device),
        "precision": args.precision,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "torch_npu": package_version("torch-npu"),
        },
        "source": source,
        "inputs": {
            "events_sha256": file_sha256(args.events),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "sid_registry_sha256": file_sha256(args.sid_registry),
            "sid_registry_version": registry.version,
        },
        "training": {
            "seed": seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "train_samples": len(train_samples),
            "learning_rate": learning_rate,
            "optimizer": "AdamW",
            "optimizer_state_loaded": optimizer_state_loaded,
            "autocast_enabled": args.precision == "bf16",
            "autocast_dtype": "bfloat16" if args.precision == "bf16" else None,
            "losses": losses,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "minimum_loss": min(losses),
            "all_losses_finite": all(math.isfinite(value) for value in losses),
            "gradient_tensors": gradient_tensors,
            "gradient_l2_first": gradient_l2[0],
            "gradient_l2_last": gradient_l2[-1],
            "gradient_max_abs": max(gradient_max_abs),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "reload_state_match": checkpoint_reload_match,
            "reload_forward_loss": float(reload_loss),
            "reload_forward_finite": reload_loss_finite,
        },
        "performance": {
            "scope": "short_training_only",
            "elapsed_seconds": elapsed,
            "mean_step_seconds": elapsed / args.steps,
            "steps_per_second": args.steps / elapsed,
            "peak_memory_allocated_bytes": max_memory_allocated(device),
        },
        "status": "passed",
        "scope_note": (
            "This validates short single-card training and checkpoint reload; it is "
            "not final accuracy alignment or performance tuning."
        ),
    }
    summary_path = args.output_dir / (
        f"v2_full_{args.platform}_{args.precision}_training_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "OK stage=v2_full_migration_training "
        f"platform={args.platform} device={device} precision={args.precision} "
        f"steps={args.steps} first_loss={losses[0]:.6f} "
        f"last_loss={losses[-1]:.6f} checkpoint_reload_match={checkpoint_reload_match} "
        f"summary={summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
