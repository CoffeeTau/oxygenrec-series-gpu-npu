#!/usr/bin/env python3
"""Compare a few matched v2 Full training steps without writing a checkpoint.

This is a numerical diagnostic, not a replacement for full-epoch training.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.data import Split, TemporalBoundaries, build_daily_listwise_samples
from oxygenrec.data.events import load_retailrocket_events
from oxygenrec.device import resolve_device, seed_torch
from oxygenrec.migration_alignment import file_sha256, git_state, load_model, restored_config
from oxygenrec.sid import SIDRegistry
from train_v2_full_migration_smoke import gradient_metrics, make_full_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("gpu", "npu"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), required=True)
    parser.add_argument("--dropout-mode", choices=("train", "disabled"), required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float)
    return parser.parse_args()


def batch_sha256(batch: dict[str, torch.Tensor]) -> str:
    """Fingerprint actual CPU batch tensors, independent of accelerator kernels."""

    digest = hashlib.sha256()
    for name, tensor in sorted(batch.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(json.dumps(tensor.tolist(), separators=(",", ":")).encode("ascii"))
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    args = parse_args()
    if min(args.steps, args.batch_size, args.max_train_samples) < 1:
        raise ValueError("steps, batch-size and max-train-samples must be positive")
    for path in (args.events, args.checkpoint, args.sid_registry):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)

    device = resolve_device(args.device)
    expected_type = "cuda" if args.platform == "gpu" else "npu"
    if device.type != expected_type:
        raise ValueError(f"platform {args.platform!r} requires {expected_type!r}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != "full":
        raise ValueError("step diagnostic requires a v2 Full checkpoint")
    registry = SIDRegistry.from_json(args.sid_registry)
    if registry.version != checkpoint.get("sid_registry_version"):
        raise ValueError("checkpoint and SID registry versions differ")
    config = restored_config(checkpoint)
    boundaries = TemporalBoundaries(**checkpoint["boundaries"])
    original_args = checkpoint.get("args", {})
    prior_run = checkpoint.get("full_training", {})
    seed = int(original_args.get("seed", 17))
    epoch = int(prior_run.get("completed_epochs", 0)) + 1
    learning_rate = (
        args.learning_rate if args.learning_rate is not None
        else float(prior_run.get("learning_rate", original_args.get("learning_rate", 3e-4)))
    )
    if learning_rate <= 0:
        raise ValueError("learning-rate must be positive")

    events = [
        event for event in load_retailrocket_events(args.events)
        if event.item_id in registry.item_to_sid
    ]
    rows = build_daily_listwise_samples(
        events,
        boundaries,
        list_size=config.max_target_items,
        max_history=config.max_history_items,
        max_samples_per_split={
            Split.TRAIN: args.max_train_samples,
            Split.VALIDATION: 1,
            Split.TEST: 1,
        },
        sample_seed=seed,
    )
    train_samples = [row for row in rows if row.split is Split.TRAIN]
    if len(train_samples) < args.batch_size:
        raise ValueError("not enough train samples for one full batch")
    order = list(range(len(train_samples)))
    random.Random(seed + epoch).shuffle(order)

    seed_torch(seed, device)
    model = load_model(checkpoint, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer_loaded = bool(checkpoint.get("optimizer_state"))
    if optimizer_loaded:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    # eval() disables dropout while keeping autograd and optimizer updates active.
    # It is diagnostic only; production training continues to use train().
    if args.dropout_mode == "disabled":
        model.eval()
    else:
        model.train()

    records = []
    for step, start in enumerate(range(0, len(order), args.batch_size), start=1):
        if step > args.steps:
            break
        selected = [train_samples[index] for index in order[start:start + args.batch_size]]
        cpu_batch = make_full_batch(selected, registry, config, torch.device("cpu"))
        fingerprint = batch_sha256(cpu_batch)
        batch = {name: tensor.to(device) for name, tensor in cpu_batch.items()}
        optimizer.zero_grad(set_to_none=True)
        context = (
            nullcontext() if args.precision == "fp32"
            else torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        )
        with context:
            output = model(**batch)
        loss = float(output.loss.detach())
        if not math.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        output.loss.backward()
        gradient_tensors, gradient_l2, gradient_max_abs = gradient_metrics(model)
        optimizer.step()
        records.append({
            "step": step,
            "batch_sha256": fingerprint,
            "loss_before_update": loss,
            "gradient_tensors": gradient_tensors,
            "gradient_l2": gradient_l2,
            "gradient_max_abs": gradient_max_abs,
        })
        print(f"step={step} loss={loss:.9f} batch_sha256={fingerprint[:12]}", flush=True)

    project_root = Path(__file__).resolve().parents[1]
    payload = {
        "protocol": "oxygenrec_v2_matched_step_diagnostic_v1",
        "platform": args.platform,
        "device": str(device),
        "precision": args.precision,
        "dropout_mode": args.dropout_mode,
        "configured_dropout": config.dropout,
        "source": git_state(project_root),
        "source_files_sha256": {
            name: file_sha256(project_root / name)
            for name in (
                "scripts/diagnose_v2_step_loss.py",
                "scripts/train_v2_full_epochs.py",
                "scripts/train_v2_full_migration_smoke.py",
                "src/oxygenrec/model.py",
                "src/oxygenrec/migration_alignment.py",
                "src/oxygenrec/data/temporal.py",
                "src/oxygenrec/data/model_inputs.py",
            )
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "torch_npu": package_version("torch-npu"),
        },
        "inputs": {
            "events_sha256": file_sha256(args.events),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "sid_registry_sha256": file_sha256(args.sid_registry),
        },
        "seed": seed,
        "epoch": epoch,
        "train_samples": len(train_samples),
        "batch_size": args.batch_size,
        "learning_rate": learning_rate,
        "optimizer_state_loaded": optimizer_loaded,
        "steps": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"OK step_diagnostic output={args.output} steps={len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
