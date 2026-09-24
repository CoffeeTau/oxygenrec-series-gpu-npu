#!/usr/bin/env python3
"""Measure reproducible single-device v2 Full BF16 training throughput.

The timed loop intentionally follows the production trainer, including one host
loss read per step.  It does not save trained weights and is not a quality run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.data import Split, TemporalBoundaries, build_daily_listwise_samples
from oxygenrec.data.events import load_retailrocket_events
from oxygenrec.device import (
    device_name,
    max_memory_allocated,
    reset_peak_memory_stats,
    resolve_device,
    seed_torch,
    synchronize,
)
from oxygenrec.migration_alignment import file_sha256, git_state, load_model, restored_config
from oxygenrec.sid import SIDRegistry
from train_v2_full_migration_smoke import make_full_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("gpu", "npu"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-samples", type=int, default=50_000)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(64, 128, 256))
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measured-steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--npu-profile-dir",
        type=Path,
        help="collect a short Ascend PyTorch Profiler trace instead of a formal baseline",
    )
    parser.add_argument("--profile-warmup-steps", type=int, default=1)
    parser.add_argument("--profile-active-steps", type=int, default=3)
    parser.add_argument(
        "--cycle-samples",
        action="store_true",
        help=(
            "repeat the shuffled training order outside the timed loop when the "
            "requested warmup and measured steps exceed the available split"
        ),
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "npu_fused_adamw"),
        default="adamw",
        help="optimizer implementation; the fused variant is NPU-only",
    )
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def aggregate_runs(runs: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    aggregates = []
    for batch_size in sorted({int(run["batch_size"]) for run in runs}):
        group = [run for run in runs if run["batch_size"] == batch_size]
        throughputs = [float(run["samples_per_second"]) for run in group]
        step_times = [float(run["mean_step_seconds"]) for run in group]
        mean_throughput = statistics.mean(throughputs)
        aggregates.append({
            "batch_size": batch_size,
            "repeats": len(group),
            "samples_per_second_median": statistics.median(throughputs),
            "samples_per_second_mean": mean_throughput,
            "samples_per_second_min": min(throughputs),
            "samples_per_second_max": max(throughputs),
            "samples_per_second_cv": (
                statistics.pstdev(throughputs) / mean_throughput
                if len(throughputs) > 1 and mean_throughput else 0.0
            ),
            "mean_step_seconds_median": statistics.median(step_times),
            "peak_memory_allocated_bytes_max": max(
                int(run["peak_memory_allocated_bytes"] or 0) for run in group
            ),
        })
    return aggregates


def build_optimizer(name: str, parameters, learning_rate: float):
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate), "AdamW"
    if name == "npu_fused_adamw":
        from torch_npu.optim import NpuFusedAdamW

        return NpuFusedAdamW(parameters, lr=learning_rate), "NpuFusedAdamW"
    raise ValueError(f"unsupported optimizer: {name}")


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.batch_sizes) < 1 or len(set(args.batch_sizes)) != len(args.batch_sizes):
        raise ValueError("batch-sizes must be unique positive integers")
    if args.max_train_samples < 1 or args.warmup_steps < 0:
        raise ValueError("max-train-samples must be positive and warmup-steps nonnegative")
    if args.measured_steps < 1 or args.repeats < 1:
        raise ValueError("measured-steps and repeats must be positive")
    if args.profile_warmup_steps < 0 or args.profile_active_steps < 1:
        raise ValueError("profile warmup must be nonnegative and active steps must be positive")
    if args.optimizer == "npu_fused_adamw" and args.platform != "npu":
        raise ValueError("npu_fused_adamw requires --platform npu")
    if args.npu_profile_dir is not None:
        if args.platform != "npu":
            raise ValueError("--npu-profile-dir requires --platform npu")
        if len(args.batch_sizes) != 1 or args.repeats != 1:
            raise ValueError("NPU profiling requires exactly one batch size and one repeat")
        profile_steps = args.profile_warmup_steps + args.profile_active_steps
        if args.measured_steps != profile_steps:
            raise ValueError(
                "measured-steps must equal profile-warmup-steps + profile-active-steps"
            )
        if args.npu_profile_dir.exists():
            raise FileExistsError(args.npu_profile_dir)
    for path in (args.events, args.checkpoint, args.sid_registry):
        if not path.is_file():
            raise FileNotFoundError(path)

    project_root = Path(__file__).resolve().parents[1]
    source = git_state(project_root)
    if source["commit"] is None or source["tracked_dirty"]:
        raise RuntimeError("performance baseline requires a known commit and clean tracked files")
    device = resolve_device(args.device)
    expected_type = "cuda" if args.platform == "gpu" else "npu"
    if device.type != expected_type:
        raise ValueError(f"platform {args.platform!r} requires {expected_type!r}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != "full":
        raise ValueError("performance baseline requires a v2 Full checkpoint")
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

    build_started = time.perf_counter()
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
    build_seconds = time.perf_counter() - build_started
    required = max(args.batch_sizes) * (args.warmup_steps + args.measured_steps)
    if len(train_samples) < max(args.batch_sizes):
        raise ValueError(
            f"need at least {max(args.batch_sizes)} train samples for one largest batch; "
            f"found {len(train_samples)}"
        )
    if len(train_samples) < required and not args.cycle_samples:
        raise ValueError(
            f"need at least {required} train samples for the requested largest batch; "
            f"found {len(train_samples)}; pass --cycle-samples for a performance-only "
            "run that repeats the shuffled sample order"
        )
    behavior_counts = Counter(row.target_behavior.value for row in train_samples)
    order = list(range(len(train_samples)))
    random.Random(seed + epoch).shuffle(order)
    if args.cycle_samples and len(order) < required:
        repetitions = math.ceil(required / len(order))
        benchmark_order = (order * repetitions)[:required]
    else:
        benchmark_order = order

    runs = []
    optimizer_name = None
    for batch_size in args.batch_sizes:
        for repeat in range(1, args.repeats + 1):
            seed_torch(seed, device)
            model = load_model(checkpoint, config, device)
            optimizer, optimizer_name = build_optimizer(
                args.optimizer,
                model.parameters(),
                learning_rate,
            )
            optimizer_loaded = bool(checkpoint.get("optimizer_state"))
            if optimizer_loaded:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            model.train()

            def step(step_index: int) -> float:
                start = step_index * batch_size
                selected = [
                    train_samples[index]
                    for index in benchmark_order[start:start + batch_size]
                ]
                batch = make_full_batch(selected, registry, config, device)
                optimizer.zero_grad(set_to_none=True)
                context = (
                    nullcontext() if args.precision == "fp32"
                    else torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                )
                with context:
                    output = model(**batch)
                loss = float(output.loss.detach())
                if not math.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite loss for batch={batch_size} repeat={repeat} step={step_index}"
                    )
                output.loss.backward()
                optimizer.step()
                return loss

            for warmup_step in range(args.warmup_steps):
                step(warmup_step)
            synchronize(device)
            reset_peak_memory_stats(device)
            synchronize(device)
            losses = []
            started = time.perf_counter()
            if args.npu_profile_dir is None:
                for measured_step in range(args.measured_steps):
                    losses.append(step(args.warmup_steps + measured_step))
            else:
                import torch_npu.profiler as npu_profiler

                with npu_profiler.profile(
                    activities=[
                        npu_profiler.ProfilerActivity.CPU,
                        npu_profiler.ProfilerActivity.NPU,
                    ],
                    schedule=npu_profiler.schedule(
                        wait=0,
                        warmup=args.profile_warmup_steps,
                        active=args.profile_active_steps,
                        repeat=1,
                    ),
                    on_trace_ready=npu_profiler.tensorboard_trace_handler(
                        str(args.npu_profile_dir)
                    ),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                    with_modules=True,
                ) as profiler:
                    for measured_step in range(args.measured_steps):
                        losses.append(step(args.warmup_steps + measured_step))
                        profiler.step()
            synchronize(device)
            elapsed = time.perf_counter() - started
            memory = max_memory_allocated(device)
            run = {
                "batch_size": batch_size,
                "repeat": repeat,
                "warmup_steps": args.warmup_steps,
                "measured_steps": args.measured_steps,
                "elapsed_seconds": elapsed,
                "mean_step_seconds": elapsed / args.measured_steps,
                "steps_per_second": args.measured_steps / elapsed,
                "samples_per_second": args.measured_steps * batch_size / elapsed,
                "peak_memory_allocated_bytes": memory,
                "first_loss": losses[0],
                "last_loss": losses[-1],
                "mean_loss": statistics.mean(losses),
                "all_losses_finite": True,
            }
            runs.append(run)
            print(
                f"platform={args.platform} precision={args.precision} "
                f"batch={batch_size} repeat={repeat}/{args.repeats} "
                f"samples_per_second={run['samples_per_second']:.3f} "
                f"step_ms={1000 * run['mean_step_seconds']:.3f}",
                flush=True,
            )

    profiling = args.npu_profile_dir is not None
    payload = {
        "schema_version": 1,
        "protocol": (
            "oxygenrec_v2_single_device_training_profile_v1"
            if profiling
            else "oxygenrec_v2_single_device_training_performance_v1"
        ),
        "scope": (
            "operator_profile_not_throughput_baseline_or_quality_training"
            if profiling
            else "steady_state_training_baseline_not_quality_training"
        ),
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
        "source_files_sha256": {
            name: file_sha256(project_root / name)
            for name in (
                "scripts/benchmark_v2_training.py",
                "scripts/train_v2_full_migration_smoke.py",
                "src/oxygenrec/model.py",
                "src/oxygenrec/device.py",
                "src/oxygenrec/migration_alignment.py",
                "src/oxygenrec/data/model_inputs.py",
                "src/oxygenrec/data/temporal.py",
            )
        },
        "inputs": {
            "events_sha256": file_sha256(args.events),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "sid_registry_sha256": file_sha256(args.sid_registry),
            "sid_registry_version": registry.version,
        },
        "workload": {
            "seed": seed,
            "epoch": epoch,
            "train_samples": len(train_samples),
            "max_train_samples": args.max_train_samples,
            "batch_sizes": args.batch_sizes,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "repeats": args.repeats,
            "cycle_samples": args.cycle_samples,
            "benchmark_order_samples": len(benchmark_order),
            "learning_rate": learning_rate,
            "optimizer": optimizer_name,
            "optimizer_state_loaded": bool(checkpoint.get("optimizer_state")),
            "loss_host_read_interval_steps": 1,
            "model_mode": "train",
            "configured_dropout": config.dropout,
            "behavior_counts": dict(sorted(behavior_counts.items())),
        },
        "profiling": ({
            "trace_dir": str(args.npu_profile_dir),
            "schedule": {
                "wait": 0,
                "warmup": args.profile_warmup_steps,
                "active": args.profile_active_steps,
                "repeat": 1,
            },
            "activities": ["CPU", "NPU"],
            "record_shapes": True,
            "profile_memory": True,
            "with_stack": False,
            "with_modules": True,
            "timing_includes_profiler_overhead": True,
        } if profiling else None),
        "data_build_seconds_once": build_seconds,
        "runs": runs,
        "aggregates": aggregate_runs(runs),
        "status": "passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stage = "training_profile" if profiling else "performance_baseline"
    print(f"OK {stage} output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
