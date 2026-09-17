#!/usr/bin/env python3
"""在EA-TOSD同一真实cohort上执行等步数SFT-only配对对照。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.data import (
    Split,
    TemporalBoundaries,
    build_daily_listwise_samples,
    load_retailrocket_events,
)
from oxygenrec.ea_tosd import load_pretrained_with_future_positions
from oxygenrec.model import OxygenRECModel
from oxygenrec.sid import PrefixTrie, SIDRegistry
from train_v2_ea_tosd_retailrocket import (
    BEHAVIOR_BY_NAME,
    common_inputs,
    make_batch,
    restored_config,
    total_gradient,
)
from train_v2_listwise_retailrocket import chunks, evaluate


PAIRED_METRICS = (
    "sid_recall",
    "sid_token_accuracy",
    "geometric_token_reward",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ea-summary", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/retailrocket_v2_sft_control_smoke"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--list-size", type=int, default=2)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--max-future-items", type=int, default=2)
    parser.add_argument(
        "--minimum-future-behavior",
        choices=tuple(BEHAVIOR_BY_NAME),
        default="view",
    )
    parser.add_argument("--max-train-samples", type=int, default=32)
    parser.add_argument("--max-validation-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--geometric-decay", type=float, default=0.9)
    return parser.parse_args()


def load_ea_summary(path: Path) -> dict:
    """读取增强EA-TOSD输出，并拒绝旧版缺少细粒度指标的summary。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    for section in ("before_metrics", "after_metrics"):
        metrics = payload.get(section)
        if not isinstance(metrics, dict):
            raise ValueError(f"EA summary is missing {section}")
        missing = [name for name in PAIRED_METRICS if name not in metrics]
        if missing:
            raise ValueError(f"EA summary {section} is missing metrics: {missing}")
    if payload.get("external_reward_model") is not False:
        raise ValueError("paired EA run must not use an external reward model")
    if payload.get("validation_cohort") != "future_eligible_same_cohort":
        raise ValueError("EA summary does not describe the required validation cohort")
    return payload


def assert_paired_before(current: dict, ea_summary: dict) -> None:
    """确保SFT与EA分支从同一checkpoint和同一validation cohort起步。"""
    ea_before = ea_summary["before_metrics"]
    if current["lists"] != ea_before.get("lists"):
        raise RuntimeError("paired validation list count differs from the EA run")
    for name in PAIRED_METRICS:
        if abs(float(current[name]) - float(ea_before[name])) > 1e-12:
            raise RuntimeError(
                f"paired before metric differs for {name}: "
                f"{current[name]} != {ea_before[name]}"
            )


def metric_comparison(before: dict, sft_after: dict, ea_after: dict) -> dict:
    """整理三点配对指标及两条更新分支相对同一起点的变化。"""
    return {
        name: {
            "before": float(before[name]),
            "sft_after": float(sft_after[name]),
            "ea_after": float(ea_after[name]),
            "sft_delta": float(sft_after[name] - before[name]),
            "ea_delta": float(ea_after[name] - before[name]),
            "ea_minus_sft": float(ea_after[name] - sft_after[name]),
        }
        for name in PAIRED_METRICS
    }


def write_comparison(output_dir: Path, summary: dict) -> None:
    """保存机器可读summary和简洁Markdown，固定当前smoke的证据边界。"""
    (output_dir / "sft_control_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# OxygenREC-v2 EA-TOSD / SFT-only 配对smoke",
        "",
        "> 两个分支使用同一预训练checkpoint、同一future-eligible样本、同一顺序、",
        "> 同一batch/epoch/learning-rate；差别仅是是否加入EA-TOSD目标。",
        "> 这是32样本单epoch诊断，不是稳定效果结论。",
        "",
        "| 指标 | 更新前 | SFT-only后 | EA-TOSD后 | SFT变化 | EA变化 | EA-SFT |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in summary["comparison"].items():
        lines.append(
            f"| {name} | {values['before']:.6f} | {values['sft_after']:.6f} | "
            f"{values['ea_after']:.6f} | {values['sft_delta']:+.6f} | "
            f"{values['ea_delta']:+.6f} | {values['ea_minus_sft']:+.6f} |"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- 若SFT和EA一起变化，优先归因于共享的SFT续训或极小样本波动。",
        "- 若EA相对SFT额外变化，才可把差异定位到EA-TOSD附加目标；仍需多seed/更长训练确认。",
        "- SID token命中是层级代理指标，不等同于原始商品级命中率。",
        "",
    ])
    (output_dir / "ea_tosd_vs_sft_control.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    positive = (
        args.list_size,
        args.max_history,
        args.max_future_items,
        args.max_train_samples,
        args.max_validation_samples,
        args.batch_size,
        args.epochs,
        args.beam_width,
    )
    if min(positive) < 1:
        raise ValueError("sizes, counts, epochs and widths must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("stage=load_checkpoint_events_and_ea_summary")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ea_summary = load_ea_summary(args.ea_summary)
    registry = SIDRegistry.from_json(args.sid_registry)
    if payload.get("sid_registry_version") != registry.version:
        raise RuntimeError("checkpoint and SID registry versions differ")
    config = restored_config(payload, args.max_future_items)
    if config.max_target_items != args.list_size:
        raise ValueError("checkpoint max_target_items does not match --list-size")
    model = OxygenRECModel(config).to(device)
    transfer = load_pretrained_with_future_positions(model, payload["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    boundaries_data = payload.get("boundaries")
    if not boundaries_data:
        raise ValueError("listwise checkpoint must contain temporal boundaries")
    boundaries = TemporalBoundaries(**boundaries_data)
    events = [
        event
        for event in load_retailrocket_events(args.events)
        if event.item_id in registry.item_to_sid
    ]
    trie = PrefixTrie.from_registry(registry)

    print("stage=build_paired_future_eligible_cohort")
    samples = build_daily_listwise_samples(
        events,
        boundaries,
        list_size=args.list_size,
        max_history=args.max_history,
        max_samples_per_split={
            Split.TRAIN: args.max_train_samples,
            Split.VALIDATION: args.max_validation_samples,
            Split.TEST: 1,
        },
        sample_seed=args.seed,
        max_future_targets=args.max_future_items,
        minimum_future_behavior=BEHAVIOR_BY_NAME[args.minimum_future_behavior],
        require_future_targets=True,
    )
    train_samples = [sample for sample in samples if sample.split is Split.TRAIN]
    validation_samples = [
        sample for sample in samples if sample.split is Split.VALIDATION
    ]
    if not train_samples or not validation_samples:
        raise RuntimeError("paired SFT cohort is empty")
    deployment_validation = [
        sample.without_privileged_future() for sample in validation_samples
    ]
    future_counts = Counter(len(sample.future_targets) for sample in train_samples)
    behavior_counts = Counter(sample.target_behavior.value for sample in train_samples)
    print(
        f"stage=paired_cohort train={len(train_samples)} "
        f"validation={len(validation_samples)} "
        f"future_counts={dict(sorted(future_counts.items()))} "
        f"behaviors={dict(sorted(behavior_counts.items()))} "
        "objective=behavior_weighted_sft_only external_reward_model=False"
    )

    before_metrics, _ = evaluate(
        model, deployment_validation, registry, trie, args, device
    )
    assert_paired_before(before_metrics, ea_summary)

    total_samples = 0
    optimizer_steps = 0
    weighted_loss = 0.0
    max_shared_gradient = 0.0
    for _epoch in range(1, args.epochs + 1):
        random.shuffle(train_samples)
        for sample_batch in chunks(train_samples, args.batch_size):
            batch = make_batch(sample_batch, registry, args, device)
            model.eval()
            supervised = model(
                **common_inputs(batch),
                target_sids=batch["target_sids"],
                token_weights=batch["token_weights"],
            )
            optimizer.zero_grad(set_to_none=True)
            supervised.loss.backward()
            gradient = total_gradient(model)
            if not torch.isfinite(supervised.loss) or gradient <= 0:
                raise RuntimeError("SFT-only control produced non-finite loss or zero gradient")
            optimizer.step()
            batch_size = len(sample_batch)
            total_samples += batch_size
            optimizer_steps += 1
            weighted_loss += float(supervised.loss.detach()) * batch_size
            max_shared_gradient = max(max_shared_gradient, gradient)

    after_metrics, _ = evaluate(
        model, deployment_validation, registry, trie, args, device
    )
    comparison = metric_comparison(
        before_metrics, after_metrics, ea_summary["after_metrics"]
    )
    summary = {
        "objective": "behavior_weighted_sft_only",
        "samples": total_samples,
        "optimizer_steps": optimizer_steps,
        "mean_sft": weighted_loss / total_samples,
        "max_shared_gradient": max_shared_gradient,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "ea_after_metrics": ea_summary["after_metrics"],
        "comparison": comparison,
        "paired_before_match": True,
        "external_reward_model": False,
        "future_rule": "strictly_after_gold+same_split",
        "minimum_future_behavior": args.minimum_future_behavior,
        "validation_cohort": "future_eligible_same_cohort",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry.to_json(args.output_dir / "sid_registry.json")
    torch.save(
        {
            "epoch": args.epochs,
            "model_config": asdict(config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "sid_registry_version": registry.version,
            "boundaries": asdict(boundaries),
            "args": vars(args),
            "pretrained_checkpoint": str(args.checkpoint),
            "paired_ea_summary": str(args.ea_summary),
            "summary": summary,
        },
        args.output_dir / f"epoch-{args.epochs}.pt",
    )
    write_comparison(args.output_dir, summary)
    print(
        "OK "
        f"device={device.type} variant=v2_sft_control train={len(train_samples)} "
        f"validation={len(validation_samples)} steps={optimizer_steps} "
        f"checkpoint_positions={transfer.copied_position_rows}+"
        f"{transfer.initialized_future_rows} mean_sft={summary['mean_sft']:.6f} "
        f"shared_grad={max_shared_gradient:.6f} paired_before_match=True "
        f"before_sid_recall={before_metrics['sid_recall']:.6f} "
        f"sft_after_sid_recall={after_metrics['sid_recall']:.6f} "
        f"ea_after_sid_recall={ea_summary['after_metrics']['sid_recall']:.6f} "
        f"before_token_accuracy={before_metrics['sid_token_accuracy']:.6f} "
        f"sft_after_token_accuracy={after_metrics['sid_token_accuracy']:.6f} "
        f"ea_after_token_accuracy={ea_summary['after_metrics']['sid_token_accuracy']:.6f} "
        f"before_geo_reward={before_metrics['geometric_token_reward']:.6f} "
        f"sft_after_geo_reward={after_metrics['geometric_token_reward']:.6f} "
        f"ea_after_geo_reward={ea_summary['after_metrics']['geometric_token_reward']:.6f} "
        f"comparison={args.output_dir / 'ea_tosd_vs_sft_control.md'} "
        "external_reward_model=False"
    )


if __name__ == "__main__":
    main()
