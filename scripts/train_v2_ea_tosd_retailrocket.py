#!/usr/bin/env python3
"""在真实RetailRocket列表代理上执行有边界的EA-TOSD后训练smoke。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, fields
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.data import (
    Behavior,
    Split,
    TemporalBoundaries,
    build_daily_listwise_samples,
    build_listwise_sid_model_batch,
    load_retailrocket_events,
)
from oxygenrec.ea_tosd import (
    EATOSDConfig,
    ea_tosd_loss,
    load_pretrained_with_future_positions,
    select_best_verifiable_trajectory,
)
from oxygenrec.model import OxygenRECConfig, OxygenRECModel
from oxygenrec.sid import PrefixTrie, SIDRegistry
from train_v2_listwise_retailrocket import BEHAVIOR_WEIGHTS, chunks, evaluate


BEHAVIOR_BY_NAME = {
    "view": Behavior.VIEW,
    "addtocart": Behavior.ADD_TO_CART,
    "transaction": Behavior.TRANSACTION,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/retailrocket_v2_ea_tosd_smoke"),
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
        help="RetailRocket无exposure；view是论文click的公开代理。",
    )
    parser.add_argument("--max-train-samples", type=int, default=32)
    parser.add_argument("--max-validation-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--geometric-decay", type=float, default=0.9)
    parser.add_argument("--verifiable-weight", type=float, default=0.1)
    parser.add_argument("--self-distillation-weight", type=float, default=0.01)
    parser.add_argument("--forward-kl-weight", type=float, default=0.01)
    parser.add_argument("--supervised-weight", type=float, default=1.0)
    parser.add_argument("--low-entropy-threshold", type=float, default=0.75)
    parser.add_argument("--high-entropy-threshold", type=float, default=2.6)
    return parser.parse_args()


def restored_config(payload: dict, max_future_items: int) -> OxygenRECConfig:
    """恢复预训练结构，仅显式增加Teacher未来位置容量。"""
    known = {field.name for field in fields(OxygenRECConfig)}
    values = {
        key: value
        for key, value in payload["model_config"].items()
        if key in known
    }
    values["max_future_items"] = max_future_items
    return OxygenRECConfig(**values)


def make_batch(samples, registry, args, device):
    """映射真实history/gold/future，并构造behavior-weighted SFT权重。"""
    raw = build_listwise_sid_model_batch(
        samples,
        registry,
        max_history_items=args.max_history,
        max_future_items=args.max_future_items,
    )
    behavior_ids = torch.tensor(
        raw.target_behavior_ids, dtype=torch.long, device=device
    )
    sid_tokens = args.list_size * registry.levels
    behavior_weights = torch.tensor(
        BEHAVIOR_WEIGHTS, dtype=torch.float32, device=device
    )
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
        "token_weights": behavior_weights[behavior_ids].unsqueeze(1).expand(
            -1, sid_tokens
        ),
        "future_sids": torch.tensor(raw.future_sids, dtype=torch.long, device=device),
        "future_padding_mask": torch.tensor(
            raw.future_padding_mask, dtype=torch.bool, device=device
        ),
    }


def common_inputs(batch: dict) -> dict:
    """取Student、Teacher、rollout共同可见的部署输入。"""
    return {
        "history_sids": batch["history_sids"],
        "history_padding_mask": batch["history_padding_mask"],
        "history_behavior_ids": batch["history_behavior_ids"],
        "behavior_instruction_ids": batch["behavior_instruction_ids"],
    }


def max_logits_delta(left, right) -> float:
    """汇总共享模型在Student/Teacher条件下的最大logit差。"""
    return max(
        float((a.detach() - b.detach()).abs().max())
        for a, b in zip(left, right)
    )


def total_gradient(model: torch.nn.Module) -> float:
    """计算一次EA-TOSD更新前共享backbone的梯度L1范数。"""
    return float(sum(
        parameter.grad.abs().sum().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    ))


def select_review_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """固定选择零/非零reward、稀有行为和两类熵门代表轨迹。"""
    roles = (
        ("nonzero_reward", lambda row: row["selected_reward"] > 0, "selected_reward"),
        ("zero_reward", lambda row: row["selected_reward"] == 0, "future_count"),
        ("transaction", lambda row: row["behavior"] == "transaction", "selected_reward"),
        (
            "most_low_entropy",
            lambda row: row["low_gate_tokens"] > 0,
            "low_gate_tokens",
        ),
        (
            "most_high_entropy",
            lambda row: row["high_gate_tokens"] > 0,
            "high_gate_tokens",
        ),
        ("largest_privilege_gap", lambda row: True, "max_abs_advantage"),
    )
    selected_by_index: dict[int, dict] = {}
    coverage = {}
    for role, predicate, score_key in roles:
        candidates = [
            (index, row) for index, row in enumerate(rows) if predicate(row)
        ]
        if not candidates:
            coverage[role] = None
            continue
        index, row = max(
            candidates,
            key=lambda item: (item[1][score_key], -item[0]),
        )
        selected_by_index.setdefault(index, {**row, "roles": []})["roles"].append(role)
        coverage[role] = index
    selected = []
    case_by_index = {}
    for number, (index, row) in enumerate(sorted(selected_by_index.items()), start=1):
        case_id = f"ea-tosd-review-{number:03d}"
        selected.append({"case_id": case_id, **row})
        case_by_index[index] = case_id
    return selected, {
        role: case_by_index[index] if index is not None else None
        for role, index in coverage.items()
    }


def write_review(output_dir: Path, rows: list[dict], summary: dict) -> None:
    """导出匿名轨迹、reward、未来前缀、advantage和熵门供人工复核。"""
    selected, coverage = select_review_rows(rows)
    jsonl_path = output_dir / "ea_tosd_representative_trajectories.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    lines = [
        "# OxygenREC-v2真实EA-TOSD代表轨迹",
        "",
        "> 仅含SID、行为和聚合诊断，不含用户ID、原始商品ID或CSV行号。",
        "> RetailRocket view是click代理；未来目标严格晚于gold列表且不跨split。",
        "",
        "## 聚合",
        "",
        f"- summary：`{json.dumps(summary, ensure_ascii=False, sort_keys=True)}`",
        f"- 固定角色覆盖：`{json.dumps(coverage, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    for row in selected:
        lines.extend([
            f"## {row['case_id']}",
            "",
            f"- 代表角色：`{row['roles']}`",
            f"- 目标行为：`{row['behavior']}`",
            f"- gold SID列表：`{row['gold_sids']}`",
            f"- Teacher未来SID：`{row['future_sids']}`",
            f"- G条候选：`{row['candidate_sids']}`",
            f"- 候选reward：`{row['candidate_rewards']}`",
            f"- best索引：{row['best_index']}",
            f"- best SID：`{row['selected_sids']}`",
            f"- best reward：{row['selected_reward']:.6f}",
            f"- Teacher entropy：`{row['teacher_entropy']}`",
            f"- privilege advantage：`{row['privilege_advantage']}`",
            f"- low/high gate token：{row['low_gate_tokens']}/{row['high_gate_tokens']}",
            "",
            "### 人工Review",
            "",
            "- [ ] future SID均能在该用户gold列表之后的真实行为中找到",
            "- [ ] reward只由候选SID与gold SID逐token命中解释",
            "- [ ] best-of-G确实选择当前组最高reward轨迹",
            "- [ ] 低熵与高熵门和Teacher entropy一致",
            "- [ ] 未把SID代理命中误写成商品级工业指标",
            "",
        ])
    markdown_path = output_dir / "ea_tosd_representative_trajectories.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_path = output_dir / "ea_tosd_summary.json"
    summary_path.write_text(
        json.dumps(
            {**summary, "coverage": coverage, "representative": len(selected)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
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
        args.group_size,
        args.beam_width,
    )
    if min(positive) < 1:
        raise ValueError("sizes, counts, epochs and widths must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("stage=load_checkpoint_and_events")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    registry = SIDRegistry.from_json(args.sid_registry)
    if payload.get("sid_registry_version") != registry.version:
        raise RuntimeError("checkpoint and SID registry versions differ")
    config = restored_config(payload, args.max_future_items)
    if config.max_target_items != args.list_size:
        raise ValueError("checkpoint max_target_items does not match --list-size")
    if config.behavior_instruction_vocab_size < 1:
        raise ValueError("EA-TOSD requires a v2 behavior-instruction checkpoint")
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

    print("stage=build_real_future_prefixes")
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
        raise RuntimeError(
            "real EA-TOSD cohort is empty; lower the future behavior threshold or list size"
        )
    # evaluate()复用部署Student的listwise批处理，不接受也不应看到F。
    # 保留validation的相同history/gold队列，只显式移除Teacher专用字段。
    deployment_validation_samples = [
        sample.without_privileged_future() for sample in validation_samples
    ]
    future_counts = Counter(len(sample.future_targets) for sample in train_samples)
    behavior_counts = Counter(sample.target_behavior.value for sample in train_samples)
    print(
        f"stage=real_ea_cohort train={len(train_samples)} validation={len(validation_samples)} "
        f"future_counts={dict(sorted(future_counts.items()))} "
        f"behaviors={dict(sorted(behavior_counts.items()))} "
        f"future_threshold={args.minimum_future_behavior} "
        "future_rule=strictly_after_gold+same_split external_reward_model=False"
    )

    # 固定同一批“存在未来反馈”的validation样本做前后比较；这是结构smoke，
    # 存在future-eligible选择条件，不能当作无偏泛化收益。
    before_metrics, _ = evaluate(
        model, deployment_validation_samples, registry, trie, args, device
    )
    objective_config = EATOSDConfig(
        geometric_decay=args.geometric_decay,
        verifiable_weight=args.verifiable_weight,
        self_distillation_weight=args.self_distillation_weight,
        forward_kl_weight=args.forward_kl_weight,
        supervised_weight=args.supervised_weight,
        low_entropy_threshold=args.low_entropy_threshold,
        high_entropy_threshold=args.high_entropy_threshold,
    )

    review_rows = []
    totals = Counter()
    component_sums = Counter()
    max_shared_gradient = 0.0
    max_teacher_student_delta = 0.0
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_samples)
        for sample_batch in chunks(train_samples, args.batch_size):
            batch = make_batch(sample_batch, registry, args, device)
            shared = common_inputs(batch)

            # rollout和Teacher均关闭dropout；Student仍保留梯度，三者来自同一当前策略。
            model.eval()
            candidates = model.sample_trajectories(
                shared["history_sids"],
                shared["history_padding_mask"],
                trie,
                group_size=args.group_size,
                history_behavior_ids=shared["history_behavior_ids"],
                behavior_instruction_ids=shared["behavior_instruction_ids"],
                output_items=args.list_size,
                temperature=args.temperature,
            )
            gold_tokens = batch["target_sids"].reshape(len(sample_batch), -1)
            selection = select_best_verifiable_trajectory(
                candidates,
                gold_tokens,
                geometric_decay=args.geometric_decay,
            )
            selected_targets = selection.selected_tokens.reshape(
                len(sample_batch), args.list_size, registry.levels
            )
            student = model(**shared, target_sids=selected_targets)
            with torch.no_grad():
                teacher = model(
                    **shared,
                    target_sids=selected_targets,
                    privileged_future_sids=batch["future_sids"],
                    privileged_future_padding_mask=batch["future_padding_mask"],
                )
            supervised = model(
                **shared,
                target_sids=batch["target_sids"],
                token_weights=batch["token_weights"],
            )
            objective = ea_tosd_loss(
                student.logits,
                teacher.logits,
                selection.selected_tokens,
                selection.selected_rewards,
                supervised.loss,
                config=objective_config,
            )
            optimizer.zero_grad(set_to_none=True)
            objective.loss.backward()
            gradient = total_gradient(model)
            if not torch.isfinite(objective.loss) or gradient <= 0:
                raise RuntimeError("EA-TOSD produced a non-finite loss or zero gradient")
            optimizer.step()

            batch_size = len(sample_batch)
            totals.update({
                "samples": batch_size,
                "nonzero_reward": int(selection.selected_rewards.gt(0).sum()),
                "zero_reward_groups": int(selection.rewards.max(dim=1).values.eq(0).sum()),
                "low_gate_tokens": int((objective.low_entropy_weights > 0).sum()),
                "high_gate_tokens": int((objective.high_entropy_weights > 0).sum()),
                "tokens": batch_size * gold_tokens.shape[1],
                "future_items": int((~batch["future_padding_mask"]).sum()),
            })
            component_sums.update({
                "loss": float(objective.loss.detach()) * batch_size,
                "vr": float(objective.verifiable_loss.detach()) * batch_size,
                "sd": float(objective.self_distillation_loss.detach()) * batch_size,
                "fkl": float(objective.forward_kl_loss.detach()) * batch_size,
                "sft": float(objective.supervised_loss.detach()) * batch_size,
                "reward": float(selection.selected_rewards.sum()),
            })
            max_shared_gradient = max(max_shared_gradient, gradient)
            max_teacher_student_delta = max(
                max_teacher_student_delta,
                max_logits_delta(student.logits, teacher.logits),
            )

            candidate_rows = candidates.detach().cpu().tolist()
            reward_rows = selection.rewards.detach().cpu().tolist()
            entropy_rows = objective.teacher_entropy.detach().cpu().tolist()
            advantage_rows = objective.privilege_advantage.detach().cpu().tolist()
            low_rows = (objective.low_entropy_weights > 0).detach().cpu().tolist()
            high_rows = (objective.high_entropy_weights > 0).detach().cpu().tolist()
            future_rows = batch["future_sids"].detach().cpu().tolist()
            future_masks = batch["future_padding_mask"].detach().cpu().tolist()
            gold_rows = gold_tokens.detach().cpu().tolist()
            best_rows = selection.best_indices.detach().cpu().tolist()
            selected_rows = selection.selected_tokens.detach().cpu().tolist()
            selected_rewards = selection.selected_rewards.detach().cpu().tolist()
            for index, sample in enumerate(sample_batch):
                valid_future = [
                    sid for sid, masked in zip(future_rows[index], future_masks[index])
                    if not masked
                ]
                review_rows.append({
                    "behavior": sample.target_behavior.value,
                    "gold_sids": gold_rows[index],
                    "future_sids": valid_future,
                    "future_count": len(valid_future),
                    "candidate_sids": candidate_rows[index],
                    "candidate_rewards": reward_rows[index],
                    "best_index": best_rows[index],
                    "selected_sids": selected_rows[index],
                    "selected_reward": selected_rewards[index],
                    "teacher_entropy": entropy_rows[index],
                    "privilege_advantage": advantage_rows[index],
                    "low_gate_tokens": sum(low_rows[index]),
                    "high_gate_tokens": sum(high_rows[index]),
                    "max_abs_advantage": max(abs(value) for value in advantage_rows[index]),
                })

    after_metrics, _ = evaluate(
        model, deployment_validation_samples, registry, trie, args, device
    )
    sample_count = totals["samples"]
    summary = {
        "samples": sample_count,
        "group_size": args.group_size,
        "mean_selected_reward": component_sums["reward"] / sample_count,
        "nonzero_reward_rate": totals["nonzero_reward"] / sample_count,
        "all_zero_group_rate": totals["zero_reward_groups"] / sample_count,
        "low_gate_rate": totals["low_gate_tokens"] / totals["tokens"],
        "high_gate_rate": totals["high_gate_tokens"] / totals["tokens"],
        "mean_future_items": totals["future_items"] / sample_count,
        "mean_loss": component_sums["loss"] / sample_count,
        "mean_vr": component_sums["vr"] / sample_count,
        "mean_sd": component_sums["sd"] / sample_count,
        "mean_fkl": component_sums["fkl"] / sample_count,
        "mean_sft": component_sums["sft"] / sample_count,
        "max_shared_gradient": max_shared_gradient,
        "max_teacher_student_delta": max_teacher_student_delta,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
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
            "ea_tosd": asdict(objective_config),
            "summary": summary,
        },
        args.output_dir / f"epoch-{args.epochs}.pt",
    )
    write_review(args.output_dir, review_rows, summary)
    print(
        "OK "
        f"device={device.type} variant=v2_ea_tosd train={len(train_samples)} "
        f"validation={len(validation_samples)} group={args.group_size} "
        f"checkpoint_positions={transfer.copied_position_rows}+{transfer.initialized_future_rows} "
        f"mean_reward={summary['mean_selected_reward']:.6f} "
        f"nonzero_reward_rate={summary['nonzero_reward_rate']:.6f} "
        f"zero_group_rate={summary['all_zero_group_rate']:.6f} "
        f"low_gate_rate={summary['low_gate_rate']:.6f} "
        f"high_gate_rate={summary['high_gate_rate']:.6f} "
        f"shared_grad={max_shared_gradient:.6f} "
        f"teacher_student_delta={max_teacher_student_delta:.6f} "
        f"before_sid_recall={before_metrics['sid_recall']:.6f} "
        f"after_sid_recall={after_metrics['sid_recall']:.6f} "
        f"review={args.output_dir / 'ea_tosd_representative_trajectories.md'} "
        "future_same_split=True external_reward_model=False"
    )


if __name__ == "__main__":
    main()
