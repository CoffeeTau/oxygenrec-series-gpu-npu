#!/usr/bin/env python3
"""Run a bounded RetailRocket SA-GCPO before/after checkpoint comparison."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from dataclasses import asdict, fields
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.alignment import sa_gcpo_loss
from oxygenrec.data import (
    Split, TemporalBoundaries, build_long_short_sid_model_batch,
    build_next_item_samples, build_sid_model_batch, load_retailrocket_events,
)
from oxygenrec.evaluation import evaluate_sid_ranking
from oxygenrec.instruction_cache import (
    instruction_sample_key,
    load_instruction_feature_cache,
)
from oxygenrec.model import OxygenRECConfig, OxygenRECModel
from oxygenrec.rewards import map_public_rewards
from oxygenrec.sa_review import ROLE_LABELS, select_sa_gcpo_trajectories
from oxygenrec.sid import PrefixTrie, SIDRegistry


IGR_VARIANTS = frozenset({
    "igr", "igr_q2i", "igr_qwen_q2i",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("data/raw/retailrocket/events.csv"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sid-registry", type=Path)
    parser.add_argument(
        "--instruction-feature-cache", type=Path,
        help="Required when post-training an igr_qwen_q2i checkpoint.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sample-seed", type=int,
        help="Override the checkpoint seed for cohort sampling and SA-GCPO updates.",
    )
    parser.add_argument("--alignment-samples", type=int, default=200)
    parser.add_argument("--validation-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--target-injection", choices=("none", "always"), default="none",
        help="Teacher-augment missing targets or keep strictly old-policy beams.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--trajectory-output", type=Path,
        help="Optional anonymous JSONL/Markdown representative trajectory output.",
    )
    return parser.parse_args()


def latest_checkpoint(root: Path = Path("checkpoints")) -> Path:
    candidates = list(root.glob("**/epoch-*.pt"))
    if not candidates:
        raise FileNotFoundError("no checkpoints/**/epoch-*.pt found; pass --checkpoint")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def restored_config(payload: dict) -> OxygenRECConfig:
    known = {field.name for field in fields(OxygenRECConfig)}
    return OxygenRECConfig(**{
        key: value for key, value in payload["model_config"].items() if key in known
    })


def checkpoint_arg(payload: dict, name: str, default):
    saved = payload.get("args", {})
    return saved.get(name, default) if isinstance(saved, dict) else default


def make_batch(samples, registry, payload, device, instruction_cache=None):
    """恢复监督训练阶段的同口径输入，避免RL时丢失Instruction或IGR。"""
    variant = checkpoint_arg(payload, "variant", "base")
    max_history = int(checkpoint_arg(payload, "max_history", 20))
    igr_top_k = int(checkpoint_arg(payload, "igr_top_k", 10))
    if variant in IGR_VARIANTS:
        raw = build_long_short_sid_model_batch(
            samples, registry, short_history_items=max_history,
            long_history_items=int(checkpoint_arg(payload, "long_history", 100)),
            minimum_long_history_items=igr_top_k,
        )
        batch = {
            "history_sids": torch.tensor(raw.short_history_sids, device=device),
            "history_padding_mask": torch.tensor(raw.short_history_padding_mask, dtype=torch.bool, device=device),
            "target_sids": torch.tensor(raw.target_sids, device=device),
            "scenario_ids": torch.tensor(raw.scenario_ids, device=device),
            "long_history_sids": torch.tensor(raw.long_history_sids, device=device),
            "long_history_padding_mask": torch.tensor(raw.long_history_padding_mask, dtype=torch.bool, device=device),
        }
        batch["trigger_sids"] = batch["history_sids"][:, -1]
        if variant == "igr_qwen_q2i":
            if instruction_cache is None:
                raise ValueError("igr_qwen_q2i requires an instruction feature cache")
            features, sample_index = instruction_cache
            keys = [instruction_sample_key(sample) for sample in samples]
            missing = [key for key in keys if key not in sample_index]
            if missing:
                raise KeyError(
                    "instruction feature cache does not cover SA-GCPO batch: "
                    f"missing={len(missing)}"
                )
            rows = [sample_index[key] for key in keys]
            batch["instruction_features"] = features[rows].to(
                device=device, dtype=torch.float32,
            )
            # 真实推理不知道target行为，必须与监督训练阶段保持通用scenario一致。
            batch["scenario_ids"] = torch.zeros_like(batch["scenario_ids"])
        return batch
    raw = build_sid_model_batch(samples, registry, max_history_items=max_history)
    batch = {
        "history_sids": torch.tensor(raw.history_sids, device=device),
        "history_padding_mask": torch.tensor(raw.history_padding_mask, dtype=torch.bool, device=device),
        "target_sids": torch.tensor(raw.target_sids, device=device),
    }
    if variant in {"instruction", "q2i"}:
        scenario = {"view": 0, "addtocart": 1, "transaction": 2}
        batch["scenario_ids"] = torch.tensor(
            [scenario[sample.target.behavior.value] for sample in samples], device=device,
        )
        batch["trigger_sids"] = batch["history_sids"][:, -1]
    return batch


def conditioning(batch: dict) -> dict:
    return {
        key: value for key, value in batch.items()
        if key not in {"history_sids", "history_padding_mask", "target_sids"}
    }


@torch.no_grad()
def evaluate(model, batches, registry, trie, beam_width):
    model.eval()
    predictions, targets = [], []
    for samples, batch in batches:
        output = model.beam_search(
            batch["history_sids"], batch["history_padding_mask"], trie,
            beam_width=beam_width, **conditioning(batch),
        )
        predictions.extend(output.semantic_ids.cpu().tolist())
        targets.extend(sample.target.item_id for sample in samples)
    available = min(len(row) for row in predictions)
    ks = tuple(k for k in (1, 5, 10) if k <= available)
    metrics = evaluate_sid_ranking(predictions, targets, registry, ks=ks)
    ranks = []
    for ranking, target in zip(predictions, targets, strict=True):
        rank = 0
        for index, sid in enumerate(ranking, start=1):
            if target in registry.items_for(sid):
                rank = index
                break
        ranks.append(rank)
    return metrics, ranks


def format_metrics(label, metrics) -> str:
    return (
        f"{label}_hr={dict(metrics.hit_rate)} {label}_mrr={metrics.mrr:.6f} "
        f"{label}_ndcg={metrics.ndcg:.6f} {label}_legal={metrics.legal_sid_rate:.6f}"
    )


def render_trajectory_markdown(records, summary) -> str:
    """把匿名SA-GCPO候选、reward、advantage和ratio渲染成人工检查表。"""
    lines = [
        "# OxygenREC-v1 Qwen主线SA-GCPO代表轨迹",
        "",
        "> 固定规则从完整alignment cohort选择，不包含用户ID、原始item ID或CSV行号。",
        "> Reward Mapping为公开代理，不等于论文私有统一ranking service。",
        "",
        "## 覆盖",
        "",
        "| 角色 | 说明 | 案例 |",
        "|---|---|---|",
    ]
    for role, label in ROLE_LABELS.items():
        lines.append(f"| `{role}` | {label} | {summary['coverage'].get(role) or '本cohort不存在'} |")
    aggregate = summary["aggregate"]
    lines.extend([
        "",
        "## 聚合",
        "",
        f"- alignment样本：{aggregate['alignment_samples']}",
        f"- old-policy目标覆盖：{aggregate['target_covered']}",
        f"- objective：`{aggregate['objective_before']:.6f} -> {aggregate['objective_after']:.6f}`",
        "",
    ])
    for record in records:
        lines.extend([
            f"## {record['case_id']}",
            "",
            f"- 代表角色：{'；'.join(record['selection_labels'])}",
            f"- 目标行为：`{record['target_behavior']}`",
            f"- 目标SID：`{record['target_sid']}`",
            f"- old-policy候选含目标：`{record['target_covered']}`",
            f"- reward spread：`{record['reward_spread']:.6f}`",
            f"- policy shift：`{record['policy_shift']:.6f}`",
            f"- 被真实目标reward阈值抑制的候选：{record['suppressed_count']}",
            "",
            "| rank | SID | total | format | relative | ranking | diversity | advantage | thresholded | ratio mean |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for candidate in record["candidates"]:
            lines.append(
                f"| {candidate['rank']} | `{candidate['sid']}` | {candidate['reward_total']:.6f} | "
                f"{candidate['reward_format']:.6f} | {candidate['reward_relative']:.6f} | "
                f"{candidate['reward_ranking']:.6f} | {candidate['reward_diversity']:.6f} | "
                f"{candidate['advantage']:.6f} | {candidate['thresholded_advantage']:.6f} | "
                f"{candidate['importance_ratio_mean']:.6f} |"
            )
        lines.extend([
            "",
            "### 人工Review",
            "",
            "- [ ] 每个reward分量都能由候选SID和目标SID解释",
            "- [ ] target reward阈值只抑制论文公式规定的假阳性优势",
            "- [ ] importance ratio为有限值且后训练确实改变策略",
            "- [ ] 候选全部来自old-policy约束beam，未静默注入target",
            "",
            "---",
            "",
        ])
    return "\n".join(lines)


def export_trajectory_review(path, trajectory_rows, *, alignment_samples, covered,
                             objective_before, objective_after):
    """按固定角色导出少量匿名轨迹，并如实记录不存在的正例。"""
    selected, coverage_indices = select_sa_gcpo_trajectories(trajectory_rows)
    records = []
    case_by_index = {}
    for number, row in enumerate(selected, start=1):
        case_id = f"sa-review-{number:03d}"
        case_by_index[int(row["cohort_index"])] = case_id
        roles = list(row.pop("selection_roles"))
        records.append({
            "case_id": case_id,
            "selection_roles": roles,
            "selection_labels": [ROLE_LABELS[role] for role in roles],
            **row,
        })
    coverage = {
        role: case_by_index.get(index) if index is not None else None
        for role, index in coverage_indices.items()
    }
    summary = {
        "selection_policy": "fixed_target_reward_shift_and_threshold_roles",
        "coverage": coverage,
        "aggregate": {
            "alignment_samples": alignment_samples,
            "representative_cases": len(records),
            "target_covered": covered,
            "objective_before": objective_before,
            "objective_after": objective_after,
        },
        "privacy": "no user IDs, raw item IDs, or CSV source rows",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    markdown_path = path.with_suffix(".md")
    summary_path = path.with_suffix(".summary.json")
    markdown_path.write_text(render_trajectory_markdown(records, summary), encoding="utf-8")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return len(records), coverage, markdown_path, summary_path


def main() -> int:
    args = parse_args()
    if min(args.alignment_samples, args.validation_samples, args.batch_size) < 1 or args.beam_width < 2:
        raise ValueError("sample counts/batch-size must be positive and beam-width >= 2")
    if args.trajectory_output is not None and args.target_injection != "none":
        raise ValueError(
            "representative trajectories require --target-injection none"
        )
    checkpoint_path = args.checkpoint or latest_checkpoint()
    device = torch.device(args.device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    registry_path = args.sid_registry or checkpoint_path.parent / "sid_registry.json"
    registry = SIDRegistry.from_json(registry_path)
    if payload.get("sid_registry_version") != registry.version:
        raise RuntimeError("checkpoint and SID registry versions differ")
    config = restored_config(payload)
    policy = OxygenRECModel(config).to(device)
    policy.load_state_dict(payload["model_state"])
    old_policy = copy.deepcopy(policy).eval()
    for parameter in old_policy.parameters():
        parameter.requires_grad_(False)

    seed = (
        args.sample_seed
        if args.sample_seed is not None
        else int(checkpoint_arg(payload, "seed", 17))
    )
    random.seed(seed)
    torch.manual_seed(seed)
    events = list(load_retailrocket_events(args.events))
    boundaries_data = payload.get("boundaries")
    if boundaries_data:
        boundaries = TemporalBoundaries(**boundaries_data)
    else:
        minimum = min(event.timestamp_ms for event in events)
        duration = max(event.timestamp_ms for event in events) - minimum + 1
        boundaries = TemporalBoundaries(
            train_end_ms=minimum + duration * 8 // 10,
            validation_end_ms=minimum + duration * 9 // 10,
        )
    filtered = [event for event in events if event.item_id in registry.item_to_sid]
    variant = checkpoint_arg(payload, "variant", "base")
    instruction_cache = None
    cache_metadata = None
    if variant == "igr_qwen_q2i":
        if args.instruction_feature_cache is None:
            raise ValueError(
                "--instruction-feature-cache is required for igr_qwen_q2i"
            )
        features, sample_index, cache_metadata = load_instruction_feature_cache(
            args.instruction_feature_cache
        )
        expected_metadata = {
            "sid_registry_version": registry.version,
            "boundaries": asdict(boundaries),
            "sample_seed": seed,
            "short_history": int(checkpoint_arg(payload, "max_history", 20)),
            "long_history": int(checkpoint_arg(payload, "long_history", 100)),
            "igr_top_k": int(checkpoint_arg(payload, "igr_top_k", 10)),
        }
        mismatches = {
            name: {"expected": expected, "actual": cache_metadata.get(name)}
            for name, expected in expected_metadata.items()
            if cache_metadata.get(name) != expected
        }
        if int(features.shape[1]) != config.instruction_feature_size:
            mismatches["feature_size"] = {
                "expected": config.instruction_feature_size,
                "actual": int(features.shape[1]),
            }
        if mismatches:
            raise ValueError(
                "instruction feature cache metadata mismatch: "
                + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            )
        split_counts = cache_metadata.get("split_counts")
        if not isinstance(split_counts, dict):
            raise ValueError("instruction feature cache has no split_counts metadata")
        expected_counts = {
            "train": args.alignment_samples,
            "validation": args.validation_samples,
        }
        actual_counts = {
            name: int(split_counts.get(name, -1)) for name in expected_counts
        }
        if actual_counts != expected_counts:
            raise ValueError(
                "SA-GCPO cohort sizes must exactly match the Qwen cache: "
                f"cache={actual_counts} requested={expected_counts}"
            )
        instruction_cache = (features, sample_index)
        print(
            f"stage=instruction_cache path={args.instruction_feature_cache} "
            f"rows={features.shape[0]} feature_size={features.shape[1]}"
        )
    elif args.instruction_feature_cache is not None:
        raise ValueError(
            "--instruction-feature-cache is only valid for igr_qwen_q2i"
        )
    matched = variant in IGR_VARIANTS or bool(
        checkpoint_arg(payload, "matched_igr_cohort", False)
    )
    max_history = int(checkpoint_arg(payload, "max_history", 20))
    igr_top_k = int(checkpoint_arg(payload, "igr_top_k", 10))
    long_history = int(checkpoint_arg(payload, "long_history", 100))
    samples = build_next_item_samples(
        filtered, boundaries,
        min_history=max_history + igr_top_k if matched else 1,
        max_history=max_history + long_history if matched else max_history,
        max_samples_per_split={
            Split.TRAIN: args.alignment_samples,
            Split.VALIDATION: args.validation_samples,
            Split.TEST: 1,
        },
        sample_seed=seed,
    )
    by_split = defaultdict(list)
    for sample in samples:
        by_split[sample.split].append(sample)
    alignment = by_split[Split.TRAIN]
    validation = by_split[Split.VALIDATION]
    if not alignment or not validation:
        raise RuntimeError("alignment or held-out validation cohort is empty")
    if instruction_cache is not None:
        required_keys = {
            instruction_sample_key(sample) for sample in alignment + validation
        }
        missing = required_keys.difference(instruction_cache[1])
        if missing:
            raise ValueError(
                "instruction feature cache does not cover SA-GCPO cohort: "
                f"missing={len(missing)}"
            )

    def tensor_batches(cohort):
        return [
            (cohort[start:start + args.batch_size], make_batch(
                cohort[start:start + args.batch_size], registry, payload, device,
                instruction_cache=instruction_cache,
            ))
            for start in range(0, len(cohort), args.batch_size)
        ]

    alignment_batches = tensor_batches(alignment)
    validation_batches = tensor_batches(validation)
    trie = PrefixTrie.from_registry(registry)
    before, before_ranks = evaluate(
        policy, validation_batches, registry, trie, args.beam_width
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    first_objective = last_objective = 0.0
    injected = 0
    covered = 0
    rollouts = []
    cohort_offset = 0
    for sample_rows, batch in alignment_batches:
        with torch.no_grad():
            candidates = old_policy.beam_search(
                batch["history_sids"], batch["history_padding_mask"], trie,
                beam_width=args.beam_width, **conditioning(batch),
            ).semantic_ids
            target = batch["target_sids"]
            present = (candidates == target[:, None]).all(dim=-1).any(dim=1)
            covered += int(present.sum())
            if args.target_injection == "always" and (~present).any():
                candidates = candidates.clone()
                candidates[~present, -1] = target[~present]
                injected += int((~present).sum())
            relative = (candidates == target[:, None]).float().mean(dim=-1)
            ranking = (candidates == target[:, None]).all(dim=-1).float()
            mapped = map_public_rewards(
                candidates, legal_mask=torch.ones_like(ranking, dtype=torch.bool),
                relative_scores=relative, ranking_scores=ranking,
            )
            target_rewards = torch.full(
                (target.shape[0],), 3.0, dtype=mapped.total.dtype, device=device
            )
            old_log_probs = old_policy.candidate_log_probs(
                batch["history_sids"], batch["history_padding_mask"], candidates,
                **conditioning(batch),
            )
        rollouts.append((
            cohort_offset, sample_rows, batch, candidates, mapped,
            target_rewards, old_log_probs,
        ))
        cohort_offset += len(sample_rows)
    policy.train()
    for update in range(args.updates):
        objective_sum = 0.0
        for _, _, batch, candidates, mapped, target_rewards, old_log_probs in rollouts:
            optimizer.zero_grad(set_to_none=True)
            current_log_probs = policy.candidate_log_probs(
                batch["history_sids"], batch["history_padding_mask"], candidates,
                **conditioning(batch),
            )
            aligned = sa_gcpo_loss(
                current_log_probs, old_log_probs, mapped.total, target_rewards,
            )
            aligned.loss.backward()
            optimizer.step()
            objective_sum += float(aligned.objective.detach())
        epoch_objective = objective_sum / len(alignment_batches)
        if update == 0:
            first_objective = epoch_objective
        last_objective = epoch_objective
    after, after_ranks = evaluate(
        policy, validation_batches, registry, trie, args.beam_width
    )
    improved = sum(
        after_rank > 0 and (before_rank == 0 or after_rank < before_rank)
        for before_rank, after_rank in zip(before_ranks, after_ranks, strict=True)
    )
    worsened = sum(
        before_rank > 0 and (after_rank == 0 or after_rank > before_rank)
        for before_rank, after_rank in zip(before_ranks, after_ranks, strict=True)
    )
    unchanged = len(before_ranks) - improved - worsened
    output_path = args.output or checkpoint_path.parent / (
        f"sa_gcpo-retailrocket-{args.target_injection}-seed{seed}.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": payload.get("epoch"),
        "model_config": payload["model_config"],
        "model_state": policy.state_dict(),
        "source_checkpoint": str(checkpoint_path),
        "sid_registry_version": registry.version,
        "boundaries": payload.get("boundaries"),
        "args": payload.get("args"),
        "alignment_samples": len(alignment),
        "validation_samples": len(validation),
        "updates": args.updates,
        "learning_rate": args.learning_rate,
        "sample_seed": seed,
        "target_injection": args.target_injection,
        "objective_before": first_objective,
        "objective_after": last_objective,
        "instruction_feature_cache": (
            str(args.instruction_feature_cache)
            if args.instruction_feature_cache is not None else None
        ),
    }, output_path)
    trajectory_message = ""
    if args.trajectory_output is not None:
        trajectory_rows = []
        with torch.no_grad():
            for offset, sample_rows, batch, candidates, mapped, target_rewards, old_log_probs in rollouts:
                current_log_probs = policy.candidate_log_probs(
                    batch["history_sids"], batch["history_padding_mask"], candidates,
                    **conditioning(batch),
                )
                diagnostic = sa_gcpo_loss(
                    current_log_probs, old_log_probs, mapped.total, target_rewards,
                )
                for row_index, sample in enumerate(sample_rows):
                    rewards = mapped.total[row_index]
                    shift = (current_log_probs[row_index] - old_log_probs[row_index]).abs().mean()
                    target_sid = batch["target_sids"][row_index]
                    target_matches = (candidates[row_index] == target_sid.unsqueeze(0)).all(dim=-1)
                    candidate_rows = []
                    for rank in range(candidates.shape[1]):
                        candidate_rows.append({
                            "rank": rank + 1,
                            "sid": candidates[row_index, rank].cpu().tolist(),
                            "reward_total": float(mapped.total[row_index, rank].cpu()),
                            "reward_format": float(mapped.format[row_index, rank].cpu()),
                            "reward_relative": float(mapped.relative[row_index, rank].cpu()),
                            "reward_ranking": float(mapped.ranking[row_index, rank].cpu()),
                            "reward_diversity": float(mapped.diversity[row_index, rank].cpu()),
                            "old_logprob_sum": float(old_log_probs[row_index, rank].sum().cpu()),
                            "current_logprob_sum": float(current_log_probs[row_index, rank].sum().cpu()),
                            "advantage": float(diagnostic.normalized_advantage[row_index, rank].cpu()),
                            "thresholded_advantage": float(diagnostic.thresholded_advantage[row_index, rank].cpu()),
                            "importance_ratio_mean": float(diagnostic.importance_ratio[row_index, rank].mean().cpu()),
                        })
                    trajectory_rows.append({
                        "cohort_index": offset + row_index,
                        "target_behavior": sample.target.behavior.value,
                        "target_sid": target_sid.cpu().tolist(),
                        "target_covered": bool(target_matches.any()),
                        "reward_spread": float((rewards.max() - rewards.min()).cpu()),
                        "policy_shift": float(shift.cpu()),
                        "suppressed_count": int(
                            (
                                (diagnostic.normalized_advantage[row_index] > 0)
                                & (diagnostic.thresholded_advantage[row_index] == 0)
                            ).sum().cpu()
                        ),
                        "candidates": candidate_rows,
                    })
        count, coverage, markdown_path, summary_path = export_trajectory_review(
            args.trajectory_output, trajectory_rows,
            alignment_samples=len(alignment), covered=covered,
            objective_before=first_objective, objective_after=last_objective,
        )
        trajectory_message = (
            f" trajectories={count} trajectory_coverage="
            f"{json.dumps(coverage, sort_keys=True)} trajectory_jsonl={args.trajectory_output} "
            f"trajectory_markdown={markdown_path} trajectory_summary={summary_path}"
        )
    print(
        f"OK device={device} checkpoint={checkpoint_path} variant={variant} seed={seed} "
        f"alignment={len(alignment)} heldout_validation={len(validation)} "
        f"alignment_target_coverage={covered} "
        f"target_injection={args.target_injection} injected_targets={injected} "
        f"objective={first_objective:.6f}->{last_objective:.6f} output={output_path}"
        f"{trajectory_message}"
    )
    print(format_metrics("before", before))
    print(format_metrics("after", after))
    print(
        f"paired_ranks improved={improved} worsened={worsened} "
        f"unchanged={unchanged}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
