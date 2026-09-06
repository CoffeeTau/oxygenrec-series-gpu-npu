#!/usr/bin/env python3
"""导出OxygenREC-v1真实Qwen论文主线的代表性正反案例。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import fields
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from oxygenrec.data import (
    Split,
    TemporalBoundaries,
    build_long_short_sid_model_batch,
    build_next_item_samples,
    load_retailrocket_events,
)
from oxygenrec.instruction_cache import (
    instruction_sample_key,
    load_instruction_feature_cache,
)
from oxygenrec.llm_reasoning import contextual_instruction_text
from oxygenrec.model import OxygenRECConfig, OxygenRECModel
from oxygenrec.review_selection import ROLE_LABELS, select_representative_rows
from oxygenrec.sid import PrefixTrie, SIDRegistry


def parse_args() -> argparse.Namespace:
    """定义checkpoint、真实cohort缓存和匿名review产物。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument("--instruction-feature-cache", type=Path, required=True)
    parser.add_argument("--reasoning-input", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/review/qwen_mainline_representative_cases.jsonl"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beam-width", type=int, default=10)
    return parser.parse_args()


def history_evidence(sample) -> dict[str, object]:
    """重新计算target前聚合证据，用于校验缓存Reasoning没有错位。"""
    behaviors = [event.behavior.value for event in sample.history]
    item_counts = Counter(event.item_id for event in sample.history)
    return {
        "history_length": len(sample.history),
        "behavior_counts": dict(Counter(behaviors)),
        "recent_behaviors": behaviors[-5:],
        "repeated_item_kinds": sum(count > 1 for count in item_counts.values()),
    }


def load_reasoning(path: Path) -> dict[str, dict[str, object]]:
    """按稳定样本键加载并校验离线Qwen Reasoning记录。"""
    records = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("sample_key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"reasoning line {line_number} has no sample_key")
        if key in records:
            raise ValueError(f"duplicate reasoning sample_key: {key}")
        if row.get("target_excluded") is not True:
            raise ValueError(f"reasoning sample did not assert target exclusion: {key}")
        records[key] = row
    return records


def render_markdown(
    records: list[dict[str, object]], summary: dict[str, object],
) -> str:
    """生成可人工勾选的中文端到端案例报告。"""
    lines = [
        "# OxygenREC-v1 真实Qwen论文主线代表性案例",
        "",
        "> 本报告按预定义规则同时保留成功与失败样本，不按结果人工挑选。",
        "> 不包含用户ID或原始商品ID；`retrieval_plan`不进入paper_igr模型输入。",
        "",
        "## 代表性覆盖",
        "",
        "| 角色 | 说明 | 案例 |",
        "|---|---|---|",
    ]
    coverage = summary["coverage"]
    for role, label in ROLE_LABELS.items():
        lines.append(f"| `{role}` | {label} | {coverage.get(role) or '本cohort不存在'} |")
    aggregate = summary["aggregate"]
    lines.extend([
        "",
        "## Cohort聚合",
        "",
        f"- validation样本：{aggregate['evaluated']}",
        f"- repeat-eligible：{aggregate['repeat_eligible']}",
        f"- IGR目标SID命中：{aggregate['igr_hits']}",
        f"- beam目标商品命中：{aggregate['beam_hits']}",
        f"- Q2I cosine范围：`{aggregate['q2i_min']:.6f} ~ {aggregate['q2i_max']:.6f}`",
        "",
    ])
    for record in records:
        lines.extend([
            f"## {record['case_id']}",
            "",
            f"- 代表角色：{'；'.join(record['selection_labels'])}",
            f"- 目标行为：`{record['target']['behavior']}`",
            f"- 目标SID：`{record['target']['sid']}`",
            f"- Q2I cosine：`{record['q2i']['cosine']:.6f}`",
            f"- IGR目标SID命中：`{record['igr']['target_sid_hit']}`",
            f"- beam目标商品命中rank：`{record['generation']['target_hit_rank']}`",
            "",
            "### Target前行为证据",
            "",
            f"- 历史长度：{record['history']['history_length']}",
            f"- 行为计数：`{record['history']['behavior_counts']}`",
            f"- 最近行为：`{record['history']['recent_behaviors']}`",
            f"- 重复商品种类：{record['history']['repeated_item_kinds']}",
            "",
            "### Qwen Contextual Reasoning Instruction",
            "",
            f"- Intent：{record['reasoning']['intent']}",
            f"- Evidence：`{record['reasoning']['evidence']}`",
            f"- Retrieval strategy：{record['reasoning']['retrieval_strategy']}",
            f"- Instruction实际文本：\n\n```text\n{record['reasoning']['instruction_text']}\n```",
            "- 结构化Retrieval Plan：存在于原Qwen输出，但本轮`paper_igr`明确未消费。",
            "",
            "### IGR Top-K",
            "",
            "| rank | long index | score | SID | behavior | hours ago | target SID |",
            "|---:|---:|---:|---|---|---:|---|",
        ])
        for item in record["igr"]["selected"]:
            lines.append(
                f"| {item['rank']} | {item['long_index']} | {item['score']:.6f} | "
                f"`{item['sid']}` | {item['behavior']} | {item['hours_before_target']:.2f} | "
                f"{item['target_sid_match']} |"
            )
        lines.extend([
            "",
            "### Constrained Beam",
            "",
            "| rank | SID | score | legal | collision | target hit |",
            "|---:|---|---:|---|---:|---|",
        ])
        for item in record["generation"]["beam"]:
            lines.append(
                f"| {item['rank']} | `{item['sid']}` | {item['score']:.6f} | "
                f"{item['legal']} | {item['collision_size']} | {item['target_hit']} |"
            )
        lines.extend([
            "",
            "### 人工Review",
            "",
            "- [ ] Reasoning中的每条Evidence都能由上方聚合输入支持",
            "- [ ] Retrieval strategy与当前意图及行为强度一致",
            "- [ ] IGR候选与Instruction语义存在可解释联系",
            "- [ ] Q2I cosine与该案例的检索表现没有明显矛盾",
            "- [ ] beam全部合法；若未命中，能区分表示、检索与生成阶段问题",
            "",
            "---",
            "",
        ])
    return "\n".join(lines)


def main() -> int:
    """评估完整validation cohort，再按固定规则导出少量正反例。"""
    args = parse_args()
    if args.beam_width < 1:
        raise ValueError("beam-width must be positive")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    registry = SIDRegistry.from_json(args.sid_registry)
    if checkpoint.get("sid_registry_version") != registry.version:
        raise ValueError("checkpoint and SID registry versions do not match")
    known_config = {field.name for field in fields(OxygenRECConfig)}
    config = OxygenRECConfig(**{
        name: value for name, value in checkpoint["model_config"].items()
        if name in known_config
    })
    if config.igr_top_k < 1 or config.q2i_weight <= 0 or config.instruction_feature_size < 1:
        raise ValueError("review checkpoint must contain Qwen features, paper IGR and Q2I")
    model = OxygenRECModel(config).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    boundaries = TemporalBoundaries(**checkpoint["boundaries"])
    features, feature_index, metadata = load_instruction_feature_cache(
        args.instruction_feature_cache
    )
    if metadata.get("sid_registry_version") != registry.version:
        raise ValueError("instruction cache and SID registry versions do not match")
    if metadata.get("boundaries") != checkpoint["boundaries"]:
        raise ValueError("instruction cache and checkpoint boundaries do not match")
    if int(features.shape[1]) != config.instruction_feature_size:
        raise ValueError("instruction cache feature size does not match checkpoint")
    reasoning_by_key = load_reasoning(args.reasoning_input)

    split_counts = metadata.get("split_counts")
    if not isinstance(split_counts, dict):
        raise ValueError("instruction cache has no split_counts metadata")
    short_history = int(metadata["short_history"])
    long_history = int(metadata["long_history"])
    sample_seed = int(metadata["sample_seed"])
    if short_history != config.max_history_items or int(metadata["igr_top_k"]) != config.igr_top_k:
        raise ValueError("instruction cache history parameters do not match checkpoint")
    events = [
        event for event in load_retailrocket_events(args.events)
        if event.item_id in registry.item_to_sid
    ]
    samples = build_next_item_samples(
        events, boundaries,
        min_history=short_history + config.igr_top_k,
        max_history=short_history + long_history,
        max_samples_per_split={
            Split.TRAIN: int(split_counts["train"]),
            Split.VALIDATION: int(split_counts["validation"]),
            Split.TEST: 1,
        },
        sample_seed=sample_seed,
    )
    by_split = defaultdict(list)
    for sample in samples:
        by_split[sample.split].append(sample)
    validation = by_split[Split.VALIDATION]
    if len(validation) != int(split_counts["validation"]):
        raise RuntimeError("validation cohort size does not match instruction cache")

    keys = [instruction_sample_key(sample) for sample in validation]
    missing_features = [key for key in keys if key not in feature_index]
    missing_reasoning = [key for key in keys if key not in reasoning_by_key]
    if missing_features or missing_reasoning:
        raise ValueError(
            f"review inputs do not cover validation: feature_missing={len(missing_features)} "
            f"reasoning_missing={len(missing_reasoning)}"
        )
    for sample, key in zip(validation, keys, strict=True):
        stored = reasoning_by_key[key]
        if stored.get("input_evidence") != history_evidence(sample):
            raise ValueError(f"reasoning evidence mismatch for {key}")
        if contextual_instruction_text(stored["reasoning"]) != stored.get("instruction_text"):
            raise ValueError(f"instruction text mismatch for {key}")

    raw = build_long_short_sid_model_batch(
        validation, registry,
        short_history_items=short_history,
        long_history_items=long_history,
        minimum_long_history_items=config.igr_top_k,
    )
    short = torch.tensor(raw.short_history_sids, dtype=torch.long, device=device)
    short_mask = torch.tensor(raw.short_history_padding_mask, dtype=torch.bool, device=device)
    long = torch.tensor(raw.long_history_sids, dtype=torch.long, device=device)
    long_mask = torch.tensor(raw.long_history_padding_mask, dtype=torch.bool, device=device)
    targets = torch.tensor(raw.target_sids, dtype=torch.long, device=device)
    instruction_features = features[
        [feature_index[key] for key in keys]
    ].to(device=device, dtype=torch.float32)
    scenarios = torch.zeros(len(validation), dtype=torch.long, device=device)
    triggers = short[:, -1, :]
    trie = PrefixTrie.from_registry(registry)
    with torch.inference_mode():
        diagnostic = model(
            short, short_mask, target_sids=targets,
            scenario_ids=scenarios, instruction_features=instruction_features,
            trigger_sids=triggers, long_history_sids=long,
            long_history_padding_mask=long_mask,
        )
        beam = model.beam_search(
            short, short_mask, trie, beam_width=args.beam_width,
            scenario_ids=scenarios, instruction_features=instruction_features,
            trigger_sids=triggers, long_history_sids=long,
            long_history_padding_mask=long_mask,
        )
    if diagnostic.q2i_cosine is None:
        raise RuntimeError("checkpoint did not return per-sample Q2I cosine")

    selection_rows = []
    payload_by_key = {}
    repeat_eligible_count = igr_hit_count = beam_hit_count = 0
    for row_index, (sample, key) in enumerate(zip(validation, keys, strict=True)):
        evidence = history_evidence(sample)
        target_sid = tuple(raw.target_sids[row_index])
        selected_indices = diagnostic.igr_indices[row_index].cpu().tolist()
        selected_scores = diagnostic.igr_scores[row_index].cpu().tolist()
        selected_sids = [tuple(raw.long_history_sids[row_index][index]) for index in selected_indices]
        known_events = [
            event for event in sample.history if event.item_id in registry.item_to_sid
        ]
        short_events = known_events[-short_history:]
        long_events = known_events[:-len(short_events)][-long_history:]
        long_padding = long_history - len(long_events)
        selected_records = []
        for rank, (index, score, sid) in enumerate(
            zip(selected_indices, selected_scores, selected_sids, strict=True), start=1,
        ):
            event = long_events[index - long_padding]
            selected_records.append({
                "rank": rank,
                "long_index": index,
                "score": score,
                "sid": list(sid),
                "behavior": event.behavior.value,
                "hours_before_target": (
                    sample.target.timestamp_ms - event.timestamp_ms
                ) / 3_600_000.0,
                "target_sid_match": sid == target_sid,
            })
        valid_long_sids = [
            tuple(sid) for sid, masked in zip(
                raw.long_history_sids[row_index], raw.long_history_padding_mask[row_index],
                strict=True,
            ) if not masked
        ]
        repeat_eligible = target_sid in valid_long_sids
        igr_hit = target_sid in selected_sids
        beam_records = []
        beam_hit_rank = None
        for rank, (sid, score) in enumerate(zip(
            beam.semantic_ids[row_index].cpu().tolist(),
            beam.scores[row_index].cpu().tolist(), strict=True,
        ), start=1):
            matched_items = registry.items_for(sid)
            target_hit = sample.target.item_id in matched_items
            if target_hit and beam_hit_rank is None:
                beam_hit_rank = rank
            beam_records.append({
                "rank": rank,
                "sid": sid,
                "score": score,
                "legal": bool(matched_items),
                "collision_size": len(matched_items),
                "target_hit": target_hit,
            })
        q2i_cosine = float(diagnostic.q2i_cosine[row_index].cpu())
        reasoning_row = reasoning_by_key[key]
        reasoning = reasoning_row["reasoning"]
        payload_by_key[key] = {
            "target": {"behavior": sample.target.behavior.value, "sid": list(target_sid)},
            "history": evidence,
            "reasoning": {
                "intent": reasoning["intent"],
                "evidence": reasoning["evidence"],
                "retrieval_strategy": reasoning["retrieval_strategy"],
                "constraints": reasoning["constraints"],
                "instruction_text": reasoning_row["instruction_text"],
                "retrieval_plan_present_but_ignored": True,
            },
            "q2i": {"cosine": q2i_cosine},
            "igr": {
                "repeat_eligible": repeat_eligible,
                "target_sid_hit": igr_hit,
                "selected": selected_records,
            },
            "generation": {"target_hit_rank": beam_hit_rank, "beam": beam_records},
        }
        selection_rows.append({
            "sample_key": key,
            "target_behavior": sample.target.behavior.value,
            "q2i_cosine": q2i_cosine,
            "repeat_eligible": repeat_eligible,
            "igr_hit": igr_hit,
            "browse_only": set(evidence["behavior_counts"]) == {"view"},
            "beam_hit_rank": beam_hit_rank,
            "history_length": evidence["history_length"],
        })
        repeat_eligible_count += int(repeat_eligible)
        igr_hit_count += int(igr_hit)
        beam_hit_count += int(beam_hit_rank is not None)

    selected_rows, coverage_keys = select_representative_rows(selection_rows)
    records = []
    case_id_by_key = {}
    for index, row in enumerate(selected_rows, start=1):
        key = str(row["sample_key"])
        case_id = f"v1-review-{index:03d}"
        case_id_by_key[key] = case_id
        roles = list(row["selection_roles"])
        records.append({
            "case_id": case_id,
            "selection_roles": roles,
            "selection_labels": [ROLE_LABELS[role] for role in roles],
            **payload_by_key[key],
        })
    coverage = {
        role: case_id_by_key.get(key) if key is not None else None
        for role, key in coverage_keys.items()
    }
    q2i_values = [float(row["q2i_cosine"]) for row in selection_rows]
    summary = {
        "selection_policy": "fixed_positive_negative_behavior_and_alignment_roles",
        "coverage": coverage,
        "aggregate": {
            "evaluated": len(validation),
            "representative_cases": len(records),
            "repeat_eligible": repeat_eligible_count,
            "igr_hits": igr_hit_count,
            "beam_hits": beam_hit_count,
            "q2i_min": min(q2i_values),
            "q2i_max": max(q2i_values),
        },
        "privacy": "no user IDs or raw item IDs",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    markdown_path = args.output.with_suffix(".md")
    summary_path = args.output.with_suffix(".summary.json")
    markdown_path.write_text(render_markdown(records, summary), encoding="utf-8")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(
        f"OK device={device} evaluated={len(validation)} representative={len(records)} "
        f"repeat_eligible={repeat_eligible_count} igr_hits={igr_hit_count} "
        f"beam_hits={beam_hit_count} q2i_min={min(q2i_values):.6f} "
        f"q2i_max={max(q2i_values):.6f} coverage={json.dumps(coverage, sort_keys=True)} "
        f"jsonl={args.output} markdown={markdown_path} summary={summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
