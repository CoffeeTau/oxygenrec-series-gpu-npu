#!/usr/bin/env python3
"""Export auditable Reasoning -> IGR -> Generation review cases as JSONL."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.data import (
    Split, TemporalBoundaries, build_long_short_sid_model_batch,
    build_next_item_samples, load_retailrocket_events,
)
from oxygenrec.model import OxygenRECConfig, OxygenRECModel
from oxygenrec.sid import PrefixTrie, SIDRegistry


def anonymize(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def reasoning_proxy(sample) -> dict:
    recent = sample.history[-5:]
    counts = Counter(event.behavior.value for event in sample.history)
    high_intent = [
        event for event in recent if event.behavior.value in {"addtocart", "transaction"}
    ]
    repeated = Counter(event.item_id for event in sample.history)
    repeated_count = sum(count > 1 for count in repeated.values())
    if high_intent:
        intent = "最近存在加购或购买等高意图行为，优先检索相关历史商品。"
    else:
        intent = "最近以浏览行为为主，结合重复访问和长期历史判断兴趣。"
    return {
        "kind": "deterministic_public_proxy_not_slow_llm_output",
        "observations": {
            "history_length": len(sample.history),
            "behavior_counts": dict(sorted(counts.items())),
            "recent_high_intent_events": len(high_intent),
            "repeated_item_kinds": repeated_count,
        },
        "instruction_zh": intent,
    }


def render_markdown(records: list[dict]) -> str:
    lines = [
        "# OxygenREC LLM 交叉链路样例 Review",
        "",
        "> 本报告用于人工检查 Reasoning -> Retrieval -> Alignment -> Generation。",
        "> reasoning 是确定性公开代理；实际模型未调用 Slow LLM。",
        "",
    ]
    for record in records:
        target = record["target"]
        proxy = record["reasoning_review_proxy"]
        conditioning = record["actual_model_conditioning"]
        igr = record["igr"]
        lines.extend([
            f"## {record['case_id']}", "",
            f"- 匿名用户：`{record['anonymized_user']}`",
            f"- 目标行为：`{target['behavior']}`",
            f"- 目标 SID：`{target['sid']}`",
            f"- Reasoning 类型：`{proxy['kind']}`",
            f"- Review instruction：{proxy['instruction_zh']}",
            f"- 模型实际 reasoning source：`{conditioning['reasoning_source']}`",
            f"- Slow LLM 文本实际输入：`{conditioning['slow_llm_text_used']}`",
            "",
            "### 行为证据", "",
            f"- 历史长度：{proxy['observations']['history_length']}",
            f"- 行为计数：`{proxy['observations']['behavior_counts']}`",
            f"- 最近高意图事件：{proxy['observations']['recent_high_intent_events']}",
            f"- 重复商品种类：{proxy['observations']['repeated_item_kinds']}",
            "", "### IGR Top-K", "",
            "| rank | long index | score | selected SID | target levels | behavior | hours ago | high intent |",
            "|---:|---:|---:|---|---:|---|---:|---|",
        ])
        for rank, evidence in enumerate(igr["selected_evidence"], start=1):
            event = evidence["event"] or {}
            hours = event.get("hours_before_target")
            lines.append(
                f"| {rank} | {evidence['long_history_index']} | {evidence['score']:.6f} | "
                f"`{evidence['sid']}` | {evidence['sid_level_matches_target']}/3 | "
                f"{event.get('behavior', 'padding')} | "
                f"{hours:.2f} | {event.get('high_intent', False)} |"
                if hours is not None else
                f"| {rank} | {evidence['long_history_index']} | {evidence['score']:.6f} | "
                f"`{evidence['sid']}` | {evidence['sid_level_matches_target']}/3 | padding | - | False |"
            )
        lines.extend([
            "",
            f"- IGR 是否取回目标 SID：`{igr['retrieved_target_sid']}`",
            f"- Q2I alignment loss：`{record['q2i']['batch_alignment_loss']}`",
            f"- Q2I cosine alignment proxy：`{record['q2i']['cosine_alignment_proxy']}`",
            "", "### Constrained Beam", "",
            "| rank | SID | score | legal | collision | target hit |",
            "|---:|---|---:|---|---:|---|",
        ])
        for candidate in record["generation"]["beam"]:
            lines.append(
                f"| {candidate['rank']} | `{candidate['sid']}` | "
                f"{candidate['score']:.6f} | {candidate['legal']} | "
                f"{candidate['collision_size']} | {candidate['target_hit']} |"
            )
        lines.extend(["", "### 人工检查问题", ""])
        lines.extend(f"- [ ] {question}" for question in record["review_questions"])
        lines.extend(["", "---", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=5)
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    registry = SIDRegistry.from_json(args.sid_registry)
    known = {field.name for field in fields(OxygenRECConfig)}
    config = OxygenRECConfig(**{
        key: value for key, value in checkpoint["model_config"].items() if key in known
    })
    if config.igr_top_k < 1:
        raise ValueError("review checkpoint must contain IGR")
    model = OxygenRECModel(config).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    boundaries = TemporalBoundaries(**checkpoint["boundaries"])
    events = [
        event for event in load_retailrocket_events(args.events)
        if event.item_id in registry.item_to_sid
    ]
    long_history = int(checkpoint["args"].get("long_history", 100))
    samples = build_next_item_samples(
        events, boundaries,
        min_history=config.max_history_items + config.igr_top_k,
        max_history=config.max_history_items + long_history,
        max_samples_per_split={Split.TRAIN: 1, Split.VALIDATION: args.cases, Split.TEST: 1},
        sample_seed=2026,
    )
    by_split = defaultdict(list)
    for sample in samples:
        by_split[sample.split].append(sample)
    validation = by_split[Split.VALIDATION]
    if len(validation) != args.cases:
        raise RuntimeError(f"expected {args.cases} cases, got {len(validation)}")
    trie = PrefixTrie.from_registry(registry)
    records = []
    repeat_hits = beam_hits = 0
    for case_index, sample in enumerate(validation, start=1):
        raw = build_long_short_sid_model_batch(
            [sample], registry, short_history_items=config.max_history_items,
            long_history_items=long_history,
            minimum_long_history_items=config.igr_top_k,
        )
        short = torch.tensor(raw.short_history_sids, device=device)
        short_mask = torch.tensor(raw.short_history_padding_mask, dtype=torch.bool, device=device)
        long = torch.tensor(raw.long_history_sids, device=device)
        long_mask = torch.tensor(raw.long_history_padding_mask, dtype=torch.bool, device=device)
        target = torch.tensor(raw.target_sids, device=device)
        scenario = torch.tensor(raw.scenario_ids, device=device)
        trigger = short[:, -1]
        with torch.inference_mode():
            diagnostic = model(
                short, short_mask, target_sids=target, scenario_ids=scenario,
                trigger_sids=trigger, long_history_sids=long,
                long_history_padding_mask=long_mask,
            )
            beam = model.beam_search(
                short, short_mask, trie, beam_width=args.beam_width,
                scenario_ids=scenario, trigger_sids=trigger,
                long_history_sids=long, long_history_padding_mask=long_mask,
            )
        selected_indices = diagnostic.igr_indices[0].cpu().tolist()
        selected_scores = diagnostic.igr_scores[0].cpu().tolist()
        selected_sids = [raw.long_history_sids[0][index] for index in selected_indices]
        target_codes = tuple(raw.target_sids[0])
        known_events = [
            event for event in sample.history if event.item_id in registry.item_to_sid
        ]
        short_events = known_events[-config.max_history_items:]
        long_events = known_events[:-len(short_events)][-long_history:]
        long_padding = long_history - len(long_events)
        selected_evidence = []
        for index, score, sid in zip(
            selected_indices, selected_scores, selected_sids, strict=True
        ):
            event = long_events[index - long_padding] if index >= long_padding else None
            selected_evidence.append({
                "long_history_index": index,
                "score": score,
                "sid": list(sid),
                "sid_level_matches_target": sum(
                    left == right for left, right in zip(sid, target_codes, strict=True)
                ),
                "event": None if event is None else {
                    "anonymized_item": anonymize(event.item_id),
                    "behavior": event.behavior.value,
                    "hours_before_target": (
                        sample.target.timestamp_ms - event.timestamp_ms
                    ) / 3_600_000.0,
                    "high_intent": event.behavior.value in {"addtocart", "transaction"},
                },
            })
        repeat_hit = target_codes in selected_sids
        beam_rows = []
        for rank, (sid, score) in enumerate(
            zip(beam.semantic_ids[0].cpu().tolist(), beam.scores[0].cpu().tolist(), strict=True),
            start=1,
        ):
            items = registry.items_for(sid)
            hit = sample.target.item_id in items
            beam_rows.append({
                "rank": rank, "sid": sid, "score": score,
                "legal": bool(items), "target_hit": hit,
                "collision_size": len(items),
            })
        repeat_hits += int(repeat_hit)
        beam_hits += int(any(row["target_hit"] for row in beam_rows))
        records.append({
            "case_id": f"review-{case_index:03d}",
            "anonymized_user": anonymize(sample.user_id),
            "target": {
                "anonymized_item": anonymize(sample.target.item_id),
                "behavior": sample.target.behavior.value,
                "sid": list(target_codes),
            },
            "reasoning_review_proxy": reasoning_proxy(sample),
            "actual_model_conditioning": {
                "scenario_id": raw.scenario_ids[0],
                "trigger_sid": list(raw.short_history_sids[0][-1]),
                "reasoning_source": "learned_fallback_plus_history_context_attention",
                "slow_llm_text_used": False,
            },
            "igr": {
                "top_k": config.igr_top_k,
                "selected_long_history_indices": selected_indices,
                "scores": selected_scores,
                "selected_sids": [list(sid) for sid in selected_sids],
                "selected_evidence": selected_evidence,
                "retrieved_target_sid": repeat_hit,
            },
            "q2i": {
                "batch_alignment_loss": (
                    float(diagnostic.q2i_alignment_loss)
                    if diagnostic.q2i_alignment_loss is not None else None
                ),
                "cosine_alignment_proxy": (
                    -float(diagnostic.q2i_alignment_loss)
                    if diagnostic.q2i_alignment_loss is not None else None
                ),
                "note": "single-case cosine alignment proxy; not a search-query label",
            },
            "generation": {"beam": beam_rows},
            "review_questions": [
                "结构化reasoning是否被行为证据支持？",
                "IGR top-k是否包含与目标或高意图行为相关的历史SID？",
                "beam候选是否合法，排序是否与检索证据一致？",
            ],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(render_markdown(records), encoding="utf-8")
    print(
        f"OK device={device} cases={len(records)} igr_repeat_hits={repeat_hits} "
        f"beam_target_hits={beam_hits} jsonl={args.output} markdown={markdown_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
