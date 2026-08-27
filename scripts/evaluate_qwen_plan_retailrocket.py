#!/usr/bin/env python3
"""Paired baseline/Plan IGR evaluation on real RetailRocket histories."""

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

from oxygenrec.data.events import load_retailrocket_events
from oxygenrec.data.model_inputs import build_long_short_sid_model_batch
from oxygenrec.data.temporal import Split, TemporalBoundaries, build_next_item_samples
from oxygenrec.llm_features import build_behavior_prompt
from oxygenrec.llm_reasoning import FrozenLLMReasoningGenerator
from oxygenrec.model import OxygenRECConfig, OxygenRECModel
from oxygenrec.retrieval_planning import compile_retrieval_plan
from oxygenrec.sid import SIDRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("outputs/review/qwen_plan_retailrocket.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    registry = SIDRegistry.from_json(args.sid_registry)
    known_config = {field.name for field in fields(OxygenRECConfig)}
    config = OxygenRECConfig(**{
        key: value for key, value in payload["model_config"].items() if key in known_config
    })
    if config.igr_top_k < 1:
        raise ValueError("checkpoint does not contain IGR")
    model = OxygenRECModel(config).to(device).eval()
    model.load_state_dict(payload["model_state"])
    boundaries = TemporalBoundaries(**payload["boundaries"])
    events = [
        event for event in load_retailrocket_events(args.events)
        if event.item_id in registry.item_to_sid
    ]
    long_history = int(payload["args"].get("long_history", 100))
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
        raise RuntimeError(f"expected {args.cases} validation cases, got {len(validation)}")

    evidence_rows = []
    prompts = []
    for sample in validation:
        behaviors = [event.behavior.value for event in sample.history]
        item_counts = Counter(event.item_id for event in sample.history)
        evidence = {
            "history_length": len(sample.history),
            "behavior_counts": dict(Counter(behaviors)),
            "recent_behaviors": tuple(behaviors[-5:]),
            "repeated_item_kinds": sum(count > 1 for count in item_counts.values()),
        }
        evidence_rows.append(evidence)
        prompts.append(build_behavior_prompt(**evidence))
    generator = FrozenLLMReasoningGenerator(args.model_path, device=args.device, dtype="bfloat16")
    reasoning_outputs = generator.generate(prompts, max_new_tokens=384)
    del generator
    torch.cuda.empty_cache()

    records = []
    changed = before_target_hits = after_target_hits = 0
    before_high_intent = after_high_intent = 0
    for index, (sample, evidence, reasoning) in enumerate(
        zip(validation, evidence_rows, reasoning_outputs, strict=True), start=1,
    ):
        raw = build_long_short_sid_model_batch(
            [sample], registry, short_history_items=config.max_history_items,
            long_history_items=long_history, minimum_long_history_items=config.igr_top_k,
        )
        short = torch.tensor(raw.short_history_sids, device=device)
        short_mask = torch.tensor(raw.short_history_padding_mask, dtype=torch.bool, device=device)
        long = torch.tensor(raw.long_history_sids, device=device)
        long_mask = torch.tensor(raw.long_history_padding_mask, dtype=torch.bool, device=device)
        long_behaviors = torch.tensor(raw.long_history_behavior_ids, device=device)
        scenario = torch.tensor(raw.scenario_ids, device=device)
        target = tuple(raw.target_sids[0])
        plan = compile_retrieval_plan(reasoning.parsed["retrieval_plan"], evidence["behavior_counts"])
        with torch.inference_mode():
            baseline = model(
                short, short_mask, scenario_ids=scenario,
                long_history_sids=long, long_history_padding_mask=long_mask,
            )
            planned = model(
                short, short_mask, scenario_ids=scenario,
                long_history_sids=long, long_history_padding_mask=long_mask,
                long_history_behavior_ids=long_behaviors, retrieval_plans=[plan],
            )
        before = baseline.igr_indices[0].tolist()
        after = planned.igr_indices[0].tolist()
        before_sids = [tuple(raw.long_history_sids[0][item]) for item in before]
        after_sids = [tuple(raw.long_history_sids[0][item]) for item in after]
        before_behaviors = [raw.long_history_behavior_ids[0][item] for item in before]
        after_behaviors = [raw.long_history_behavior_ids[0][item] for item in after]
        changed += int(before != after)
        before_target_hits += int(target in before_sids)
        after_target_hits += int(target in after_sids)
        before_high_intent += sum(item in {1, 2} for item in before_behaviors)
        after_high_intent += sum(item in {1, 2} for item in after_behaviors)
        records.append({
            "case": index, "input_evidence": evidence,
            "reasoning": reasoning.parsed, "baseline_indices": before,
            "planned_indices": after, "target_sid_hit_before": target in before_sids,
            "target_sid_hit_after": target in after_sids,
            "high_intent_selected_before": sum(item in {1, 2} for item in before_behaviors),
            "high_intent_selected_after": sum(item in {1, 2} for item in after_behaviors),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    print(
        f"OK device={device} cases={len(records)} changed={changed} "
        f"target_hits={before_target_hits}->{after_target_hits} "
        f"high_intent_selected={before_high_intent}->{after_high_intent} output={args.output}"
    )


if __name__ == "__main__":
    main()
