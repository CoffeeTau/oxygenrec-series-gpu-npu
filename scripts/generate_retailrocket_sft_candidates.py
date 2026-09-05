#!/usr/bin/env python3
"""从 RetailRocket train split 批量生成无目标泄漏的 Qwen SFT 待审候选。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from oxygenrec.data.events import load_retailrocket_events
from oxygenrec.data.temporal import Split, TemporalBoundaries, build_next_item_samples
from oxygenrec.llm_features import build_behavior_prompt
from oxygenrec.llm_reasoning import FrozenLLMReasoningGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--max-history", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cases < 1 or args.batch_size < 1 or args.max_history < 1:
        raise ValueError("cases, batch-size and max-history must be positive")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    boundaries = TemporalBoundaries(**payload["boundaries"])
    samples = build_next_item_samples(
        load_retailrocket_events(args.events), boundaries,
        min_history=1, max_history=args.max_history,
        max_samples_per_split={
            Split.TRAIN: args.cases,
            Split.VALIDATION: 1,
            Split.TEST: 1,
        },
        sample_seed=args.sample_seed,
    )
    train_samples = [sample for sample in samples if sample.split is Split.TRAIN]
    if len(train_samples) != args.cases:
        raise RuntimeError(f"expected {args.cases} train cases, got {len(train_samples)}")

    generator = FrozenLLMReasoningGenerator(
        args.model_path, device=args.device, dtype="bfloat16",
    )
    records = []
    for start in range(0, len(train_samples), args.batch_size):
        batch = train_samples[start:start + args.batch_size]
        evidence_rows = []
        prompts = []
        for sample in batch:
            behaviors = [event.behavior.value for event in sample.history]
            item_counts = Counter(event.item_id for event in sample.history)
            evidence = {
                "history_length": len(sample.history),
                "behavior_counts": dict(Counter(behaviors)),
                "recent_behaviors": behaviors[-5:],
                "repeated_item_kinds": sum(count > 1 for count in item_counts.values()),
            }
            evidence_rows.append(evidence)
            prompts.append(build_behavior_prompt(**evidence))
        generated = generator.generate(prompts, max_new_tokens=384)
        for offset, (evidence, prompt, output) in enumerate(
            zip(evidence_rows, prompts, generated, strict=True), start=start + 1,
        ):
            records.append({
                "case_id": f"train-{offset:06d}",
                "source": "retailrocket_train_split",
                "review_status": "pending",
                "input_evidence": evidence,
                "prompt": prompt,
                "reasoning": output.parsed,
                "raw_text": output.raw_text,
                "sampling": {"sample_seed": args.sample_seed, "split": "train"},
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(
        f"OK device={args.device} candidates={len(records)} split=train "
        f"review_status=pending output={args.output}"
    )


if __name__ == "__main__":
    main()
