#!/usr/bin/env python3
"""Summarize paired Qwen Plan retrieval cases without rerunning either model."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("paired evaluation file is empty")
    plan_counts = Counter()
    priority_counts = Counter()
    target_gains = []
    target_losses = []
    high_intent_deltas = []
    cases = []
    for row in rows:
        plan = row["reasoning"]["retrieval_plan"]
        plan_counts[(plan["recency"], plan["prefer_repeated_items"], plan["diversity"])] += 1
        priority_counts[tuple(plan["priority_behaviors"])] += 1
        before_hit = bool(row["target_sid_hit_before"])
        after_hit = bool(row["target_sid_hit_after"])
        if after_hit and not before_hit:
            target_gains.append(row["case"])
        if before_hit and not after_hit:
            target_losses.append(row["case"])
        delta = int(row["high_intent_selected_after"]) - int(row["high_intent_selected_before"])
        if delta:
            high_intent_deltas.append((row["case"], delta))
        cases.append({
            "case": row["case"],
            "priority": plan["priority_behaviors"],
            "recency": plan["recency"],
            "repeat": plan["prefer_repeated_items"],
            "diversity": plan["diversity"],
            "target": f"{int(before_hit)}->{int(after_hit)}",
            "high_intent_delta": delta,
        })
    summary = {
        "cases": len(rows),
        "priority_counts": {str(key): value for key, value in priority_counts.items()},
        "control_counts": {str(key): value for key, value in plan_counts.items()},
        "target_gain_cases": target_gains,
        "target_loss_cases": target_losses,
        "high_intent_delta_cases": high_intent_deltas,
        "paired_cases": cases,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
