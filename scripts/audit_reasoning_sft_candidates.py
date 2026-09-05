#!/usr/bin/env python3
"""审计 SFT Reasoning 候选的格式、可执行性和分布，只输出聚合统计。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oxygenrec.llm_reasoning import parse_reasoning_json
from oxygenrec.retrieval_planning import compile_retrieval_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-samples", type=int, default=6)
    return parser.parse_args()


def audit(records: list[dict[str, object]], review_samples: int) -> dict[str, object]:
    if not records or review_samples < 1:
        raise ValueError("records and review-samples must be non-empty/positive")
    statuses: Counter[str] = Counter()
    priorities: Counter[tuple[str, ...]] = Counter()
    recencies: Counter[str] = Counter()
    diversities: Counter[str] = Counter()
    repeats: Counter[bool] = Counter()
    target_behaviors: Counter[str] = Counter()
    evidence_cohorts: Counter[str] = Counter()
    assistant_outputs: Counter[str] = Counter()
    evidence_lengths = []
    review_pool: list[tuple[int, int, int, str]] = []

    for record in records:
        case_id = record.get("case_id")
        evidence = record.get("input_evidence")
        reasoning = record.get("reasoning")
        if not isinstance(case_id, str) or not isinstance(evidence, dict) or not isinstance(reasoning, dict):
            raise ValueError("each record needs case_id, input_evidence and reasoning objects")
        parsed = parse_reasoning_json(json.dumps(reasoning, ensure_ascii=False))
        counts = evidence.get("behavior_counts")
        if not isinstance(counts, dict):
            raise ValueError(f"case {case_id}: missing behavior_counts")
        compile_retrieval_plan(parsed["retrieval_plan"], counts)
        plan = parsed["retrieval_plan"]
        statuses[str(record.get("review_status", "pending"))] += 1
        evidence_cohorts[str(record.get("evidence_cohort", "unspecified"))] += 1
        priorities[tuple(plan["priority_behaviors"])] += 1
        recencies[str(plan["recency"])] += 1
        diversities[str(plan["diversity"])] += 1
        repeats[bool(plan["prefer_repeated_items"])] += 1
        for behavior, count in counts.items():
            if int(count) > 0:
                target_behaviors[str(behavior)] += 1
        canonical = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
        assistant_outputs[canonical] += 1
        evidence_lengths.append(len(parsed["evidence"]))
        high_intent = int(counts.get("transaction", 0)) + int(counts.get("addtocart", 0))
        repeated = int(evidence.get("repeated_item_kinds", 0))
        history = int(evidence.get("history_length", 0))
        review_pool.append((high_intent, repeated, history, case_id))

    # 优先抽查高意图、重复访问和长历史样例；只输出匿名 case_id，不输出原始行为文本。
    review_pool.sort(reverse=True)
    selected = [item[3] for item in review_pool[:review_samples]]
    duplicate_rows = sum(count - 1 for count in assistant_outputs.values() if count > 1)
    return {
        "cases": len(records),
        "schema_valid": len(records),
        "executable_plans": len(records),
        "review_status": dict(sorted(statuses.items())),
        "priority_patterns": {str(key): value for key, value in sorted(priorities.items())},
        "recency": dict(sorted(recencies.items())),
        "diversity": dict(sorted(diversities.items())),
        "prefer_repeated_items": {str(key): value for key, value in sorted(repeats.items())},
        "behavior_presence": dict(sorted(target_behaviors.items())),
        "evidence_cohorts": dict(sorted(evidence_cohorts.items())),
        "unique_reasoning_rate": len(assistant_outputs) / len(records),
        "duplicate_reasoning_rows": duplicate_rows,
        "evidence_count_min": min(evidence_lengths),
        "evidence_count_max": max(evidence_lengths),
        "review_sample_case_ids": selected,
    }


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = audit(records, args.review_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
