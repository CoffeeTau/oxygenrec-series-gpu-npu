#!/usr/bin/env python3
"""Generate and export a few auditable Qwen reasoning instructions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from oxygenrec.llm_features import build_behavior_prompt
from oxygenrec.llm_reasoning import FrozenLLMReasoningGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/review/qwen_reasoning_cases.jsonl"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    cases = [
        {
            "case_id": "high-intent",
            "history_length": 41,
            "behavior_counts": {"view": 38, "addtocart": 2, "transaction": 1},
            "recent_behaviors": ("view", "addtocart", "view", "transaction", "view"),
            "repeated_item_kinds": 10,
        },
        {
            "case_id": "browse-only",
            "history_length": 41,
            "behavior_counts": {"view": 41},
            "recent_behaviors": ("view", "view", "view", "view", "view"),
            "repeated_item_kinds": 2,
        },
        {
            "case_id": "repeat-browse",
            "history_length": 120,
            "behavior_counts": {"view": 117, "addtocart": 2, "transaction": 1},
            "recent_behaviors": ("view", "view", "view", "view", "view"),
            "repeated_item_kinds": 22,
        },
    ]
    prompts = [build_behavior_prompt(**{
        key: value for key, value in case.items() if key != "case_id"
    }) for case in cases]
    generator = FrozenLLMReasoningGenerator(
        args.model_path, device=args.device, dtype=args.dtype,
    )
    outputs = generator.generate(prompts, max_new_tokens=384)
    records = []
    for case, prompt, output in zip(cases, prompts, outputs, strict=True):
        records.append({
            "case_id": case["case_id"], "input_evidence": case,
            "prompt": prompt, "reasoning": output.parsed,
            "raw_text": output.raw_text,
            "review_questions": [
                "evidence是否全部来自输入？",
                "是否泄漏或猜测下一行为/目标商品？",
                "retrieval_plan是否与证据一致且可直接转换为检索约束？",
            ],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    markdown = ["# Qwen结构化Reasoning Review", ""]
    for record in records:
        reasoning = record["reasoning"]
        markdown.extend([
            f"## {record['case_id']}", "",
            f"- 输入证据：`{record['input_evidence']}`",
            f"- Intent：{reasoning['intent']}",
            f"- Evidence：`{reasoning['evidence']}`",
            f"- Retrieval strategy：{reasoning['retrieval_strategy']}",
            f"- Retrieval plan：`{reasoning['retrieval_plan']}`",
            f"- Constraints：`{reasoning['constraints']}`", "",
            "### 人工检查", "",
        ])
        markdown.extend(f"- [ ] {item}" for item in record["review_questions"])
        markdown.extend(["", "---", ""])
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    peak = (
        torch.cuda.max_memory_allocated(args.device) / 1024**3
        if args.device.startswith("cuda") else 0.0
    )
    print(
        f"OK device={args.device} cases={len(records)} schema_valid={len(records)} "
        f"peak_allocated_gib={peak:.3f} jsonl={args.output} markdown={markdown_path}"
    )


if __name__ == "__main__":
    main()
