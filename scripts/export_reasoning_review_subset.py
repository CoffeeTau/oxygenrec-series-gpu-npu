#!/usr/bin/env python3
"""按聚合审计选出的匿名 case_id 导出少量 Reasoning 人工复核页。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = {
        row["case_id"]: row
        for row in (
            json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    case_ids = audit["review_sample_case_ids"]
    missing = [case_id for case_id in case_ids if case_id not in records]
    if missing:
        raise ValueError(f"audit references missing cases: {missing}")
    markdown = ["# Qwen SFT 分层候选人工复核", ""]
    for case_id in case_ids:
        row = records[case_id]
        reasoning = row["reasoning"]
        markdown.extend([
            f"## {case_id}", "",
            f"- 证据分层：`{row.get('evidence_cohort', 'unspecified')}`",
            f"- 输入聚合证据：`{row['input_evidence']}`",
            f"- Intent：{reasoning['intent']}",
            f"- Evidence：`{reasoning['evidence']}`",
            f"- Retrieval strategy：{reasoning['retrieval_strategy']}",
            f"- Retrieval plan：`{reasoning['retrieval_plan']}`",
            f"- Constraints：`{reasoning['constraints']}`", "",
            "- [ ] Evidence均可由输入逐项支持",
            "- [ ] 没有猜测下一行为或目标商品",
            "- [ ] Plan与该证据分层相符",
            "- [ ] 与其他样例相比不是无意义模板复制", "", "---", "",
        ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(markdown), encoding="utf-8")
    print(f"OK review_cases={len(case_ids)} output={args.output}")


if __name__ == "__main__":
    main()
