#!/usr/bin/env python3
"""把 Qwen Reasoning review JSONL 转成可审计的 messages 格式 SFT JSONL。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oxygenrec.sft_data import build_reasoning_sft_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-pending", action="store_true",
        help="只用于生成待审核候选集；正式训练集不得使用此参数。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lines = args.input.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("input JSONL is empty")
    examples = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            examples.append(build_reasoning_sft_example(
                record, require_approved=not args.allow_pending,
            ))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid SFT source at line {line_number}: {error}") from error
    if not examples:
        raise ValueError("no valid SFT examples were produced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in examples),
        encoding="utf-8",
    )
    status = "candidate_pending_review" if args.allow_pending else "train_ready_approved"
    print(f"OK examples={len(examples)} status={status} output={args.output}")


if __name__ == "__main__":
    main()
