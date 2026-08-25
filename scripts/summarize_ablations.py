#!/usr/bin/env python3
"""Summarize final-epoch ablation metrics across random seeds."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    grouped = defaultdict(list)
    for path in sorted(args.root.glob("seed-*/**/metrics.jsonl")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if records:
            grouped[records[-1]["variant"]].append(records[-1])
    expected = {"base", "instruction", "igr", "igr_q2i"}
    if set(grouped) != expected:
        raise RuntimeError(f"incomplete variants: found={sorted(grouped)}")
    summary = {}
    for variant in ("base", "instruction", "igr", "igr_q2i"):
        rows = grouped[variant]
        seeds = sorted(row["seed"] for row in rows)
        if len(seeds) < 2:
            raise RuntimeError(f"{variant} needs at least two seeds")
        values = {
            "hr1": [row["hit_rate"]["1"] for row in rows],
            "hr5": [row["hit_rate"]["5"] for row in rows],
            "mrr": [row["mrr"] for row in rows],
            "ndcg": [row["ndcg"] for row in rows],
            "ntp_loss": [row["ntp_loss"] for row in rows],
        }
        summary[variant] = {"seeds": seeds}
        fields = []
        for name, samples in values.items():
            mean = statistics.fmean(samples)
            std = statistics.stdev(samples)
            summary[variant][name] = {"mean": mean, "std": std}
            fields.append(f"{name}={mean:.6f}+/-{std:.6f}")
        print(f"variant={variant} seeds={seeds} " + " ".join(fields))
    (args.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
