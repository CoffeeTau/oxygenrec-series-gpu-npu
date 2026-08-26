#!/usr/bin/env python3
"""Evaluate frozen property-side-information retrieval before model integration."""

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from oxygenrec.data import Split, TemporalBoundaries, build_next_item_samples, load_retailrocket_events
from oxygenrec.sid import SIDRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--short-history", type=int, default=20)
    parser.add_argument("--long-history", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--validation-samples", type=int, default=2000)
    args = parser.parse_args()

    registry = SIDRegistry.from_json(args.sid_registry)
    item_ids = (args.embedding_dir / "item_ids.txt").read_text(encoding="utf-8").splitlines()
    vectors = np.load(args.embedding_dir / "item_embeddings.npy", allow_pickle=False)
    if vectors.shape[0] != len(item_ids):
        raise ValueError("item ID and embedding row counts differ")
    row_for = {item_id: row for row, item_id in enumerate(item_ids)}
    all_events = list(load_retailrocket_events(args.events))
    minimum = min(event.timestamp_ms for event in all_events)
    maximum = max(event.timestamp_ms for event in all_events)
    duration = maximum - minimum + 1
    boundaries = TemporalBoundaries(
        minimum + duration * 8 // 10, minimum + duration * 9 // 10
    )
    events = [
        event for event in all_events
        if event.item_id in registry.item_to_sid and event.item_id in row_for
    ]
    samples = build_next_item_samples(
        events, boundaries,
        min_history=args.short_history + args.top_k,
        max_history=args.short_history + args.long_history,
        max_samples_per_split={Split.TRAIN: 1, Split.VALIDATION: args.validation_samples, Split.TEST: 1},
        sample_seed=args.seed,
    )
    validation = [sample for sample in samples if sample.split is Split.VALIDATION]
    eligible = retrieved = recent_hits = 0
    random_expected_hits = 0.0
    for sample in validation:
        history_ids = [event.item_id for event in sample.history]
        short_ids = history_ids[-args.short_history:]
        long_ids = history_ids[:-len(short_ids)][-args.long_history:]
        query = vectors[[row_for[item] for item in short_ids]].mean(axis=0)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            continue
        long_vectors = vectors[[row_for[item] for item in long_ids]]
        scores = long_vectors @ (query / query_norm)
        selected = np.argpartition(scores, -args.top_k)[-args.top_k:]
        target_sid = registry.sid_for(sample.target.item_id).codes
        matches = np.array([registry.sid_for(item).codes == target_sid for item in long_ids])
        if not matches.any():
            continue
        eligible += 1
        retrieved += int(matches[selected].any())
        recent_hits += int(matches[-args.top_k:].any())
        valid_count = len(long_ids)
        match_count = int(matches.sum())
        misses = valid_count - match_count
        miss_probability = (
            math.comb(misses, args.top_k) / math.comb(valid_count, args.top_k)
            if misses >= args.top_k else 0.0
        )
        random_expected_hits += 1.0 - miss_probability
    if not eligible:
        raise RuntimeError("no repeat-eligible validation samples")
    recall = retrieved / eligible
    random_recall = random_expected_hits / eligible
    print(
        f"OK property_retrieval repeat_recall={recall:.6f} "
        f"repeat_recent={recent_hits / eligible:.6f} "
        f"repeat_random={random_recall:.6f} "
        f"repeat_lift={recall / random_recall:.6f} repeat_eligible={eligible}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
