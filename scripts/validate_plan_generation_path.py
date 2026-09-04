#!/usr/bin/env python3
"""验证 Retrieval Plan 会进入 greedy、beam 和候选 log-prob 三条推理路径。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from oxygenrec.model import OxygenRECConfig, OxygenRECModel
from oxygenrec.retrieval_planning import compile_retrieval_plan
from oxygenrec.sid import PrefixTrie


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(41)
    model = OxygenRECModel(OxygenRECConfig(
        sid_width=16, hidden_size=16, attention_heads=4,
        encoder_layers=1, decoder_layers=1, feedforward_size=32,
        dropout=0.0, max_history_items=3, igr_top_k=2,
        q2i_dimension=8,
    )).to(device).eval()

    short = torch.tensor([
        [[1, 2, 3], [4, 5, 6]],
        [[2, 3, 4], [5, 6, 7]],
    ], device=device)
    short_mask = torch.zeros(2, 2, dtype=torch.bool, device=device)
    long = torch.tensor([
        [[7, 8, 9], [3, 4, 5], [6, 7, 8], [0, 0, 0]],
        [[8, 9, 10], [1, 3, 5], [4, 6, 8], [0, 0, 0]],
    ], device=device)
    long_mask = torch.tensor([[False, False, False, True]] * 2, device=device)
    long_behaviors = torch.tensor([[0, 2, 0, 0]] * 2, device=device)
    plans = [compile_retrieval_plan(
        {"priority_behaviors": ["transaction"], "recency": "balanced",
         "prefer_repeated_items": False, "diversity": "low"},
        {"view": 2, "transaction": 1},
    )] * 2
    trie = PrefixTrie([(1, 2, 3), (7, 8, 9), (8, 9, 10)])
    shared = dict(
        long_history_sids=long,
        long_history_padding_mask=long_mask,
        long_history_behavior_ids=long_behaviors,
        retrieval_plans=plans,
    )

    calls: list[str] = []
    original = model._augment_history

    def recording_augment(self, *positional, **keywords):
        if keywords.get("retrieval_plans") is None:
            raise AssertionError("generation path dropped retrieval_plans")
        if keywords.get("long_history_behavior_ids") is None:
            raise AssertionError("generation path dropped long-history behaviors")
        calls.append("plan")
        return original(*positional, **keywords)

    model._augment_history = MethodType(recording_augment, model)
    generated = model.generate(short, short_mask, trie, **shared)
    beams = model.beam_search(short, short_mask, trie, beam_width=2, **shared)
    candidate_log_probs = model.candidate_log_probs(
        short, short_mask, beams.semantic_ids, **shared,
    )

    if len(calls) != 3:
        raise AssertionError(f"expected three planned IGR calls, got {len(calls)}")
    if not all(trie.contains(row) for row in generated.tolist()):
        raise AssertionError("greedy generation returned an illegal SID")
    if not torch.isfinite(beams.scores).all() or not torch.isfinite(candidate_log_probs).all():
        raise AssertionError("beam or candidate log-prob contains non-finite values")
    print(
        f"OK device={device} planned_paths={len(calls)} "
        f"generated_shape={tuple(generated.shape)} "
        f"beams_shape={tuple(beams.semantic_ids.shape)} "
        f"log_probs_shape={tuple(candidate_log_probs.shape)} legal=True"
    )


if __name__ == "__main__":
    main()
