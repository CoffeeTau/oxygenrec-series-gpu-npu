#!/usr/bin/env python3
"""Validate deterministic execution of Qwen retrieval plans on CUDA."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from oxygenrec.retrieval_planning import compile_retrieval_plan, execute_retrieval_plan
from oxygenrec.model import OxygenRECConfig, OxygenRECModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/review/qwen_reasoning_cases.jsonl"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    plans = [compile_retrieval_plan(row["reasoning"]["retrieval_plan"], row["input_evidence"]["behavior_counts"]) for row in records]
    device = torch.device(args.device)
    base_scores = torch.tensor([[.90, .89, .88, .87, .86]] * len(plans), device=device)
    sids = torch.tensor([[[1,1,1], [1,1,1], [2,2,2], [3,3,3], [4,4,4]]] * len(plans), device=device)
    behaviors = torch.tensor([[0, 0, 2, 1, 0]] * len(plans), device=device)
    mask = torch.zeros_like(base_scores, dtype=torch.bool)
    baseline = base_scores.topk(3, dim=1).indices
    selected, selected_scores, adjusted = execute_retrieval_plan(
        base_scores, sids, behaviors, mask, plans, top_k=3,
    )
    changed = (selected != baseline).any(dim=1)
    if not changed.any():
        raise RuntimeError("retrieval plans did not change any controlled selection")
    if not torch.isfinite(selected_scores).all() or not torch.isfinite(adjusted).all():
        raise RuntimeError("retrieval plan execution produced non-finite scores")
    torch.manual_seed(17)
    model = OxygenRECModel(OxygenRECConfig(
        sid_width=8, hidden_size=16, attention_heads=4, encoder_layers=1,
        decoder_layers=1, feedforward_size=32, dropout=0.0,
        max_history_items=2, instruction_feature_size=4,
        q2i_dimension=8, igr_top_k=3,
    )).to(device).eval()
    short = torch.tensor([[[5,5,5], [6,6,6]]] * len(plans), device=device)
    short_mask = torch.zeros(len(plans), 2, dtype=torch.bool, device=device)
    with torch.inference_mode():
        model_output = model(
            short, short_mask,
            instruction_features=torch.randn(len(plans), 4, device=device),
            long_history_sids=sids,
            long_history_padding_mask=mask,
            long_history_behavior_ids=behaviors,
            retrieval_plans=plans,
        )
    if model_output.igr_indices is None or model_output.igr_indices.shape != (len(plans), 3):
        raise RuntimeError("model IGR did not return plan-controlled indices")
    print(
        f"OK device={device} cases={len(records)} plan_changed={changed.tolist()} "
        f"baseline={baseline.tolist()} selected={selected.tolist()} "
        f"model_igr={model_output.igr_indices.tolist()}"
    )


if __name__ == "__main__":
    main()
