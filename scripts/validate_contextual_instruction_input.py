#!/usr/bin/env python3
"""CUDA smoke test for the real proxy-text -> fast-model instruction path."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from oxygenrec.instructions import encode_instructions
from oxygenrec.model import OxygenRECConfig, OxygenRECModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(17)
    texts = [
        "最近存在加购或购买等高意图行为，优先检索相关历史商品。",
        "近期以浏览为主，扩大检索范围并探索新的商品类别。",
    ]
    features = torch.tensor(
        encode_instructions(texts, args.feature_size), dtype=torch.float32, device=device,
    )
    model = OxygenRECModel(OxygenRECConfig(
        sid_width=32, hidden_size=32, attention_heads=4,
        encoder_layers=1, decoder_layers=1, feedforward_size=64,
        dropout=0.0, max_history_items=4, scenario_vocab_size=2,
        instruction_feature_size=args.feature_size, q2i_dimension=16,
        igr_top_k=2,
    )).to(device)
    model.eval()
    short = torch.tensor([[[1, 2, 3], [4, 5, 6]]] * 2, device=device)
    short_mask = torch.zeros(2, 2, dtype=torch.bool, device=device)
    long = torch.tensor([[
        [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18],
    ]] * 2, device=device)
    long_mask = torch.zeros(2, 4, dtype=torch.bool, device=device)
    # Keep every non-instruction input identical, so any output difference is
    # causally attributable to the encoded instruction text.
    targets = torch.tensor([[7, 8, 9], [7, 8, 9]], device=device)
    output = model(
        short, short_mask, target_sids=targets,
        scenario_ids=torch.tensor([0, 0], device=device),
        instruction_features=features,
        long_history_sids=long, long_history_padding_mask=long_mask,
    )
    output.loss.backward()
    feature_delta = torch.linalg.vector_norm(features[0] - features[1]).item()
    logit_delta = sum(
        torch.linalg.vector_norm(level[0] - level[1]).item() for level in output.logits
    )
    adapter_grad = model.instruction_feature_adapter.weight.grad.norm().item()
    retrieval_changed = not torch.equal(output.igr_indices[0], output.igr_indices[1])
    if feature_delta <= 0 or logit_delta <= 0 or adapter_grad <= 0:
        raise AssertionError("instruction text did not affect the differentiable model path")
    print(
        f"OK device={device} feature_shape={tuple(features.shape)} "
        f"feature_delta={feature_delta:.6f} logit_delta={logit_delta:.6f} "
        f"adapter_grad={adapter_grad:.6f} retrieval_changed={retrieval_changed} "
        f"igr_indices={output.igr_indices.tolist()}"
    )


if __name__ == "__main__":
    main()
