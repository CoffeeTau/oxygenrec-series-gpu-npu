#!/usr/bin/env python3
"""Show that distinct proxy instructions can learn distinct IGR targets."""

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
    parser.add_argument("--steps", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(23)
    feature_size = 64
    features = torch.tensor(encode_instructions([
        "用户表现出购买意图，检索历史中的目标商品甲。",
        "用户正在探索另一需求，检索历史中的目标商品乙。",
    ], feature_size), dtype=torch.float32, device=device)
    model = OxygenRECModel(OxygenRECConfig(
        sid_width=32, hidden_size=32, attention_heads=4,
        encoder_layers=1, decoder_layers=1, feedforward_size=64,
        dropout=0.0, max_history_items=3, scenario_vocab_size=1,
        instruction_feature_size=feature_size, q2i_dimension=16,
        q2i_weight=1.0, q2i_variance_weight=0.0,
        q2i_decorrelation_weight=0.0, igr_top_k=1,
    )).to(device)
    # Both samples see the same candidate pool. Only their instruction and
    # supervised target differ, so successful top-1 separation is causal.
    short = torch.tensor([[[20, 21, 22], [23, 24, 25]]] * 2, device=device)
    short_mask = torch.zeros(2, 2, dtype=torch.bool, device=device)
    candidates = torch.tensor([[
        [1, 2, 3], [7, 8, 9], [13, 14, 15],
    ]] * 2, device=device)
    candidate_mask = torch.zeros(2, 3, dtype=torch.bool, device=device)
    targets = torch.tensor([[1, 2, 3], [7, 8, 9]], device=device)
    trainable = list(model.instruction_feature_adapter.parameters())
    trainable += list(model.query_adapter.parameters())
    optimizer = torch.optim.Adam(trainable, lr=3e-2)

    def forward():
        return model(
            short, short_mask, target_sids=targets,
            instruction_features=features,
            long_history_sids=candidates,
            long_history_padding_mask=candidate_mask,
        )

    model.train()
    initial = forward()
    initial_alignment = initial.q2i_alignment_loss.item()
    initial_indices = initial.igr_indices.squeeze(1).tolist()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = forward()
        output.q2i_alignment_loss.backward()
        optimizer.step()

    model.eval()
    final = forward()
    final_alignment = final.q2i_alignment_loss.item()
    final_indices = final.igr_indices.squeeze(1).tolist()
    expected = [0, 1]
    if final_alignment >= initial_alignment or final_indices != expected:
        raise AssertionError(
            f"instruction retrieval control failed: alignment "
            f"{initial_alignment:.6f}->{final_alignment:.6f}, "
            f"indices={final_indices}, expected={expected}"
        )
    print(
        f"OK device={device} alignment={initial_alignment:.6f}->{final_alignment:.6f} "
        f"initial_indices={initial_indices} final_indices={final_indices} "
        f"expected_indices={expected} controlled_retrieval=True"
    )


if __name__ == "__main__":
    main()
