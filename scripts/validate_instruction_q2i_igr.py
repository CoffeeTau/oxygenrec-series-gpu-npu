#!/usr/bin/env python3
"""CUDA smoke/optimization test for instruction fusion, Q2I and IGR."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from oxygenrec.model import OxygenRECConfig, OxygenRECModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(17)
    model = OxygenRECModel(OxygenRECConfig(
        sid_width=16, hidden_size=32, attention_heads=4,
        encoder_layers=1, decoder_layers=1, feedforward_size=64,
        dropout=0.0, max_history_items=3, scenario_vocab_size=2,
        instruction_feature_size=12, q2i_dimension=16,
        q2i_weight=0.2, igr_top_k=2,
    )).to(device)
    short = torch.tensor([[[1,2,3],[4,5,6]], [[2,3,4],[5,6,7]], [[3,4,5],[6,7,8]], [[4,5,6],[7,8,9]]], device=device)
    short_mask = torch.zeros(4, 2, dtype=torch.bool, device=device)
    long = torch.tensor([
        [[7,8,9],[3,4,5],[6,7,8],[0,0,0]],
        [[8,9,10],[1,3,5],[4,6,8],[0,0,0]],
        [[9,10,11],[2,4,6],[5,7,9],[0,0,0]],
        [[10,11,12],[3,5,7],[6,8,10],[0,0,0]],
    ], device=device)
    long_mask = torch.tensor([[False, False, False, True]] * 4, device=device)
    targets = torch.tensor([[7,8,9],[8,9,10],[9,10,11],[10,11,12]], device=device)
    scenarios = torch.tensor([0,1,0,1], device=device)
    features = torch.randn(4, 12, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            short, short_mask, target_sids=targets, scenario_ids=scenarios,
            instruction_features=features, long_history_sids=long,
            long_history_padding_mask=long_mask,
        )
        if not torch.isfinite(output.loss):
            raise RuntimeError("non-finite joint loss")
        output.loss.backward()
        optimizer.step()
        losses.append(float(output.loss.detach()))
    if not losses[-1] < losses[0]:
        raise RuntimeError("joint loss did not decrease")
    if (output.igr_indices >= 3).any():
        raise RuntimeError("IGR selected a padded long-history item")
    if not (output.igr_scores[:, :-1] >= output.igr_scores[:, 1:]).all():
        raise RuntimeError("IGR scores are not ranked")
    model.eval()
    with torch.no_grad():
        zero_logits = model(short, short_mask, instruction_features=torch.zeros_like(features)).logits[0]
        one_logits = model(short, short_mask, instruction_features=torch.ones_like(features)).logits[0]
    instruction_delta = float((zero_logits - one_logits).abs().mean())
    if instruction_delta <= 0:
        raise RuntimeError("instruction adapter does not affect logits")
    print(
        f"OK device={device} joint_loss={losses[0]:.6f}->{losses[-1]:.6f} "
        f"ntp={float(output.ntp_loss):.6f} q2i={float(output.q2i_loss):.6f} "
        f"igr_shape={tuple(output.igr_indices.shape)} instruction_delta={instruction_delta:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
