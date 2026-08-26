#!/usr/bin/env python3
"""CUDA validation for SA-GCPO equations, thresholds and smooth gates."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from oxygenrec.alignment import sa_gcpo_loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(17)
    current = torch.nn.Parameter(torch.zeros(2, 4, 3, device=device))
    old = torch.zeros_like(current.detach())
    rewards = torch.tensor([[0.2, 0.6, 0.9, 1.2], [0.1, 0.4, 0.8, 1.1]], device=device)
    targets = torch.tensor([0.8, 0.7], device=device)
    mask = torch.tensor([[[1,1,1],[1,1,1],[1,1,0],[1,1,1]]] * 2, dtype=torch.bool, device=device)
    optimizer = torch.optim.Adam([current], lr=0.05)
    objectives = []
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        output = sa_gcpo_loss(
            current, old, rewards, targets, token_mask=mask,
            tau_positive=2.0, tau_negative=5.0,
        )
        output.loss.backward()
        optimizer.step()
        objectives.append(float(output.objective.detach()))
    if not objectives[-1] > objectives[0]:
        raise RuntimeError("SA-GCPO objective did not improve")
    suppressed = int((output.thresholded_advantage == 0).sum())
    print(
        f"OK device={device} objective={objectives[0]:.6f}->{objectives[-1]:.6f} "
        f"suppressed={suppressed} ratio_min={float(output.importance_ratio.min()):.6f} "
        f"ratio_max={float(output.importance_ratio.max()):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
