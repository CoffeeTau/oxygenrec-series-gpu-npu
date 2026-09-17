#!/usr/bin/env python3
"""CUDA validation for the paper-structured multimodal fusion module."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.multimodal import MultimodalItemEncoder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(17)
    model = MultimodalItemEncoder(
        text_size=32, image_size=24, hidden_size=64, query_tokens=8,
        qformer_layers=2, attention_heads=8, output_size=32,
    ).to(device)
    text = torch.randn(16, 6, 32, device=device)
    image = torch.randn(16, 4, 24, device=device)
    target = torch.randn(16, 32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        output = model(text, image)
        loss = torch.nn.functional.mse_loss(output.item_embedding, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    if not losses[-1] < losses[0] * 0.2:
        raise RuntimeError("multimodal fusion did not overfit the fixed batch")
    with torch.no_grad():
        baseline = model(text, image).item_embedding
        text_delta = (baseline - model(torch.zeros_like(text), image).item_embedding).abs().mean()
        image_delta = (baseline - model(text, torch.zeros_like(image)).item_embedding).abs().mean()
    if float(text_delta) == 0.0 or float(image_delta) == 0.0:
        raise RuntimeError("one modality has no observable influence")
    print(
        f"OK device={device} item_shape={tuple(output.item_embedding.shape)} "
        f"query_shape={tuple(output.query_tokens.shape)} loss={losses[0]:.6f}->{losses[-1]:.6f} "
        f"text_delta={float(text_delta):.6f} image_delta={float(image_delta):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
