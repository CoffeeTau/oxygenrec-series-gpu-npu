#!/usr/bin/env python3
"""CUDA smoke test for the scalable residual K-Means backend."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.quantization_torch import TorchResidualKMeans, TorchResidualKMeansModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    generator = torch.Generator(device="cpu").manual_seed(23)
    centers = torch.randn(8, 16, generator=generator)
    labels = torch.arange(512) % 8
    vectors = centers[labels] + 0.1 * torch.randn(512, 16, generator=generator)
    vectors = vectors.to(device)
    fitter = TorchResidualKMeans(
        levels=3,
        width=8,
        max_iterations=15,
        seed=11,
        assignment_chunk_size=97,
    )
    model = fitter.fit(vectors)
    codes = model.encode(vectors, chunk_size=83)
    reconstruction = model.reconstruct(codes)
    baseline_mse = float(torch.mean(vectors**2))
    reconstruction_mse = float(torch.mean((vectors - reconstruction) ** 2))
    if reconstruction_mse >= baseline_mse:
        raise RuntimeError("residual quantization did not improve reconstruction")
    registry = model.registry_for(
        [f"synthetic-{index}" for index in range(len(vectors))],
        vectors,
        version="cuda-smoke-v1",
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "codebooks.pt"
        model.save(path, version="cuda-smoke-v1")
        _, restored = TorchResidualKMeansModel.load(
            path, device=device, expected_version="cuda-smoke-v1"
        )
        torch.testing.assert_close(restored.codebooks, model.codebooks)
    print(
        f"OK device={device} shape={tuple(model.codebooks.shape)} "
        f"mse={baseline_mse:.6f}->{reconstruction_mse:.6f} "
        f"collision_rate={registry.collision_rate():.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
