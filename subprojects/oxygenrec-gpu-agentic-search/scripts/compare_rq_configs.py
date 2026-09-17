#!/usr/bin/env python3
"""Complete the final 2x2 RQ width/initialization controlled comparison."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from oxygenrec.quantization_torch import TorchResidualKMeans
from oxygenrec.sid_metrics import compute_sid_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/retailrocket_sid"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/rq_comparison"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def load_inputs(input_dir: Path) -> tuple[tuple[str, ...], np.ndarray]:
    vectors = np.load(input_dir / "item_embeddings.npy", allow_pickle=False)
    item_ids = tuple(
        line.strip()
        for line in (input_dir / "item_ids.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if vectors.ndim != 2 or len(item_ids) != vectors.shape[0]:
        raise ValueError("item IDs and embedding matrix are not aligned")
    return item_ids, vectors


def fit_one(item_ids, vectors, *, width, initialization, args):
    device = torch.device(args.device)
    tensor = torch.from_numpy(vectors).to(device)
    fitter = TorchResidualKMeans(
        levels=3,
        width=width,
        max_iterations=args.iterations,
        seed=args.seed,
        assignment_chunk_size=args.chunk_size,
        initialization=initialization,
    )
    model = fitter.fit(tensor)
    version = f"rq-control-l3-w{width}-s{args.seed}-init{initialization}"
    registry = model.registry_for(
        item_ids, tensor, version=version, chunk_size=args.chunk_size
    )
    codes = model.encode(tensor, chunk_size=args.chunk_size)
    reconstruction = model.reconstruct(codes)
    diagnostics = compute_sid_diagnostics(registry)
    result = {
        "width": width,
        "initialization": initialization,
        "mse": float(torch.mean((tensor - reconstruction) ** 2)),
        "collision_rate": diagnostics.colliding_item_rate,
        "unique_sid_count": diagnostics.unique_sid_count,
        "prefix_occupied": [item.occupied for item in diagnostics.prefix_coverage],
        "diagnostics": asdict(diagnostics),
    }
    output = args.output_dir / f"w{width}_{initialization.replace('+', 'p')}"
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "rq_codebooks.pt", version=version)
    registry.to_json(output / "sid_registry.json")
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"RESULT width={width} init={initialization} mse={result['mse']:.6f} "
        f"collision_rate={result['collision_rate']:.6f} "
        f"unique_sids={result['unique_sid_count']} "
        f"prefix_occupied={result['prefix_occupied']}"
    )
    return result


def main() -> int:
    args = parse_args()
    item_ids, vectors = load_inputs(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"items={len(item_ids)} dimension={vectors.shape[1]}")
    results = [
        fit_one(
            item_ids,
            vectors,
            width=256,
            initialization="kmeans++",
            args=args,
        ),
        fit_one(
            item_ids,
            vectors,
            width=512,
            initialization="random",
            args=args,
        ),
    ]
    (args.output_dir / "missing_cells_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("OK comparison_complete=2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
