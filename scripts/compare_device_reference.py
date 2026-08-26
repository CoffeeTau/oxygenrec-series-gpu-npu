#!/usr/bin/env python3
"""Compare one accelerator against an exported GPU reference artifact."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.model import OxygenRECConfig, OxygenRECModel
from oxygenrec.sid import PrefixTrie, SIDRegistry


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    if reference.get("schema_version") != 1:
        raise RuntimeError("unsupported reference schema")
    actual_hash = file_sha256(args.checkpoint)
    if actual_hash != reference["checkpoint_sha256"]:
        raise RuntimeError("checkpoint SHA-256 does not match GPU reference")
    registry = SIDRegistry.from_json(args.sid_registry)
    if registry.version != reference["sid_registry_version"]:
        raise RuntimeError("registry version does not match GPU reference")
    known = {field.name for field in fields(OxygenRECConfig)}
    config = OxygenRECConfig(**{
        key: value for key, value in reference["model_config"].items() if key in known
    })
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = OxygenRECModel(config).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    history = reference["history_sids"].to(device)
    padding = reference["history_padding_mask"].to(device)
    targets = reference["target_sids"].to(device)
    trie = PrefixTrie.from_registry(registry)
    with torch.inference_mode():
        output = model(history, padding, target_sids=targets)
        beam = model.beam_search(
            history, padding, trie, beam_width=int(reference["beam_width"])
        )
    max_abs = max(
        float((actual.cpu() - expected).abs().max())
        for actual, expected in zip(output.logits, reference["logits"], strict=True)
    )
    loss_delta = abs(float(output.loss.cpu()) - float(reference["loss"]))
    logits_close = all(
        torch.allclose(actual.cpu(), expected, atol=args.atol, rtol=args.rtol)
        for actual, expected in zip(output.logits, reference["logits"], strict=True)
    )
    beam_match = torch.equal(beam.semantic_ids.cpu(), reference["beam_sids"])
    if not logits_close or not beam_match:
        raise RuntimeError(
            f"device alignment failed: logits_close={logits_close} beam_match={beam_match}"
        )
    print(
        f"OK device={device} logits_close={logits_close} max_abs={max_abs:.6e} "
        f"loss_delta={loss_delta:.6e} beam_match={beam_match}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
