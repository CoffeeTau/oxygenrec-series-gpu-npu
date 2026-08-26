#!/usr/bin/env python3
"""Export deterministic RetailRocket logits/loss/beams for device alignment."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import fields
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.data import (
    Split, TemporalBoundaries, build_next_item_samples, build_sid_model_batch,
    load_retailrocket_events,
)
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
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    registry = SIDRegistry.from_json(args.sid_registry)
    if checkpoint.get("sid_registry_version") != registry.version:
        raise RuntimeError("checkpoint and registry versions differ")
    known = {field.name for field in fields(OxygenRECConfig)}
    config_data = {
        key: value for key, value in checkpoint["model_config"].items() if key in known
    }
    config = OxygenRECConfig(**config_data)
    if config.behavior_vocab_size:
        raise ValueError("GPU reference exporter currently expects the frozen base checkpoint")
    model = OxygenRECModel(config).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])

    boundaries = TemporalBoundaries(**checkpoint["boundaries"])
    events = [
        event for event in load_retailrocket_events(args.events)
        if event.item_id in registry.item_to_sid
    ]
    samples = build_next_item_samples(
        events, boundaries, min_history=1, max_history=config.max_history_items,
        max_samples_per_split={Split.TRAIN: 1, Split.VALIDATION: args.samples, Split.TEST: 1},
        sample_seed=1729,
    )
    by_split = defaultdict(list)
    for sample in samples:
        by_split[sample.split].append(sample)
    validation = by_split[Split.VALIDATION]
    if len(validation) != args.samples:
        raise RuntimeError(f"expected {args.samples} validation samples, got {len(validation)}")
    raw = build_sid_model_batch(
        validation, registry, max_history_items=config.max_history_items
    )
    history = torch.tensor(raw.history_sids, dtype=torch.long, device=device)
    padding = torch.tensor(raw.history_padding_mask, dtype=torch.bool, device=device)
    targets = torch.tensor(raw.target_sids, dtype=torch.long, device=device)
    trie = PrefixTrie.from_registry(registry)
    with torch.inference_mode():
        output = model(history, padding, target_sids=targets)
        beam = model.beam_search(history, padding, trie, beam_width=args.beam_width)
    payload = {
        "schema_version": 1,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "sid_registry_version": registry.version,
        "model_config": config_data,
        "history_sids": history.cpu(),
        "history_padding_mask": padding.cpu(),
        "target_sids": targets.cpu(),
        "logits": tuple(tensor.cpu() for tensor in output.logits),
        "loss": output.loss.cpu(),
        "beam_sids": beam.semantic_ids.cpu(),
        "beam_scores": beam.scores.cpu(),
        "beam_width": args.beam_width,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"OK device={device} samples={args.samples} loss={float(output.loss):.6f} "
        f"beam_shape={tuple(beam.semantic_ids.shape)} checkpoint_sha256={payload['checkpoint_sha256']} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
