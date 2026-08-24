#!/usr/bin/env python3
"""Train the Phase-1 dense model on a bounded RetailRocket experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.data import (
    Split,
    TemporalBoundaries,
    build_frequency_bootstrap_registry,
    build_next_item_samples,
    build_sid_model_batch,
    load_retailrocket_events,
)
from oxygenrec.evaluation import evaluate_sid_ranking
from oxygenrec.model import OxygenRECConfig, OxygenRECModel
from oxygenrec.sid import PrefixTrie


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("data/raw/retailrocket/events.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/retailrocket_phase1"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-items", type=int, default=50_000)
    parser.add_argument("--sid-width", type=int, default=64)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--max-train-samples", type=int, default=100_000)
    parser.add_argument("--max-validation-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=10)
    return parser.parse_args()


def tensor_batch(samples, registry, max_history, device):
    batch = build_sid_model_batch(samples, registry, max_history_items=max_history)
    return (
        torch.tensor(batch.history_sids, dtype=torch.long, device=device),
        torch.tensor(batch.history_padding_mask, dtype=torch.bool, device=device),
        torch.tensor(batch.target_sids, dtype=torch.long, device=device),
    )


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


@torch.no_grad()
def validate(model, samples, registry, trie, args, device):
    model.eval()
    predictions = []
    targets = []
    for sample_batch in chunks(samples, args.batch_size):
        history, padding, _ = tensor_batch(
            sample_batch, registry, args.max_history, device
        )
        output = model.beam_search(
            history, padding, trie, beam_width=args.beam_width
        )
        predictions.extend(output.semantic_ids.cpu().tolist())
        targets.extend(sample.target.item_id for sample in sample_batch)
    available = min(len(ranking) for ranking in predictions)
    ks = tuple(k for k in (1, 5, 10) if k <= available)
    return evaluate_sid_ranking(predictions, targets, registry, ks=ks)


def main() -> int:
    args = parse_args()
    if not args.events.is_file():
        raise FileNotFoundError(args.events)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print("stage=load_events")
    events = list(load_retailrocket_events(args.events))
    minimum = min(event.timestamp_ms for event in events)
    maximum = max(event.timestamp_ms for event in events)
    duration = maximum - minimum + 1
    boundaries = TemporalBoundaries(
        train_end_ms=minimum + duration * 8 // 10,
        validation_end_ms=minimum + duration * 9 // 10,
    )
    registry = build_frequency_bootstrap_registry(
        events,
        boundaries,
        max_items=args.max_items,
        width=args.sid_width,
    )
    in_vocabulary = registry.item_to_sid
    filtered_events = [event for event in events if event.item_id in in_vocabulary]
    del events
    print("stage=build_samples")
    samples = build_next_item_samples(
        filtered_events,
        boundaries,
        max_history=args.max_history,
        max_samples_per_split={
            Split.TRAIN: args.max_train_samples,
            Split.VALIDATION: args.max_validation_samples,
            Split.TEST: 1,
        },
        sample_seed=args.seed,
    )
    by_split = defaultdict(list)
    for sample in samples:
        by_split[sample.split].append(sample)
    train_samples = by_split[Split.TRAIN]
    validation_samples = by_split[Split.VALIDATION]
    if not train_samples or not validation_samples:
        raise RuntimeError("bounded experiment produced an empty train or validation split")

    config = OxygenRECConfig(
        sid_width=registry.width,
        sid_levels=registry.levels,
        hidden_size=args.hidden_size,
        attention_heads=args.attention_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        feedforward_size=args.hidden_size * 4,
        max_history_items=args.max_history,
    )
    model = OxygenRECModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry.to_json(args.output_dir / "sid_registry.json")
    trie = PrefixTrie.from_registry(registry)

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_samples)
        total_loss = 0.0
        batches = 0
        for sample_batch in chunks(train_samples, args.batch_size):
            history, padding, targets = tensor_batch(
                sample_batch, registry, args.max_history, device
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(history, padding, target_sids=targets)
            output.loss.backward()
            optimizer.step()
            total_loss += float(output.loss.detach())
            batches += 1
        metrics = validate(
            model, validation_samples, registry, trie, args, device
        )
        checkpoint = {
            "epoch": epoch,
            "model_config": asdict(config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "sid_registry_version": registry.version,
            "boundaries": asdict(boundaries),
            "args": vars(args),
        }
        torch.save(checkpoint, args.output_dir / f"epoch-{epoch}.pt")
        print(
            f"epoch={epoch} train_loss={total_loss / batches:.6f} "
            f"hr={dict(metrics.hit_rate)} mrr={metrics.mrr:.6f} "
            f"legal_sid_rate={metrics.legal_sid_rate:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
