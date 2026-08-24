#!/usr/bin/env python3
"""Train controlled OxygenREC ablations on bounded RetailRocket data."""

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
    build_long_short_sid_model_batch,
    build_next_item_samples,
    build_sid_model_batch,
    load_retailrocket_events,
)
from oxygenrec.evaluation import evaluate_sid_ranking
from oxygenrec.model import OxygenRECConfig, OxygenRECModel
from oxygenrec.sid import PrefixTrie
from oxygenrec.sid import SIDRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("data/raw/retailrocket/events.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/retailrocket_phase1"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-items", type=int, default=50_000)
    parser.add_argument("--sid-width", type=int, default=64)
    parser.add_argument(
        "--sid-registry",
        type=Path,
        default=None,
        help="Use a fitted RQ registry instead of the temporary frequency bootstrap.",
    )
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--long-history", type=int, default=100)
    parser.add_argument("--igr-top-k", type=int, default=10)
    parser.add_argument(
        "--variant", choices=("base", "instruction", "igr", "igr_q2i"),
        default="base",
    )
    parser.add_argument("--q2i-weight", type=float, default=0.2)
    parser.add_argument(
        "--matched-igr-cohort", action="store_true",
        help="Use the IGR-eligible sample universe for every ablation variant.",
    )
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


def tensor_batch(samples, registry, args, device):
    uses_igr = args.variant in {"igr", "igr_q2i"}
    if uses_igr:
        batch = build_long_short_sid_model_batch(
            samples, registry, short_history_items=args.max_history,
            long_history_items=args.long_history,
            minimum_long_history_items=args.igr_top_k,
        )
        return {
            "history_sids": torch.tensor(batch.short_history_sids, dtype=torch.long, device=device),
            "history_padding_mask": torch.tensor(batch.short_history_padding_mask, dtype=torch.bool, device=device),
            "target_sids": torch.tensor(batch.target_sids, dtype=torch.long, device=device),
            "scenario_ids": torch.tensor(batch.scenario_ids, dtype=torch.long, device=device),
            "long_history_sids": torch.tensor(batch.long_history_sids, dtype=torch.long, device=device),
            "long_history_padding_mask": torch.tensor(batch.long_history_padding_mask, dtype=torch.bool, device=device),
        }
    batch = build_sid_model_batch(samples, registry, max_history_items=args.max_history)
    result = {
        "history_sids": torch.tensor(batch.history_sids, dtype=torch.long, device=device),
        "history_padding_mask": torch.tensor(batch.history_padding_mask, dtype=torch.bool, device=device),
        "target_sids": torch.tensor(batch.target_sids, dtype=torch.long, device=device),
    }
    if args.variant == "instruction":
        scenario = {"view": 0, "addtocart": 1, "transaction": 2}
        result["scenario_ids"] = torch.tensor(
            [scenario[sample.target.behavior.value] for sample in samples],
            dtype=torch.long, device=device,
        )
    return result


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


@torch.no_grad()
def validate(model, samples, registry, trie, args, device):
    model.eval()
    predictions = []
    targets = []
    for sample_batch in chunks(samples, args.batch_size):
        batch = tensor_batch(sample_batch, registry, args, device)
        batch.pop("target_sids")
        output = model.beam_search(
            batch.pop("history_sids"), batch.pop("history_padding_mask"), trie,
            beam_width=args.beam_width, **batch,
        )
        predictions.extend(output.semantic_ids.cpu().tolist())
        targets.extend(sample.target.item_id for sample in sample_batch)
    available = min(len(ranking) for ranking in predictions)
    ks = tuple(k for k in (1, 5, 10) if k <= available)
    return evaluate_sid_ranking(predictions, targets, registry, ks=ks)


def main() -> int:
    args = parse_args()
    if args.igr_top_k > args.long_history:
        raise ValueError("igr-top-k cannot exceed long-history")
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
    if args.sid_registry is None:
        registry = build_frequency_bootstrap_registry(
            events,
            boundaries,
            max_items=args.max_items,
            width=args.sid_width,
        )
    else:
        registry = SIDRegistry.from_json(args.sid_registry)
    in_vocabulary = registry.item_to_sid
    filtered_events = [event for event in events if event.item_id in in_vocabulary]
    del events
    print("stage=build_samples")
    uses_igr = args.variant in {"igr", "igr_q2i"}
    matched_cohort = uses_igr or args.matched_igr_cohort
    samples = build_next_item_samples(
        filtered_events,
        boundaries,
        min_history=args.max_history + args.igr_top_k if matched_cohort else 1,
        max_history=(args.max_history + args.long_history) if matched_cohort else args.max_history,
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
    print(
        f"stage=samples variant={args.variant} matched_cohort={matched_cohort} "
        f"train={len(train_samples)} validation={len(validation_samples)}"
    )

    config = OxygenRECConfig(
        sid_width=registry.width,
        sid_levels=registry.levels,
        hidden_size=args.hidden_size,
        attention_heads=args.attention_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        feedforward_size=args.hidden_size * 4,
        max_history_items=args.max_history,
        scenario_vocab_size=3 if args.variant != "base" else 1,
        igr_top_k=args.igr_top_k if uses_igr else 0,
        q2i_weight=args.q2i_weight if args.variant == "igr_q2i" else 0.0,
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
        total_ntp_loss = 0.0
        total_q2i_loss = 0.0
        batches = 0
        for sample_batch in chunks(train_samples, args.batch_size):
            batch = tensor_batch(sample_batch, registry, args, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            total_loss += float(output.loss.detach())
            total_ntp_loss += float(output.ntp_loss.detach())
            if output.q2i_loss is not None:
                total_q2i_loss += float(output.q2i_loss.detach())
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
        q2i_summary = (
            f" q2i_loss={total_q2i_loss / batches:.6f}"
            if args.variant == "igr_q2i" else ""
        )
        print(
            f"variant={args.variant} epoch={epoch} train_loss={total_loss / batches:.6f} "
            f"ntp_loss={total_ntp_loss / batches:.6f}{q2i_summary} "
            f"hr={dict(metrics.hit_rate)} mrr={metrics.mrr:.6f} "
            f"legal_sid_rate={metrics.legal_sid_rate:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
