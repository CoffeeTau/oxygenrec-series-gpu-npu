#!/usr/bin/env python3
"""Train controlled OxygenREC ablations on bounded RetailRocket data."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import math
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
        "--variant", choices=("base", "instruction", "q2i", "igr", "igr_q2i"),
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
    parser.add_argument(
        "--eval-only-checkpoint", type=Path, default=None,
        help="Load a checkpoint and run validation/retrieval diagnostics without training.",
    )
    parser.add_argument(
        "--retriever-init-checkpoint", type=Path, default=None,
        help="Initialize SID/instruction/Q2I modules from a Q2I-only checkpoint.",
    )
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
    if args.variant in {"instruction", "q2i"}:
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
    repeat_eligible = 0
    repeat_retrieved = 0
    repeat_recent_retrieved = 0
    repeat_random_expected_hits = 0.0
    q2i_alignments = []
    for sample_batch in chunks(samples, args.batch_size):
        batch = tensor_batch(sample_batch, registry, args, device)
        target_sids = batch.pop("target_sids")
        if args.variant in {"igr", "igr_q2i"}:
            diagnostic = model(target_sids=target_sids, **batch)
            long_sids = batch["long_history_sids"]
            long_mask = batch["long_history_padding_mask"]
            selected = long_sids.gather(
                1,
                diagnostic.igr_indices.unsqueeze(-1).expand(
                    -1, -1, long_sids.shape[-1]
                ),
            )
            target_matches = (long_sids == target_sids.unsqueeze(1)).all(dim=-1) & ~long_mask
            selected_matches = (selected == target_sids.unsqueeze(1)).all(dim=-1)
            repeat_eligible += int(target_matches.any(dim=1).sum())
            repeat_retrieved += int(
                (target_matches.any(dim=1) & selected_matches.any(dim=1)).sum()
            )
            recent_matches = target_matches[:, -args.igr_top_k :].any(dim=1)
            eligible_rows = target_matches.any(dim=1)
            repeat_recent_retrieved += int((eligible_rows & recent_matches).sum())
            for row in range(target_matches.shape[0]):
                if not bool(eligible_rows[row]):
                    continue
                valid_count = int((~long_mask[row]).sum())
                match_count = int(target_matches[row].sum())
                misses = valid_count - match_count
                miss_probability = (
                    math.comb(misses, args.igr_top_k)
                    / math.comb(valid_count, args.igr_top_k)
                    if misses >= args.igr_top_k else 0.0
                )
                repeat_random_expected_hits += 1.0 - miss_probability
            if diagnostic.q2i_alignment_loss is not None:
                q2i_alignments.append(float(diagnostic.q2i_alignment_loss))
        output = model.beam_search(
            batch.pop("history_sids"), batch.pop("history_padding_mask"), trie,
            beam_width=args.beam_width, **batch,
        )
        predictions.extend(output.semantic_ids.cpu().tolist())
        targets.extend(sample.target.item_id for sample in sample_batch)
    available = min(len(ranking) for ranking in predictions)
    ks = tuple(k for k in (1, 5, 10) if k <= available)
    metrics = evaluate_sid_ranking(predictions, targets, registry, ks=ks)
    retrieval = {
        "repeat_eligible": repeat_eligible,
        "repeat_retrieved": repeat_retrieved,
        "repeat_recall": (
            repeat_retrieved / repeat_eligible if repeat_eligible else None
        ),
        "repeat_recent_recall": (
            repeat_recent_retrieved / repeat_eligible if repeat_eligible else None
        ),
        "repeat_random_expected_recall": (
            repeat_random_expected_hits / repeat_eligible if repeat_eligible else None
        ),
        "repeat_lift_over_random": (
            (repeat_retrieved / repeat_random_expected_hits)
            if repeat_random_expected_hits else None
        ),
        "q2i_alignment": (
            sum(q2i_alignments) / len(q2i_alignments) if q2i_alignments else None
        ),
    }
    return metrics, retrieval


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
        q2i_weight=args.q2i_weight if args.variant in {"q2i", "igr_q2i"} else 0.0,
    )
    model = OxygenRECModel(config).to(device)
    if args.retriever_init_checkpoint is not None:
        if args.variant not in {"igr", "igr_q2i"}:
            raise ValueError("retriever initialization is only valid for IGR variants")
        source = torch.load(
            args.retriever_init_checkpoint, map_location=device, weights_only=False
        )["model_state"]
        prefixes = (
            "sid_embeddings.", "instruction_embeddings.", "scenario_embeddings.",
            "instruction_feature_adapter.", "query_adapter.", "item_adapter.",
        )
        current = model.state_dict()
        transferred = {
            name: tensor
            for name, tensor in source.items()
            if name.startswith(prefixes)
            and name in current
            and current[name].shape == tensor.shape
        }
        if not transferred:
            raise RuntimeError("retriever init checkpoint has no compatible parameters")
        incompatible = model.load_state_dict(transferred, strict=False)
        print(
            f"stage=retriever_init transferred_tensors={len(transferred)} "
            f"missing_tensors={len(incompatible.missing_keys)}"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry.to_json(args.output_dir / "sid_registry.json")
    trie = PrefixTrie.from_registry(registry)
    epoch_records = []

    if args.eval_only_checkpoint is not None:
        checkpoint = torch.load(
            args.eval_only_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"])
        metrics, retrieval = validate(
            model, validation_samples, registry, trie, args, device
        )
        print(
            f"EVAL variant={args.variant} checkpoint_epoch={checkpoint['epoch']} "
            f"hr={dict(metrics.hit_rate)} mrr={metrics.mrr:.6f} "
            f"repeat_recall={retrieval['repeat_recall']} "
            f"repeat_recent={retrieval['repeat_recent_recall']} "
            f"repeat_random={retrieval['repeat_random_expected_recall']} "
            f"repeat_lift={retrieval['repeat_lift_over_random']} "
            f"repeat_eligible={retrieval['repeat_eligible']}"
        )
        return 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_samples)
        total_loss = 0.0
        total_ntp_loss = 0.0
        total_q2i_loss = 0.0
        max_loss_identity_error = 0.0
        batches = 0
        for sample_batch in chunks(train_samples, args.batch_size):
            batch = tensor_batch(sample_batch, registry, args, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            expected_loss = output.ntp_loss
            if output.q2i_loss is not None:
                expected_loss = expected_loss + config.q2i_weight * output.q2i_loss
            identity_error = float((output.loss - expected_loss).abs().detach())
            if identity_error > 1e-5:
                raise RuntimeError(
                    "joint-loss identity failed: "
                    f"loss={float(output.loss.detach()):.6f} "
                    f"ntp={float(output.ntp_loss.detach()):.6f} "
                    f"q2i={float(output.q2i_loss.detach()) if output.q2i_loss is not None else None} "
                    f"weight={config.q2i_weight}"
                )
            output.loss.backward()
            optimizer.step()
            total_loss += float(output.loss.detach())
            total_ntp_loss += float(output.ntp_loss.detach())
            if output.q2i_loss is not None:
                total_q2i_loss += float(output.q2i_loss.detach())
            max_loss_identity_error = max(max_loss_identity_error, identity_error)
            batches += 1
        metrics, retrieval = validate(
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
            if args.variant in {"q2i", "igr_q2i"} else ""
        )
        mean_loss = total_loss / batches
        mean_ntp_loss = total_ntp_loss / batches
        epoch_identity_error = 0.0
        if args.variant in {"q2i", "igr_q2i"}:
            mean_q2i_loss = total_q2i_loss / batches
            reconstructed = mean_ntp_loss + config.q2i_weight * mean_q2i_loss
            epoch_identity_error = abs(mean_loss - reconstructed)
            if epoch_identity_error > 1e-5:
                raise RuntimeError(
                    "epoch joint-loss identity failed: "
                    f"loss={mean_loss:.6f} ntp={mean_ntp_loss:.6f} "
                    f"q2i={mean_q2i_loss:.6f} weight={config.q2i_weight}"
                )
        print(
            f"variant={args.variant} epoch={epoch} train_loss={mean_loss:.6f} "
            f"ntp_loss={mean_ntp_loss:.6f}{q2i_summary} "
            f"q2i_weight={config.q2i_weight:.6f} "
            f"loss_identity_error={max_loss_identity_error:.3e} "
            f"epoch_identity_error={epoch_identity_error:.3e} "
            f"hr={dict(metrics.hit_rate)} mrr={metrics.mrr:.6f} "
            f"legal_sid_rate={metrics.legal_sid_rate:.6f} "
            f"repeat_recall={retrieval['repeat_recall']} "
            f"repeat_recent={retrieval['repeat_recent_recall']} "
            f"repeat_random={retrieval['repeat_random_expected_recall']} "
            f"repeat_lift={retrieval['repeat_lift_over_random']} "
            f"repeat_eligible={retrieval['repeat_eligible']}"
        )
        epoch_records.append({
            "variant": args.variant,
            "seed": args.seed,
            "epoch": epoch,
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "train_loss": mean_loss,
            "ntp_loss": mean_ntp_loss,
            "q2i_loss": (
                total_q2i_loss / batches
                if args.variant in {"q2i", "igr_q2i"} else None
            ),
            "q2i_weight": config.q2i_weight,
            "loss_identity_error": max_loss_identity_error,
            "epoch_identity_error": epoch_identity_error,
            "hit_rate": dict(metrics.hit_rate),
            "mrr": metrics.mrr,
            "ndcg": metrics.ndcg,
            "legal_sid_rate": metrics.legal_sid_rate,
            "retrieval": retrieval,
        })
        (args.output_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in epoch_records),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
