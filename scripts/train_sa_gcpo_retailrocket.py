#!/usr/bin/env python3
"""Run a bounded RetailRocket SA-GCPO before/after checkpoint comparison."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from dataclasses import fields
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.alignment import sa_gcpo_loss
from oxygenrec.data import (
    Split, TemporalBoundaries, build_long_short_sid_model_batch,
    build_next_item_samples, build_sid_model_batch, load_retailrocket_events,
)
from oxygenrec.evaluation import evaluate_sid_ranking
from oxygenrec.model import OxygenRECConfig, OxygenRECModel
from oxygenrec.rewards import map_public_rewards
from oxygenrec.sid import PrefixTrie, SIDRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("data/raw/retailrocket/events.csv"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sid-registry", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sample-seed", type=int,
        help="Override the checkpoint seed for cohort sampling and SA-GCPO updates.",
    )
    parser.add_argument("--alignment-samples", type=int, default=200)
    parser.add_argument("--validation-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--target-injection", choices=("none", "always"), default="none",
        help="Teacher-augment missing targets or keep strictly old-policy beams.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def latest_checkpoint(root: Path = Path("checkpoints")) -> Path:
    candidates = list(root.glob("**/epoch-*.pt"))
    if not candidates:
        raise FileNotFoundError("no checkpoints/**/epoch-*.pt found; pass --checkpoint")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def restored_config(payload: dict) -> OxygenRECConfig:
    known = {field.name for field in fields(OxygenRECConfig)}
    return OxygenRECConfig(**{
        key: value for key, value in payload["model_config"].items() if key in known
    })


def checkpoint_arg(payload: dict, name: str, default):
    saved = payload.get("args", {})
    return saved.get(name, default) if isinstance(saved, dict) else default


def make_batch(samples, registry, payload, device):
    variant = checkpoint_arg(payload, "variant", "base")
    max_history = int(checkpoint_arg(payload, "max_history", 20))
    igr_top_k = int(checkpoint_arg(payload, "igr_top_k", 10))
    if variant in {"igr", "igr_q2i"}:
        raw = build_long_short_sid_model_batch(
            samples, registry, short_history_items=max_history,
            long_history_items=int(checkpoint_arg(payload, "long_history", 100)),
            minimum_long_history_items=igr_top_k,
        )
        batch = {
            "history_sids": torch.tensor(raw.short_history_sids, device=device),
            "history_padding_mask": torch.tensor(raw.short_history_padding_mask, dtype=torch.bool, device=device),
            "target_sids": torch.tensor(raw.target_sids, device=device),
            "scenario_ids": torch.tensor(raw.scenario_ids, device=device),
            "long_history_sids": torch.tensor(raw.long_history_sids, device=device),
            "long_history_padding_mask": torch.tensor(raw.long_history_padding_mask, dtype=torch.bool, device=device),
        }
        batch["trigger_sids"] = batch["history_sids"][:, -1]
        return batch
    raw = build_sid_model_batch(samples, registry, max_history_items=max_history)
    batch = {
        "history_sids": torch.tensor(raw.history_sids, device=device),
        "history_padding_mask": torch.tensor(raw.history_padding_mask, dtype=torch.bool, device=device),
        "target_sids": torch.tensor(raw.target_sids, device=device),
    }
    if variant in {"instruction", "q2i"}:
        scenario = {"view": 0, "addtocart": 1, "transaction": 2}
        batch["scenario_ids"] = torch.tensor(
            [scenario[sample.target.behavior.value] for sample in samples], device=device,
        )
        batch["trigger_sids"] = batch["history_sids"][:, -1]
    return batch


def conditioning(batch: dict) -> dict:
    return {
        key: value for key, value in batch.items()
        if key not in {"history_sids", "history_padding_mask", "target_sids"}
    }


@torch.no_grad()
def evaluate(model, batches, registry, trie, beam_width):
    model.eval()
    predictions, targets = [], []
    for samples, batch in batches:
        output = model.beam_search(
            batch["history_sids"], batch["history_padding_mask"], trie,
            beam_width=beam_width, **conditioning(batch),
        )
        predictions.extend(output.semantic_ids.cpu().tolist())
        targets.extend(sample.target.item_id for sample in samples)
    available = min(len(row) for row in predictions)
    ks = tuple(k for k in (1, 5, 10) if k <= available)
    metrics = evaluate_sid_ranking(predictions, targets, registry, ks=ks)
    ranks = []
    for ranking, target in zip(predictions, targets, strict=True):
        rank = 0
        for index, sid in enumerate(ranking, start=1):
            if target in registry.items_for(sid):
                rank = index
                break
        ranks.append(rank)
    return metrics, ranks


def format_metrics(label, metrics) -> str:
    return (
        f"{label}_hr={dict(metrics.hit_rate)} {label}_mrr={metrics.mrr:.6f} "
        f"{label}_ndcg={metrics.ndcg:.6f} {label}_legal={metrics.legal_sid_rate:.6f}"
    )


def main() -> int:
    args = parse_args()
    if min(args.alignment_samples, args.validation_samples, args.batch_size) < 1 or args.beam_width < 2:
        raise ValueError("sample counts/batch-size must be positive and beam-width >= 2")
    checkpoint_path = args.checkpoint or latest_checkpoint()
    device = torch.device(args.device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    registry_path = args.sid_registry or checkpoint_path.parent / "sid_registry.json"
    registry = SIDRegistry.from_json(registry_path)
    if payload.get("sid_registry_version") != registry.version:
        raise RuntimeError("checkpoint and SID registry versions differ")
    config = restored_config(payload)
    policy = OxygenRECModel(config).to(device)
    policy.load_state_dict(payload["model_state"])
    old_policy = copy.deepcopy(policy).eval()
    for parameter in old_policy.parameters():
        parameter.requires_grad_(False)

    seed = (
        args.sample_seed
        if args.sample_seed is not None
        else int(checkpoint_arg(payload, "seed", 17))
    )
    random.seed(seed)
    torch.manual_seed(seed)
    events = list(load_retailrocket_events(args.events))
    boundaries_data = payload.get("boundaries")
    if boundaries_data:
        boundaries = TemporalBoundaries(**boundaries_data)
    else:
        minimum = min(event.timestamp_ms for event in events)
        duration = max(event.timestamp_ms for event in events) - minimum + 1
        boundaries = TemporalBoundaries(
            train_end_ms=minimum + duration * 8 // 10,
            validation_end_ms=minimum + duration * 9 // 10,
        )
    filtered = [event for event in events if event.item_id in registry.item_to_sid]
    variant = checkpoint_arg(payload, "variant", "base")
    matched = variant in {"igr", "igr_q2i"} or bool(
        checkpoint_arg(payload, "matched_igr_cohort", False)
    )
    max_history = int(checkpoint_arg(payload, "max_history", 20))
    igr_top_k = int(checkpoint_arg(payload, "igr_top_k", 10))
    long_history = int(checkpoint_arg(payload, "long_history", 100))
    samples = build_next_item_samples(
        filtered, boundaries,
        min_history=max_history + igr_top_k if matched else 1,
        max_history=max_history + long_history if matched else max_history,
        max_samples_per_split={
            Split.TRAIN: args.alignment_samples,
            Split.VALIDATION: args.validation_samples,
            Split.TEST: 1,
        },
        sample_seed=seed,
    )
    by_split = defaultdict(list)
    for sample in samples:
        by_split[sample.split].append(sample)
    alignment = by_split[Split.TRAIN]
    validation = by_split[Split.VALIDATION]
    if not alignment or not validation:
        raise RuntimeError("alignment or held-out validation cohort is empty")

    def tensor_batches(cohort):
        return [
            (cohort[start:start + args.batch_size], make_batch(
                cohort[start:start + args.batch_size], registry, payload, device
            ))
            for start in range(0, len(cohort), args.batch_size)
        ]

    alignment_batches = tensor_batches(alignment)
    validation_batches = tensor_batches(validation)
    trie = PrefixTrie.from_registry(registry)
    before, before_ranks = evaluate(
        policy, validation_batches, registry, trie, args.beam_width
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    first_objective = last_objective = 0.0
    injected = 0
    covered = 0
    rollouts = []
    for _, batch in alignment_batches:
        with torch.no_grad():
            candidates = old_policy.beam_search(
                batch["history_sids"], batch["history_padding_mask"], trie,
                beam_width=args.beam_width, **conditioning(batch),
            ).semantic_ids
            target = batch["target_sids"]
            present = (candidates == target[:, None]).all(dim=-1).any(dim=1)
            covered += int(present.sum())
            if args.target_injection == "always" and (~present).any():
                candidates = candidates.clone()
                candidates[~present, -1] = target[~present]
                injected += int((~present).sum())
            relative = (candidates == target[:, None]).float().mean(dim=-1)
            ranking = (candidates == target[:, None]).all(dim=-1).float()
            mapped = map_public_rewards(
                candidates, legal_mask=torch.ones_like(ranking, dtype=torch.bool),
                relative_scores=relative, ranking_scores=ranking,
            )
            target_rewards = torch.full(
                (target.shape[0],), 3.0, dtype=mapped.total.dtype, device=device
            )
            old_log_probs = old_policy.candidate_log_probs(
                batch["history_sids"], batch["history_padding_mask"], candidates,
                **conditioning(batch),
            )
        rollouts.append((batch, candidates, mapped.total, target_rewards, old_log_probs))
    policy.train()
    for update in range(args.updates):
        objective_sum = 0.0
        for batch, candidates, rewards, target_rewards, old_log_probs in rollouts:
            optimizer.zero_grad(set_to_none=True)
            current_log_probs = policy.candidate_log_probs(
                batch["history_sids"], batch["history_padding_mask"], candidates,
                **conditioning(batch),
            )
            aligned = sa_gcpo_loss(
                current_log_probs, old_log_probs, rewards, target_rewards,
            )
            aligned.loss.backward()
            optimizer.step()
            objective_sum += float(aligned.objective.detach())
        epoch_objective = objective_sum / len(alignment_batches)
        if update == 0:
            first_objective = epoch_objective
        last_objective = epoch_objective
    after, after_ranks = evaluate(
        policy, validation_batches, registry, trie, args.beam_width
    )
    improved = sum(
        after_rank > 0 and (before_rank == 0 or after_rank < before_rank)
        for before_rank, after_rank in zip(before_ranks, after_ranks, strict=True)
    )
    worsened = sum(
        before_rank > 0 and (after_rank == 0 or after_rank > before_rank)
        for before_rank, after_rank in zip(before_ranks, after_ranks, strict=True)
    )
    unchanged = len(before_ranks) - improved - worsened
    output_path = args.output or checkpoint_path.parent / (
        f"sa_gcpo-retailrocket-{args.target_injection}-seed{seed}.pt"
    )
    torch.save({
        "model_state": policy.state_dict(),
        "source_checkpoint": str(checkpoint_path),
        "sid_registry_version": registry.version,
        "alignment_samples": len(alignment),
        "validation_samples": len(validation),
        "updates": args.updates,
        "learning_rate": args.learning_rate,
        "sample_seed": seed,
        "target_injection": args.target_injection,
        "objective_before": first_objective,
        "objective_after": last_objective,
    }, output_path)
    print(
        f"OK device={device} checkpoint={checkpoint_path} variant={variant} seed={seed} "
        f"alignment={len(alignment)} heldout_validation={len(validation)} "
        f"alignment_target_coverage={covered} "
        f"target_injection={args.target_injection} injected_targets={injected} "
        f"objective={first_objective:.6f}->{last_objective:.6f} output={output_path}"
    )
    print(format_metrics("before", before))
    print(format_metrics("after", after))
    print(
        f"paired_ranks improved={improved} worsened={worsened} "
        f"unchanged={unchanged}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
