#!/usr/bin/env python3
"""在真实RetailRocket代理数据上验证v2 daily列表式行为训练。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.data import (
    Split,
    TemporalBoundaries,
    build_daily_listwise_samples,
    build_listwise_sid_model_batch,
    load_retailrocket_events,
)
from oxygenrec.model import OxygenRECConfig, OxygenRECModel
from oxygenrec.evaluation import evaluate_sid_token_match
from oxygenrec.sid import PrefixTrie, SIDRegistry


BEHAVIOR_WEIGHTS = (1.2, 1.5, 2.0)  # view/click代理、cart、order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("checkpoints/retailrocket_v2_listwise_smoke"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--list-size", type=int, default=2)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--max-train-samples", type=int, default=5_000)
    parser.add_argument("--max-validation-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=5)
    return parser.parse_args()


def chunks(items, size):
    """把列表切成连续且不超过size的mini-batch。"""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def tensor_batch(samples, registry, args, device):
    """把ListwiseTargetSample转换为v2 forward需要的Tensor。"""
    batch = build_listwise_sid_model_batch(
        samples, registry, max_history_items=args.max_history
    )
    target_behavior_ids = torch.tensor(
        batch.target_behavior_ids, dtype=torch.long, device=device
    )
    sid_tokens = args.list_size * registry.levels
    behavior_weights = torch.tensor(
        BEHAVIOR_WEIGHTS, dtype=torch.float32, device=device
    )
    return {
        "history_sids": torch.tensor(
            batch.history_sids, dtype=torch.long, device=device
        ),
        "history_padding_mask": torch.tensor(
            batch.history_padding_mask, dtype=torch.bool, device=device
        ),
        "history_behavior_ids": torch.tensor(
            batch.history_behavior_ids, dtype=torch.long, device=device
        ),
        "target_sids": torch.tensor(
            batch.target_sids, dtype=torch.long, device=device
        ),
        "behavior_instruction_ids": target_behavior_ids,
        "token_weights": behavior_weights[target_behavior_ids].unsqueeze(1).expand(
            -1, sid_tokens
        ),
    }


def sid_chunks(flat_path: list[int], levels: int) -> list[list[int]]:
    """把展平的3N token恢复为N个SID，便于评测和审阅。"""
    return [
        flat_path[start : start + levels]
        for start in range(0, len(flat_path), levels)
    ]


def evaluate(model, samples, registry, trie, args, device):
    """计算列表SID代理指标，并收集不含用户/商品原始ID的案例。"""
    model.eval()
    totals = Counter()
    behavior_totals: dict[str, Counter] = defaultdict(Counter)
    rows = []
    with torch.no_grad():
        for sample_batch in chunks(samples, args.batch_size):
            batch = tensor_batch(sample_batch, registry, args, device)
            targets = batch.pop("target_sids")
            batch.pop("token_weights")
            generated = model.generate(
                batch.pop("history_sids"),
                batch.pop("history_padding_mask"),
                trie,
                output_items=args.list_size,
                **batch,
            )
            # beam_search会重复编码，因此重新构造一次输入，保持接口清晰。
            beam_batch = tensor_batch(sample_batch, registry, args, device)
            beam_batch.pop("target_sids")
            beam_batch.pop("token_weights")
            beams = model.beam_search(
                beam_batch.pop("history_sids"),
                beam_batch.pop("history_padding_mask"),
                trie,
                beam_width=args.beam_width,
                output_items=args.list_size,
                **beam_batch,
            )

            for sample, predicted, target, beam_paths in zip(
                sample_batch,
                generated.cpu().tolist(),
                targets.reshape(targets.shape[0], -1).cpu().tolist(),
                beams.semantic_ids.cpu().tolist(),
                strict=True,
            ):
                predicted_items = sid_chunks(predicted, registry.levels)
                target_items = sid_chunks(target, registry.levels)
                predicted_set = {tuple(item) for item in predicted_items}
                target_set = {tuple(item) for item in target_items}
                overlap = len(predicted_set & target_set)
                position_hits = sum(
                    left == right
                    for left, right in zip(predicted_items, target_items, strict=True)
                )
                exact = predicted == target
                beam_rank = next(
                    (
                        rank
                        for rank, path in enumerate(beam_paths, start=1)
                        if path == target
                    ),
                    None,
                )
                legal_items = sum(trie.contains(item) for item in predicted_items)
                token_match = evaluate_sid_token_match(
                    predicted,
                    target,
                    geometric_decay=getattr(args, "geometric_decay", 0.9),
                )
                behavior = sample.target_behavior.value
                totals.update({
                    "lists": 1,
                    "targets": args.list_size,
                    "unique_target_sids": len(target_set),
                    "overlap": overlap,
                    "position_hits": position_hits,
                    "exact_lists": int(exact),
                    "beam_exact": int(beam_rank is not None),
                    "legal_items": legal_items,
                    "sid_token_hits": sum(token_match.hits),
                    "sid_tokens": len(token_match.hits),
                    "geometric_token_reward": token_match.geometric_reward,
                })
                behavior_totals[behavior].update({
                    "lists": 1,
                    "targets": args.list_size,
                    "overlap": overlap,
                })
                rows.append({
                    "behavior": behavior,
                    "utc_day": sample.utc_day,
                    "history_length": len(sample.history),
                    "history_behavior_counts": dict(sorted(Counter(
                        event.behavior.value for event in sample.history
                    ).items())),
                    "target_sids": target_items,
                    "generated_sids": predicted_items,
                    "sid_token_hits": token_match.hits,
                    "sid_token_accuracy": token_match.accuracy,
                    "geometric_token_reward": token_match.geometric_reward,
                    "sid_recall": overlap / args.list_size,
                    "position_accuracy": position_hits / args.list_size,
                    "exact_list": exact,
                    "beam_target_rank": beam_rank,
                    "all_generated_items_legal": legal_items == args.list_size,
                    "target_sid_collision": len(target_set) < args.list_size,
                })

    if not totals["lists"]:
        raise RuntimeError("validation produced no listwise rows")
    metrics = {
        "lists": totals["lists"],
        "list_size": args.list_size,
        "target_sid_unique_rate": totals["unique_target_sids"] / totals["targets"],
        "sid_recall": totals["overlap"] / totals["targets"],
        "position_accuracy": totals["position_hits"] / totals["targets"],
        "exact_list_rate": totals["exact_lists"] / totals["lists"],
        "beam_exact_rate": totals["beam_exact"] / totals["lists"],
        "legal_item_rate": totals["legal_items"] / totals["targets"],
        "sid_token_accuracy": totals["sid_token_hits"] / totals["sid_tokens"],
        "geometric_token_reward": (
            totals["geometric_token_reward"] / totals["lists"]
        ),
        "behavior": {
            behavior: {
                "lists": counts["lists"],
                "sid_recall": counts["overlap"] / counts["targets"],
            }
            for behavior, counts in sorted(behavior_totals.items())
        },
    }
    return metrics, rows


def select_representative_rows(rows):
    """按固定角色选择稀有行为、最好和最差列表，并合并重复案例。"""
    roles = (
        ("transaction_list", lambda row: row["behavior"] == "transaction", True),
        ("addtocart_list", lambda row: row["behavior"] == "addtocart", True),
        ("view_list", lambda row: row["behavior"] == "view", True),
        ("target_sid_collision", lambda row: row["target_sid_collision"], False),
        ("best_sid_recall", lambda row: True, True),
        ("worst_sid_recall", lambda row: True, False),
    )
    chosen: dict[int, dict] = {}
    coverage = {}
    for role, predicate, maximize in roles:
        candidates = [
            (index, row) for index, row in enumerate(rows) if predicate(row)
        ]
        if not candidates:
            coverage[role] = None
            continue
        key = lambda item: (
            item[1]["sid_recall"],
            item[1]["history_length"],
            -item[0],
        )
        index, row = (max if maximize else min)(candidates, key=key)
        chosen.setdefault(index, {**row, "roles": []})["roles"].append(role)
        coverage[role] = index
    selected = []
    old_to_case = {}
    for case_number, (old_index, row) in enumerate(sorted(chosen.items()), start=1):
        case_id = f"v2-list-review-{case_number:03d}"
        selected.append({"case_id": case_id, **row})
        old_to_case[old_index] = case_id
    return selected, {
        role: old_to_case[index] if index is not None else None
        for role, index in coverage.items()
    }


def write_review_artifacts(output_dir: Path, metrics: dict, rows: list[dict]) -> None:
    """保存匿名JSONL、Markdown和summary，供后续固定案例人工复核。"""
    selected, coverage = select_representative_rows(rows)
    jsonl_path = output_dir / "v2_listwise_representative_cases.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    lines = [
        "# OxygenREC-v2真实列表式代表案例",
        "",
        "> 仅含SID和聚合行为，不含用户ID、原始商品ID或源CSV行号。",
        "> RetailRocket的view是click公开代理；daily/行为同质分组是公开数据工程协议。",
        "",
        "## 聚合",
        "",
        f"- validation列表：{metrics['lists']}",
        f"- 列表长度N：{metrics['list_size']}",
        f"- SID recall：{metrics['sid_recall']:.6f}",
        f"- 目标SID唯一率：{metrics['target_sid_unique_rate']:.6f}",
        f"- 逐位置准确率：{metrics['position_accuracy']:.6f}",
        f"- exact-list rate：{metrics['exact_list_rate']:.6f}",
        f"- beam exact rate：{metrics['beam_exact_rate']:.6f}",
        f"- 合法商品率：{metrics['legal_item_rate']:.6f}",
        f"- SID token准确率：{metrics['sid_token_accuracy']:.6f}",
        f"- 几何token reward：{metrics['geometric_token_reward']:.6f}",
        f"- 固定角色覆盖：`{json.dumps(coverage, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    for row in selected:
        lines.extend([
            f"## {row['case_id']}",
            "",
            f"- 代表角色：`{row['roles']}`",
            f"- 目标行为：`{row['behavior']}`",
            f"- UTC日编号：`{row['utc_day']}`",
            f"- 历史长度：{row['history_length']}",
            f"- 历史行为计数：`{row['history_behavior_counts']}`",
            f"- 目标SID列表：`{row['target_sids']}`",
            f"- greedy SID列表：`{row['generated_sids']}`",
            f"- SID token命中：`{row['sid_token_hits']}`",
            f"- SID token准确率：{row['sid_token_accuracy']:.6f}",
            f"- 几何token reward：{row['geometric_token_reward']:.6f}",
            f"- SID recall：{row['sid_recall']:.6f}",
            f"- 逐位置准确率：{row['position_accuracy']:.6f}",
            f"- exact list：`{row['exact_list']}`",
            f"- beam目标列表命中rank：`{row['beam_target_rank']}`",
            f"- 全部生成SID合法：`{row['all_generated_items_legal']}`",
            f"- 目标商品发生SID碰撞：`{row['target_sid_collision']}`",
            "",
            "### 人工Review",
            "",
            "- [ ] 行为优先级折叠与目标列表一致",
            "- [ ] 历史聚合不包含列表目标后的未来行为",
            "- [ ] 生成列表中的成功或失败可由当前欠训练状态解释",
            "- [ ] 未把SID代理指标误写成原始商品级工业指标",
            "",
        ])
    markdown_path = output_dir / "v2_listwise_representative_cases.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_path = output_dir / "v2_listwise_summary.json"
    summary_path.write_text(
        json.dumps(
            {"metrics": metrics, "coverage": coverage, "representative": len(selected)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.list_size < 1 or args.max_history < 1:
        raise ValueError("list-size and max-history must be positive")
    if args.beam_width < 1:
        raise ValueError("beam-width must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("stage=load_events")
    events = list(load_retailrocket_events(args.events))
    minimum = min(event.timestamp_ms for event in events)
    maximum = max(event.timestamp_ms for event in events)
    duration = maximum - minimum + 1
    boundaries = TemporalBoundaries(
        train_end_ms=minimum + duration * 8 // 10,
        validation_end_ms=minimum + duration * 9 // 10,
    )
    registry = SIDRegistry.from_json(args.sid_registry)
    events = [event for event in events if event.item_id in registry.item_to_sid]

    print("stage=build_daily_lists")
    samples = build_daily_listwise_samples(
        events,
        boundaries,
        list_size=args.list_size,
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
        raise RuntimeError(
            "daily list construction produced an empty train or validation split; "
            "reduce --list-size before changing the grouping protocol"
        )
    train_behaviors = Counter(sample.target_behavior.value for sample in train_samples)
    validation_behaviors = Counter(
        sample.target_behavior.value for sample in validation_samples
    )
    print(
        f"stage=daily_lists train={len(train_samples)} validation={len(validation_samples)} "
        f"list_size={args.list_size} sid_tokens={args.list_size * registry.levels} "
        f"train_behaviors={dict(sorted(train_behaviors.items()))} "
        f"validation_behaviors={dict(sorted(validation_behaviors.items()))} "
        "dedup_priority=transaction>addtocart>view grouping=user+utc_day+behavior "
        "instruction_mode=learned_fallback"
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
        max_target_items=args.list_size,
        behavior_vocab_size=3,
        behavior_instruction_vocab_size=3,
    )
    model = OxygenRECModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    trie = PrefixTrie.from_registry(registry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry.to_json(args.output_dir / "sid_registry.json")

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_samples)
        total_loss = 0.0
        max_identity_error = 0.0
        max_adapter_grad = 0.0
        max_reserved_token_grad = 0.0
        batches = 0
        for sample_batch in chunks(train_samples, args.batch_size):
            batch = tensor_batch(sample_batch, registry, args, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            identity_error = float((output.loss - output.ntp_loss).abs().detach())
            if identity_error > 1e-6:
                raise RuntimeError("v2 listwise loss must equal weighted NTP loss")
            output.loss.backward()
            max_adapter_grad = max(
                max_adapter_grad,
                sum(
                    float(parameter.grad.abs().sum())
                    for parameter in model.behavior_instruction_adapter.parameters()
                    if parameter.grad is not None
                ),
            )
            embedding_grad = model.sid_embeddings[0].weight.grad
            if embedding_grad is not None:
                max_reserved_token_grad = max(
                    max_reserved_token_grad,
                    float(embedding_grad[registry.width :].abs().sum()),
                )
            optimizer.step()
            total_loss += float(output.loss.detach())
            max_identity_error = max(max_identity_error, identity_error)
            batches += 1

        metrics, rows = evaluate(
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
            "grouping_protocol": "user+utc_day+strongest_item_behavior+homogeneous_behavior",
        }
        torch.save(checkpoint, args.output_dir / f"epoch-{epoch}.pt")
        write_review_artifacts(args.output_dir, metrics, rows)
        print(
            f"variant=v2_listwise epoch={epoch} train_loss={total_loss / batches:.6f} "
            f"loss_identity_error={max_identity_error:.3e} "
            f"ib_adapter_grad={max_adapter_grad:.6f} "
            f"reserved_token_grad={max_reserved_token_grad:.6f} "
            f"metrics={json.dumps(metrics, sort_keys=True)} "
            f"reviews={args.output_dir / 'v2_listwise_representative_cases.md'}"
        )


if __name__ == "__main__":
    main()
