#!/usr/bin/env python3
"""为真实RetailRocket样本生成并缓存论文式Qwen Instruction特征。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from oxygenrec.data.events import load_retailrocket_events
from oxygenrec.data.temporal import Split, TemporalBoundaries, build_next_item_samples
from oxygenrec.instruction_cache import (
    instruction_sample_key,
    save_instruction_feature_cache,
)
from oxygenrec.llm_features import build_behavior_prompt
from oxygenrec.llm_reasoning import (
    FrozenLLMReasoningGenerator,
    contextual_instruction_text,
)
from oxygenrec.sid import SIDRegistry


def parse_args() -> argparse.Namespace:
    """定义固定样本cohort、Qwen批量生成和缓存输出参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/qwen_instruction_features.pt"),
    )
    parser.add_argument(
        "--reasoning-output", type=Path,
        default=Path("data/processed/qwen_instruction_reasoning.jsonl"),
    )
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-validation-samples", type=int, default=64)
    parser.add_argument("--short-history", type=int, default=20)
    parser.add_argument("--long-history", type=int, default=100)
    parser.add_argument("--igr-top-k", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    return parser.parse_args()


def evidence_from_history(sample) -> dict[str, object]:
    """只聚合严格早于target的历史，绝不读取target行为或商品。"""
    behaviors = [event.behavior.value for event in sample.history]
    item_counts = Counter(event.item_id for event in sample.history)
    return {
        "history_length": len(sample.history),
        "behavior_counts": dict(Counter(behaviors)),
        "recent_behaviors": behaviors[-5:],
        "repeated_item_kinds": sum(count > 1 for count in item_counts.values()),
    }


def main() -> None:
    """冻结Qwen完成近线生成与编码，输出训练期只读特征缓存。"""
    args = parse_args()
    positive = (
        args.max_train_samples, args.max_validation_samples,
        args.short_history, args.long_history, args.igr_top_k, args.batch_size,
        args.max_new_tokens,
    )
    if min(positive) < 1:
        raise ValueError("sample limits, history sizes, top-k and batch-size must be positive")
    if args.igr_top_k > args.long_history:
        raise ValueError("igr-top-k cannot exceed long-history")
    if args.output.exists() or args.reasoning_output.exists():
        raise FileExistsError("refusing to overwrite an existing instruction cache output")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    boundaries = TemporalBoundaries(**checkpoint["boundaries"])
    registry = SIDRegistry.from_json(args.sid_registry)
    events = [
        event for event in load_retailrocket_events(args.events)
        if event.item_id in registry.item_to_sid
    ]
    samples = build_next_item_samples(
        events,
        boundaries,
        min_history=args.short_history + args.igr_top_k,
        max_history=args.short_history + args.long_history,
        max_samples_per_split={
            Split.TRAIN: args.max_train_samples,
            Split.VALIDATION: args.max_validation_samples,
            Split.TEST: 1,
        },
        sample_seed=args.sample_seed,
    )
    selected = [
        sample for sample in samples
        if sample.split in {Split.TRAIN, Split.VALIDATION}
    ]
    split_counts = Counter(sample.split.value for sample in selected)
    if split_counts["train"] != args.max_train_samples:
        raise RuntimeError("bounded cache cohort did not fill the requested train samples")
    if split_counts["validation"] != args.max_validation_samples:
        raise RuntimeError("bounded cache cohort did not fill the requested validation samples")

    llm = FrozenLLMReasoningGenerator(
        args.model_path, device=args.device, dtype=args.dtype,
        max_input_length=512,
    )
    sample_keys: list[str] = []
    feature_batches = []
    reasoning_records = []
    instruction_texts_seen: set[str] = set()
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start:start + args.batch_size]
        evidence_rows = [evidence_from_history(sample) for sample in batch]
        prompts = [build_behavior_prompt(**evidence) for evidence in evidence_rows]
        generated = llm.generate(prompts, max_new_tokens=args.max_new_tokens)
        instruction_texts = [
            contextual_instruction_text(output.parsed) for output in generated
        ]
        encoded = llm.encode_instruction_texts(
            instruction_texts, pooling="last_token",
        )
        feature_batches.append(encoded.features.detach().to(device="cpu", dtype=torch.float16))
        for sample, evidence, output, text, token_count in zip(
            batch, evidence_rows, generated, instruction_texts,
            encoded.token_counts, strict=True,
        ):
            key = instruction_sample_key(sample)
            sample_keys.append(key)
            instruction_texts_seen.add(text)
            reasoning_records.append({
                "sample_key": key,
                "split": sample.split.value,
                "input_evidence": evidence,
                "reasoning": output.parsed,
                "instruction_text": text,
                "instruction_tokens": token_count,
                "target_excluded": True,
            })
        print(f"stage=cache_qwen completed={len(sample_keys)}/{len(selected)}")

    features = torch.cat(feature_batches, dim=0)
    save_instruction_feature_cache(
        args.output,
        sample_keys=sample_keys,
        features=features,
        metadata={
            "source": "qwen_generated_contextual_reasoning_instruction",
            "model_directory_name": args.model_path.name,
            "pooling": "last_token",
            "dtype": "float16",
            "sid_registry_version": registry.version,
            "boundaries": asdict(boundaries),
            "sample_seed": args.sample_seed,
            "short_history": args.short_history,
            "long_history": args.long_history,
            "igr_top_k": args.igr_top_k,
            "max_new_tokens": args.max_new_tokens,
            "split_counts": dict(split_counts),
        },
    )
    args.reasoning_output.parent.mkdir(parents=True, exist_ok=True)
    args.reasoning_output.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in reasoning_records
        ),
        encoding="utf-8",
    )
    print(
        f"OK device={args.device} cached={len(sample_keys)} "
        f"split_counts={dict(split_counts)} feature_shape={tuple(features.shape)} "
        f"unique_instruction_rate={len(instruction_texts_seen) / len(sample_keys):.6f} "
        f"cache={args.output} reasoning={args.reasoning_output}"
    )


if __name__ == "__main__":
    main()
