#!/usr/bin/env python3
"""在单卡上验证OxygenREC-v2列表式3N SID训练与约束生成闭环。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.model import OxygenRECConfig, OxygenRECModel
from oxygenrec.sid import PrefixTrie


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    return parser.parse_args()


def total_gradient(module: torch.nn.Module) -> float:
    """汇总模块当前梯度的L1范数，用于验证反向链实际接通。"""
    return float(sum(
        parameter.grad.abs().sum().item()
        for parameter in module.parameters()
        if parameter.grad is not None
    ))


def chunks_are_legal(paths: torch.Tensor, trie: PrefixTrie) -> bool:
    """逐三个SID token切分展平列表，并检查每个商品都在Trie中。"""
    return all(
        trie.contains(path[start : start + 3])
        for path in paths.reshape(-1, paths.shape[-1]).tolist()
        for start in range(0, len(path), 3)
    )


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")

    torch.manual_seed(37)
    model = OxygenRECModel(OxygenRECConfig(
        sid_width=16,
        behavior_vocab_size=3,
        behavior_instruction_vocab_size=3,
        hidden_size=32,
        attention_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        feedforward_size=64,
        dropout=0.0,
        max_history_items=4,
        max_target_items=2,
    )).to(device)

    # 一个I_b描述整条列表，因此该最小闭环使用行为同质的二商品列表。
    one_history = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]],
        dtype=torch.long,
        device=device,
    )
    history = one_history.unsqueeze(0).expand(12, -1, -1).clone()
    padding = torch.tensor(
        [[False, False, False, True]], device=device
    ).expand(12, -1).clone()
    history_behaviors = torch.tensor(
        [[0, 1, 2, 0]], dtype=torch.long, device=device
    ).expand(12, -1).clone()
    behavior_ids = torch.tensor(
        [0] * 4 + [1] * 4 + [2] * 4, dtype=torch.long, device=device
    )
    target_lists = torch.tensor(
        [[[1, 2, 3], [4, 5, 6]]] * 4
        + [[[7, 8, 9], [10, 11, 12]]] * 4
        + [[[13, 14, 15], [2, 4, 6]]] * 4,
        dtype=torch.long,
        device=device,
    )
    behavior_weights = torch.tensor(
        [1.2, 1.5, 2.0], dtype=torch.float32, device=device
    )
    token_weights = behavior_weights[behavior_ids].unsqueeze(1).expand(-1, 6)
    trie = PrefixTrie([
        (1, 2, 3), (4, 5, 6), (7, 8, 9),
        (10, 11, 12), (13, 14, 15), (2, 4, 6),
    ])

    model.eval()
    with torch.no_grad():
        initial = model(
            history,
            padding,
            target_sids=target_lists,
            history_behavior_ids=history_behaviors,
            behavior_instruction_ids=behavior_ids,
            token_weights=token_weights,
        )
        initial_loss = float(initial.loss)
        changed_targets = target_lists.clone()
        changed_targets[:, 1] = torch.tensor(
            [10, 11, 12], dtype=torch.long, device=device
        )
        changed = model(
            history,
            padding,
            target_sids=changed_targets,
            history_behavior_ids=history_behaviors,
            behavior_instruction_ids=behavior_ids,
            token_weights=token_weights,
        )
        causal_delta = max(
            float((left - right).abs().max())
            for left, right in zip(initial.logits[:3], changed.logits[:3])
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    max_adapter_grad = 0.0
    max_reserved_token_grad = 0.0
    model.train()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            history,
            padding,
            target_sids=target_lists,
            history_behavior_ids=history_behaviors,
            behavior_instruction_ids=behavior_ids,
            token_weights=token_weights,
        )
        output.loss.backward()
        max_adapter_grad = max(
            max_adapter_grad, total_gradient(model.behavior_instruction_adapter)
        )
        embedding_grad = model.sid_embeddings[0].weight.grad
        if embedding_grad is not None:
            max_reserved_token_grad = max(
                max_reserved_token_grad,
                float(embedding_grad[model.config.sid_width :].abs().sum()),
            )
        optimizer.step()

    review_rows = torch.tensor([0, 4, 8], device=device)
    model.eval()
    with torch.no_grad():
        final = model(
            history,
            padding,
            target_sids=target_lists,
            history_behavior_ids=history_behaviors,
            behavior_instruction_ids=behavior_ids,
            token_weights=token_weights,
        )
        final_loss = float(final.loss)
        generated = model.generate(
            history[review_rows],
            padding[review_rows],
            trie,
            history_behavior_ids=history_behaviors[review_rows],
            behavior_instruction_ids=behavior_ids[review_rows],
            output_items=2,
        )
        beams = model.beam_search(
            history[review_rows],
            padding[review_rows],
            trie,
            beam_width=3,
            history_behavior_ids=history_behaviors[review_rows],
            behavior_instruction_ids=behavior_ids[review_rows],
            output_items=2,
        )
        flat_targets = target_lists[review_rows].reshape(3, 6)
        candidates = torch.stack((flat_targets, flat_targets.flip(1)), dim=1)
        candidate_scores = model.candidate_log_probs(
            history[review_rows],
            padding[review_rows],
            candidates,
            history_behavior_ids=history_behaviors[review_rows],
            behavior_instruction_ids=behavior_ids[review_rows],
        )

    expected = target_lists[review_rows].reshape(3, 6)
    if len(final.logits) != 6:
        raise AssertionError(f"expected 6 SID logits, got {len(final.logits)}")
    if causal_delta > 1e-7:
        raise AssertionError(f"future item leaked into first item: {causal_delta}")
    if not final_loss < initial_loss * 0.25:
        raise AssertionError(
            f"listwise toy batch did not converge: {initial_loss} -> {final_loss}"
        )
    if max_adapter_grad <= 0.0 or max_reserved_token_grad <= 0.0:
        raise AssertionError("I_b projection or reserved token received no gradient")
    if not torch.equal(generated, expected):
        raise AssertionError(f"listwise generation mismatch: {generated.tolist()}")
    if not chunks_are_legal(generated, trie) or not chunks_are_legal(
        beams.semantic_ids, trie
    ):
        raise AssertionError("listwise constrained decoding emitted an illegal SID")
    if candidate_scores.shape != (3, 2, 6) or not torch.isfinite(candidate_scores).all():
        raise AssertionError("listwise candidate log-probability shape/value is invalid")

    print(
        "OK "
        f"device={device.type} "
        "items=2 sid_tokens=6 "
        f"loss={initial_loss:.6f}->{final_loss:.6f} "
        f"causal_delta={causal_delta:.3e} "
        f"adapter_grad={max_adapter_grad:.6f} "
        f"reserved_token_grad={max_reserved_token_grad:.6f} "
        f"generated={generated.tolist()} "
        f"beam_shape={tuple(beams.semantic_ids.shape)} "
        f"candidate_shape={tuple(candidate_scores.shape)} "
        "all_item_chunks_legal=True grouping=homogeneous_behavior"
    )


if __name__ == "__main__":
    main()
