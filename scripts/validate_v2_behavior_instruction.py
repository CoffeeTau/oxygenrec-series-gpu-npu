#!/usr/bin/env python3
"""在单卡上验证 OxygenREC-v2 Decoder Behavior Instruction 的最小闭环。"""

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
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    return parser.parse_args()


def total_gradient(module: torch.nn.Module) -> float:
    """汇总一个模块当前梯度的 L1 范数，便于确认反向链是否接通。"""
    return float(sum(
        parameter.grad.abs().sum().item()
        for parameter in module.parameters()
        if parameter.grad is not None
    ))


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")

    torch.manual_seed(31)
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
    )).to(device)

    # 三组样本拥有完全相同的用户历史，只改变 Decoder 端目标行为 I_b。
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
    target_paths = torch.tensor(
        [[1, 2, 3]] * 4 + [[7, 8, 9]] * 4 + [[12, 13, 14]] * 4,
        dtype=torch.long,
        device=device,
    )
    trie = PrefixTrie([(1, 2, 3), (7, 8, 9), (12, 13, 14)])

    # I_b 不进入 Encoder：同一组 Encoder 输入重复计算应严格一致。
    model.eval()
    with torch.no_grad():
        encoder_a = model._encode(history[:3], padding[:3], history_behaviors[:3])
        encoder_b = model._encode(history[:3], padding[:3], history_behaviors[:3])
        encoder_delta = float((encoder_a - encoder_b).abs().max())
        initial_loss = float(model(
            history,
            padding,
            target_sids=target_paths,
            history_behavior_ids=history_behaviors,
            behavior_instruction_ids=behavior_ids,
        ).loss)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    max_adapter_grad = 0.0
    max_reserved_token_grad = 0.0
    model.train()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            history,
            padding,
            target_sids=target_paths,
            history_behavior_ids=history_behaviors,
            behavior_instruction_ids=behavior_ids,
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

    model.eval()
    with torch.no_grad():
        final_loss = float(model(
            history,
            padding,
            target_sids=target_paths,
            history_behavior_ids=history_behaviors,
            behavior_instruction_ids=behavior_ids,
        ).loss)
        fixed_target = target_paths[:3]
        click_logits = model(
            history[:3],
            padding[:3],
            target_sids=fixed_target,
            history_behavior_ids=history_behaviors[:3],
            behavior_instruction_ids=torch.zeros(3, dtype=torch.long, device=device),
        ).logits
        order_logits = model(
            history[:3],
            padding[:3],
            target_sids=fixed_target,
            history_behavior_ids=history_behaviors[:3],
            behavior_instruction_ids=torch.full(
                (3,), 2, dtype=torch.long, device=device
            ),
        ).logits
        logit_delta = max(
            float((left - right).abs().max())
            for left, right in zip(click_logits, order_logits)
        )
        generated = model.generate(
            history[[0, 4, 8]],
            padding[[0, 4, 8]],
            trie,
            history_behavior_ids=history_behaviors[[0, 4, 8]],
            behavior_instruction_ids=torch.tensor([0, 1, 2], device=device),
        )
        beams = model.beam_search(
            history[[0, 4, 8]],
            padding[[0, 4, 8]],
            trie,
            beam_width=3,
            history_behavior_ids=history_behaviors[[0, 4, 8]],
            behavior_instruction_ids=torch.tensor([0, 1, 2], device=device),
        )

    expected = torch.tensor(
        [[1, 2, 3], [7, 8, 9], [12, 13, 14]], device=device
    )
    if not final_loss < initial_loss * 0.25:
        raise AssertionError(f"I_b overfit did not converge: {initial_loss} -> {final_loss}")
    if encoder_delta != 0.0:
        raise AssertionError(f"I_b unexpectedly changed Encoder memory: {encoder_delta}")
    if logit_delta <= 1e-5:
        raise AssertionError("switching I_b did not change Decoder logits")
    if max_adapter_grad <= 0.0 or max_reserved_token_grad <= 0.0:
        raise AssertionError("I_b projection or reserved behavior token received no gradient")
    if not torch.equal(generated, expected):
        raise AssertionError(
            f"behavior-conditioned generation mismatch: {generated.tolist()}"
        )
    if not all(trie.contains(path) for rows in beams.semantic_ids.tolist() for path in rows):
        raise AssertionError("beam search returned an illegal SID")

    print(
        "OK "
        f"device={device.type} "
        f"loss={initial_loss:.6f}->{final_loss:.6f} "
        f"encoder_delta={encoder_delta:.6e} "
        f"logit_delta={logit_delta:.6f} "
        f"adapter_grad={max_adapter_grad:.6f} "
        f"reserved_token_grad={max_reserved_token_grad:.6f} "
        f"generated={generated.tolist()} "
        f"beam_shape={tuple(beams.semantic_ids.shape)} "
        "prefix=[BOS,I_s,I_r,I_b]"
    )


if __name__ == "__main__":
    main()
