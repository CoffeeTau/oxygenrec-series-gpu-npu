#!/usr/bin/env python3
"""在单卡上验证OxygenREC-v2 EA-TOSD论文主链，不调用外部Reward Model。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.ea_tosd import (  # noqa: E402
    EATOSDConfig,
    ea_tosd_loss,
    select_best_verifiable_trajectory,
)
from oxygenrec.model import OxygenRECConfig, OxygenRECModel  # noqa: E402
from oxygenrec.sid import PrefixTrie  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrain-steps", type=int, default=300)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def total_gradient(module: torch.nn.Module) -> float:
    """汇总共享Student backbone当前梯度L1范数。"""
    return float(sum(
        parameter.grad.abs().sum().item()
        for parameter in module.parameters()
        if parameter.grad is not None
    ))


def logits_max_delta(left, right) -> float:
    """比较两组逐SID位置logits的最大绝对差。"""
    return max(float((a - b).abs().max()) for a, b in zip(left, right))


def paths_are_legal(paths: torch.Tensor, trie: PrefixTrie) -> bool:
    """把``[B,G,3N]``逐商品切分，验证每组三层SID都属于registry trie。"""
    return all(
        trie.contains(path[start : start + 3])
        for path in paths.reshape(-1, paths.shape[-1]).tolist()
        for start in range(0, len(path), 3)
    )


def main() -> None:
    args = parse_args()
    if args.pretrain_steps < 1 or args.group_size < 1:
        raise ValueError("pretrain steps and group size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")

    torch.manual_seed(43)
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
        max_future_items=2,
    )).to(device)
    trie = PrefixTrie([
        (1, 2, 3), (4, 5, 6), (7, 8, 9),
        (10, 11, 12), (13, 14, 15), (2, 4, 6),
    ])

    base_history = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]],
        dtype=torch.long,
        device=device,
    )
    history = base_history.unsqueeze(0).expand(6, -1, -1).clone()
    padding = torch.tensor(
        [[False, False, False, True]], device=device
    ).expand(6, -1).clone()
    history_behaviors = torch.tensor(
        [[0, 1, 2, 0]], dtype=torch.long, device=device
    ).expand(6, -1).clone()
    behavior_ids = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
    targets = torch.tensor(
        [[[1, 2, 3], [4, 5, 6]]] * 2
        + [[[7, 8, 9], [10, 11, 12]]] * 2
        + [[[13, 14, 15], [2, 4, 6]]] * 2,
        dtype=torch.long,
        device=device,
    )
    behavior_weights = torch.tensor([1.2, 1.5, 2.0], device=device)
    token_weights = behavior_weights[behavior_ids].unsqueeze(1).expand(-1, 6)

    # 先得到一个可采出非零verifiable reward的行为条件Student。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    for _ in range(args.pretrain_steps):
        optimizer.zero_grad(set_to_none=True)
        pretrain = model(
            history,
            padding,
            target_sids=targets,
            history_behavior_ids=history_behaviors,
            behavior_instruction_ids=behavior_ids,
            token_weights=token_weights,
        )
        pretrain.loss.backward()
        optimizer.step()

    model.eval()
    candidates = model.sample_trajectories(
        history,
        padding,
        trie,
        group_size=args.group_size,
        history_behavior_ids=history_behaviors,
        behavior_instruction_ids=behavior_ids,
        output_items=2,
        temperature=args.temperature,
    )
    gold_tokens = targets.reshape(targets.shape[0], -1)
    selection = select_best_verifiable_trajectory(
        candidates, gold_tokens, geometric_decay=0.9
    )
    selected_targets = selection.selected_tokens.reshape(-1, 2, 3)

    # F含一个有效未来目标和一个固定padding位置；Teacher只在训练中读取F。
    future = torch.tensor(
        [
            [[7, 8, 9], [1, 2, 3]],
            [[10, 11, 12], [4, 5, 6]],
            [[13, 14, 15], [7, 8, 9]],
            [[2, 4, 6], [10, 11, 12]],
            [[1, 2, 3], [13, 14, 15]],
            [[4, 5, 6], [2, 4, 6]],
        ],
        dtype=torch.long,
        device=device,
    )
    future_mask = torch.tensor(
        [[False, True]], dtype=torch.bool, device=device
    ).expand(6, -1).clone()
    changed_padding = future.clone()
    changed_padding[:, 1] = torch.tensor([10, 11, 12], device=device)

    with torch.no_grad():
        teacher = model(
            history,
            padding,
            target_sids=selected_targets,
            history_behavior_ids=history_behaviors,
            behavior_instruction_ids=behavior_ids,
            privileged_future_sids=future,
            privileged_future_padding_mask=future_mask,
        )
        teacher_changed_padding = model(
            history,
            padding,
            target_sids=selected_targets,
            history_behavior_ids=history_behaviors,
            behavior_instruction_ids=behavior_ids,
            privileged_future_sids=changed_padding,
            privileged_future_padding_mask=future_mask,
        )
    padding_delta = logits_max_delta(teacher.logits, teacher_changed_padding.logits)

    # 同一次更新同时计算selected trajectory三项损失和gold SFT anchor。
    model.train()
    optimizer.zero_grad(set_to_none=True)
    student = model(
        history,
        padding,
        target_sids=selected_targets,
        history_behavior_ids=history_behaviors,
        behavior_instruction_ids=behavior_ids,
    )
    supervised = model(
        history,
        padding,
        target_sids=targets,
        history_behavior_ids=history_behaviors,
        behavior_instruction_ids=behavior_ids,
        token_weights=token_weights,
    )
    objective = ea_tosd_loss(
        student.logits,
        teacher.logits,
        selection.selected_tokens,
        selection.selected_rewards,
        supervised.loss,
        config=EATOSDConfig(),
    )
    teacher_student_delta = logits_max_delta(student.logits, teacher.logits)
    objective.loss.backward()
    gradient = total_gradient(model)
    optimizer.step()

    # 独立构造低熵与高熵token，确保两个门控公式都在CUDA上实际执行。
    branch_student = torch.zeros(1, 2, 16, device=device, requires_grad=True)
    branch_teacher = torch.stack(
        (
            torch.tensor([12.0] + [-12.0] * 15, device=device),
            torch.zeros(16, device=device),
        ),
        dim=0,
    ).unsqueeze(0)
    branch = ea_tosd_loss(
        branch_student,
        branch_teacher,
        torch.tensor([[0, 1]], dtype=torch.long, device=device),
        torch.zeros(1, device=device),
        branch_student.square().mean(),
        config=EATOSDConfig(),
    )
    branch.loss.backward()

    if not paths_are_legal(candidates, trie):
        raise AssertionError("on-policy rollout emitted an illegal SID")
    if float(selection.selected_rewards.min()) <= 0:
        raise AssertionError("fixture produced no non-zero verifiable reward")
    if padding_delta > 1e-6:
        raise AssertionError(f"masked future SID changed teacher logits: {padding_delta}")
    if teacher_student_delta <= 1e-6:
        raise AssertionError("privileged future prefix did not change teacher logits")
    if gradient <= 0 or not torch.isfinite(objective.loss):
        raise AssertionError("EA-TOSD objective did not produce a finite shared-model gradient")
    if int((branch.low_entropy_weights > 0).sum()) != 1:
        raise AssertionError("low-entropy self-distillation branch was not activated")
    if int((branch.high_entropy_weights > 0).sum()) != 1:
        raise AssertionError("high-entropy forward-KL branch was not activated")

    print(
        "OK "
        f"device={device.type} group={args.group_size} candidates={tuple(candidates.shape)} "
        f"reward_min={float(selection.selected_rewards.min()):.6f} "
        f"reward_max={float(selection.selected_rewards.max()):.6f} "
        f"best_indices={selection.best_indices.tolist()} "
        f"loss={float(objective.loss.detach()):.6f} "
        f"vr={float(objective.verifiable_loss.detach()):.6f} "
        f"sd={float(objective.self_distillation_loss.detach()):.6f} "
        f"fkl={float(objective.forward_kl_loss.detach()):.6f} "
        f"sft={float(objective.supervised_loss.detach()):.6f} "
        f"teacher_student_delta={teacher_student_delta:.6e} "
        f"masked_future_delta={padding_delta:.3e} "
        f"shared_grad={gradient:.6f} "
        "low_gate=True high_gate=True all_rollouts_legal=True external_reward_model=False"
    )


if __name__ == "__main__":
    main()
