#!/usr/bin/env python3
"""比较同起点EA-TOSD与SFT-only checkpoint的连续策略差异。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.nn import functional as F

from oxygenrec.data import (
    Split,
    TemporalBoundaries,
    build_daily_listwise_samples,
    load_retailrocket_events,
)
from oxygenrec.model import OxygenRECConfig, OxygenRECModel
from oxygenrec.sid import PrefixTrie, SIDRegistry
from train_v2_ea_tosd_retailrocket import BEHAVIOR_BY_NAME, common_inputs, make_batch
from train_v2_listwise_retailrocket import chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument("--ea-checkpoint", type=Path, required=True)
    parser.add_argument("--sft-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/review/v2_ea_tosd_checkpoint_comparison"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--list-size", type=int, default=2)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--max-future-items", type=int, default=2)
    parser.add_argument(
        "--minimum-future-behavior",
        choices=tuple(BEHAVIOR_BY_NAME),
        default="view",
    )
    parser.add_argument("--max-train-samples", type=int, default=32)
    parser.add_argument("--max-validation-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def checkpoint_config(payload: dict) -> OxygenRECConfig:
    """仅恢复当前代码认识的配置字段，并拒绝缺失关键结构。"""
    known = {field.name for field in fields(OxygenRECConfig)}
    values = {
        key: value
        for key, value in payload["model_config"].items()
        if key in known
    }
    return OxygenRECConfig(**values)


def parameter_delta(left: dict, right: dict) -> dict:
    """对两个model state按元素汇总绝对差、L2差和变化覆盖率。"""
    if set(left) != set(right):
        raise ValueError("checkpoint model-state keys differ")
    elements = 0
    changed = 0
    changed_tensors = 0
    absolute_sum = 0.0
    squared_sum = 0.0
    reference_squared_sum = 0.0
    maximum = 0.0
    per_tensor = []
    for name in sorted(left):
        first = left[name]
        second = right[name]
        if first.shape != second.shape:
            raise ValueError(f"checkpoint tensor shape differs for {name}")
        if not first.is_floating_point():
            if not torch.equal(first, second):
                raise ValueError(f"non-floating checkpoint tensor differs for {name}")
            continue
        difference = (first.detach() - second.detach()).float()
        count = difference.numel()
        nonzero = int(torch.count_nonzero(difference).item())
        tensor_max = float(difference.abs().max().item()) if count else 0.0
        tensor_l2 = float(torch.linalg.vector_norm(difference).item())
        elements += count
        changed += nonzero
        changed_tensors += int(nonzero > 0)
        absolute_sum += float(difference.abs().sum().item())
        squared_sum += tensor_l2 * tensor_l2
        reference_squared_sum += float(
            torch.linalg.vector_norm(second.detach().float()).item()
        ) ** 2
        maximum = max(maximum, tensor_max)
        if nonzero:
            per_tensor.append({
                "name": name,
                "elements": count,
                "changed_elements": nonzero,
                "mean_abs": float(difference.abs().mean().item()),
                "max_abs": tensor_max,
                "l2": tensor_l2,
            })
    per_tensor.sort(key=lambda row: row["l2"], reverse=True)
    l2 = squared_sum ** 0.5
    reference_l2 = reference_squared_sum ** 0.5
    return {
        "floating_tensors": sum(tensor.is_floating_point() for tensor in left.values()),
        "changed_tensors": changed_tensors,
        "elements": elements,
        "changed_elements": changed,
        "changed_fraction": changed / elements if elements else 0.0,
        "mean_abs": absolute_sum / elements if elements else 0.0,
        "max_abs": maximum,
        "l2": l2,
        "relative_l2_to_sft": l2 / reference_l2 if reference_l2 else 0.0,
        "largest_l2_tensors": per_tensor[:10],
    }


def compare_models(
    ea_model,
    sft_model,
    samples,
    registry,
    trie,
    args,
    device,
) -> dict:
    """在同一部署输入上比较连续logits与最终约束生成。"""
    ea_model.eval()
    sft_model.eval()
    totals = Counter()
    sums = Counter()
    maximum_logit_delta = 0.0
    with torch.no_grad():
        for sample_batch in chunks(samples, args.batch_size):
            batch = make_batch(sample_batch, registry, args, device)
            shared = common_inputs(batch)
            targets = batch["target_sids"]
            flat_targets = targets.reshape(len(sample_batch), -1)

            ea_output = ea_model(**shared, target_sids=targets)
            sft_output = sft_model(**shared, target_sids=targets)
            ea_logits = torch.stack(ea_output.logits, dim=1)
            sft_logits = torch.stack(sft_output.logits, dim=1)
            difference = ea_logits - sft_logits
            ea_log_probs = F.log_softmax(ea_logits, dim=-1)
            sft_log_probs = F.log_softmax(sft_logits, dim=-1)
            ea_probs = ea_log_probs.exp()
            sft_probs = sft_log_probs.exp()
            symmetric_kl = 0.5 * (
                (ea_probs * (ea_log_probs - sft_log_probs)).sum(dim=-1)
                + (sft_probs * (sft_log_probs - ea_log_probs)).sum(dim=-1)
            )
            token_count = flat_targets.numel()
            sums.update({
                "absolute_logit_delta": float(difference.abs().sum().item()),
                "symmetric_kl": float(symmetric_kl.sum().item()),
                "ea_gold_log_prob": float(
                    ea_log_probs.gather(2, flat_targets.unsqueeze(-1)).sum().item()
                ),
                "sft_gold_log_prob": float(
                    sft_log_probs.gather(2, flat_targets.unsqueeze(-1)).sum().item()
                ),
            })
            maximum_logit_delta = max(
                maximum_logit_delta, float(difference.abs().max().item())
            )
            totals.update({
                "samples": len(sample_batch),
                "tokens": token_count,
                "logit_elements": difference.numel(),
                "teacher_forcing_argmax_changes": int(
                    ea_logits.argmax(dim=-1).ne(sft_logits.argmax(dim=-1)).sum().item()
                ),
            })

            ea_generated = ea_model.generate(
                shared["history_sids"],
                shared["history_padding_mask"],
                trie,
                history_behavior_ids=shared["history_behavior_ids"],
                behavior_instruction_ids=shared["behavior_instruction_ids"],
                output_items=args.list_size,
            )
            sft_generated = sft_model.generate(
                shared["history_sids"],
                shared["history_padding_mask"],
                trie,
                history_behavior_ids=shared["history_behavior_ids"],
                behavior_instruction_ids=shared["behavior_instruction_ids"],
                output_items=args.list_size,
            )
            token_changes = ea_generated.ne(sft_generated)
            ea_hits = ea_generated.eq(flat_targets)
            sft_hits = sft_generated.eq(flat_targets)
            totals.update({
                "greedy_token_changes": int(token_changes.sum().item()),
                "greedy_list_changes": int(token_changes.any(dim=1).sum().item()),
                "ea_only_target_token_hits": int((ea_hits & ~sft_hits).sum().item()),
                "sft_only_target_token_hits": int((sft_hits & ~ea_hits).sum().item()),
            })

    if not totals["samples"]:
        raise RuntimeError("validation produced no comparison rows")
    return {
        "samples": totals["samples"],
        "tokens": totals["tokens"],
        "mean_abs_logit_delta": (
            sums["absolute_logit_delta"] / totals["logit_elements"]
        ),
        "max_abs_logit_delta": maximum_logit_delta,
        "mean_symmetric_kl": sums["symmetric_kl"] / totals["tokens"],
        "ea_mean_gold_log_prob": sums["ea_gold_log_prob"] / totals["tokens"],
        "sft_mean_gold_log_prob": sums["sft_gold_log_prob"] / totals["tokens"],
        "ea_minus_sft_gold_log_prob": (
            sums["ea_gold_log_prob"] - sums["sft_gold_log_prob"]
        ) / totals["tokens"],
        "teacher_forcing_argmax_change_rate": (
            totals["teacher_forcing_argmax_changes"] / totals["tokens"]
        ),
        "greedy_token_change_rate": (
            totals["greedy_token_changes"] / totals["tokens"]
        ),
        "greedy_list_change_rate": (
            totals["greedy_list_changes"] / totals["samples"]
        ),
        "ea_only_target_token_hits": totals["ea_only_target_token_hits"],
        "sft_only_target_token_hits": totals["sft_only_target_token_hits"],
    }


def write_report(output_dir: Path, summary: dict) -> None:
    """保存连续差异诊断及其解释边界。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoint_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    parameters = summary["parameter_delta"]
    behavior = summary["behavior_delta"]
    lines = [
        "# OxygenREC-v2 EA-TOSD / SFT-only checkpoint差异",
        "",
        "> 本报告只比较两条已完成训练分支，不执行新的参数更新。",
        "> validation仍是32条future-eligible公开代理cohort，不是最终效果评测。",
        "",
        "## 参数差异",
        "",
        f"- changed tensors：{parameters['changed_tensors']}/{parameters['floating_tensors']}",
        f"- changed elements：{parameters['changed_elements']}/{parameters['elements']}",
        f"- mean/max absolute delta：{parameters['mean_abs']:.9e} / {parameters['max_abs']:.9e}",
        f"- L2 / relative L2：{parameters['l2']:.9e} / {parameters['relative_l2_to_sft']:.9e}",
        "",
        "## 同validation输入上的策略差异",
        "",
        f"- mean/max absolute logit delta：{behavior['mean_abs_logit_delta']:.9e} / {behavior['max_abs_logit_delta']:.9e}",
        f"- mean symmetric KL：{behavior['mean_symmetric_kl']:.9e}",
        f"- EA-SFT mean gold log-prob：{behavior['ea_minus_sft_gold_log_prob']:+.9e}",
        f"- teacher-forcing argmax change rate：{behavior['teacher_forcing_argmax_change_rate']:.6f}",
        f"- greedy token/list change rate：{behavior['greedy_token_change_rate']:.6f} / {behavior['greedy_list_change_rate']:.6f}",
        f"- EA-only / SFT-only target token hits：{behavior['ea_only_target_token_hits']} / {behavior['sft_only_target_token_hits']}",
        "",
        "## 解释规则",
        "",
        "- 参数与logits不同、greedy不变：EA已改变连续策略，但尚未跨过解码决策边界。",
        "- greedy改变、聚合命中相同：EA与SFT变化在小cohort上相互抵消。",
        "- 参数差异接近数值零：再检查附加loss权重、门控和梯度贡献。",
        "- 任一种情况都不能由本smoke推出EA-TOSD有稳定质量收益。",
        "",
    ]
    (output_dir / "checkpoint_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if min(
        args.list_size,
        args.max_history,
        args.max_future_items,
        args.max_train_samples,
        args.max_validation_samples,
        args.batch_size,
    ) < 1:
        raise ValueError("sizes and counts must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")

    print("stage=load_paired_checkpoints")
    ea_payload = torch.load(args.ea_checkpoint, map_location=device, weights_only=False)
    sft_payload = torch.load(args.sft_checkpoint, map_location=device, weights_only=False)
    if ea_payload.get("sid_registry_version") != sft_payload.get("sid_registry_version"):
        raise ValueError("EA and SFT checkpoint registry versions differ")
    if ea_payload.get("boundaries") != sft_payload.get("boundaries"):
        raise ValueError("EA and SFT checkpoint temporal boundaries differ")
    if ea_payload.get("pretrained_checkpoint") != sft_payload.get("pretrained_checkpoint"):
        raise ValueError("EA and SFT branches did not use the same pretrained checkpoint")
    if "ea_tosd" not in ea_payload:
        raise ValueError("--ea-checkpoint does not contain EA-TOSD configuration")
    if sft_payload.get("summary", {}).get("objective") != "behavior_weighted_sft_only":
        raise ValueError("--sft-checkpoint is not the paired SFT-only control")
    ea_config = checkpoint_config(ea_payload)
    sft_config = checkpoint_config(sft_payload)
    if ea_config != sft_config:
        raise ValueError("EA and SFT checkpoint model configurations differ")
    if ea_config.max_target_items != args.list_size:
        raise ValueError("checkpoint max_target_items does not match --list-size")
    if ea_config.max_future_items != args.max_future_items:
        raise ValueError("checkpoint max_future_items does not match the requested value")

    registry = SIDRegistry.from_json(args.sid_registry)
    if ea_payload.get("sid_registry_version") != registry.version:
        raise ValueError("checkpoint and SID registry versions differ")
    ea_model = OxygenRECModel(ea_config).to(device)
    sft_model = OxygenRECModel(sft_config).to(device)
    ea_model.load_state_dict(ea_payload["model_state"], strict=True)
    sft_model.load_state_dict(sft_payload["model_state"], strict=True)
    parameters = parameter_delta(ea_model.state_dict(), sft_model.state_dict())

    print("stage=build_paired_validation_cohort")
    boundaries = TemporalBoundaries(**ea_payload["boundaries"])
    events = [
        event
        for event in load_retailrocket_events(args.events)
        if event.item_id in registry.item_to_sid
    ]
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
        max_future_targets=args.max_future_items,
        minimum_future_behavior=BEHAVIOR_BY_NAME[args.minimum_future_behavior],
        require_future_targets=True,
    )
    validation = [
        sample.without_privileged_future()
        for sample in samples
        if sample.split is Split.VALIDATION
    ]
    if len(validation) != args.max_validation_samples:
        raise RuntimeError("paired validation cohort does not have the requested size")
    trie = PrefixTrie.from_registry(registry)
    behavior = compare_models(
        ea_model, sft_model, validation, registry, trie, args, device
    )
    summary = {
        "parameter_delta": parameters,
        "behavior_delta": behavior,
        "validation_cohort": "future_eligible_same_cohort",
        "validation_samples": len(validation),
        "external_reward_model": False,
        "read_only": True,
    }
    write_report(args.output_dir, summary)
    print(
        "OK "
        f"device={device.type} variant=v2_ea_vs_sft_checkpoint_readonly "
        f"validation={len(validation)} "
        f"changed_tensors={parameters['changed_tensors']}/"
        f"{parameters['floating_tensors']} "
        f"parameter_mean_abs={parameters['mean_abs']:.9e} "
        f"parameter_max_abs={parameters['max_abs']:.9e} "
        f"parameter_relative_l2={parameters['relative_l2_to_sft']:.9e} "
        f"logit_mean_abs={behavior['mean_abs_logit_delta']:.9e} "
        f"logit_max_abs={behavior['max_abs_logit_delta']:.9e} "
        f"symmetric_kl={behavior['mean_symmetric_kl']:.9e} "
        f"gold_logprob_delta={behavior['ea_minus_sft_gold_log_prob']:+.9e} "
        f"teacher_argmax_change={behavior['teacher_forcing_argmax_change_rate']:.6f} "
        f"greedy_token_change={behavior['greedy_token_change_rate']:.6f} "
        f"greedy_list_change={behavior['greedy_list_change_rate']:.6f} "
        f"ea_only_hits={behavior['ea_only_target_token_hits']} "
        f"sft_only_hits={behavior['sft_only_target_token_hits']} "
        f"report={args.output_dir / 'checkpoint_comparison.md'} "
        "read_only=True external_reward_model=False"
    )


if __name__ == "__main__":
    main()
