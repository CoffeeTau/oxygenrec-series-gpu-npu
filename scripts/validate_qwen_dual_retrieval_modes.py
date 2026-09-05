#!/usr/bin/env python3
"""单卡验证论文式Instruction IGR与Agentic Plan IGR可显式切换。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

from oxygenrec.llm_features import build_behavior_prompt
from oxygenrec.llm_reasoning import (
    FrozenLLMReasoningGenerator,
    contextual_instruction_text,
)
from oxygenrec.model import OxygenRECConfig, OxygenRECModel
from oxygenrec.retrieval_planning import compile_retrieval_plan


def parse_args() -> argparse.Namespace:
    """定义本地Qwen路径以及单卡精度参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    return parser.parse_args()


def main() -> None:
    """生成Instruction，编码为query输入，并在同一模型上切换两种IGR。"""
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # 两例都只包含目标事件之前的聚合证据，而且三种行为均真实出现，确保
    # Qwen生成的任意白名单priority都能通过事实编译器。
    evidence_rows = [
        {
            "history_length": 41,
            "behavior_counts": {"view": 38, "addtocart": 2, "transaction": 1},
            "recent_behaviors": ("view", "addtocart", "view", "transaction", "view"),
            "repeated_item_kinds": 10,
        },
        {
            "history_length": 120,
            "behavior_counts": {"view": 97, "addtocart": 12, "transaction": 11},
            "recent_behaviors": ("transaction", "view", "view", "view", "addtocart"),
            "repeated_item_kinds": 19,
        },
    ]
    prompts = [build_behavior_prompt(**row) for row in evidence_rows]
    llm = FrozenLLMReasoningGenerator(
        args.model_path, device=args.device, dtype=args.dtype,
        max_input_length=args.max_input_length,
    )
    generated = llm.generate(prompts, max_new_tokens=args.max_new_tokens)

    # 论文分支只编码自然语言Instruction/Reason，不读取retrieval_plan字段。
    instruction_texts = [
        contextual_instruction_text(row.parsed) for row in generated
    ]
    feature_batch = llm.encode_instruction_texts(
        instruction_texts, pooling="last_token",
    )
    features = feature_batch.features
    plans = [
        compile_retrieval_plan(row.parsed["retrieval_plan"], evidence["behavior_counts"])
        for row, evidence in zip(generated, evidence_rows, strict=True)
    ]

    torch.manual_seed(47)
    model = OxygenRECModel(OxygenRECConfig(
        sid_width=32,
        hidden_size=32,
        attention_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        feedforward_size=64,
        dropout=0.0,
        max_history_items=2,
        instruction_feature_size=features.shape[1],
        q2i_dimension=16,
        q2i_weight=0.2,
        igr_top_k=3,
    )).to(device)
    short = torch.tensor(
        [[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]],
        device=device,
    )
    short_mask = torch.zeros(2, 2, dtype=torch.bool, device=device)
    long = torch.tensor([
        [[7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21]],
        [[8, 9, 10], [11, 12, 13], [14, 15, 16], [17, 18, 19], [20, 21, 22]],
    ], device=device)
    long_mask = torch.zeros(2, 5, dtype=torch.bool, device=device)
    long_behaviors = torch.tensor(
        [[0, 1, 2, 0, 1], [2, 0, 1, 0, 2]], device=device,
    )
    targets = torch.tensor([[7, 8, 9], [8, 9, 10]], device=device)
    shared = dict(
        target_sids=targets,
        instruction_features=features,
        long_history_sids=long,
        long_history_padding_mask=long_mask,
        long_history_behavior_ids=long_behaviors,
    )

    # paper_igr：Qwen文本特征进入query，严格执行余弦Top-K。
    paper = model(short, short_mask, retrieval_mode="paper_igr", **shared)
    # agentic_plan：保持相同query和候选池，只在相同语义分数上增加Plan控制。
    agentic = model(
        short, short_mask, retrieval_mode="agentic_plan",
        retrieval_plans=plans, **shared,
    )
    paper.loss.backward()

    feature_delta = torch.linalg.vector_norm(features[0] - features[1]).item()
    feature_cosine = F.cosine_similarity(features[0:1], features[1:2]).item()
    adapter_grad = model.instruction_feature_adapter.weight.grad.norm().item()
    selection_changed = (paper.igr_indices != agentic.igr_indices).any(dim=1)
    identity_error = (
        paper.loss - (paper.ntp_loss + model.config.q2i_weight * paper.q2i_loss)
    ).abs().item()

    # 两个负例保证调用方不能把Plan静默混入论文分支，也不能声称使用
    # agentic_plan却遗漏Plan。
    paper_rejected_plan = False
    try:
        model(
            short, short_mask, retrieval_mode="paper_igr",
            retrieval_plans=plans, **shared,
        )
    except ValueError as error:
        paper_rejected_plan = "does not consume" in str(error)
    agentic_required_plan = False
    try:
        model(short, short_mask, retrieval_mode="agentic_plan", **shared)
    except ValueError as error:
        agentic_required_plan = "requires retrieval_plans" in str(error)

    if feature_delta <= 0 or adapter_grad <= 0:
        raise AssertionError("generated instruction did not enter the trainable query path")
    if identity_error > 1e-5:
        raise AssertionError("paper Q2I/NTP joint-loss identity failed")
    if not paper_rejected_plan or not agentic_required_plan:
        raise AssertionError("retrieval mode contract is not explicit")
    if not torch.isfinite(paper.loss) or not torch.isfinite(agentic.loss):
        raise AssertionError("retrieval mode produced non-finite loss")

    peak_gib = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda" else 0.0
    )
    print(
        f"OK device={device} generated={len(generated)} "
        f"instruction_shape={tuple(features.shape)} tokens={feature_batch.token_counts} "
        f"feature_delta={feature_delta:.6f} feature_cosine={feature_cosine:.6f} "
        f"q2i={float(paper.q2i_loss.detach()):.6f} "
        f"loss_identity_error={identity_error:.3e} adapter_grad={adapter_grad:.6f} "
        f"paper_indices={paper.igr_indices.tolist()} "
        f"agentic_indices={agentic.igr_indices.tolist()} "
        f"selection_changed={selection_changed.tolist()} "
        f"mode_contracts=True peak_allocated_gib={peak_gib:.3f}"
    )


if __name__ == "__main__":
    main()
