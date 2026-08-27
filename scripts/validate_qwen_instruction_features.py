#!/usr/bin/env python3
"""Validate local Qwen hidden states entering OxygenREC on one GPU."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from oxygenrec.llm_features import FrozenLLMInstructionEncoder, build_behavior_prompt
from oxygenrec.model import OxygenRECConfig, OxygenRECModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    prompts = [
        build_behavior_prompt(
            history_length=41,
            behavior_counts={"view": 38, "addtocart": 2, "transaction": 1},
            recent_behaviors=("view", "addtocart", "view", "transaction", "view"),
            repeated_item_kinds=10,
        ),
        build_behavior_prompt(
            history_length=41,
            behavior_counts={"view": 41},
            recent_behaviors=("view", "view", "view", "view", "view"),
            repeated_item_kinds=2,
        ),
    ]
    encoder = FrozenLLMInstructionEncoder(
        args.model_path, device=args.device, dtype=args.dtype,
        max_length=args.max_length,
    )
    mean_first = encoder.encode(prompts, pooling="mean")
    mean_second = encoder.encode(prompts, pooling="mean")
    last = encoder.encode(prompts, pooling="last_token")
    determinism_error = (
        mean_first.features - mean_second.features
    ).abs().max().item()
    mean_delta = torch.linalg.vector_norm(
        mean_first.features[0] - mean_first.features[1]
    ).item()
    mean_cosine = torch.dot(
        mean_first.features[0], mean_first.features[1]
    ).item()
    last_delta = torch.linalg.vector_norm(
        last.features[0] - last.features[1]
    ).item()
    last_cosine = torch.dot(last.features[0], last.features[1]).item()

    # Keep all recommendation inputs identical. Any logit difference must come
    # from the real LLM features passed through the existing adapter boundary.
    torch.manual_seed(29)
    model = OxygenRECModel(OxygenRECConfig(
        sid_width=32, hidden_size=32, attention_heads=4,
        encoder_layers=1, decoder_layers=1, feedforward_size=64,
        dropout=0.0, max_history_items=2,
        instruction_feature_size=encoder.hidden_size,
    )).to(args.device)
    history = torch.tensor([[[1, 2, 3], [4, 5, 6]]] * 2, device=args.device)
    mask = torch.zeros(2, 2, dtype=torch.bool, device=args.device)
    targets = torch.tensor([[7, 8, 9], [7, 8, 9]], device=args.device)
    output = model(
        history, mask, target_sids=targets,
        instruction_features=last.features,
    )
    output.loss.backward()
    logit_delta = sum(
        torch.linalg.vector_norm(level[0] - level[1]).item()
        for level in output.logits
    )
    adapter_grad = model.instruction_feature_adapter.weight.grad.norm().item()
    peak_gib = (
        torch.cuda.max_memory_allocated(args.device) / 1024**3
        if args.device.startswith("cuda") else 0.0
    )
    if determinism_error > 1e-6 or mean_delta <= 0 or last_delta <= 0 or logit_delta <= 0 or adapter_grad <= 0:
        raise AssertionError("Qwen feature integration failed")
    print(
        f"OK device={args.device} hidden_size={encoder.hidden_size} "
        f"tokens={mean_first.token_counts} mean_delta={mean_delta:.6f} "
        f"mean_cosine={mean_cosine:.6f} last_delta={last_delta:.6f} "
        f"last_cosine={last_cosine:.6f} determinism_error={determinism_error:.3e} "
        f"logit_delta={logit_delta:.6f} adapter_grad={adapter_grad:.6f} "
        f"peak_allocated_gib={peak_gib:.3f}"
    )


if __name__ == "__main__":
    main()
