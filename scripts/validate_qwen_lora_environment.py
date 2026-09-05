#!/usr/bin/env python3
"""在不加载模型权重的情况下检查 Qwen LoRA SFT 最小环境。"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    import transformers
    try:
        import accelerate
        import peft
    except ImportError as error:
        raise RuntimeError(
            f"missing LoRA dependency: {error.name}; install only peft from the "
            "company-approved PyPI mirror, and do not upgrade torch/transformers"
        ) from error
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"local model directory not found: {args.model_path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if not tokenizer.chat_template:
        raise RuntimeError("Qwen tokenizer has no chat template")
    bf16 = torch.cuda.is_bf16_supported() if args.device.startswith("cuda") else False
    if args.device.startswith("cuda") and not bf16:
        raise RuntimeError("selected CUDA device does not support BF16")
    memory_gib = (
        torch.cuda.get_device_properties(args.device).total_memory / 1024**3
        if args.device.startswith("cuda") else 0.0
    )
    print(
        f"OK device={args.device} torch={torch.__version__} "
        f"transformers={transformers.__version__} accelerate={accelerate.__version__} "
        f"peft={peft.__version__} model_type={config.model_type} "
        f"layers={config.num_hidden_layers} hidden_size={config.hidden_size} "
        f"bf16={bf16} memory_gib={memory_gib:.3f} local_only=True"
    )


if __name__ == "__main__":
    main()
