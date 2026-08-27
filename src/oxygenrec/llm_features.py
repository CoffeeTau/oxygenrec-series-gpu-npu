"""Frozen Hugging Face LLM features for OxygenREC instructions.

This optional module replaces the deterministic hash proxy with contextual
hidden states from a locally supplied open-weight model.  It does not download
weights implicitly and keeps the LLM frozen during the first integration stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class LLMFeatureBatch:
    features: object
    token_counts: tuple[int, ...]


def build_behavior_prompt(
    *, history_length: int, behavior_counts: dict[str, int],
    recent_behaviors: Sequence[str], repeated_item_kinds: int,
) -> str:
    """Build a structured, target-free prompt from strictly prior evidence."""

    if history_length < 1 or not recent_behaviors:
        raise ValueError("history evidence must not be empty")
    counts = {name: int(behavior_counts.get(name, 0)) for name in (
        "view", "addtocart", "transaction"
    )}
    return (
        "你是电商检索规划器。只能依据目标事件之前的用户行为证据，"
        "生成用于长期历史检索的简短意图表示；不得猜测下一次真实行为或商品。\n"
        f"历史长度: {history_length}\n"
        f"行为计数: view={counts['view']}, addtocart={counts['addtocart']}, "
        f"transaction={counts['transaction']}\n"
        f"最近行为: {', '.join(recent_behaviors)}\n"
        f"重复访问商品种类: {repeated_item_kinds}\n"
        "检索目标: 总结长期兴趣、高意图线索、时效性和候选多样性。"
    )


class FrozenLLMInstructionEncoder:
    """Mean-pool normalized hidden states from a local Transformers model."""

    def __init__(
        self, model_path: str | Path, *, device: str = "cuda",
        dtype: str = "bfloat16", max_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(
                f"local model directory not found: {path}; weights are never downloaded implicitly"
            )
        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in dtype_by_name:
            raise ValueError(f"unsupported dtype: {dtype}")
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False,
        )
        self.model = AutoModel.from_pretrained(
            path, local_files_only=True, trust_remote_code=False,
            dtype=dtype_by_name[dtype], low_cpu_mem_usage=True,
        ).to(self.device).eval()
        self.model.requires_grad_(False)
        self.hidden_size = int(self.model.config.hidden_size)

    def encode(self, texts: Sequence[str], *, pooling: str = "mean") -> LLMFeatureBatch:
        import torch
        import torch.nn.functional as F

        if not texts or any(not text.strip() for text in texts):
            raise ValueError("texts must contain non-empty strings")
        if pooling not in {"mean", "last_token"}:
            raise ValueError("pooling must be mean or last_token")
        formatted = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False,
                add_generation_prompt=False,
            )
            if self.tokenizer.chat_template else text
            for text in texts
        ]
        tokens = self.tokenizer(
            formatted, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        tokens = {name: value.to(self.device) for name, value in tokens.items()}
        with torch.inference_mode():
            hidden = self.model(**tokens, return_dict=True).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            if pooling == "mean":
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            else:
                positions = torch.arange(hidden.shape[1], device=self.device)
                last_indices = (
                    tokens["attention_mask"] * positions.unsqueeze(0)
                ).argmax(dim=1)
                pooled = hidden[
                    torch.arange(hidden.shape[0], device=self.device), last_indices
                ]
            features = F.normalize(pooled.float(), dim=-1)
        # Tensors created in inference_mode cannot be saved for backward by
        # the trainable OxygenREC adapter.  Clone after leaving the context to
        # produce an ordinary detached tensor while keeping Qwen frozen.
        features = features.clone()
        counts = tuple(int(value) for value in tokens["attention_mask"].sum(dim=1).tolist())
        return LLMFeatureBatch(features=features, token_counts=counts)
