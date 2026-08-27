"""Structured Slow-LLM reasoning generation with explicit audit boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence


REQUIRED_REASONING_KEYS = (
    "intent", "evidence", "retrieval_strategy", "constraints",
)


@dataclass(frozen=True)
class GeneratedReasoning:
    raw_text: str
    parsed: dict[str, object]


def reasoning_system_prompt() -> str:
    return (
        "你是电商检索规划器。只能使用用户提供的历史证据，不得猜测下一次真实行为、"
        "目标商品或未提供的属性。只输出一个JSON对象，不输出Markdown。JSON必须包含"
        "intent、evidence、retrieval_strategy、constraints四个字段；evidence和constraints"
        "必须是字符串数组，其余字段必须是简短字符串。"
    )


def parse_reasoning_json(text: str) -> dict[str, object]:
    """Extract one JSON object while tolerating accidental surrounding text."""

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("generated text does not contain a JSON object")
    parsed = json.loads(text[start:end + 1])
    missing = [key for key in REQUIRED_REASONING_KEYS if key not in parsed]
    if missing:
        raise ValueError(f"generated reasoning is missing keys: {missing}")
    if not isinstance(parsed["intent"], str) or not isinstance(
        parsed["retrieval_strategy"], str
    ):
        raise ValueError("intent and retrieval_strategy must be strings")
    for key in ("evidence", "constraints"):
        if not isinstance(parsed[key], list) or not all(
            isinstance(value, str) for value in parsed[key]
        ):
            raise ValueError(f"{key} must be a list of strings")
    return parsed


class FrozenLLMReasoningGenerator:
    """Deterministic local generation from a frozen causal language model."""

    def __init__(
        self, model_path: str | Path, *, device: str = "cuda",
        dtype: str = "bfloat16", max_input_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"local model directory not found: {path}")
        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.device = torch.device(device)
        self.max_input_length = max_input_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, trust_remote_code=False,
            dtype=dtype_by_name[dtype], low_cpu_mem_usage=True,
        ).to(self.device).eval()
        self.model.requires_grad_(False)

    def generate(
        self, evidence_prompts: Sequence[str], *, max_new_tokens: int = 192,
    ) -> list[GeneratedReasoning]:
        import torch

        rendered = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": reasoning_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False, add_generation_prompt=True,
            )
            for prompt in evidence_prompts
        ]
        tokens = self.tokenizer(
            rendered, padding=True, truncation=True,
            max_length=self.max_input_length, return_tensors="pt",
        )
        tokens = {name: value.to(self.device) for name, value in tokens.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **tokens, do_sample=False, max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        prompt_length = tokens["input_ids"].shape[1]
        results = []
        for row in generated:
            raw = self.tokenizer.decode(row[prompt_length:], skip_special_tokens=True).strip()
            results.append(GeneratedReasoning(raw_text=raw, parsed=parse_reasoning_json(raw)))
        return results
