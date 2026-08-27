"""Structured Slow-LLM reasoning generation with explicit audit boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence


REQUIRED_REASONING_KEYS = (
    "intent", "evidence", "retrieval_strategy", "retrieval_plan", "constraints",
)


@dataclass(frozen=True)
class GeneratedReasoning:
    raw_text: str
    parsed: dict[str, object]


def reasoning_system_prompt() -> str:
    return (
        "你是电商检索规划器。只能使用用户提供的历史证据，不得猜测下一次真实行为、"
        "目标商品或未提供的属性。只输出一个JSON对象，不输出Markdown。JSON必须包含"
        "intent、evidence、retrieval_strategy、retrieval_plan、constraints五个字段。"
        "evidence必须是1到6条聚合事实，不能逐条重复view；constraints必须是字符串数组；"
        "intent和retrieval_strategy必须是简短字符串。retrieval_plan必须是JSON对象，且只包含"
        "priority_behaviors（只能从view/addtocart/transaction选择的数组）、recency"
        "（recent/long_term/balanced之一）、prefer_repeated_items（布尔值）和diversity"
        "（low/medium/high之一）。"
    )


def parse_reasoning_json(text: str) -> dict[str, object]:
    """Extract one JSON object while tolerating accidental surrounding text."""

    start = text.find("{")
    if start < 0:
        raise ValueError("generated text does not contain a JSON object")
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text, idx=start)
    except json.JSONDecodeError as error:
        raise ValueError(
            "generated JSON is incomplete or invalid: "
            f"{error.msg} at line {error.lineno} column {error.colno}"
        ) from error
    if not isinstance(parsed, dict):
        raise ValueError("generated JSON root must be an object")
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
    if not 1 <= len(parsed["evidence"]) <= 6:
        raise ValueError("evidence must contain 1 to 6 aggregated facts")
    plan = parsed["retrieval_plan"]
    if not isinstance(plan, dict) or set(plan) != {
        "priority_behaviors", "recency", "prefer_repeated_items", "diversity",
    }:
        raise ValueError("retrieval_plan has the wrong fields")
    priorities = plan["priority_behaviors"]
    if not isinstance(priorities, list) or not priorities or any(
        value not in {"view", "addtocart", "transaction"} for value in priorities
    ):
        raise ValueError("retrieval_plan priority_behaviors is invalid")
    if plan["recency"] not in {"recent", "long_term", "balanced"}:
        raise ValueError("retrieval_plan recency is invalid")
    if not isinstance(plan["prefer_repeated_items"], bool):
        raise ValueError("retrieval_plan prefer_repeated_items must be boolean")
    if plan["diversity"] not in {"low", "medium", "high"}:
        raise ValueError("retrieval_plan diversity is invalid")
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
        self.tokenizer.padding_side = "left"
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
            generation_config = self.model.generation_config
            generation_config.do_sample = False
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None
            generated = self.model.generate(
                **tokens, generation_config=generation_config,
                max_new_tokens=max_new_tokens, use_cache=True,
            )
        prompt_length = tokens["input_ids"].shape[1]
        results = []
        for index, row in enumerate(generated):
            raw = self.tokenizer.decode(row[prompt_length:], skip_special_tokens=True).strip()
            try:
                parsed = parse_reasoning_json(raw)
            except ValueError as error:
                hit_token_limit = row.shape[0] >= prompt_length + max_new_tokens
                raise ValueError(
                    f"reasoning case {index} failed schema parsing; "
                    f"hit_token_limit={hit_token_limit}; generated_chars={len(raw)}; {error}"
                ) from error
            results.append(GeneratedReasoning(raw_text=raw, parsed=parsed))
        return results
