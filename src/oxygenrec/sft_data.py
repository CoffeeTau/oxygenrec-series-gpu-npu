"""把已生成并人工审核的 Retrieval Reasoning 转成 Qwen SFT 对话样本。"""

from __future__ import annotations

import json
from typing import Mapping

from .llm_reasoning import parse_reasoning_json, reasoning_system_prompt
from .retrieval_planning import compile_retrieval_plan


def build_reasoning_sft_example(
    record: Mapping[str, object], *, require_approved: bool = True,
) -> dict[str, object]:
    """校验一条 review 记录并生成标准 messages；默认拒绝未审核标签。"""

    case_id = record.get("case_id")
    prompt = record.get("prompt")
    evidence = record.get("input_evidence")
    reasoning = record.get("reasoning")
    review_status = record.get("review_status", "pending")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id must be a non-empty string")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"case {case_id}: prompt must be a non-empty string")
    if not isinstance(evidence, Mapping):
        raise ValueError(f"case {case_id}: input_evidence must be an object")
    if not isinstance(reasoning, Mapping):
        raise ValueError(f"case {case_id}: reasoning must be an object")
    if review_status not in {"pending", "approved", "rejected"}:
        raise ValueError(f"case {case_id}: unsupported review_status={review_status!r}")
    if require_approved and review_status != "approved":
        raise ValueError(f"case {case_id}: SFT label is not approved")

    # 复用线上生成时的严格 schema，保证训练标签和推理输出协议完全一致。
    assistant_text = json.dumps(reasoning, ensure_ascii=False, separators=(",", ":"))
    parsed = parse_reasoning_json(assistant_text)
    behavior_counts = evidence.get("behavior_counts")
    if not isinstance(behavior_counts, Mapping):
        raise ValueError(f"case {case_id}: behavior_counts must be an object")
    # Plan 还必须能被确定性 IGR 执行器编译，不能只满足 JSON 外形。
    compile_retrieval_plan(parsed["retrieval_plan"], behavior_counts)

    return {
        "messages": [
            {"role": "system", "content": reasoning_system_prompt()},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_text},
        ],
        "metadata": {
            "case_id": case_id,
            "review_status": review_status,
            "source": record.get("source", "oxygenrec_reasoning_review"),
        },
    }
