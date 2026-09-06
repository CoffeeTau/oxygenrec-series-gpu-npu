#!/usr/bin/env bash
set -euo pipefail

# OxygenREC-v1论文主线收口：从完整validation cohort按固定正反例规则
# 导出Qwen Reasoning -> paper IGR -> Q2I -> constrained beam代表性案例。
# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

events_path="${EVENTS_PATH:-data/raw/retailrocket/events.csv}"
checkpoint="${MAINLINE_CHECKPOINT:-checkpoints/retailrocket_qwen_instruction_mainline_smoke/epoch-3.pt}"
sid_registry="${SID_REGISTRY:-data/processed/rq_comparison/w256_kmeanspp/sid_registry.json}"
feature_cache="${INSTRUCTION_FEATURE_CACHE:-data/processed/qwen_instruction_features_train32_val32.pt}"
reasoning_output="${INSTRUCTION_REASONING_OUTPUT:-outputs/review/qwen_instruction_reasoning_train32_val32.jsonl}"
review_output="${MAINLINE_REVIEW_OUTPUT:-outputs/review/qwen_mainline_representative_cases.jsonl}"

for required_path in "$events_path" "$checkpoint" "$sid_registry" "$feature_cache" "$reasoning_output"; do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR missing required path: $required_path" >&2
        exit 1
    fi
done

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/export_qwen_mainline_review_cases.py \
    --events "$events_path" \
    --checkpoint "$checkpoint" \
    --sid-registry "$sid_registry" \
    --instruction-feature-cache "$feature_cache" \
    --reasoning-input "$reasoning_output" \
    --output "$review_output" \
    --device cuda \
    --beam-width 10
