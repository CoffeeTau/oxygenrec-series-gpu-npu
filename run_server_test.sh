#!/usr/bin/env bash
set -euo pipefail

# OxygenREC-v1最终统一闭环：从真实Qwen论文主线checkpoint继续执行SA-GCPO，
# 严格复用Instruction缓存与paper IGR，并导出匿名候选/reward/advantage/ratio轨迹。
# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

events_path="${EVENTS_PATH:-data/raw/retailrocket/events.csv}"
checkpoint="${MAINLINE_CHECKPOINT:-checkpoints/retailrocket_qwen_instruction_mainline_smoke/epoch-3.pt}"
sid_registry="${SID_REGISTRY:-data/processed/rq_comparison/w256_kmeanspp/sid_registry.json}"
feature_cache="${INSTRUCTION_FEATURE_CACHE:-data/processed/qwen_instruction_features_train32_val32.pt}"
sa_checkpoint="${SA_GCPO_OUTPUT:-checkpoints/retailrocket_qwen_instruction_mainline_smoke/sa_gcpo-qwen-mainline-none.pt}"
trajectory_output="${SA_GCPO_TRAJECTORY_OUTPUT:-outputs/review/qwen_mainline_sa_gcpo_trajectories.jsonl}"

for required_path in "$events_path" "$checkpoint" "$sid_registry" "$feature_cache"; do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR missing required path: $required_path" >&2
        exit 1
    fi
done

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/train_sa_gcpo_retailrocket.py \
    --events "$events_path" \
    --checkpoint "$checkpoint" \
    --sid-registry "$sid_registry" \
    --instruction-feature-cache "$feature_cache" \
    --alignment-samples 32 \
    --validation-samples 32 \
    --batch-size 8 \
    --beam-width 5 \
    --updates 10 \
    --learning-rate 1e-5 \
    --target-injection none \
    --output "$sa_checkpoint" \
    --trajectory-output "$trajectory_output" \
    --device cuda
