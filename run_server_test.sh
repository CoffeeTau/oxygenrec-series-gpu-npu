#!/usr/bin/env bash
set -euo pipefail

# OxygenREC-v2 第八步：只读比较EA-TOSD与SFT-only checkpoint的连续策略差异。
# 不执行新的训练更新；同validation上检查参数、logits、argmax与greedy变化。
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

events_path="${EVENTS_PATH:-data/raw/retailrocket/events.csv}"
sid_registry="${SID_REGISTRY:-data/processed/rq_comparison/w256_kmeanspp/sid_registry.json}"
ea_checkpoint="${V2_EA_TOSD_CHECKPOINT:-checkpoints/retailrocket_v2_ea_tosd_smoke/epoch-1.pt}"
sft_checkpoint="${V2_SFT_CONTROL_CHECKPOINT:-checkpoints/retailrocket_v2_sft_control_smoke/epoch-1.pt}"
output_dir="${V2_CHECKPOINT_COMPARISON_OUTPUT:-outputs/review/v2_ea_tosd_checkpoint_comparison}"

for required_path in "$events_path" "$sid_registry" "$ea_checkpoint" "$sft_checkpoint"; do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR missing required path: $required_path" >&2
        exit 1
    fi
done

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python \
    scripts/compare_v2_posttraining_checkpoints.py \
    --events "$events_path" \
    --sid-registry "$sid_registry" \
    --ea-checkpoint "$ea_checkpoint" \
    --sft-checkpoint "$sft_checkpoint" \
    --output-dir "$output_dir" \
    --device cuda \
    --list-size 2 \
    --max-history 20 \
    --max-future-items 2 \
    --minimum-future-behavior "${EA_TOSD_FUTURE_BEHAVIOR:-view}" \
    --max-train-samples 32 \
    --max-validation-samples 32 \
    --batch-size 4
