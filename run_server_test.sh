#!/usr/bin/env bash
set -euo pipefail

# OxygenREC-v2 第七步：在EA-TOSD完全相同的cohort上跑等步数SFT-only对照。
# 脚本会校验训练前指标与已完成EA run一致，再输出三点配对比较。
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

events_path="${EVENTS_PATH:-data/raw/retailrocket/events.csv}"
sid_registry="${SID_REGISTRY:-data/processed/rq_comparison/w256_kmeanspp/sid_registry.json}"
checkpoint="${V2_LISTWISE_CHECKPOINT:-checkpoints/retailrocket_v2_listwise_smoke/epoch-1.pt}"
ea_summary="${V2_EA_TOSD_SUMMARY:-checkpoints/retailrocket_v2_ea_tosd_smoke/ea_tosd_summary.json}"
output_dir="${V2_SFT_CONTROL_OUTPUT:-checkpoints/retailrocket_v2_sft_control_smoke}"

for required_path in "$events_path" "$sid_registry" "$checkpoint" "$ea_summary"; do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR missing required path: $required_path" >&2
        exit 1
    fi
done

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python \
    scripts/train_v2_sft_control_retailrocket.py \
    --events "$events_path" \
    --sid-registry "$sid_registry" \
    --checkpoint "$checkpoint" \
    --ea-summary "$ea_summary" \
    --output-dir "$output_dir" \
    --device cuda \
    --list-size 2 \
    --max-history 20 \
    --max-future-items 2 \
    --minimum-future-behavior "${EA_TOSD_FUTURE_BEHAVIOR:-view}" \
    --max-train-samples "${EA_TOSD_TRAIN_SAMPLES:-32}" \
    --max-validation-samples 32 \
    --batch-size 4 \
    --epochs 1 \
    --beam-width 5
