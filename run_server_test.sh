#!/usr/bin/env bash
set -euo pipefail

# OxygenREC-v2 第六步：把真实daily checkpoint接入future prefix与EA-TOSD后训练。
# 未来事件严格晚于gold列表且不跨split；仍不加载或调用任何外部Reward Model。
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

events_path="${EVENTS_PATH:-data/raw/retailrocket/events.csv}"
sid_registry="${SID_REGISTRY:-data/processed/rq_comparison/w256_kmeanspp/sid_registry.json}"
checkpoint="${V2_LISTWISE_CHECKPOINT:-checkpoints/retailrocket_v2_listwise_smoke/epoch-1.pt}"
output_dir="${V2_EA_TOSD_OUTPUT:-checkpoints/retailrocket_v2_ea_tosd_smoke}"

for required_path in "$events_path" "$sid_registry" "$checkpoint"; do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR missing required path: $required_path" >&2
        exit 1
    fi
done

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python \
    scripts/train_v2_ea_tosd_retailrocket.py \
    --events "$events_path" \
    --sid-registry "$sid_registry" \
    --checkpoint "$checkpoint" \
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
    --group-size "${EA_TOSD_GROUP_SIZE:-4}" \
    --beam-width 5
