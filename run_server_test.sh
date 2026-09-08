#!/usr/bin/env bash
set -euo pipefail

# OxygenREC-v2 第二步：在真实RetailRocket事件上验证目标行为标签和token级权重。
# view作为论文click的公开代理；每个目标的三个SID token共同继承1.2/1.5/2.0权重。
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

events_path="${EVENTS_PATH:-data/raw/retailrocket/events.csv}"
sid_registry="${SID_REGISTRY:-data/processed/rq_comparison/w256_kmeanspp/sid_registry.json}"
output_dir="${V2_BEHAVIOR_OUTPUT:-checkpoints/retailrocket_v2_behavior_weight_smoke}"

for required_path in "$events_path" "$sid_registry"; do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR missing required path: $required_path" >&2
        exit 1
    fi
done

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/train_retailrocket.py \
    --events "$events_path" \
    --sid-registry "$sid_registry" \
    --variant v2_behavior \
    --max-history 20 \
    --max-train-samples 5000 \
    --max-validation-samples 32 \
    --batch-size 64 \
    --epochs 1 \
    --beam-width 10 \
    --output-dir "$output_dir" \
    --device cuda \
    --seed 17
