#!/usr/bin/env bash
set -euo pipefail

# OxygenREC-v2 第四步：在真实RetailRocket代理数据上验证daily列表式行为训练。
# 同商品保留最强行为，再按用户、UTC日、行为组成同质列表；输出匿名代表案例。
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

events_path="${EVENTS_PATH:-data/raw/retailrocket/events.csv}"
sid_registry="${SID_REGISTRY:-data/processed/rq_comparison/w256_kmeanspp/sid_registry.json}"
output_dir="${V2_LISTWISE_OUTPUT:-checkpoints/retailrocket_v2_listwise_smoke}"

for required_path in "$events_path" "$sid_registry"; do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR missing required path: $required_path" >&2
        exit 1
    fi
done

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python \
    scripts/train_v2_listwise_retailrocket.py \
    --events "$events_path" \
    --sid-registry "$sid_registry" \
    --output-dir "$output_dir" \
    --list-size "${V2_LIST_SIZE:-2}" \
    --max-history 20 \
    --max-train-samples 5000 \
    --max-validation-samples 32 \
    --batch-size 64 \
    --epochs 1 \
    --beam-width 5 \
    --device cuda \
    --seed 17
