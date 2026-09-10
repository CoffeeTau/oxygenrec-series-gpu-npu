#!/usr/bin/env bash
set -euo pipefail

# OxygenREC-v2 第九步：同cohort、同预算复现预训练三组消融。
# Base(w/o I_b) -> +I_b -> +行为加权NTP，同时报告生成列表SID唯一率。
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

events_path="${EVENTS_PATH:-data/raw/retailrocket/events.csv}"
sid_registry="${SID_REGISTRY:-data/processed/rq_comparison/w256_kmeanspp/sid_registry.json}"
output_dir="${V2_PRETRAIN_ABLATION_OUTPUT:-checkpoints/retailrocket_v2_pretraining_ablation_smoke}"

for required_path in "$events_path" "$sid_registry"; do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR missing required path: $required_path" >&2
        exit 1
    fi
done

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python \
    scripts/train_v2_pretraining_ablation_retailrocket.py \
    --events "$events_path" \
    --sid-registry "$sid_registry" \
    --output-dir "$output_dir" \
    --device cuda \
    --list-size 2 \
    --max-history 20 \
    --max-train-samples 5000 \
    --max-validation-samples 32 \
    --batch-size 64 \
    --epochs 1 \
    --learning-rate 3e-4 \
    --hidden-size 128 \
    --attention-heads 4 \
    --encoder-layers 2 \
    --decoder-layers 2
