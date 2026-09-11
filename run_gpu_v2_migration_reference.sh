#!/usr/bin/env bash
set -euo pipefail

# 冻结v2 Full的GPU FP32参考：静态迁移清单 + 固定batch前向/生成/反向/单步更新。
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python_bin="${PYTHON_BIN:-python}"
events_path="${EVENTS_PATH:-data/raw/retailrocket/events.csv}"
registry_path="${SID_REGISTRY_PATH:-data/processed/rq_comparison/w256_kmeanspp/sid_registry.json}"
checkpoint_path="${V2_FULL_CHECKPOINT:-checkpoints/retailrocket_v2_pretraining_ablation_smoke/full-epoch-1.pt}"
output_root="${V2_ALIGNMENT_ROOT:-checkpoints/device_alignment/v2_full}"
cuda_device="${CUDA_DEVICE:-cuda:0}"

"$python_bin" scripts/collect_v2_migration_inventory.py \
    --output-dir "$output_root/support_inventory/gpu"

"$python_bin" scripts/export_v2_gpu_reference.py \
    --events "$events_path" \
    --checkpoint "$checkpoint_path" \
    --sid-registry "$registry_path" \
    --output-dir "$output_root/gpu" \
    --device "$cuda_device"
