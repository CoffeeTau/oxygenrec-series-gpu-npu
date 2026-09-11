#!/usr/bin/env bash
set -euo pipefail

# 用GPU生成的同一参考包在NPU执行v2 Full固定batch FP32精度对齐。
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python_bin="${PYTHON_BIN:-python}"
registry_path="${SID_REGISTRY_PATH:-data/processed/rq_comparison/w256_kmeanspp/sid_registry.json}"
checkpoint_path="${V2_FULL_CHECKPOINT:-checkpoints/retailrocket_v2_pretraining_ablation_smoke/full-epoch-1.pt}"
output_root="${V2_ALIGNMENT_ROOT:-checkpoints/device_alignment/v2_full}"
reference_path="${V2_GPU_REFERENCE:-$output_root/gpu/v2_full_gpu_reference.pt}"
npu_device="${NPU_DEVICE:-npu:0}"

"$python_bin" scripts/collect_v2_migration_inventory.py \
    --output-dir "$output_root/support_inventory/npu"

"$python_bin" scripts/compare_device_reference.py \
    --reference "$reference_path" \
    --checkpoint "$checkpoint_path" \
    --sid-registry "$registry_path" \
    --output "$output_root/npu/v2_full_comparison.json" \
    --device "$npu_device"
