#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

platform="${1:-}"
if [[ "$platform" != "gpu" && "$platform" != "npu" ]]; then
    echo "usage: bash run_v2_device_alignment.sh <gpu|npu>" >&2
    exit 2
fi

python_bin="${PYTHON_BIN:-python}"
events_path="${EVENTS_PATH:-data/raw/retailrocket/events.csv}"
registry_path="${SID_REGISTRY_PATH:-data/processed/rq_comparison/w256_kmeanspp/sid_registry.json}"
checkpoint_path="${V2_FULL_CHECKPOINT:-checkpoints/retailrocket_v2_pretraining_ablation_smoke/full-epoch-1.pt}"
output_root="${V2_ALIGNMENT_ROOT:-checkpoints/device_alignment/v2_full}"

if [[ "$platform" == "gpu" ]]; then
    device="${CUDA_DEVICE:-cuda:0}"
else
    device="${NPU_DEVICE:-npu:0}"
fi

"$python_bin" scripts/collect_v2_migration_inventory.py \
    --output-dir "$output_root/support_inventory/$platform"

"$python_bin" scripts/run_v2_device_probe.py \
    --platform "$platform" \
    --device "$device" \
    --events "$events_path" \
    --checkpoint "$checkpoint_path" \
    --sid-registry "$registry_path" \
    --output "$output_root/$platform/v2_full_${platform}_probe.json"
