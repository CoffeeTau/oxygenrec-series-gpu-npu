#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

platform="${1:-}"
precision="${2:-fp32}"
if [[ "$platform" != "gpu" && "$platform" != "npu" ]]; then
    echo "usage: bash run_v2_training.sh <gpu|npu> [fp32|bf16]" >&2
    exit 2
fi
if [[ "$precision" != "fp32" && "$precision" != "bf16" ]]; then
    echo "usage: bash run_v2_training.sh <gpu|npu> [fp32|bf16]" >&2
    exit 2
fi

python_bin="${PYTHON_BIN:-python}"
events_path="${EVENTS_PATH:-data/raw/retailrocket/events.csv}"
registry_path="${SID_REGISTRY_PATH:-data/processed/rq_comparison/w256_kmeanspp/sid_registry.json}"
checkpoint_path="${V2_FULL_CHECKPOINT:-checkpoints/retailrocket_v2_pretraining_ablation_smoke/full-epoch-1.pt}"
output_root="${V2_TRAINING_ROOT:-checkpoints/device_training/v2_full}"
steps="${V2_TRAINING_STEPS:-20}"
batch_size="${V2_TRAINING_BATCH_SIZE:-64}"

if [[ "$platform" == "gpu" ]]; then
    device="${CUDA_DEVICE:-cuda:0}"
else
    device="${NPU_DEVICE:-npu:0}"
fi

missing=0
for required in "$events_path" "$registry_path" "$checkpoint_path"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR missing required input: $required" >&2
        missing=1
    fi
done
if [[ "$missing" -ne 0 ]]; then
    exit 3
fi

output_dir="$output_root/$platform/$precision"
mkdir -p "$output_dir"

"$python_bin" scripts/train_v2_full_migration_smoke.py \
    --platform "$platform" \
    --device "$device" \
    --precision "$precision" \
    --events "$events_path" \
    --checkpoint "$checkpoint_path" \
    --sid-registry "$registry_path" \
    --steps "$steps" \
    --batch-size "$batch_size" \
    --output-dir "$output_dir" \
    2>&1 | tee "$output_dir/training.log"
