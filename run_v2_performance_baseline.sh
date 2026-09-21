#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

platform="${1:-}"
precision="${2:-bf16}"
if [[ "$platform" != "gpu" && "$platform" != "npu" ]]; then
    echo "usage: bash run_v2_performance_baseline.sh <gpu|npu> [fp32|bf16]" >&2
    exit 2
fi
if [[ "$precision" != "fp32" && "$precision" != "bf16" ]]; then
    echo "usage: bash run_v2_performance_baseline.sh <gpu|npu> [fp32|bf16]" >&2
    exit 2
fi

python_bin="${PYTHON_BIN:-python}"
events_path="${EVENTS_PATH:-data/raw/retailrocket/events.csv}"
registry_path="${SID_REGISTRY_PATH:-data/processed/rq_comparison/w256_kmeanspp/sid_registry.json}"
checkpoint_path="${V2_FULL_TRAIN_CHECKPOINT:-checkpoints/retailrocket_v2_pretraining_ablation_smoke/full-epoch-1.pt}"
output_root="${V2_PERF_OUTPUT_ROOT:-checkpoints/performance_baseline/v2_full}"
max_samples="${V2_PERF_MAX_SAMPLES:-50000}"
warmup_steps="${V2_PERF_WARMUP_STEPS:-20}"
measured_steps="${V2_PERF_MEASURED_STEPS:-100}"
repeats="${V2_PERF_REPEATS:-3}"
cycle_samples="${V2_PERF_CYCLE_SAMPLES:-0}"
read -r -a batch_sizes <<< "${V2_PERF_BATCH_SIZES:-64 128 256}"

if [[ "$cycle_samples" != "0" && "$cycle_samples" != "1" ]]; then
    echo "V2_PERF_CYCLE_SAMPLES must be 0 or 1" >&2
    exit 2
fi

if [[ "$platform" == "gpu" ]]; then
    device="${CUDA_DEVICE:-cuda:0}"
else
    device="${NPU_DEVICE:-npu:0}"
fi

output="$output_root/${platform}_${precision}_performance.json"
command=("$python_bin" scripts/benchmark_v2_training.py \
    --platform "$platform" \
    --device "$device" \
    --precision "$precision" \
    --events "$events_path" \
    --checkpoint "$checkpoint_path" \
    --sid-registry "$registry_path" \
    --output "$output" \
    --max-train-samples "$max_samples" \
    --batch-sizes "${batch_sizes[@]}" \
    --warmup-steps "$warmup_steps" \
    --measured-steps "$measured_steps" \
    --repeats "$repeats")
if [[ "$cycle_samples" == "1" ]]; then
    command+=(--cycle-samples)
fi
"${command[@]}"
