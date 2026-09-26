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
optimizer="${V2_PERF_OPTIMIZER:-adamw}"
zero_grad_mode="${V2_PERF_ZERO_GRAD_MODE:-set_to_none}"
npu_internal_format="${V2_PERF_NPU_INTERNAL_FORMAT:-default}"
read -r -a batch_sizes <<< "${V2_PERF_BATCH_SIZES:-64 128 256}"

if [[ "$cycle_samples" != "0" && "$cycle_samples" != "1" ]]; then
    echo "V2_PERF_CYCLE_SAMPLES must be 0 or 1" >&2
    exit 2
fi
if [[ "$optimizer" != "adamw" && "$optimizer" != "npu_fused_adamw" ]]; then
    echo "V2_PERF_OPTIMIZER must be adamw or npu_fused_adamw" >&2
    exit 2
fi
if [[ "$zero_grad_mode" != "set_to_none" && "$zero_grad_mode" != "zero" ]]; then
    echo "V2_PERF_ZERO_GRAD_MODE must be set_to_none or zero" >&2
    exit 2
fi
if [[ "$npu_internal_format" != "default" && "$npu_internal_format" != "disable" && "$npu_internal_format" != "enable" ]]; then
    echo "V2_PERF_NPU_INTERNAL_FORMAT must be default, disable, or enable" >&2
    exit 2
fi
if [[ "$platform" != "npu" && "$npu_internal_format" != "default" ]]; then
    echo "V2_PERF_NPU_INTERNAL_FORMAT is only valid for the npu platform" >&2
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
    --repeats "$repeats" \
    --optimizer "$optimizer" \
    --zero-grad-mode "$zero_grad_mode" \
    --npu-internal-format "$npu_internal_format")
if [[ "$cycle_samples" == "1" ]]; then
    command+=(--cycle-samples)
fi
"${command[@]}"
