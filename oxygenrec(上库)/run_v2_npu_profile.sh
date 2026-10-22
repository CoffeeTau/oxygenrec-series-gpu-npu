#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python_bin="${PYTHON_BIN:-python}"
events_path="${EVENTS_PATH:-data/raw/retailrocket/events.csv}"
registry_path="${SID_REGISTRY_PATH:-data/processed/rq_comparison/w256_kmeanspp/sid_registry.json}"
checkpoint_path="${V2_FULL_TRAIN_CHECKPOINT:-checkpoints/retailrocket_v2_pretraining_ablation_smoke/full-epoch-1.pt}"
output_root="${V2_PROFILE_OUTPUT_ROOT:-checkpoints/performance_profiling/v2_full_npu_bs4096}"
max_samples="${V2_PROFILE_MAX_SAMPLES:-130000}"
batch_size="${V2_PROFILE_BATCH_SIZE:-4096}"
outer_warmup_steps="${V2_PROFILE_OUTER_WARMUP_STEPS:-100}"
profile_warmup_steps="${V2_PROFILE_WARMUP_STEPS:-1}"
profile_active_steps="${V2_PROFILE_ACTIVE_STEPS:-3}"
measured_steps=$((profile_warmup_steps + profile_active_steps))

"$python_bin" scripts/benchmark_v2_training.py \
    --platform npu \
    --device "${NPU_DEVICE:-npu:0}" \
    --precision bf16 \
    --events "$events_path" \
    --checkpoint "$checkpoint_path" \
    --sid-registry "$registry_path" \
    --output "$output_root/npu_bf16_profile_summary.json" \
    --max-train-samples "$max_samples" \
    --batch-sizes "$batch_size" \
    --warmup-steps "$outer_warmup_steps" \
    --measured-steps "$measured_steps" \
    --repeats 1 \
    --cycle-samples \
    --npu-profile-dir "$output_root/trace" \
    --profile-warmup-steps "$profile_warmup_steps" \
    --profile-active-steps "$profile_active_steps"
