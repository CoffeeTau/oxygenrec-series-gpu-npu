#!/usr/bin/env bash
set -euo pipefail

# OxygenREC-v1论文主线：真实历史 -> Qwen Reasoning/hidden state缓存
# -> paper IGR -> Encoder-Decoder GR -> NTP+Q2I。此脚本不执行Agentic Plan。
# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

events_path="${EVENTS_PATH:-data/raw/retailrocket/events.csv}"
source_checkpoint="${SOURCE_CHECKPOINT:-checkpoints/retailrocket_attention_context_instruction/igr_q2i/epoch-2.pt}"
sid_registry="${SID_REGISTRY:-data/processed/rq_comparison/w256_kmeanspp/sid_registry.json}"
qwen_model_path="${QWEN_MODEL_PATH:-models/Qwen3-4B-Instruct-2507}"
feature_cache="${INSTRUCTION_FEATURE_CACHE:-data/processed/qwen_instruction_features_train32_val32.pt}"
reasoning_output="${INSTRUCTION_REASONING_OUTPUT:-outputs/review/qwen_instruction_reasoning_train32_val32.jsonl}"
training_output="${TRAINING_OUTPUT_DIR:-checkpoints/retailrocket_qwen_instruction_mainline_smoke}"

for required_path in "$events_path" "$source_checkpoint" "$sid_registry" "$qwen_model_path"; do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR missing required path: $required_path" >&2
        exit 1
    fi
done

if [[ ! -f "$feature_cache" && ! -f "$reasoning_output" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/cache_qwen_instructions_retailrocket.py \
        --events "$events_path" \
        --checkpoint "$source_checkpoint" \
        --sid-registry "$sid_registry" \
        --model-path "$qwen_model_path" \
        --output "$feature_cache" \
        --reasoning-output "$reasoning_output" \
        --max-train-samples 32 \
        --max-validation-samples 32 \
        --short-history 20 \
        --long-history 100 \
        --igr-top-k 10 \
        --sample-seed 17 \
        --batch-size 4 \
        --max-new-tokens 384 \
        --device cuda \
        --dtype bfloat16
elif [[ -f "$feature_cache" && -f "$reasoning_output" ]]; then
    echo "stage=reuse_instruction_cache cache=$feature_cache reasoning=$reasoning_output"
else
    echo "ERROR cache outputs are incomplete; keep or remove both files together" >&2
    exit 1
fi

# 缓存进程结束后Qwen显存已释放；下面只训练小型Fast模型及Qwen特征adapter。
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/train_retailrocket.py \
    --events "$events_path" \
    --sid-registry "$sid_registry" \
    --variant igr_qwen_q2i \
    --instruction-feature-cache "$feature_cache" \
    --output-dir "$training_output" \
    --device cuda \
    --seed 17 \
    --max-history 20 \
    --long-history 100 \
    --igr-top-k 10 \
    --max-train-samples 32 \
    --max-validation-samples 32 \
    --batch-size 16 \
    --epochs 3 \
    --beam-width 10
