#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/generate_retailrocket_sft_candidates.py \
    --events data/raw/retailrocket/events.csv \
    --checkpoint checkpoints/retailrocket_attention_context_instruction/igr_q2i/epoch-2.pt \
    --model-path models/Qwen3-4B-Instruct-2507 \
    --output data/sft/retailrocket_reasoning_review_stratified.jsonl \
    --cases 32 \
    --sampling-pool 4096 \
    --batch-size 4 \
    --device cuda

python scripts/audit_reasoning_sft_candidates.py \
    --input data/sft/retailrocket_reasoning_review_stratified.jsonl \
    --output data/sft/retailrocket_reasoning_audit_stratified.json \
    --review-samples 6
