#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python scripts/evaluate_qwen_plan_retailrocket.py \
    --events data/raw/retailrocket/events.csv \
    --checkpoint checkpoints/retailrocket_attention_context_instruction/igr_q2i/epoch-2.pt \
    --sid-registry checkpoints/retailrocket_attention_context_instruction/igr_q2i/sid_registry.json \
    --model-path "${QWEN_MODEL_PATH:-models/Qwen3-4B-Instruct-2507}" \
    --cases 8 \
    --device cuda \
    --output outputs/review/qwen_plan_retailrocket.jsonl
