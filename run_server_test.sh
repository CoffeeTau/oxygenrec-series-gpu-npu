#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python scripts/export_llm_review_cases.py \
    --events data/raw/retailrocket/events.csv \
    --checkpoint checkpoints/retailrocket_attention_context_instruction/igr_q2i/epoch-2.pt \
    --sid-registry checkpoints/retailrocket_attention_context_instruction/igr_q2i/sid_registry.json \
    --output outputs/review/llm_cross_cases.jsonl \
    --device cuda \
    --cases 5 \
    --beam-width 5
