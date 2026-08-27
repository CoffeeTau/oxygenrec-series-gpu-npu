#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python scripts/train_retailrocket.py \
    --events data/raw/retailrocket/events.csv \
    --sid-registry checkpoints/retailrocket_attention_context_instruction/igr_q2i/sid_registry.json \
    --output-dir checkpoints/retailrocket_text_instruction/igr_text_q2i \
    --device cuda \
    --variant igr_text_q2i \
    --history-context-instruction \
    --history-context-pooling attention \
    --max-history 20 \
    --long-history 100 \
    --igr-top-k 10 \
    --max-train-samples 20000 \
    --max-validation-samples 500 \
    --batch-size 128 \
    --epochs 2 \
    --beam-width 10
