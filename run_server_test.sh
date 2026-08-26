#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

experiment_root="checkpoints/retailrocket_attention_context_instruction"
for epoch in 1 2; do
  python scripts/train_retailrocket.py \
    --events data/raw/retailrocket/events.csv \
    --sid-registry data/processed/rq_comparison/w256_kmeanspp/sid_registry.json \
    --device cuda \
    --seed 17 \
    --variant igr_q2i \
    --matched-igr-cohort \
    --history-context-instruction \
    --history-context-pooling attention \
    --max-history 20 \
    --long-history 50 \
    --igr-top-k 5 \
    --max-train-samples 1 \
    --max-validation-samples 2000 \
    --batch-size 128 \
    --beam-width 5 \
    --output-dir "$experiment_root/igr_q2i" \
    --eval-only-checkpoint "$experiment_root/igr_q2i/epoch-$epoch.pt"
done
