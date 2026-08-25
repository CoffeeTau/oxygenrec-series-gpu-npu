#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

experiment_root="checkpoints/retailrocket_retrieval_diagnostic"
python scripts/train_retailrocket.py \
  --events data/raw/retailrocket/events.csv \
  --sid-registry data/processed/rq_comparison/w256_kmeanspp/sid_registry.json \
  --device cuda \
  --seed 17 \
  --variant igr_q2i \
  --matched-igr-cohort \
  --max-history 20 \
  --long-history 50 \
  --igr-top-k 5 \
  --max-train-samples 20000 \
  --max-validation-samples 500 \
  --batch-size 128 \
  --epochs 2 \
  --beam-width 5 \
  --retriever-init-checkpoint "$experiment_root/q2i/epoch-2.pt" \
  --output-dir "$experiment_root/igr_q2i_q2i_initialized"
