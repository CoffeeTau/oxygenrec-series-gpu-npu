#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python scripts/train_retailrocket.py \
  --events data/raw/retailrocket/events.csv \
  --sid-registry data/processed/rq_comparison/w256_kmeanspp/sid_registry.json \
  --device cuda \
  --variant igr_q2i \
  --matched-igr-cohort \
  --max-history 20 \
  --long-history 50 \
  --igr-top-k 5 \
  --max-train-samples 5000 \
  --max-validation-samples 20 \
  --batch-size 128 \
  --epochs 1 \
  --beam-width 5 \
  --output-dir checkpoints/retailrocket_igr_q2i_smoke
