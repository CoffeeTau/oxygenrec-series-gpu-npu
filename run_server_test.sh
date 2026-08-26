#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python scripts/evaluate_property_retrieval.py \
  --events data/raw/retailrocket/events.csv \
  --sid-registry data/processed/rq_comparison/w256_kmeanspp/sid_registry.json \
  --embedding-dir data/processed/retailrocket_sid \
  --seed 17 \
  --short-history 20 \
  --long-history 50 \
  --top-k 5 \
  --validation-samples 2000
