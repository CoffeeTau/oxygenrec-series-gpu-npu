#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python scripts/fit_retailrocket_sid.py \
  --events data/raw/retailrocket/events.csv \
  --properties \
    data/raw/retailrocket/item_properties_part1.csv \
    data/raw/retailrocket/item_properties_part2.csv \
  --device cuda \
  --max-items 5000 \
  --dimension 256 \
  --levels 3 \
  --width 64 \
  --iterations 10 \
  --chunk-size 2048

python scripts/train_retailrocket.py \
  --events data/raw/retailrocket/events.csv \
  --sid-registry data/processed/retailrocket_sid/sid_registry.json \
  --output-dir checkpoints/retailrocket_rq_smoke \
  --device cuda \
  --max-history 20 \
  --max-train-samples 2000 \
  --max-validation-samples 20 \
  --batch-size 128 \
  --epochs 1 \
  --beam-width 5
