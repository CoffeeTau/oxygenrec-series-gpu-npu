#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python scripts/train_retailrocket.py \
  --events data/raw/retailrocket/events.csv \
  --device cuda \
  --max-items 5000 \
  --max-train-samples 2000 \
  --max-validation-samples 20 \
  --batch-size 128 \
  --epochs 1 \
  --beam-width 5
