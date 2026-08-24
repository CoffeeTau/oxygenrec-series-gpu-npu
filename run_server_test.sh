#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python scripts/compare_rq_configs.py \
  --input-dir data/processed/retailrocket_sid \
  --output-dir data/processed/rq_comparison \
  --device cuda \
  --iterations 15 \
  --chunk-size 2048 \
  --seed 17
