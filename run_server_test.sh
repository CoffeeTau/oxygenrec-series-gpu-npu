#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python scripts/train_sa_gcpo_retailrocket.py \
    --events data/raw/retailrocket/events.csv \
    --device cuda \
    --alignment-samples 200 \
    --validation-samples 1000 \
    --batch-size 16 \
    --beam-width 5 \
    --updates 20 \
    --target-injection none
