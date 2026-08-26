#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

for variant in base behavior_strength_decay; do
    python scripts/train_retailrocket.py \
        --events data/raw/retailrocket/events.csv \
        --sid-registry checkpoints/retailrocket_attention_context_instruction/igr_q2i/sid_registry.json \
        --output-dir "checkpoints/retailrocket_behavior/${variant}_eval5000" \
        --device cuda \
        --variant "$variant" \
        --seed 17 \
        --max-history 20 \
        --max-train-samples 1 \
        --max-validation-samples 5000 \
        --batch-size 128 \
        --beam-width 5 \
        --eval-only-checkpoint "checkpoints/retailrocket_behavior/${variant}/epoch-2.pt"
done
