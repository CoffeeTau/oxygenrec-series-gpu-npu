#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

experiment_root="checkpoints/retailrocket_ablation_20k_3seed"
for seed in 17 23 41; do
  for variant in base instruction igr igr_q2i; do
    python scripts/train_retailrocket.py \
      --events data/raw/retailrocket/events.csv \
      --sid-registry data/processed/rq_comparison/w256_kmeanspp/sid_registry.json \
      --device cuda \
      --seed "$seed" \
      --variant "$variant" \
      --matched-igr-cohort \
      --max-history 20 \
      --long-history 50 \
      --igr-top-k 5 \
      --max-train-samples 20000 \
      --max-validation-samples 200 \
      --batch-size 128 \
      --epochs 2 \
      --beam-width 5 \
      --output-dir "$experiment_root/seed-$seed/$variant"
  done
done

python scripts/summarize_ablations.py --root "$experiment_root"
