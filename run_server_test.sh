#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python scripts/audit_reasoning_sft_candidates.py \
    --input data/sft/retailrocket_reasoning_review.jsonl \
    --output data/sft/retailrocket_reasoning_audit.json \
    --review-samples 6
