#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python scripts/export_reasoning_review_subset.py \
    --input data/sft/retailrocket_reasoning_review_stratified.jsonl \
    --audit data/sft/retailrocket_reasoning_audit_stratified.json \
    --output outputs/review/qwen_sft_stratified_subset.md
