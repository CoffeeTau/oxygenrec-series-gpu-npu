#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python scripts/prepare_qwen_sft_data.py \
    --input outputs/review/qwen_reasoning_cases.jsonl \
    --output data/sft/qwen_reasoning_candidates.jsonl \
    --allow-pending
