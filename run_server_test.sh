#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python scripts/validate_retrieval_plan_execution.py \
    --input outputs/review/qwen_reasoning_cases.jsonl \
    --device cuda
