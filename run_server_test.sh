#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python scripts/summarize_qwen_plan_retailrocket.py \
    --input outputs/review/qwen_plan_retailrocket.jsonl
