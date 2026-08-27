#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

qwen_model_path="${QWEN_MODEL_PATH:-models/Qwen3-4B-Instruct-2507}"

python scripts/validate_qwen_instruction_features.py \
    --model-path "$qwen_model_path" \
    --device cuda \
    --dtype bfloat16 \
    --max-length 256
