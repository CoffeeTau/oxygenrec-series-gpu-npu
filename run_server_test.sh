#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere after the complete project has been uploaded to the server.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/validate_qwen_dual_retrieval_modes.py \
    --model-path models/Qwen3-4B-Instruct-2507 \
    --device cuda
