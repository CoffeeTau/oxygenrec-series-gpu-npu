#!/usr/bin/env bash
set -euo pipefail

# NPU迁移Stage-0：只检查环境与单卡基础计算，不读取训练数据或模型checkpoint。
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python_bin="${PYTHON_BIN:-python}"
output_dir="${NPU_STAGE0_OUTPUT:-checkpoints/npu_stage0}"
npu_device="${NPU_DEVICE:-npu:0}"

"$python_bin" scripts/collect_npu_env.py \
    --output "$output_dir/environment.json"

"$python_bin" scripts/validate_npu_stage0.py \
    --device "$npu_device" \
    --output "$output_dir/single_card.json"
