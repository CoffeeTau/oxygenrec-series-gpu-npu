#!/usr/bin/env bash
set -euo pipefail

# NPU迁移Stage-0：只检查环境与单卡基础计算，不读取训练数据或模型checkpoint。
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python_bin="${PYTHON_BIN:-python}"
output_dir="${NPU_STAGE0_OUTPUT:-checkpoints/npu_stage0}"
npu_device="${NPU_DEVICE:-npu:0}"
snapshot_date="${NPU_SNAPSHOT_DATE:-$(date -u +%F)}"

"$python_bin" scripts/collect_npu_env.py \
    --output "$output_dir/npu_server_environment_snapshot_${snapshot_date}.json"

"$python_bin" scripts/validate_npu_stage0.py \
    --device "$npu_device" \
    --output "$output_dir/npu_single_card_validation_${snapshot_date}.json"
