#!/usr/bin/env bash
set -euo pipefail

# OxygenREC-v2 第五步：验证EA-TOSD公式、Teacher未来前缀和共享backbone更新。
# 该fixture不加载或调用任何外部Reward Model；真实daily checkpoint在本关通过后接入。
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python \
    scripts/validate_ea_tosd.py \
    --device cuda \
    --pretrain-steps "${EA_TOSD_PRETRAIN_STEPS:-300}" \
    --group-size "${EA_TOSD_GROUP_SIZE:-8}"
