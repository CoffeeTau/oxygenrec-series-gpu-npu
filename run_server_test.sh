#!/usr/bin/env bash
set -euo pipefail

# OxygenREC-v2 第三步：验证一个I_b条件下的列表式3N SID训练与约束解码。
# 当前先用行为同质的二商品列表验证模型主干；真实数据分组规则在下一步接入。
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python \
    scripts/validate_v2_listwise_generation.py \
    --device cuda \
    --steps "${V2_LISTWISE_STEPS:-400}"
