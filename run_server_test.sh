#!/usr/bin/env bash
set -euo pipefail

# OxygenREC-v2 第一步：验证真正位于 Decoder prefix 的目标行为指令 I_b。
# fixture 固定用户历史，只切换 view/click、cart、order，检查前向、反向、
# 受控过拟合、greedy 和 constrained beam；不再继续 v1 SA-GCPO reward 目标。
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python \
    scripts/validate_v2_behavior_instruction.py \
    --device cuda \
    --steps "${V2_IB_STEPS:-240}"
