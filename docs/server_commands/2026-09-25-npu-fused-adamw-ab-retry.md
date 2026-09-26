# 2026-09-25 NPU融合AdamW匹配zero-grad重试

状态：**待执行**

目的：修复首轮融合优化器不支持`set_to_none=True`的问题。control和treatment均使用
`zero_grad(set_to_none=False)`，确保唯一实验变量仍是优化器实现。

本轮使用新目录，不覆盖2026-09-24的失败实验。118与119机器按用户确认视为几乎等价，可进行
历史横向参考；本轮优化收益仍以同一台118机器内的成对结果为准。

## NPU服务器：可整段复制

```bash
set -euo pipefail
cd /home/h50061831/oxygenrec-series-gpu-npu

git pull origin main
test -z "$(git status --porcelain)" || { git status --short; exit 1; }

export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/cann/set_env.sh

AB_ROOT=checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925
mkdir -p "$AB_ROOT"

git rev-parse HEAD | tee "$AB_ROOT/git_commit.txt"
git status --short | tee "$AB_ROOT/git_status_short.txt"
npu-smi info | tee "$AB_ROOT/npu_smi_before.txt"

NPU_DEVICE=npu:0 \
V2_PERF_OUTPUT_ROOT="$AB_ROOT/control_adamw" \
V2_PERF_MAX_SAMPLES=130000 \
V2_PERF_BATCH_SIZES="4096" \
V2_PERF_WARMUP_STEPS=100 \
V2_PERF_MEASURED_STEPS=100 \
V2_PERF_REPEATS=3 \
V2_PERF_CYCLE_SAMPLES=1 \
V2_PERF_OPTIMIZER=adamw \
V2_PERF_ZERO_GRAD_MODE=zero \
bash run_v2_performance_baseline.sh npu bf16 \
2>&1 | tee "$AB_ROOT/control_adamw_terminal.log"
test -f "$AB_ROOT/control_adamw/npu_bf16_performance.json" || {
  echo "control result missing; stop before treatment" >&2
  exit 1
}

NPU_DEVICE=npu:0 \
V2_PERF_OUTPUT_ROOT="$AB_ROOT/treatment_npu_fused_adamw" \
V2_PERF_MAX_SAMPLES=130000 \
V2_PERF_BATCH_SIZES="4096" \
V2_PERF_WARMUP_STEPS=100 \
V2_PERF_MEASURED_STEPS=100 \
V2_PERF_REPEATS=3 \
V2_PERF_CYCLE_SAMPLES=1 \
V2_PERF_OPTIMIZER=npu_fused_adamw \
V2_PERF_ZERO_GRAD_MODE=zero \
bash run_v2_performance_baseline.sh npu bf16 \
2>&1 | tee "$AB_ROOT/treatment_npu_fused_adamw_terminal.log"
test -f "$AB_ROOT/treatment_npu_fused_adamw/npu_bf16_performance.json" || {
  echo "treatment result missing; stop before comparison" >&2
  exit 1
}

python scripts/compare_v2_performance_ab.py \
  "$AB_ROOT/control_adamw/npu_bf16_performance.json" \
  "$AB_ROOT/treatment_npu_fused_adamw/npu_bf16_performance.json" \
  --output "$AB_ROOT/comparison.json" \
  2>&1 | tee "$AB_ROOT/comparison.txt"

npu-smi info | tee "$AB_ROOT/npu_smi_after.txt"
find "$AB_ROOT" -type f | sort | tee "$AB_ROOT/file_manifest.txt"
```

## 成功判据

- 两个JSON中的`workload.zero_grad_mode`均为`zero`；
- control optimizer为`AdamW`，treatment为`NpuFusedAdamW`；
- 两组loss全部有限，首步loss基本一致；
- 两组CV优先低于5%；
- `comparison.json`成功生成，只允许optimizer字段不同。

融合版本吞吐至少提高5%且无稳定性、loss或显存异常时，再进入一次独立确认；否则保留负结果并
转向layout/copy优化。

## 回传结果

成功时返回：

1. `$AB_ROOT/comparison.json`；
2. `$AB_ROOT/control_adamw/npu_bf16_performance.json`；
3. `$AB_ROOT/treatment_npu_fused_adamw/npu_bf16_performance.json`。

如果treatment失败，只返回完整的
`$AB_ROOT/treatment_npu_fused_adamw_terminal.log`。比较脚本现在会对缺失JSON给出简短提示，
不会再产生第二段长堆栈。
