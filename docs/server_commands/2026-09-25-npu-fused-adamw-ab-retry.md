# 2026-09-25 NPU融合AdamW匹配zero-grad重试

状态：**已执行并完成分析；运行通过，收益不足未采纳**

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
- 两组loss全部有限；`first_loss`是100步外部warmup后的首个测量loss，不要求两组一致；
- 两组CV优先低于5%；
- `comparison.json`成功生成，只允许optimizer字段不同。

融合版本吞吐至少提高5%且无稳定性、loss或显存异常时，再进入一次独立确认；否则保留负结果并
转向layout/copy优化。

## 回传结果

成功时返回：

1. `checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925/comparison.json`；
2. `checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925/control_adamw/npu_bf16_performance.json`；
3. `checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925/treatment_npu_fused_adamw/npu_bf16_performance.json`。

如果treatment失败，只返回完整的
`checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925/treatment_npu_fused_adamw_terminal.log`。
比较脚本现在会对缺失JSON给出简短提示，
不会再产生第二段长堆栈。

## 结果路径

本轮根目录：

```text
checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925/
```

需要回传分析的文件：

```text
checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925/comparison.json
checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925/control_adamw/npu_bf16_performance.json
checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925/treatment_npu_fused_adamw/npu_bf16_performance.json
```

服务器保留、目前无需回传的文件：

```text
checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925/comparison.txt
checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925/control_adamw_terminal.log
checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925/treatment_npu_fused_adamw_terminal.log
checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925/git_commit.txt
checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925/git_status_short.txt
checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925/npu_smi_before.txt
checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925/npu_smi_after.txt
checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925/file_manifest.txt
```

## 实际结果与结论

- control中位吞吐：`27150.77338772522 samples/s`；
- treatment中位吞吐：`27879.619557167545 samples/s`；
- treatment提升：`2.6844398096292688%`；
- 中位step时延：`150.8613 ms → 146.9174 ms`，减少约`2.61%`；
- CV：control约`0.918%`，treatment约`0.959%`，两组稳定；
- treatment最小吞吐`27394.50`仍高于control最大吞吐`27181.41`，本轮小幅收益具有一致性；
- allocated memory仅减少约`0.071%`，可忽略；
- 两组loss全部有限，但100步warmup后的loss轨迹已有差异，本性能实验不能据此声明质量提升。

结论：融合AdamW运行成功并带来稳定的小幅收益，但低于预设`5%`工程门槛，不进入第二次确认，
也不替换正式训练默认优化器。后续转向现有Profile中约`15.77%`的layout/copy热点。
