# 2026-09-24 NPU原生AdamW与融合AdamW受控A/B

状态：**待执行**

目的：在相同代码、输入、batch、预热、测量窗口和随机种子下，只替换优化器实现，验证
`torch_npu.optim.NpuFusedAdamW`能否减少当前Profile暴露的优化器标量读取、原地更新和拷贝开销。

本轮不重新采集Profiler，不修改dropout或模型结构。两组都从同一个冻结checkpoint重新开始，
因此不是连续续训关系。

## NPU服务器：可整段复制

```bash
set -euo pipefail
cd /home/h50061831/oxygenrec-series-gpu-npu

git pull origin main
test -z "$(git status --porcelain)" || { git status --short; exit 1; }

export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/cann/set_env.sh

AB_ROOT=checkpoints/performance_optimization/npu_fused_adamw_20260924
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
bash run_v2_performance_baseline.sh npu bf16 \
2>&1 | tee "$AB_ROOT/control_adamw_terminal.log"

NPU_DEVICE=npu:0 \
V2_PERF_OUTPUT_ROOT="$AB_ROOT/treatment_npu_fused_adamw" \
V2_PERF_MAX_SAMPLES=130000 \
V2_PERF_BATCH_SIZES="4096" \
V2_PERF_WARMUP_STEPS=100 \
V2_PERF_MEASURED_STEPS=100 \
V2_PERF_REPEATS=3 \
V2_PERF_CYCLE_SAMPLES=1 \
V2_PERF_OPTIMIZER=npu_fused_adamw \
bash run_v2_performance_baseline.sh npu bf16 \
2>&1 | tee "$AB_ROOT/treatment_npu_fused_adamw_terminal.log"

python scripts/compare_v2_performance_ab.py \
  "$AB_ROOT/control_adamw/npu_bf16_performance.json" \
  "$AB_ROOT/treatment_npu_fused_adamw/npu_bf16_performance.json" \
  --output "$AB_ROOT/comparison.json" \
  2>&1 | tee "$AB_ROOT/comparison.txt"

npu-smi info | tee "$AB_ROOT/npu_smi_after.txt"
find "$AB_ROOT" -type f | sort | tee "$AB_ROOT/file_manifest.txt"
```

## 结果判读

先满足硬条件：

- 两组`status=passed`且loss全部有限；
- `comparison.json`成功生成，说明commit、输入哈希、源码哈希和除优化器外的workload一致；
- treatment首步loss与control首步loss应基本一致，因为首个`optimizer.step()`发生在首步loss之后；
- 两组CV均应优先低于5%，否则暂不依据单点比值合入主训练。

性能决策：

- treatment吞吐提高至少5%，且没有loss、稳定性或显存异常：进入第二次独立确认；
- 变化在±5%内：视为当前模型收益不足，保留负结果并转向layout/copy热点；
- treatment失败：不回退或删除日志，先依据错误判断是checkpoint state兼容还是算子支持问题；
- treatment明显变慢：停止该路线，不尝试用`capturable=True`强行补救。

## 回传结果

成功时只需返回：

1. `$AB_ROOT/comparison.json`；
2. `$AB_ROOT/control_adamw/npu_bf16_performance.json`；
3. `$AB_ROOT/treatment_npu_fused_adamw/npu_bf16_performance.json`。

如果融合优化器运行失败，只返回完整的
`$AB_ROOT/treatment_npu_fused_adamw_terminal.log`。其余原始日志留在服务器归档。
