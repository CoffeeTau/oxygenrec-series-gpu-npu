# 2026-09-27 NPU逐step读取与延后读取loss A/B

状态：**已执行，无稳定收益，未采纳**

目的：验证训练循环每步执行`float(output.loss.detach())`所引入的主机同步是否限制NPU端到端吞吐。
control保持历史行为，在每个计时step内读取一次loss；treatment在设备侧保留每步detach后的标量，
计时结束且设备同步后再统一转为Python数值并检查有限性。

两组都计算、保存并校验全部100个测量loss；唯一变量是host materialization发生在计时区间内还是区间外。
模型、dropout、优化器、梯度清零、样本顺序和数学计算均不改变。

## NPU服务器：可整段复制

```bash
set -euo pipefail
cd /home/h50061831/oxygenrec-series-gpu-npu

git pull origin main
test -z "$(git status --porcelain)" || { git status --short; exit 1; }

export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/cann/set_env.sh

AB_ROOT=checkpoints/performance_optimization/npu_deferred_loss_read_20260927
mkdir -p "$AB_ROOT"

git rev-parse HEAD | tee "$AB_ROOT/git_commit.txt"
git status --short | tee "$AB_ROOT/git_status_short.txt"
npu-smi info | tee "$AB_ROOT/npu_smi_before.txt"

NPU_DEVICE=npu:0 \
V2_PERF_OUTPUT_ROOT="$AB_ROOT/control_per_step_loss_read" \
V2_PERF_MAX_SAMPLES=130000 \
V2_PERF_BATCH_SIZES="4096" \
V2_PERF_WARMUP_STEPS=100 \
V2_PERF_MEASURED_STEPS=100 \
V2_PERF_REPEATS=3 \
V2_PERF_CYCLE_SAMPLES=1 \
V2_PERF_OPTIMIZER=adamw \
V2_PERF_ZERO_GRAD_MODE=set_to_none \
V2_PERF_NPU_INTERNAL_FORMAT=disable \
V2_PERF_NPU_DROPOUT_IMPLEMENTATION=standard \
V2_PERF_LOSS_HOST_READ_MODE=per_step \
bash run_v2_performance_baseline.sh npu bf16 \
2>&1 | tee "$AB_ROOT/control_per_step_loss_read_terminal.log"
test -f "$AB_ROOT/control_per_step_loss_read/npu_bf16_performance.json" || {
  echo "control result missing; stop before treatment" >&2
  exit 1
}

NPU_DEVICE=npu:0 \
V2_PERF_OUTPUT_ROOT="$AB_ROOT/treatment_deferred_loss_read" \
V2_PERF_MAX_SAMPLES=130000 \
V2_PERF_BATCH_SIZES="4096" \
V2_PERF_WARMUP_STEPS=100 \
V2_PERF_MEASURED_STEPS=100 \
V2_PERF_REPEATS=3 \
V2_PERF_CYCLE_SAMPLES=1 \
V2_PERF_OPTIMIZER=adamw \
V2_PERF_ZERO_GRAD_MODE=set_to_none \
V2_PERF_NPU_INTERNAL_FORMAT=disable \
V2_PERF_NPU_DROPOUT_IMPLEMENTATION=standard \
V2_PERF_LOSS_HOST_READ_MODE=deferred \
bash run_v2_performance_baseline.sh npu bf16 \
2>&1 | tee "$AB_ROOT/treatment_deferred_loss_read_terminal.log"
test -f "$AB_ROOT/treatment_deferred_loss_read/npu_bf16_performance.json" || {
  echo "treatment result missing; stop before comparison" >&2
  exit 1
}

python scripts/compare_v2_loss_host_read_ab.py \
  "$AB_ROOT/control_per_step_loss_read/npu_bf16_performance.json" \
  "$AB_ROOT/treatment_deferred_loss_read/npu_bf16_performance.json" \
  --output "$AB_ROOT/comparison.json" \
  2>&1 | tee "$AB_ROOT/comparison.txt"

npu-smi info | tee "$AB_ROOT/npu_smi_after.txt"
find "$AB_ROOT" -type f | sort | tee "$AB_ROOT/file_manifest.txt"
```

## 唯一变量与判断规则

- control：`loss_host_read.mode=per_step`，每个计时step内有一次loss host materialization；
- treatment：`loss_host_read.mode=deferred`，计时step内为0次，计时结束后统一读取；
- 两组均逐项保存并检查全部测量loss，不能把“延后读取”解释为“不检查loss”；
- 两组均为标准Dropout、internal format关闭、AdamW、`set_to_none`、BF16、batch 4096、
  100步warmup、100步测量和3 repeats；
- commit、输入哈希、关键源码哈希及除`loss_host_read`外的workload必须一致；
- 两组loss必须全部有限，CV优先低于5%。

由于该改动不改变训练数值路径、没有额外运行时依赖且维护成本很低，采纳门槛设为：treatment中位吞吐
至少提升2%，三个repeat区间没有明显反转，且loss中位数无异常差异。若不足2%或波动覆盖收益，记录后停止，
不重复微调读取间隔。

## 结果路径

本轮根目录：

```text
checkpoints/performance_optimization/npu_deferred_loss_read_20260927/
```

执行成功后，请回传：

```text
checkpoints/performance_optimization/npu_deferred_loss_read_20260927/comparison.json
checkpoints/performance_optimization/npu_deferred_loss_read_20260927/control_per_step_loss_read/npu_bf16_performance.json
checkpoints/performance_optimization/npu_deferred_loss_read_20260927/treatment_deferred_loss_read/npu_bf16_performance.json
```

服务器留存、首轮无需回传：

```text
checkpoints/performance_optimization/npu_deferred_loss_read_20260927/comparison.txt
checkpoints/performance_optimization/npu_deferred_loss_read_20260927/control_per_step_loss_read_terminal.log
checkpoints/performance_optimization/npu_deferred_loss_read_20260927/treatment_deferred_loss_read_terminal.log
checkpoints/performance_optimization/npu_deferred_loss_read_20260927/git_commit.txt
checkpoints/performance_optimization/npu_deferred_loss_read_20260927/git_status_short.txt
checkpoints/performance_optimization/npu_deferred_loss_read_20260927/npu_smi_before.txt
checkpoints/performance_optimization/npu_deferred_loss_read_20260927/npu_smi_after.txt
checkpoints/performance_optimization/npu_deferred_loss_read_20260927/file_manifest.txt
```

如果某组失败，只回传对应的完整终端日志及已经生成的另一组JSON。

## 实际结果

- 中位吞吐：`27356.6887 → 27417.9353 samples/s`，仅`+0.2239%`；
- 中位step时延：`149.7257 → 149.3913 ms`，仅减少约`0.3345 ms`；
- CV：`0.7381% → 0.8982%`，两组均稳定，但吞吐范围高度重叠；
- 两组allocated memory均为`2030623232 bytes`，first/last/mean loss中位数完全一致；
- 结论：远低于预设2%门槛，不采纳，也不继续测试更多读取间隔。

归档记录：
[`2026-09-27 NPU延后loss主机读取A/B结果`](../../实验记录/案例原始记录/performance_tuning/2026-09-27-npu-deferred-loss-read-ab-result.md)
