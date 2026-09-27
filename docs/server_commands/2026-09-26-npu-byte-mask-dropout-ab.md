# 2026-09-26 NPU标准Dropout与ByteMask Dropout A/B

状态：**已执行，treatment因目标机不支持旧ByteMask算子而失败，未采纳**

目的：针对既有Profile中`aclnnDropoutV3`约`24.13%`的最大设备热点，在保持dropout概率`0.1`、
模型结构和训练数据不变的前提下，比较标准`torch.nn.Dropout`与昇腾NPU专用
`torch_npu.contrib.module.DropoutWithByteMask`。

本轮不会关闭dropout。ByteMask实现仍执行随机失活，但随机数消费顺序可能与标准实现不同，因此
loss只要求有限，不要求逐repeat一致；即使吞吐达到门槛，也必须后续做数值与质量复核。

官方接口参考：[DropoutWithByteMask](https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/apilist/ptaoplist_000207.html)

## NPU服务器：可整段复制

```bash
set -euo pipefail
cd /home/h50061831/oxygenrec-series-gpu-npu

git pull origin main
test -z "$(git status --porcelain)" || { git status --short; exit 1; }

export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/cann/set_env.sh

AB_ROOT=checkpoints/performance_optimization/npu_byte_mask_dropout_20260926
mkdir -p "$AB_ROOT"

git rev-parse HEAD | tee "$AB_ROOT/git_commit.txt"
git status --short | tee "$AB_ROOT/git_status_short.txt"
npu-smi info | tee "$AB_ROOT/npu_smi_before.txt"

NPU_DEVICE=npu:0 \
V2_PERF_OUTPUT_ROOT="$AB_ROOT/control_standard_dropout" \
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
bash run_v2_performance_baseline.sh npu bf16 \
2>&1 | tee "$AB_ROOT/control_standard_dropout_terminal.log"
test -f "$AB_ROOT/control_standard_dropout/npu_bf16_performance.json" || {
  echo "control result missing; stop before treatment" >&2
  exit 1
}

NPU_DEVICE=npu:0 \
V2_PERF_OUTPUT_ROOT="$AB_ROOT/treatment_byte_mask_dropout" \
V2_PERF_MAX_SAMPLES=130000 \
V2_PERF_BATCH_SIZES="4096" \
V2_PERF_WARMUP_STEPS=100 \
V2_PERF_MEASURED_STEPS=100 \
V2_PERF_REPEATS=3 \
V2_PERF_CYCLE_SAMPLES=1 \
V2_PERF_OPTIMIZER=adamw \
V2_PERF_ZERO_GRAD_MODE=set_to_none \
V2_PERF_NPU_INTERNAL_FORMAT=disable \
V2_PERF_NPU_DROPOUT_IMPLEMENTATION=byte_mask \
bash run_v2_performance_baseline.sh npu bf16 \
2>&1 | tee "$AB_ROOT/treatment_byte_mask_dropout_terminal.log"
test -f "$AB_ROOT/treatment_byte_mask_dropout/npu_bf16_performance.json" || {
  echo "treatment result missing; stop before comparison" >&2
  exit 1
}

python scripts/compare_v2_dropout_ab.py \
  "$AB_ROOT/control_standard_dropout/npu_bf16_performance.json" \
  "$AB_ROOT/treatment_byte_mask_dropout/npu_bf16_performance.json" \
  --output "$AB_ROOT/comparison.json" \
  2>&1 | tee "$AB_ROOT/comparison.txt"

npu-smi info | tee "$AB_ROOT/npu_smi_after.txt"
find "$AB_ROOT" -type f | sort | tee "$AB_ROOT/file_manifest.txt"
```

## 唯一变量与成功判据

- control：`npu_dropout.implementation=standard`且`replaced_modules=0`；
- treatment：`npu_dropout.implementation=byte_mask`且`replaced_modules>=1`；
- 两组均显式关闭internal format，并保持AdamW、`set_to_none`、BF16、batch 4096、100步warmup、
  100步测量和3 repeats一致；
- 两组commit、输入哈希、关键源码哈希以及除`npu_dropout`外的workload必须一致；
- 两组所有记录的loss必须有限，CV优先低于5%；
- 比较脚本必须生成`comparison.json`。

判断门槛：若treatment吞吐提升不足5%、失败或明显更不稳定，则不采纳；若至少提升5%，进入独立
性能确认和固定验证集数值复核，不直接替换正式模型实现。

## 结果路径

本轮根目录：

```text
checkpoints/performance_optimization/npu_byte_mask_dropout_20260926/
```

执行成功后，请回传：

```text
checkpoints/performance_optimization/npu_byte_mask_dropout_20260926/comparison.json
checkpoints/performance_optimization/npu_byte_mask_dropout_20260926/control_standard_dropout/npu_bf16_performance.json
checkpoints/performance_optimization/npu_byte_mask_dropout_20260926/treatment_byte_mask_dropout/npu_bf16_performance.json
```

服务器留存、首轮无需回传：

```text
checkpoints/performance_optimization/npu_byte_mask_dropout_20260926/comparison.txt
checkpoints/performance_optimization/npu_byte_mask_dropout_20260926/control_standard_dropout_terminal.log
checkpoints/performance_optimization/npu_byte_mask_dropout_20260926/treatment_byte_mask_dropout_terminal.log
checkpoints/performance_optimization/npu_byte_mask_dropout_20260926/git_commit.txt
checkpoints/performance_optimization/npu_byte_mask_dropout_20260926/git_status_short.txt
checkpoints/performance_optimization/npu_byte_mask_dropout_20260926/npu_smi_before.txt
checkpoints/performance_optimization/npu_byte_mask_dropout_20260926/npu_smi_after.txt
checkpoints/performance_optimization/npu_byte_mask_dropout_20260926/file_manifest.txt
```

如果某组失败，只回传对应的完整终端日志。

## 2026-09-27 实际结果

- control通过结果文件存在性检查后进入treatment；本次截图没有包含control聚合数值，因此不补写数值；
- treatment在第一个warmup前向经过`behavior_instruction_adapter`中的Dropout时失败，尚未进入正式测量；
- `DropoutWithByteMask`调用已弃用的`dropout_with_byte_mask`，运行时明确报告当前设备只支持
  ACLNN算子，而该旧算子没有ACLNN实现，并返回`ERR00007 PTA feature not supported`；
- `treatment result missing; stop before comparison`是前述失败导致JSON未生成后的保护性停止，不是第二个故障；
- 结论：目标Ascend 950DT / TorchNPU 2.7.1.post4栈不支持此实现，不尝试强制开启旧算子，路线停止。

归档记录：
[`2026-09-27 NPU ByteMask Dropout失败`](../../实验记录/案例原始记录/performance_tuning/2026-09-27-npu-byte-mask-dropout-failure.md)
