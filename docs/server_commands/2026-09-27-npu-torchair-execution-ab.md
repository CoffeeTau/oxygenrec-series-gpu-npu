# 2026-09-27 NPU eager与TorchAir图模式训练A/B

状态：**待执行**

目的：比较相同loss-only训练前向在eager模式与TorchAir静态图模式下的稳态吞吐。此前融合优化器、
私有格式和loss读取时机均未达到采纳门槛，且现有Profile热点分散在Dropout、copy/layout、矩阵乘与
Attention；本轮转向图级执行，以减少Python/算子下发和跨算子调度开销。

两组都使用相同的loss-only wrapper。treatment通过TorchNPU公开的
`torch_npu.dynamo.torchair.get_npu_backend()`传给`torch.compile(..., dynamic=False)`；固定batch 4096，
因此不在本轮引入动态shape变量。

官方依据：

- [TorchAir get_npu_backend API](https://www.hiascend.com/document/detail/zh/Pytorch/730/modthirdparty/torchairuseguide/torchair_00084.html)
- [Ascend PyTorch公开API清单中的torch_npu.dynamo.torchair](https://gitee.com/ascend/pytorch/blob/c1d9378d1f0872024c0efb006505d1bffe26bbcd/test/allowlist_for_publicAPI.json)

## NPU服务器：可整段复制

```bash
set -euo pipefail
cd /home/h50061831/oxygenrec-series-gpu-npu

git pull origin main
test -z "$(git status --porcelain)" || { git status --short; exit 1; }

export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/cann/set_env.sh

AB_ROOT=checkpoints/performance_optimization/npu_torchair_execution_20260927
mkdir -p "$AB_ROOT"

git rev-parse HEAD | tee "$AB_ROOT/git_commit.txt"
git status --short | tee "$AB_ROOT/git_status_short.txt"
npu-smi info | tee "$AB_ROOT/npu_smi_before.txt"
python -c "from torch_npu.dynamo import torchair; assert callable(torchair.get_npu_backend); print('TorchAir backend import OK')" \
2>&1 | tee "$AB_ROOT/torchair_import_precheck.log"

NPU_DEVICE=npu:0 \
V2_PERF_OUTPUT_ROOT="$AB_ROOT/control_eager" \
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
V2_PERF_EXECUTION_MODE=eager \
bash run_v2_performance_baseline.sh npu bf16 \
2>&1 | tee "$AB_ROOT/control_eager_terminal.log"
test -f "$AB_ROOT/control_eager/npu_bf16_performance.json" || {
  echo "control result missing; stop before treatment" >&2
  exit 1
}

NPU_DEVICE=npu:0 \
V2_PERF_OUTPUT_ROOT="$AB_ROOT/treatment_torchair" \
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
V2_PERF_EXECUTION_MODE=torchair \
bash run_v2_performance_baseline.sh npu bf16 \
2>&1 | tee "$AB_ROOT/treatment_torchair_terminal.log"
test -f "$AB_ROOT/treatment_torchair/npu_bf16_performance.json" || {
  echo "treatment result missing; stop before comparison" >&2
  exit 1
}

python scripts/compare_v2_execution_mode_ab.py \
  "$AB_ROOT/control_eager/npu_bf16_performance.json" \
  "$AB_ROOT/treatment_torchair/npu_bf16_performance.json" \
  --output "$AB_ROOT/comparison.json" \
  2>&1 | tee "$AB_ROOT/comparison.txt"

npu-smi info | tee "$AB_ROOT/npu_smi_after.txt"
find "$AB_ROOT" -type f | sort | tee "$AB_ROOT/file_manifest.txt"
```

## 唯一变量与判断规则

- control：`execution.mode=eager`；
- treatment：`execution.mode=torchair`、官方NPU backend、`dynamic=false`；
- 两组使用同一个loss-only wrapper，避免把Python dataclass输出差异混入比较；
- 两组均为逐step loss读取、标准Dropout、internal format关闭、AdamW、`set_to_none`、BF16、
  batch 4096、100步warmup、100步测量和3 repeats；
- 首次图编译发生在warmup，稳态计时从warmup完成后开始；每个repeat的完整warmup耗时仍写入JSON，
  比较结果同时报告warmup中位数，不能只看稳态吞吐而忽略编译成本；
- commit、输入哈希、关键源码哈希及除`execution`外的workload必须一致；
- loss必须有限，CV优先低于5%。

判断门槛：若TorchAir稳态吞吐提升不足5%、编译失败或明显更不稳定，则不采纳；若至少提升5%，再结合
首次编译成本计算长训练的摊销价值，并进入第二次性能确认与固定验证集数值复核。首次treatment warmup
可能因编译明显长于eager；只要没有明确报错，不要在编译过程中手动中断。

## 结果路径

本轮根目录：

```text
checkpoints/performance_optimization/npu_torchair_execution_20260927/
```

执行成功后，请回传：

```text
checkpoints/performance_optimization/npu_torchair_execution_20260927/comparison.json
checkpoints/performance_optimization/npu_torchair_execution_20260927/control_eager/npu_bf16_performance.json
checkpoints/performance_optimization/npu_torchair_execution_20260927/treatment_torchair/npu_bf16_performance.json
```

服务器留存、首轮无需回传：

```text
checkpoints/performance_optimization/npu_torchair_execution_20260927/torchair_import_precheck.log
checkpoints/performance_optimization/npu_torchair_execution_20260927/comparison.txt
checkpoints/performance_optimization/npu_torchair_execution_20260927/control_eager_terminal.log
checkpoints/performance_optimization/npu_torchair_execution_20260927/treatment_torchair_terminal.log
checkpoints/performance_optimization/npu_torchair_execution_20260927/git_commit.txt
checkpoints/performance_optimization/npu_torchair_execution_20260927/git_status_short.txt
checkpoints/performance_optimization/npu_torchair_execution_20260927/npu_smi_before.txt
checkpoints/performance_optimization/npu_torchair_execution_20260927/npu_smi_after.txt
checkpoints/performance_optimization/npu_torchair_execution_20260927/file_manifest.txt
```

若导入预检失败，只回传`torchair_import_precheck.log`。若treatment失败，回传
`treatment_torchair_terminal.log`和已生成的control JSON。
