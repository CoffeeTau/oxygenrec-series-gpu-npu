# 2026-09-26 NPU私有格式开关A/B

状态：**待执行**

目的：针对既有Profile中约`15.77%`的copy/layout kernel占比，在不改变模型结构、dropout、
优化器或训练数据的前提下，比较TorchNPU私有格式关闭/开启的稳态训练性能。

昇腾官方接口说明中，`torch_npu.npu.config.allow_internal_format`控制是否允许私有格式，且只影响
设置之后创建的tensor。本轮benchmark会在模型和batch tensor创建之前设置它。官方同时提示，开启
私有格式后某些计算路径可能出现数值差异，因此本实验除吞吐外必须保留完整loss证据；即使性能提升，
也不能直接视为质量验证通过。

官方参考：

- [allow_internal_format接口说明](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu-npu/%28beta%29torch_npu-npu-config-allow_internal_format.md)
- [开启私有格式后的精度风险说明](https://www.hiascend.com/document/detail/en/Pytorch/2610/userguide/troubleshooting/docs/en/troubleshooting/nz_format_precision_issue.md)

## NPU服务器：可整段复制

```bash
set -euo pipefail
cd /home/h50061831/oxygenrec-series-gpu-npu

git pull origin main
test -z "$(git status --porcelain)" || { git status --short; exit 1; }

export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/cann/set_env.sh

AB_ROOT=checkpoints/performance_optimization/npu_internal_format_20260926
mkdir -p "$AB_ROOT"

git rev-parse HEAD | tee "$AB_ROOT/git_commit.txt"
git status --short | tee "$AB_ROOT/git_status_short.txt"
npu-smi info | tee "$AB_ROOT/npu_smi_before.txt"

NPU_DEVICE=npu:0 \
V2_PERF_OUTPUT_ROOT="$AB_ROOT/control_internal_format_disabled" \
V2_PERF_MAX_SAMPLES=130000 \
V2_PERF_BATCH_SIZES="4096" \
V2_PERF_WARMUP_STEPS=100 \
V2_PERF_MEASURED_STEPS=100 \
V2_PERF_REPEATS=3 \
V2_PERF_CYCLE_SAMPLES=1 \
V2_PERF_OPTIMIZER=adamw \
V2_PERF_ZERO_GRAD_MODE=set_to_none \
V2_PERF_NPU_INTERNAL_FORMAT=disable \
bash run_v2_performance_baseline.sh npu bf16 \
2>&1 | tee "$AB_ROOT/control_internal_format_disabled_terminal.log"
test -f "$AB_ROOT/control_internal_format_disabled/npu_bf16_performance.json" || {
  echo "control result missing; stop before treatment" >&2
  exit 1
}

NPU_DEVICE=npu:0 \
V2_PERF_OUTPUT_ROOT="$AB_ROOT/treatment_internal_format_enabled" \
V2_PERF_MAX_SAMPLES=130000 \
V2_PERF_BATCH_SIZES="4096" \
V2_PERF_WARMUP_STEPS=100 \
V2_PERF_MEASURED_STEPS=100 \
V2_PERF_REPEATS=3 \
V2_PERF_CYCLE_SAMPLES=1 \
V2_PERF_OPTIMIZER=adamw \
V2_PERF_ZERO_GRAD_MODE=set_to_none \
V2_PERF_NPU_INTERNAL_FORMAT=enable \
bash run_v2_performance_baseline.sh npu bf16 \
2>&1 | tee "$AB_ROOT/treatment_internal_format_enabled_terminal.log"
test -f "$AB_ROOT/treatment_internal_format_enabled/npu_bf16_performance.json" || {
  echo "treatment result missing; stop before comparison" >&2
  exit 1
}

python scripts/compare_v2_internal_format_ab.py \
  "$AB_ROOT/control_internal_format_disabled/npu_bf16_performance.json" \
  "$AB_ROOT/treatment_internal_format_enabled/npu_bf16_performance.json" \
  --output "$AB_ROOT/comparison.json" \
  2>&1 | tee "$AB_ROOT/comparison.txt"

npu-smi info | tee "$AB_ROOT/npu_smi_after.txt"
find "$AB_ROOT" -type f | sort | tee "$AB_ROOT/file_manifest.txt"
```

## 唯一变量与成功判据

- control：`workload.npu_internal_format=disable`；
- treatment：`workload.npu_internal_format=enable`；
- 两组均使用AdamW、`set_to_none`、BF16、batch 4096、100步warmup、100步测量、3 repeats；
- 两组commit、输入哈希、关键源码哈希和除私有格式开关外的workload必须一致；
- 两组所有loss必须有限，CV优先低于5%；
- 比较脚本必须生成`comparison.json`，否则不作性能结论。

判断门槛：若treatment吞吐提升不足5%、运行失败或稳定性恶化，则不采纳；若提升至少5%，仍只进入
第二次确认和固定验证集数值复核，不能仅凭本性能A/B修改正式训练默认值。

## 结果路径

本轮根目录：

```text
checkpoints/performance_optimization/npu_internal_format_20260926/
```

执行成功后，请回传以下三个文件：

```text
checkpoints/performance_optimization/npu_internal_format_20260926/comparison.json
checkpoints/performance_optimization/npu_internal_format_20260926/control_internal_format_disabled/npu_bf16_performance.json
checkpoints/performance_optimization/npu_internal_format_20260926/treatment_internal_format_enabled/npu_bf16_performance.json
```

服务器留存、首轮无需回传：

```text
checkpoints/performance_optimization/npu_internal_format_20260926/comparison.txt
checkpoints/performance_optimization/npu_internal_format_20260926/control_internal_format_disabled_terminal.log
checkpoints/performance_optimization/npu_internal_format_20260926/treatment_internal_format_enabled_terminal.log
checkpoints/performance_optimization/npu_internal_format_20260926/git_commit.txt
checkpoints/performance_optimization/npu_internal_format_20260926/git_status_short.txt
checkpoints/performance_optimization/npu_internal_format_20260926/npu_smi_before.txt
checkpoints/performance_optimization/npu_internal_format_20260926/npu_smi_after.txt
checkpoints/performance_optimization/npu_internal_format_20260926/file_manifest.txt
```

如果control失败，只回传：

```text
checkpoints/performance_optimization/npu_internal_format_20260926/control_internal_format_disabled_terminal.log
```

如果treatment失败，只回传：

```text
checkpoints/performance_optimization/npu_internal_format_20260926/treatment_internal_format_enabled_terminal.log
```

