# 2026-09-21 NPU batch 4096 短窗口 Profile

状态：**待执行**

目的：在已经完成的稳态性能摸底基础上，采集 batch 4096 的3个 active训练step，定位主机下发、
同步等待、Memcpy、TransData、CPU fallback及主要训练算子耗时。本轮不是吞吐基准，也不训练
正式质量checkpoint。

采集协议：

- 设备：`npu:0`；精度：BF16；batch：4096；
- Profiler 外先执行100个正常训练step，使运行进入稳态；
- Profiler 内1个warmup step、3个active step，只生成一组trace；
- 记录CPU和NPU活动、shape、memory和module信息；
- Profile期间的step时延包含采集开销，不与性能基线比较。

## NPU服务器：可整段复制

```bash
set -euo pipefail
cd /home/h50061831/oxygenrec-series-gpu-npu

export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/cann/set_env.sh

PROFILE_ROOT=checkpoints/performance_profiling/npu_bs4096_short_20260921
mkdir -p "$PROFILE_ROOT"

git rev-parse HEAD | tee "$PROFILE_ROOT/git_commit.txt"
git status --short | tee "$PROFILE_ROOT/git_status_short.txt"
npu-smi info | tee "$PROFILE_ROOT/npu_smi_before.txt"

NPU_DEVICE=npu:0 \
V2_PROFILE_OUTPUT_ROOT="$PROFILE_ROOT" \
V2_PROFILE_MAX_SAMPLES=130000 \
V2_PROFILE_BATCH_SIZE=4096 \
V2_PROFILE_OUTER_WARMUP_STEPS=100 \
V2_PROFILE_WARMUP_STEPS=1 \
V2_PROFILE_ACTIVE_STEPS=3 \
bash run_v2_npu_profile.sh 2>&1 | tee "$PROFILE_ROOT/terminal.log"

npu-smi info | tee "$PROFILE_ROOT/npu_smi_after.txt"
find "$PROFILE_ROOT" -type f | sort | tee "$PROFILE_ROOT/file_manifest.txt"
```

## 成功判据与停止点

终端应出现：

```text
OK training_profile output=checkpoints/performance_profiling/npu_bs4096_short_20260921/npu_bf16_profile_summary.json
```

执行后暂停，把以下内容作为本轮原始证据提供：

1. `terminal.log`；
2. `npu_bf16_profile_summary.json`；
3. `file_manifest.txt`；
4. trace目录中名称包含 `operator`、`kernel`、`trace_view` 的文件；
5. 运行前后的两个 `npu_smi` 文件。

不要先手工筛选“看起来慢”的算子，也不要删除没有性能提升的原始输出。下一步直接根据
operator/kernel统计选择第一个收益最大的A/B优化点。

官方API依据：

- [torch_npu.profiler.profile](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu-profiler/torch_npu-profiler-profile.md)
- [torch_npu.profiler.schedule](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu-profiler/torch_npu-profiler-schedule.md)

