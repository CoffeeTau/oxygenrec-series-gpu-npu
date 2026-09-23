# 2026-09-21 NPU Profile CSV 热点汇总

状态：**待执行**

目的：直接解析已经生成的 `operator_details.csv` 和 `kernel_details.csv`，输出可回传的小型热点
摘要。本步骤不重跑训练、不重跑Profiler。

## NPU服务器：可整段复制

```bash
set -euo pipefail
cd /home/h50061831/oxygenrec-series-gpu-npu

PROFILE_ROOT=checkpoints/performance_profiling/npu_bs4096_short_20260921

python scripts/summarize_v2_npu_profile.py \
  "$PROFILE_ROOT/trace" \
  --output "$PROFILE_ROOT/profile_hotspots.json" \
  --top-k 30 \
  2>&1 | tee "$PROFILE_ROOT/profile_hotspots.txt"
```

## 回传结果

只需要提供：

1. `profile_hotspots.txt`；
2. `profile_hotspots.json`。

这两个文件会包含CSV原始列名、设备/主机时长口径、Top 30 operator、Top 30 kernel，以及
`_local_scalar_dense`、Memcpy、TransData、masked_fill、Adam/optimizer等候选项的累计占比。

若脚本报“找不到列”，请只提供报错和两个CSV的第一行表头，不需要重跑Profiler。

