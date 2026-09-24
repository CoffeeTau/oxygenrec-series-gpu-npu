# 2026-09-23 NPU Profile 重新生成并汇总热点

状态：**待执行**

目的：重新生成已误删的 `operator_details.csv` 和 `kernel_details.csv`，随后立即解析热点。
本流程使用新的输出目录，不覆盖、不清理旧实验目录。

执行内容：

- NPU 0、BF16、batch 4096；
- Profiler外预热100步；
- Profiler内warmup 1步、active 3步；
- 自动生成operator/kernel CSV；
- 自动输出Top 30 operator、Top 30 kernel与关键候选项累计占比。

## NPU服务器：可整段复制

```bash
set -euo pipefail
cd /home/h50061831/oxygenrec-series-gpu-npu

export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/cann/set_env.sh

PROFILE_ROOT=checkpoints/performance_profiling/npu_bs4096_short_regenerated_20260923
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

python scripts/summarize_v2_npu_profile.py \
  "$PROFILE_ROOT/trace" \
  --output "$PROFILE_ROOT/profile_hotspots.json" \
  --top-k 30 \
  2>&1 | tee "$PROFILE_ROOT/profile_hotspots.txt"

find "$PROFILE_ROOT" -type f | sort | tee "$PROFILE_ROOT/file_manifest.txt"
```

## 成功判据

终端末尾应先后出现：

```text
OK training_profile output=checkpoints/performance_profiling/npu_bs4096_short_regenerated_20260923/npu_bf16_profile_summary.json
OK profile_summary output=checkpoints/performance_profiling/npu_bs4096_short_regenerated_20260923/profile_hotspots.json
```

`file_manifest.txt`中应至少包含：

- `operator_details.csv`；
- `kernel_details.csv`；
- `profile_hotspots.txt`；
- `profile_hotspots.json`。

## 回传结果

只需要返回：

1. `profile_hotspots.txt`；
2. `profile_hotspots.json`；
3. 如果命令失败，则改为返回完整的`terminal.log`和报错位置。

原始CSV、trace、运行摘要与前后`npu-smi`继续保留在服务器，不需要首轮全部回传。

