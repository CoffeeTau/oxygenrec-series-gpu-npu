# 2026-09-21 NPU 健康状态复核

状态：**可选；不阻塞性能主线**

目的：仅在 NPU 0 报错、吞吐明显漂移或 Profile 结果异常时，获取 NPU 0 的告警详情，并检查
显示为 `Critical` 的 NPU 6。正常情况下不执行本专项检查，直接推进短窗口 Profile。
本轮是只读检查，不重置设备、不结束进程、不安装或升级软件。

华为 `npu-smi` 文档将 `Warning` 定义为一般告警、`Critical` 定义为紧急告警；详细健康查询
会返回 Error Code 和 Error Information。因此当前概览不能直接按“正常”跳过。

## 1. NPU 服务器：可整段复制

```bash
set -euo pipefail
cd /home/h50061831/oxygenrec-series-gpu-npu

export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/cann/set_env.sh

RESULT_DIR=checkpoints/performance_profiling/npu_health_recheck_20260921
mkdir -p "$RESULT_DIR"

git rev-parse HEAD | tee "$RESULT_DIR/git_commit.txt"
git status --short | tee "$RESULT_DIR/git_status_short.txt"
npu-smi info -l | tee "$RESULT_DIR/npu_list.txt"
npu-smi info | tee "$RESULT_DIR/npu_info.txt"
npu-smi info -t health -i 0 | tee "$RESULT_DIR/npu0_health_all.txt"
npu-smi info -t health -i 0 -c 0 | tee "$RESULT_DIR/npu0_chip0_health.txt"
npu-smi info -t usages -i 0 | tee "$RESULT_DIR/npu0_usages.txt"
npu-smi info -t health -i 6 | tee "$RESULT_DIR/npu6_health_all.txt"
npu-smi info -t health -i 6 -c 0 | tee "$RESULT_DIR/npu6_chip0_health.txt"
npu-smi info -t usages -i 6 | tee "$RESULT_DIR/npu6_usages.txt"
```

## 2. 停止点

执行完成后暂停，不要执行 reset、清理显存、结束其他用户进程或继续正式 Profile。请回传：

- 终端完整输出；
- `npu0_chip0_health.txt` 与 `npu6_chip0_health.txt`；
- `npu_info.txt` 中的进程区域（若存在）；
- 若命令因当前950DT版本不支持某个参数而停止，保留报错，不自行换命令。

确认 NPU 0 的 Error Code 和占用来源后，再追加 batch 4096 短窗口 Profile 命令。

官方参考：

- [查询所有芯片健康状态](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/24.1.0/re/npu/npusmi_027.html)
- [查询指定芯片健康状态](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/250RC1/re/npu/npusmi_028.html)
