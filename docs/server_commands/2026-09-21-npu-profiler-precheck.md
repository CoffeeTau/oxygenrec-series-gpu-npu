# 2026-09-21 NPU Profiler API 预检查

状态：**待执行**

目的：确认当前 NPU Python 环境已经提供 Ascend PyTorch Profiler API，并采集设备状态。
本轮只做只读检查，不安装软件、不修改模型，也不开始正式 Profile。

## 1. NPU 服务器：加载环境并检查 Profiler API

在 NPU 服务器执行。下面是一个完整代码块，可整段复制：

```bash
set -euo pipefail
cd /home/h50061831/oxygenrec-series-gpu-npu

export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/cann/set_env.sh

python -c "import torch_npu.profiler; print('torch_npu profiler import OK')"
python -c "import torch, torch_npu, torch_npu.profiler as p; print('torch=', torch.__version__); print('torch_npu=', getattr(torch_npu, '__version__', 'unknown')); print('profiler_api=', [x for x in ('profile', 'schedule', 'ProfilerActivity', 'tensorboard_trace_handler') if hasattr(p, x)])"
npu-smi info
```

## 2. 预期判断

- 第一条 Python 命令输出 `torch_npu profiler import OK`：模块可导入；
- `profiler_api` 最理想为四项全部存在：
  `profile`、`schedule`、`ProfilerActivity`、`tensorboard_trace_handler`；
- 若出现 `ModuleNotFoundError` 或 `ImportError`，才属于模块或环境问题；
- 若出现 `IndentationError`，属于命令在复制时被拆行或行首混入空格，不代表 Profiler 未安装；
- `npu-smi info` 应列出 `Ascend950DT_9572` 及设备健康、功耗、温度和显存信息。

## 3. 停止点

执行到这里后暂停，不要自行安装或升级 `torch`、`torch_npu`、CANN 或 MindStudio。
请回传两条 Python 命令和 `npu-smi info` 的完整服务器运行结果。确认 API 形态后，下一份
命令文档将给出 batch 4096 的短窗口 Profile 采集命令。
