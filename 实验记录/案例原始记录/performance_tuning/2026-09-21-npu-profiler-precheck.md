# NPU Profiler API 与设备状态预检查

> 记录类型：服务器运行结果摘录  
> 实验编号：`E139`  
> 登记时间：`2026-09-21` / 具体时分未留存  
> 服务器产物路径：本轮未生成文件  
> 完整性说明：以下为服务器终端可确认字段；`npu-smi info` 末尾进程区未提供，不能判断
> HBM 由哪些进程持有。  
> 主日志：[`E139`](../../复现实验日志.md)  
> 案例分析：[`v2 BF16 单卡性能调优分析`](../../案例分析/v2_bf16_single_card_performance_tuning.md)

## 1. Profiler Python API

```text
torch_npu profiler import OK
torch= 2.7.1+cpu
torch_npu= 2.7.1.post4
profiler_api= ['profile', 'schedule', 'ProfilerActivity', 'tensorboard_trace_handler']
```

`torch_npu.profiler` 导入成功，计划使用的四个 API 均存在。

## 2. npu-smi 概览

```text
npu-smi 25.6.1.b007
Version: 25.6.1.b007
```

| NPU ID | Name | Health | Power (W) | Temp (C) | NPU Util (%) | HBM Usage (MB) |
|---:|---|---|---:|---:|---:|---:|
| 0 | Ascend950DT | Warning | 410.3 | 36 | 0 | 48409 / 98304 |
| 1 | Ascend950DT | Warning | 412.6 | 39 | 0 | 15379 / 98304 |
| 2 | Ascend950DT | Warning | 415.1 | 38 | 0 | 15374 / 98304 |
| 3 | Ascend950DT | Warning | 413.3 | 38 | 0 | 15376 / 98304 |
| 4 | Ascend950DT | Warning | 409.6 | 37 | 0 | 15373 / 98304 |
| 5 | Ascend950DT | Warning | 411.4 | 41 | 0 | 15368 / 98304 |
| 6 | Ascend950DT | Critical | 426.9 | 41 | 0 | 15372 / 98304 |
| 7 | Ascend950DT | Warning | 410.8 | 43 | 0 | 15372 / 98304 |

## 3. 记录缺口

- 未取得 `npu-smi info -t health -i 0 -c 0` 和设备 6 的 Error Code/Error Information；
- 未取得各设备 `usages` 详情；
- 未取得进程列表，无法解释 NPU 0 在 Util=0 时约 48.4 GB HBM 占用；
- 因健康状态不是 `OK`，本轮仅确认 Profiler API 可用，不把设备环境判为健康通过。

