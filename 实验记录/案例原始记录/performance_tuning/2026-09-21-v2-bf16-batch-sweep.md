# v2 Full 单卡 BF16 batch size 性能摸底

> 记录类型：服务器运行结果摘录  
> 实验编号：`E138`  
> 登记时间：`2026-09-21` / 各轮具体时分未留存  
> 服务器产物路径：见各小节  
> 完整性说明：原始 JSON 保存在 GPU/NPU 服务器的 `checkpoints/performance_baseline/`；
> 本仓库当前只取得服务器运行结果中的 aggregate 字段，未取得 JSON 字节级原件，故以下内容
> 明确作为摘录，不补造未显示字段。  
> 主日志：[`E138`](../../复现实验日志.md)  
> 案例分析：[`v2 BF16 单卡性能调优分析`](../../案例分析/v2_bf16_single_card_performance_tuning.md)

## 1. 固定工作负载

- 模型：OxygenREC-v2 Full；
- 精度：BF16；
- 设备：GPU `NVIDIA L20`，NPU `Ascend950DT_9572`；
- 优化器：AdamW；训练模式；每步读取一次 loss 到主机；
- 输入 checkpoint、events 与 SID registry 的 SHA-256 在比较脚本中校验；
- 每组均为单卡、3 次 repeat，表中吞吐和 step 时延取 repeat 中位数；
- `CV` 是 3 次吞吐的总体标准差除以均值，不是精度指标。

## 2. 初始基线：batch 64/128/256

协议：warmup `20` 步，测量 `100` 步，repeat `3`。两端 source commit 与输入一致。

服务器原始文件：

- GPU：`checkpoints/performance_baseline/v2_full/gpu_bf16_performance.json`
- NPU：`checkpoints/performance_baseline/v2_full/npu_bf16_performance.json`

| batch | 平台 | step 中位数 (ms) | 吞吐中位数 (samples/s) | CV | 峰值 allocated (bytes) |
|---:|---|---:|---:|---:|---:|
| 64 | GPU | 28.978093 | 2208.564966 | 0.064642 | 66067968 |
| 64 | NPU | 66.135737 | 967.706763 | 0.053458 | 78941184 |
| 128 | GPU | 31.358769 | 4081.792939 | 0.024338 | 97634816 |
| 128 | NPU | 67.862969 | 1886.153841 | 0.009381 | 109660160 |
| 256 | GPU | 35.260069 | 7260.337381 | 0.028908 | 158153216 |
| 256 | NPU | 74.020488 | 3458.501917 | 0.030865 | 173064192 |

## 3. batch 64 受控复核

协议：warmup `100` 步，测量 `300` 步，repeat `3`。两端 Git commit 不同，但比较脚本确认
关键 source file SHA-256 一致；该事实仅支持源码内容可比，不把 commit mismatch 隐藏掉。

服务器原始文件：

- GPU：`checkpoints/performance_baseline/recheck_gpu_20260921/gpu_bf16_performance.json`
- NPU：`checkpoints/performance_baseline/recheck_npu_20260921/npu_bf16_performance.json`

| batch | GPU samples/s | GPU step (ms) | GPU CV | NPU samples/s | NPU step (ms) | NPU CV | NPU/GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 2298.088003 | 27.849238 | 0.090779 | 3459.944930 | 18.497404 | 0.029104 | 1.505575 |

峰值 allocated：GPU `66067968` bytes，NPU `78941184` bytes。

## 4. 稳态 batch sweep：128/256/512/1024

| batch | warmup / measured | GPU samples/s | GPU step (ms) | GPU CV | NPU samples/s | NPU step (ms) | NPU CV | NPU/GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 100 / 200 | 4335.330042 | 29.524857 | 0.055395 | 5846.450750 | 21.893625 | 0.075542 | 1.348560 |
| 256 | 100 / 200 | 7657.745957 | 33.430203 | 0.014511 | 9772.966375 | 26.194708 | 0.016522 | 1.276220 |
| 512 | 100 / 100 | 14472.306525 | 35.377913 | 0.041132 | 17838.879346 | 28.701355 | 0.031589 | 1.232622 |
| 1024 | 60 / 60 | 22427.469086 | 45.658295 | 0.053325 | 27773.130213 | 36.870169 | 0.023623 | 1.238353 |

峰值 allocated：

| batch | GPU bytes | NPU bytes |
|---:|---:|---:|
| 128 | 98421248 | 109660672 |
| 256 | 158153216 | 173064192 |
| 512 | 278665728 | 304427520 |
| 1024 | 519821824 | 553057280 |

对应服务器目录：

- `steady_sweep_gpu_20260921` / `steady_sweep_npu_20260921`；
- `steady_bs512_gpu_20260921` / `steady_bs512_npu_20260921`；
- `steady_bs1024_gpu_w60_m60_20260921` / `steady_bs1024_npu_w60_m60_20260921`。

## 5. batch 2048 短窗口：保留但不作为稳定结论

协议：warmup `30` 步，测量 `30` 步，repeat `3`。

| batch | GPU samples/s | GPU step (ms) | GPU CV | NPU samples/s | NPU step (ms) | NPU CV | 表面 NPU/GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 20844.726142 | 98.250271 | 0.199185 | 33123.897464 | 61.828473 | 0.014214 | 1.589078 |

峰值 allocated：GPU `983652864` bytes，NPU `1048994816` bytes。GPU CV 达 `19.92%`，
因此这一轮保留为“测量窗口不足/环境抖动”的失败尝试，不把 `1.589×` 当成稳定性能结论。

## 6. batch 2048/4096 长窗口复核

协议：warmup `100` 步，测量 `100` 步，repeat `3`，`V2_PERF_CYCLE_SAMPLES=1`。
GPU/NPU commit 均为 `fbe005a6d98d9208902662ced8c3e61feae52ac4`，关键源码哈希一致。

服务器原始文件：

- GPU：`checkpoints/performance_baseline/steady_bs2048_4096_gpu_cycle_20260921/gpu_bf16_performance.json`
- NPU：`checkpoints/performance_baseline/steady_bs2048_4096_npu_cycle_20260921/npu_bf16_performance.json`

| batch | GPU samples/s | GPU step (ms) | GPU CV | NPU samples/s | NPU step (ms) | NPU CV | NPU/GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 23320.445802 | 87.819935 | 0.003292 | 32773.286252 | 62.489919 | 0.018072 | 1.405346 |
| 4096 | 26687.405526 | 153.480637 | 0.066257 | 36859.295357 | 111.125293 | 0.007527 | 1.381149 |

峰值 allocated：

| batch | GPU bytes | NPU bytes |
|---:|---:|---:|
| 2048 | 983652864 | 1048994816 |
| 4096 | 1928223232 | 2034717696 |

## 7. 记录缺口

- 本仓库尚未保存上述服务器 JSON 的字节级原件和 SHA-256；
- 各轮开始/结束时的完整 `npu-smi info`、主机负载与频率未系统保存；
- 表中 allocated memory 是各框架报告值，不等价于设备总 HBM 占用；
- 当前结果是不同硬件及软件栈的工程观察，不是同规格硬件比较。

