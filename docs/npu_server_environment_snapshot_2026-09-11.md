# OxygenREC NPU（Ascend 950DT）服务器环境快照

> 环境采集时间：`2026-09-11T12:26:32.957516+00:00`。  
> 整理时间：2026-09-11 CST。  
> 证据边界：本文是截图转录与分析，不是服务器JSON的字节级副本；截图没有展示或无法可靠辨认的字段不补造。

## 1. Stage-0结论

NPU Stage-0通过：TorchNPU识别8张Ascend卡，HCCL接口可用；`npu:0`完成矩阵计算、
backward、AdamW参数更新和checkpoint保存恢复。当前可以进入v2固定输入的GPU/NPU
精度对齐，但还不能据此声称OxygenREC模型已在NPU完成迁移。

`atc --version`返回255并提示不支持`version`参数。因此只能确认ATC命令存在，CANN
安装路径显示`9.0.T550`；ATC精确版本尚未由本轮命令确认。该项不影响已完成的
PyTorch/TorchNPU单卡验证，但在需要ATC离线编译时必须另行采集。

## 2. 系统与主机资源

| 项目 | 采集值 |
|---|---|
| 操作系统 | Linux |
| Platform | `Linux-6.6.0-159.4.10.164.oe2403sp4.aarch64-aarch64-with-glibc2.36` |
| Kernel release | `6.6.0-159.4.10.164.oe2403sp4.aarch64` |
| 架构 | `aarch64` |
| CPU逻辑核数 | 192 |
| 主机内存总量 | `1512560692 kB`，约1442.49 GiB |
| 采集时可用内存 | `1488729000 kB`，约1419.76 GiB |
| Swap总量/空闲 | `4194300 kB` / `4194300 kB`，约4.00 GiB |

## 3. Python与核心软件包

| 项目 | 采集值 |
|---|---|
| Python | `3.11.0`，CPython，GCC 11.2.0 |
| Python可执行文件 | `/usr/bin/python` |
| Python prefix | `/usr/local/python3.11.0` |
| PyTorch包 | `2.10.0` |
| PyTorch运行时字符串 | `2.10.0+cpu` |
| torch-npu | `2.10.0.post4.dev20260715` |
| torchvision | `0.25.0` |
| torchaudio | `2.10.0` |
| transformers | `5.5.4` |
| NumPy / SciPy | `1.26.4` / `1.13.1` |
| scikit-learn | `1.8.0` |
| sentencepiece / safetensors | `0.2.1` / `0.7.0` |
| accelerate / DeepSpeed / datasets | 未安装或采集值为`null` |

说明：`torch_version=2.10.0+cpu`在TorchNPU插件栈中不等于“只能使用CPU”。本轮
`npu:0`反向与优化器更新已实际成功，证明该PyTorch构建能通过已安装的TorchNPU
访问设备；后续仍需对OxygenREC使用的Transformer算子逐项验证。

## 4. NPU与分布式接口

共识别8张配置一致的NPU（index 0-7）：

| 项目 | 采集值 |
|---|---|
| 型号 | `Ascend950DT_9581` |
| 每卡设备内存 | `93952 MB`（按设备属性字符串记录） |
| Cube core | 32 |
| Vector core | 64 |
| L2 cache | 128 MB |
| `torch.npu.is_available()` | `True` |
| NPU device count | 8 |
| PyTorch distributed | available |
| HCCL backend | available |
| `npu-smi info` | 可用，return code 0；截图显示版本`25.6.rc2.b016` |

设备UUID不在本文保留，因为精度迁移不依赖该标识。

## 5. CANN与HCCL配置

| 项目 | 采集值 |
|---|---|
| CANN/Ascend路径 | `/usr/local/Ascend/cann-9.0.T550` |
| OPP路径 | `/usr/local/Ascend/cann-9.0.T550/opp` |
| `HCCL_BUFFSIZE` | `2048` |
| `HCCL_CONNECT_TIMEOUT` | `120` |
| `HCCL_EXEC_TIMEOUT` | `120` |
| HCCL topology file | 已配置为8卡拓扑文件；本文不保留服务器绝对路径 |
| `atc --version` | 命令存在，但参数不兼容，return code 255；精确版本未确认 |
| GCC | `11.2.0`，return code 0 |

## 6. 单卡验证结果

| 项目 | 实测值 |
|---|---:|
| Device | `npu:0` |
| Device name | `Ascend950DT_9581` |
| Loss | `93.5` |
| Gradient L1 | `738.0` |
| 参数最大绝对更新量 | `0.010100007057189941` |
| Checkpoint保存恢复一致 | `True` |

上述数值来自确定性的4×4 FP32矩阵与AdamW冒烟测试，仅用于证明计算链有效，不能
拿来与OxygenREC训练loss或GPU/NPU模型精度直接比较。

## 7. 文件命名约定

为避免和GPU环境及后续重复运行混淆，Stage-0入口以后按UTC日期输出：

```text
checkpoints/npu_stage0/npu_server_environment_snapshot_YYYY-MM-DD.json
checkpoints/npu_stage0/npu_single_card_validation_YYYY-MM-DD.json
checkpoints/npu_stage0/npu_single_card_validation_YYYY-MM-DD.pt
```

本次服务器上的旧文件可重命名为：

```bash
mv checkpoints/npu_stage0/environment.json \
  checkpoints/npu_stage0/npu_server_environment_snapshot_2026-09-11.json
mv checkpoints/npu_stage0/single_card.json \
  checkpoints/npu_stage0/npu_single_card_validation_2026-09-11.json
mv checkpoints/npu_stage0/single_card.pt \
  checkpoints/npu_stage0/npu_single_card_validation_2026-09-11.pt
```

## 8. 下一步

冻结本次环境快照，在GPU与NPU使用相同Full checkpoint、registry、固定validation
batch和生成参数，依次比较teacher-forcing logits、weighted loss、greedy/beam列表，
再进入梯度与optimizer step对齐。此时不需要先启动完整训练或8卡训练。
