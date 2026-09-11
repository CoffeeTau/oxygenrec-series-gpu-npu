# OxygenREC GPU→NPU迁移与验收计划

> 状态：Stage-0已于2026-09-11在8×Ascend 950DT服务器实测通过；Stage-0.5迁移清单和Stage-1/2固定输入对齐代码已完成本地静态验证，待GPU/NPU服务器实跑。
> GPU方法基线：[`OxygenREC-v2 GPU方法复现验收报告`](../实验记录/案例分析/OxygenREC-v2%20GPU方法复现验收报告.md)
> NPU环境：[`Ascend 950DT服务器环境快照`](npu_server_environment_snapshot_2026-09-11.md)

## 1. 迁移边界

迁移对象是本项目的PyTorch公开代理实现，不是京东生产模型。GPU侧已经完成方法级
验收，但小样本没有建立稳定推荐收益。NPU阶段首先回答“相同输入、checkpoint和算法
是否得到可接受的数值与离散输出”，之后才评估吞吐、HBM或多卡扩展。

旧的`scripts/export_gpu_reference.py`只支持v1单商品冻结checkpoint：它会拒绝带
behavior词表的配置，也不包含`I_b`、历史行为ID、`[B,N,3]`列表目标和多商品生成。
本轮新增`scripts/export_v2_gpu_reference.py`，并将
`scripts/compare_device_reference.py`扩展到v2 Full协议，同时保留旧v1 reference读取分支；
v1 reference不能冒充v2证据。

## 2. 分阶段门槛

| 阶段 | 验证内容 | 通过证据 | 当前状态 |
|---|---|---|---|
| Stage-0A | 驱动/固件、CANN、PyTorch、TorchNPU、可见设备 | 环境JSON与`npu-smi info` | `[通过]`；8卡可见、HCCL可用，ATC精确版本待补 |
| Stage-0B | 单卡矩阵计算、backward、AdamW step、checkpoint保存恢复 | `OK stage=npu_single_card` | `[通过]`；梯度非零、参数更新、checkpoint一致 |
| Stage-0.5 | 迁移入口、依赖、平台调用和精度敏感API清单 | GPU/NPU各自静态清单；官方分析工具原始报告另存 | `[代码就绪]`；项目内清单不等于算子支持证明 |
| Stage-1 | v2固定batch的logits、weighted loss、greedy和beam GPU/NPU对齐 | 同commit/checkpoint/registry哈希；误差与离散匹配报告 | `[代码就绪-待实跑]` |
| Stage-2 | v2单batch训练步：全量梯度与参数delta对齐 | 梯度有限且误差受控 | `[并入固定参考协议-待实跑]` |
| Stage-3 | 20步短训练与恢复训练 | loss曲线、checkpoint恢复一致 | `[未开始]` |
| Stage-4 | 多卡可见性与HCCL通信 | 每卡独立计算、collective通过 | `[未开始]` |
| Stage-5 | 8卡吞吐、HBM、扩展效率 | 固定配置性能报告 | `[未开始]` |

任何阶段失败都先停在该阶段，记录首个可操作原因；静态检查不算目标NPU通过。

## 3. Stage-0执行

在昇腾服务器的目标Python环境、项目根目录中运行：

```bash
bash run_npu_stage0.sh
```

可选参数：

```bash
PYTHON_BIN=python3 NPU_DEVICE=npu:0 \
NPU_STAGE0_OUTPUT=checkpoints/npu_stage0 \
bash run_npu_stage0.sh
```

生成：

- `checkpoints/npu_stage0/npu_server_environment_snapshot_YYYY-MM-DD.json`：版本栈、设备可见性和安全白名单环境变量；
- `checkpoints/npu_stage0/npu_single_card_validation_YYYY-MM-DD.json`：单卡loss、梯度、参数更新和checkpoint结果；
- `checkpoints/npu_stage0/npu_single_card_validation_YYYY-MM-DD.pt`：保存恢复测试张量。

通过条件：`torch_npu_importable=True`、`npu_available=True`、设备数至少1，且末行包含
`OK stage=npu_single_card`、有限非零梯度、非零参数delta和
`checkpoint_match=True`。

## 4. Stage-0.5至Stage-2执行

首个v2设备对齐对象使用预训练Full checkpoint：

```text
checkpoints/retailrocket_v2_pretraining_ablation_smoke/full-epoch-1.pt
```

固定项包括checkpoint SHA-256、SID registry版本、temporal boundaries、validation
reservoir seed、history/target SID、历史行为ID、目标行为`I_b`、行为token权重、list
size和beam width。优先对齐顺序为：teacher-forcing logits → weighted loss → greedy
列表 → beam列表 → 单步梯度。EA-TOSD checkpoint在Full基线通过后再作为第二个对象，
避免同时引入设备误差和后训练差异。

先在GPU服务器以待迁移代码生成唯一参考包：

```bash
CUDA_VISIBLE_DEVICES=0 bash run_gpu_v2_migration_reference.sh
```

这一步同时生成：

- `support_inventory/gpu/v2_migration_inventory.json/.md`：项目内静态迁移清单；
- `gpu/v2_full_gpu_reference.pt`：固定batch、FP32 logits/loss、greedy/beam、全量梯度和一次AdamW参数delta；
- `gpu/v2_full_gpu_reference.json`：便于人工检查的摘要。

导出器要求Git工作区干净，并把参考包自身SHA-256写入JSON摘要。把`.pt`参考包原样传到
NPU服务器，并确保两端Git commit、关键源文件、Full checkpoint和SID registry的
SHA-256一致，再运行：

```bash
NPU_DEVICE=npu:0 bash run_npu_v2_migration_compare.sh
```

NPU入口会生成自己的静态清单和
`npu/v2_full_comparison.json`。默认连续量容差为`atol=rtol=5e-3`；greedy与beam SID
要求完全一致。报告会分别保留logits、loss、各层loss、beam score、梯度和参数delta
差异。若失败，先保留报告并定位首个差异，不立即调参或改成BF16。

`collect_v2_migration_inventory.py`只是可审计的源码预检查，不替代官方PyTorch
Analyse、msProbe或目标NPU实跑。若目标环境提供官方工具，原始报告需与本项目清单并列
归档；只有数值差异出现时才进入Module/API级下钻。

## 5. 当前已知环境与未验证项

- 已知：8×`Ascend950DT_9581`、PyTorch `2.10.0`、TorchNPU
  `2.10.0.post4.dev20260715`、CANN路径`9.0.T550`，`torch.npu`与HCCL接口可用；
- 未确认：ATC精确版本、驱动/固件完整版本以及该开发版软件栈的正式兼容矩阵；
- 未验证：BF16及OxygenREC使用的Transformer算子；
- v2模型在NPU上的数值误差、生成一致性、显存和吞吐；
- checkpoint是否能在目标环境直接恢复。

Stage-0基础链已经通过，但Stage-1/2目前只是代码就绪。在GPU参考包与NPU比较报告产生
前不安装/升级依赖、不启动多卡，也不声称OxygenREC已完成NPU兼容。
