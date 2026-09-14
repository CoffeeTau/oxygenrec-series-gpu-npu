# OxygenREC GPU→NPU迁移与验收计划

> 状态：Stage-0已于2026-09-11在8×Ascend 950DT服务器实测通过；GPU/NPU两端独立探针入口已就绪，待同一`main`版本实跑比较。
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
| Stage-0.5 | 迁移入口、依赖、平台调用和精度敏感API清单 | GPU/NPU各自静态清单；官方分析工具原始报告另存 | `[GPU清单已生成-NPU待采集]`；项目内清单不等于算子支持证明 |
| Stage-1 | v2固定batch的logits、weighted loss、greedy和beam GPU/NPU对齐 | 两端独立JSON的commit/输入哈希一致；误差与离散匹配报告 | `[统一入口就绪-待两端实跑]` |
| Stage-2 | v2单batch训练步：梯度与参数delta初筛 | 两端逐张量统计量和固定采样点误差受控 | `[紧凑探针就绪-待两端实跑]` |
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

日常主流程只维护`main`。MacBook推送一次后，两台服务器分别快进同步同一份代码；
不复制代码目录、不维护GPU/NPU分支，也不在两台服务器之间传递参考包。

GPU服务器运行：

```bash
git pull origin main
CUDA_VISIBLE_DEVICES=0 bash run_v2_device_alignment.sh gpu
```

NPU服务器运行：

```bash
git pull origin main
NPU_DEVICE=npu:0 bash run_v2_device_alignment.sh npu
```

两端入口会生成各自的静态清单以及结构相同的结果：

- `support_inventory/gpu/v2_migration_inventory.json/.md`：项目内静态迁移清单；
- `support_inventory/npu/v2_migration_inventory.json/.md`：NPU环境中的静态迁移清单；
- `gpu/v2_full_gpu_probe.json`：GPU独立探针结果；
- `npu/v2_full_npu_probe.json`：NPU独立探针结果。

探针要求Git commit可定位且已跟踪文件无修改，并在JSON中记录commit、关键源文件、
events、Full checkpoint与SID registry的SHA-256。比较前先核验这些前置项完全一致，
再直接比较完整teacher-forcing logits、loss、greedy/beam SID和beam scores；梯度与参数
delta采用逐张量统计量和固定采样点进行首轮筛查。

取得两份JSON后，可在不需要PyTorch的环境运行：

```bash
python3 scripts/compare_v2_device_probes.py \
  --gpu v2_full_gpu_probe.json \
  --npu v2_full_npu_probe.json \
  --output v2_full_gpu_npu_comparison.json
```

默认连续量容差为`atol=rtol=5e-3`，greedy与beam SID要求完全一致。若输入哈希不一致，
先修复服务器资产；若前向或训练步出现差异，再使用已有
`run_gpu_v2_migration_reference.sh`与`run_npu_v2_migration_compare.sh`执行逐tensor严格诊断。
严格参考包流程是异常下钻工具，不再是日常必经步骤。失败时先保留报告并定位首个差异，
不立即调大容差、改成BF16或开始调参。

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

Stage-0基础链与GPU固定参考已经通过，但新统一入口的两端数值比较尚未执行。在比较报告通过前
不安装/升级依赖、不启动短训练或多卡，也不声称OxygenREC已完成NPU兼容。
