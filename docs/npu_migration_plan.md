# OxygenREC GPU→NPU迁移与验收计划

> 状态：Stage-0与v2 Full单batch NPU功能冒烟已通过；短训练、模型保存恢复和BF16仍待目标服务器验证。
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

## 2. 按官方流程划分的门槛

| 阶段 | 验证内容 | 通过证据 | 当前状态 |
|---|---|---|---|
| Stage-0A | 驱动/固件、CANN、PyTorch、TorchNPU、可见设备 | 环境JSON与`npu-smi info` | `[通过]`；8卡可见、HCCL可用，ATC精确版本待补 |
| Stage-0B | 单卡矩阵计算、backward、AdamW step、checkpoint保存恢复 | `OK stage=npu_single_card` | `[通过]`；梯度非零、参数更新、checkpoint一致 |
| 迁移分析 | 依赖、平台调用、算子支持和已知不支持场景 | 项目静态清单、目标机运行信息；官方分析报告另存 | `[部分完成]`；单batch发现Transformer融合算子CPU fallback，官方分析工具待补 |
| 模型迁移 | `torch_npu`注册、`npu`设备选择、模型与数据移动 | 同一代码可选择`cuda`或`npu`，不含目标路径CUDA硬编码 | `[v2 Full主链完成]`；不代表其他实验脚本全部完成 |
| FP32模型训练 | Full checkpoint续训20步并保存、恢复、再次前向 | loss/梯度有限，checkpoint逐值恢复且可继续前向 | `[入口就绪-待服务器实跑]` |
| BF16特性适配 | 在同一训练入口开启autocast | BF16 loss/梯度有限，checkpoint可保存恢复 | `[代码就绪-FP32通过后执行]` |
| 精度调试 | 比较GPU/NPU训练摘要；异常时下钻Module/API/tensor | 输入一致、loss趋势和验证指标误差可解释 | `[未开始]`；单batch大JSON仅作为诊断资产 |
| 多卡与性能 | HCCL训练、吞吐、HBM与扩展效率 | 多卡正确性与固定配置性能报告 | `[未开始]` |

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

## 4. v2 Full模型迁移与训练执行

首个迁移对象只选择预训练Full checkpoint：

```text
checkpoints/retailrocket_v2_pretraining_ablation_smoke/full-epoch-1.pt
```

EA-TOSD、Qwen、多卡和性能优化均不与本阶段同时展开，避免把多种迁移问题混在一起。
Full训练脚本通过统一设备层选择GPU/NPU；`model.py`保持平台无关，不加入设备分支。

日常主流程只维护`main`。MacBook推送一次后，两台服务器分别快进同步同一份代码；
不复制代码目录、不维护GPU/NPU分支，也不在两台服务器之间传递参考包。

先在GPU服务器执行FP32短训练：

```bash
git pull --ff-only origin main
CUDA_VISIBLE_DEVICES=0 bash run_v2_training.sh gpu fp32
```

再在NPU服务器执行同配置FP32短训练：

```bash
git pull --ff-only origin main
source /usr/local/Ascend/cann/set_env.sh
NPU_DEVICE=npu:0 bash run_v2_training.sh npu fp32
```

统一入口默认从相同Full checkpoint继续训练`20`步，并验证模型与优化器保存恢复。日常只需
检查终端最后一行和小型摘要：

- `checkpoints/device_training/v2_full/gpu/fp32/v2_full_gpu_fp32_training_summary.json`；
- `checkpoints/device_training/v2_full/npu/fp32/v2_full_npu_fp32_training_summary.json`。

摘要只包含代码与输入哈希、20个loss、首末梯度、耗时、显存和checkpoint恢复结果。
checkpoint与完整`training.log`保留在服务器，不作为日常交付内容。步数和batch可通过
`V2_TRAINING_STEPS`、`V2_TRAINING_BATCH_SIZE`覆盖。

两端FP32均通过后，再验证BF16 autocast：

```bash
CUDA_VISIBLE_DEVICES=0 bash run_v2_training.sh gpu bf16
NPU_DEVICE=npu:0 bash run_v2_training.sh npu bf16
```

FP32用于隔离设备迁移问题；BF16是单独的混合精度特性验证，不能用BF16绕过FP32失败。
GPU/NPU独立训练经过多步后不要求checkpoint逐值相等，先比较输入一致性、loss趋势、有限性、
保存恢复和验证指标。只有出现异常时才运行`run_v2_device_alignment.sh`或msProbe进行详细
数值下钻；现有大JSON探针不再是日常流程。

`collect_v2_migration_inventory.py`只是可审计的源码预检查，不替代官方PyTorch
Analyse、msProbe或目标NPU实跑。若目标环境提供官方工具，原始报告需与本项目清单并列
归档；只有数值差异出现时才进入Module/API级下钻。

## 5. 当前已知环境与未验证项

- 已知：8×`Ascend950DT_9581`、PyTorch `2.10.0`、TorchNPU
  `2.10.0.post4.dev20260715`、CANN路径`9.0.T550`，`torch.npu`与HCCL接口可用；
- 未确认：ATC精确版本、驱动/固件完整版本以及该开发版软件栈的正式兼容矩阵；
- 已验证：Full checkpoint可在NPU加载；单batch FP32前向、生成、反向与AdamW step完成，loss为`6.092293`；
- 已知问题：`aten::_transformer_encoder_layer_fwd`在本次eval路径回退CPU，CANN/HDK也输出版本相关提示；
- 未验证：真实`model.train()`路径是否仍有CPU fallback、BF16、短训练保存恢复、显存和吞吐。

当前只能标记为`[v2 Full单batch NPU功能冒烟通过]`。FP32短训练及保存恢复通过前不称为
模型迁移完成；两端训练摘要尚未比较前不进入正式精度结论；CPU fallback未处理前不进入性能调优。
