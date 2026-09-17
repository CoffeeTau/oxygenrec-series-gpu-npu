# OxygenREC GPU→NPU迁移与验收计划

> 状态：Stage-0及v2 Full GPU/NPU FP32、BF16短训练保存恢复均已通过；下一步执行固定验证集精度比较。
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
| FP32模型训练 | Full checkpoint续训20步并保存、恢复、再次前向 | loss/梯度有限，checkpoint逐值恢复且可继续前向 | `[通过]`；两端同commit/输入，20步训练及保存恢复均通过 |
| BF16特性适配 | 在同一训练入口开启autocast | BF16 loss/梯度有限，checkpoint可保存恢复 | `[通过]`；两端20步训练、梯度及保存恢复均通过 |
| 固定验证集比较 | 同一冻结checkpoint、validation cohort和解码配置 | 列表指标接近，目标指纹一致，离散输出指纹可解释 | `[入口就绪-待服务器实跑]` |
| 精度调试 | 仅在固定验证差异异常时下钻Module/API/tensor | 找到首个超差模块/算子并形成处置结论 | `[条件未触发]`；大JSON仅作为诊断资产 |
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
export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}
source /usr/local/Ascend/driver/bin/setenv.bash
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
export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/cann/set_env.sh
NPU_DEVICE=npu:0 bash run_v2_training.sh npu bf16
```

FP32用于隔离设备迁移问题；BF16是单独的混合精度特性验证，不能用BF16绕过FP32失败。
GPU/NPU独立训练经过多步后不要求checkpoint逐值相等，先比较输入一致性、loss趋势、有限性、
保存恢复和验证指标。只有出现异常时才运行`run_v2_device_alignment.sh`或msProbe进行详细
数值下钻；现有大JSON探针不再是日常流程。

两端BF16短训练通过后，使用同一冻结Full checkpoint和固定32条validation列表分别执行
FP32、BF16正确性验证。GPU服务器执行：

```bash
git pull --ff-only origin main
CUDA_VISIBLE_DEVICES=0 bash run_v2_validation.sh gpu all
```

NPU服务器执行：

```bash
git pull --ff-only origin main
export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/cann/set_env.sh
NPU_DEVICE=npu:0 bash run_v2_validation.sh npu all
```

验证入口使用v2列表任务原生指标：SID recall、逐位置准确率、exact-list、beam exact-list
hit/MRR/NDCG、SID token准确率和合法SID率。摘要还记录target、greedy、beam三个稳定指纹；
指纹一致可直接证明离散结果一致，若不一致才提取详细案例。

服务器实跑前冻结如下工程验收门槛，避免根据结果事后调整标准：

- 硬门槛：commit、三个输入哈希、validation元数据及target指纹一致；logits有限；
  greedy/beam合法SID率均为`1.0`；
- 连续量：相同精度的GPU/NPU mean loss相对差不超过`1%`，同一设备BF16相对FP32也不超过`1%`；
- 离散聚合：SID recall和position accuracy允许最多一个目标SID的量化差，即`1/64=0.015625`；
  exact-list及beam exact-list hit允许最多一个列表的量化差，即`1/32=0.03125`；
- 离散指纹：greedy/beam指纹完全一致是最强通过证据；若指纹不同但聚合指标仍在上述范围内，
  不直接判失败，而是只下钻发生变化的固定案例。

为隔离旧eval融合算子的CPU fallback，正确性协议在GPU/NPU两端都关闭PyTorch MHA fastpath，
执行相同的非融合Transformer公式。该设置只用于本阶段的跨设备正确性比较，不等于最终部署
路径，也不能用于正式性能结论。生成四份小型摘要：

- `checkpoints/device_validation/v2_full/gpu/fp32/v2_full_gpu_fp32_validation_summary.json`；
- `checkpoints/device_validation/v2_full/gpu/bf16/v2_full_gpu_bf16_validation_summary.json`；
- `checkpoints/device_validation/v2_full/npu/fp32/v2_full_npu_fp32_validation_summary.json`；
- `checkpoints/device_validation/v2_full/npu/bf16/v2_full_npu_bf16_validation_summary.json`。

`collect_v2_migration_inventory.py`只是可审计的源码预检查，不替代官方PyTorch
Analyse、msProbe或目标NPU实跑。若目标环境提供官方工具，原始报告需与本项目清单并列
归档；只有数值差异出现时才进入Module/API级下钻。

## 5. 当前已知环境与未验证项

- 本次FP32短训练：`Ascend950DT_9572`、Python `3.11.6`、PyTorch `2.10.0+cpu`、
  TorchNPU `2.10.0.post5.dev20260910`；硬件仍为950DT，但TorchNPU版本已不同于旧环境快照，
  最终验收前需重新采集完整NPU环境信息；
- 未确认：ATC精确版本、驱动/固件完整版本以及该开发版软件栈的正式兼容矩阵；
- 已验证：Full checkpoint可在NPU加载；单batch FP32前向、生成、反向与AdamW step完成，loss为`6.092293`；
- 已验证：GPU/NPU使用commit `9633639329cc569c18b458683980ecef38d0c9f5`和完全相同的
  events、起始checkpoint及SID registry，分别完成20步FP32训练、梯度计算、checkpoint保存恢复与恢复后前向；
- FP32摘要：GPU/NPU平均loss分别为`5.707635`/`5.707190`，逐步loss平均绝对差
  `0.008988`，最大绝对差`0.023938`；这些结果通过短训练迁移门槛，但不等价于最终精度对齐；
- BF16摘要：两端autocast均启用为`bfloat16`，GPU/NPU平均loss分别为`5.707227`/`5.710366`，
  逐步loss平均绝对差`0.007879`，最大绝对差`0.013265`；两端所有loss/梯度有限且保存恢复通过；
- 已知问题：旧的eval单batch路径报告`aten::_transformer_encoder_layer_fwd`回退CPU；本次短训练摘要
  不含算子级执行位置，是否完全消除fallback仍需完整运行日志或Profiler确认；
- 环境前提：本次NPU运行曾因TorchNPU后端动态库未正确加载而失败，补充`/usr/lib64`并依次加载
  driver与CANN环境后成功；
- 固定验证服务器实跑：GPU/NPU 的 FP32 target、greedy、beam 指纹均一致；BF16
  target 与 greedy 一致但 beam 指纹不同。四次运行均通过输入、有限性、合法性、
  mean-loss 与聚合指标门槛；BF16 须按预定规则下钻变化案例。具体数值和边界见
  [`2026-09-17 v2固定验证记录`](../实验记录/案例原始记录/gpu_npu_alignment/2026-09-17-v2-fixed-validation.md)。
- 未验证：BF16 beam 逐案例差异原因、正式显存/吞吐和多卡。

当前标记为`[v2 Full FP32/BF16迁移短训练通过；FP32固定验证通过；BF16聚合指标通过、beam差异待定位]`。
独立设备训练使用各自随机数实现，输出checkpoint哈希不要求相同；只要求起始代码和输入一致、
训练量有限且保存恢复成功。不能把聚合指标相同扩展成BF16全部生成路径相同。CPU fallback
未确认前不进入正式性能调优。
