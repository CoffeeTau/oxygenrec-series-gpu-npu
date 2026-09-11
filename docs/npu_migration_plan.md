# OxygenREC GPU→NPU迁移与验收计划

> 状态：Stage-0代码已准备，目标Ascend环境尚未实测。  
> GPU方法基线：[`OxygenREC-v2 GPU方法复现验收报告`](../实验记录/案例分析/OxygenREC-v2%20GPU方法复现验收报告.md)

## 1. 迁移边界

迁移对象是本项目的PyTorch公开代理实现，不是京东生产模型。GPU侧已经完成方法级
验收，但小样本没有建立稳定推荐收益。NPU阶段首先回答“相同输入、checkpoint和算法
是否得到可接受的数值与离散输出”，之后才评估吞吐、HBM或多卡扩展。

现有`scripts/export_gpu_reference.py`和`scripts/compare_device_reference.py`只支持
v1单商品冻结checkpoint：前者会拒绝带behavior词表的配置，也不包含`I_b`、历史行为
ID、`[B,N,3]`列表目标和多商品生成。因此它们不能作为v2对齐证据。Stage-0通过后需
新增v2专用固定输入reference，不能绕过该差异复用v1结论。

## 2. 分阶段门槛

| 阶段 | 验证内容 | 通过证据 | 当前状态 |
|---|---|---|---|
| Stage-0A | 驱动/固件、CANN、PyTorch、TorchNPU、可见设备 | 环境JSON与`npu-smi info` | `[待目标机]` |
| Stage-0B | 单卡矩阵计算、backward、AdamW step、checkpoint保存恢复 | `OK stage=npu_single_card` | `[待目标机]` |
| Stage-1 | v2固定batch的logits、weighted loss、greedy和beam GPU/NPU对齐 | 同checkpoint哈希；误差与离散匹配报告 | `[未开始]` |
| Stage-2 | v2单batch训练步：关键梯度与参数delta对齐 | 梯度有限且误差受控 | `[未开始]` |
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

- `checkpoints/npu_stage0/environment.json`：版本栈、设备可见性和安全白名单环境变量；
- `checkpoints/npu_stage0/single_card.json`：单卡loss、梯度、参数更新和checkpoint结果；
- `checkpoints/npu_stage0/single_card.pt`：保存恢复测试张量。

通过条件：`torch_npu_importable=True`、`npu_available=True`、设备数至少1，且末行包含
`OK stage=npu_single_card`、有限非零梯度、非零参数delta和
`checkpoint_match=True`。

## 4. Stage-1预定冻结对象

首个v2设备对齐对象使用预训练Full checkpoint：

```text
checkpoints/retailrocket_v2_pretraining_ablation_smoke/full-epoch-1.pt
```

固定项包括checkpoint SHA-256、SID registry版本、temporal boundaries、validation
reservoir seed、history/target SID、历史行为ID、目标行为`I_b`、行为token权重、list
size和beam width。优先对齐顺序为：teacher-forcing logits → weighted loss → greedy
列表 → beam列表 → 单步梯度。EA-TOSD checkpoint在Full基线通过后再作为第二个对象，
避免同时引入设备误差和后训练差异。

## 5. 当前未验证项

- 目标服务器硬件型号、卡数、驱动/固件和CANN版本；
- PyTorch与TorchNPU版本是否属于兼容矩阵；
- `torch.npu`、HCCL、BF16及Transformer算子是否可用；
- v2模型在NPU上的数值误差、生成一致性、显存和吞吐；
- checkpoint是否能在目标环境直接恢复。

在取得Stage-0输出之前，不安装/升级依赖，不启动多卡，也不声称NPU兼容。
