# OxygenREC 模型复现与 GPU/NPU 迁移说明

## 适配说明

本项目基于 OxygenREC 论文公开的方法，自实现 PyTorch 生成式推荐模型，使用 RetailRocket
公开数据构建代理实验，并开展 NVIDIA GPU 到昇腾 NPU 的迁移、数值对齐和性能调优。

本项目不是京东生产系统或论文私有数据指标的严格复现：工业特征、私有数据、Reward Service
及线上服务不可得，对应部分使用明确标注的公开代理实现。

### 当前状态（2026-10-22）

| 范围 | 状态与边界 |
| --- | --- |
| GPU 方法实现 | v1/v2 在公开代理范围内完成方法级验证；包含 SID、Encoder-Decoder、Instruction、Q2I、IGR、SA-GCPO，以及 v2 行为条件、列表生成和蒸馏路径，不代表已获得论文质量收益 |
| NPU 基础适配 | Stage-0 单卡张量、反向、优化器及 checkpoint 检查通过；设备可见或 HCCL 可用不等于多卡训练已验证 |
| 训练对齐 | v2 Full 的 GPU/NPU FP32、BF16 固定输入 20-step 训练和 checkpoint 重载门槛通过 |
| 固定验证集 | 输入、有限值、合法性、平均 loss 和聚合指标门槛通过；FP32 指纹一致，BF16 beam 指纹存在差异，不宣称逐位一致 |
| 性能调优 | 已建立单卡稳态基线、Profile 和单变量 A/B 流程；已测局部优化未形成满足采纳条件的方案 |
| TorchAir | A/B 入口已实现，尚无目标机成功收益结论 |
| FlashAttention | 下一阶段方向：按 Attention 分支归因，再评估定向替换；尚未实现为已验证优化。已有 Profile 出现 FA 算子，不能称为当前完全未启用 FA |
| 未覆盖范围 | 多卡扩展性能、MoE、工业部署及长期训练质量收益；全 epoch 入口存在不等于完整训练质量已验收 |

详细证据见[复现实验日志](实验记录/复现实验日志.md)。
较早的[复现进度](复现进度.md)用于方法追溯，最新性能状态以实验日志和对应记录为准。

## 代码结构说明

```text
src/oxygenrec/                 # 模型、数据、SID、解码、检索、奖励及设备适配
scripts/                      # 数据处理、训练、验证、迁移和性能工具
tests/                        # 单元测试与结果契约测试
configs/assumptions.yaml       # 公开代理协议及实现假设
docs/                         # 数据/模型协议与迁移说明
docs/server_commands/         # 完整服务器执行步骤及产物路径
实验记录/                     # 经审查允许留存的实验记录与分析
run_npu_stage0.sh              # NPU 环境与单卡功能检查
run_v2_training.sh            # 短训练及 checkpoint 重载验证
run_v2_validation.sh          # 固定验证集评测
run_v2_full_training.sh       # 从已有 checkpoint 继续全 epoch 训练
run_v2_performance_baseline.sh # 单卡稳态训练性能测试
run_v2_npu_profile.sh         # NPU 算子性能采集
```

主线数据流：

```text
行为日志 → 时间切分 → SID registry → 历史 SID 与指令条件
        → Encoder-Decoder → 受约束 SID 生成 → 商品候选与推荐指标
```

调用链见[代码阅读与数据流指南](代码阅读与数据流指南.md)。

## 运行环境准备

| 环境 | 用途 | 准备要求 |
| --- | --- | --- |
| 本地 Python / CPU | 基础测试与静态检查 | 标准库可运行基础测试；PyTorch 相关测试需相应依赖，跳过不代表通过 |
| NVIDIA GPU | 方法验证、迁移参考与基线 | 已有验证设备为 L20，使用与服务器 CUDA 兼容的 PyTorch |
| 昇腾 NPU | 训练迁移与性能调优 | 已有验证设备为 Ascend 950DT，使用配套驱动、固件、CANN、PyTorch 和 torch_npu |
| Qwen 可选路径 | 指令生成与特征缓存 | 另需匹配的 Transformers 环境及 Qwen 权重，不属于每轮性能测试的在线依赖 |

本仓库采用源码脚本入口，目前没有统一安装包或锁定依赖文件，不使用 `pip install -e .`。
应在经批准的服务器环境中运行，不按其他示例覆盖安装软件栈。
不同轮次存在不同环境版本，复测须记录实际版本，不把早期快照当作全部实验的统一配置。

- [服务器环境索引](docs/server_environment.md)
- [NPU 迁移计划](docs/npu_migration_plan.md)
- [Qwen 服务器接入操作指南](Qwen服务器接入操作指南.md)

基础测试在仓库根目录执行：

```bash
python3 -m unittest discover -s tests -v
```

测试通过仅代表当前环境覆盖的检查通过，不代替目标 GPU/NPU 验证。

## 数据与模型准备

仓库不分发数据、权重和运行产物。服务器执行前须准备：

| 资产 | 脚本默认路径 | 要求 |
| --- | --- | --- |
| RetailRocket 事件数据 | `data/raw/retailrocket/events.csv` | 在服务器内按批准流程准备，遵循时间切分及防泄漏协议 |
| SID registry | `data/processed/rq_comparison/w256_kmeanspp/sid_registry.json` | 与 checkpoint 的词表、层数和版本一致 |
| v2 Full checkpoint | `checkpoints/retailrocket_v2_pretraining_ablation_smoke/full-epoch-1.pt` | 迁移、验证和性能脚本的默认输入，不随代码提供 |

默认路径不是自动下载入口。缺少 registry/checkpoint 时，应先完成数据处理、SID 拟合和对应
v2 预训练，不能直接运行后续性能测试，也不能用随机资产替代固定对齐输入。

数据和训练流程见[数据协议](docs/data_protocol.md)、[SID 协议](docs/sid_protocol.md)、
[模型协议](docs/model_protocol.md)、[训练协议](docs/training_protocol.md)。

常用路径变量为 `EVENTS_PATH`、`SID_REGISTRY_PATH`。
短训练/验证使用 `V2_FULL_CHECKPOINT`，全 epoch/性能入口使用 `V2_FULL_TRAIN_CHECKPOINT`；
两者名称不同，请按具体入口配置。

## 运行方式与结果路径

以下为入口速查，不是要求逐项连续执行。完成输入准备后，按
[服务器执行命令索引](docs/server_commands/README.md)中的完整步骤运行。
每轮指定独立输出目录，避免覆盖旧结果；不要将早期 `tmp_command.txt` 当作当前执行入口。

| 任务 | 根目录入口 | 默认输出位置 |
| --- | --- | --- |
| NPU 单卡检查 | `bash run_npu_stage0.sh` | `checkpoints/npu_stage0/` |
| GPU 短训练 | `bash run_v2_training.sh gpu bf16` | `checkpoints/device_training/v2_full/gpu/bf16/` |
| NPU 短训练 | `bash run_v2_training.sh npu bf16` | `checkpoints/device_training/v2_full/npu/bf16/` |
| NPU 固定验证 | `bash run_v2_validation.sh npu bf16` | `checkpoints/device_validation/v2_full/npu/bf16/` |
| NPU 全 epoch 续训 | `bash run_v2_full_training.sh npu bf16` | `checkpoints/full_training/v2_full/npu/bf16/` |
| NPU 稳态基线 | `bash run_v2_performance_baseline.sh npu bf16` | `checkpoints/performance_baseline/v2_full/npu_bf16_performance.json` |

训练、验证和性能入口支持 `gpu`/`npu`，精度支持 `fp32`/`bf16`；验证还支持 `all`。
`PYTHON_BIN` 指定解释器，`CUDA_DEVICE`/`NPU_DEVICE` 指定设备。
上表采用脚本默认参数，**不等同于历史 batch4096 匹配实验配置**。
预热、计时、样本循环、重复次数、Profile 配置及完整结果文件路径，以对应日期命令文档为准。

## 精度、性能对比

以下汇总仓库已有历史实验，供评审参考，不代表当前版本重新实测。精度验证与性能测试为两次
独立实验，分别列出，避免把固定小样本指标误写为训练一个 epoch 后的质量。

### 精度参考（固定验证集，BF16）

| 设备 | 卡数 | 验证样本数 | Mean loss | SID recall | Greedy / Beam 合法率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVIDIA L20 | 1 | 32 | 5.818978 | 1.5625% | 100% / 100% |
| Ascend 950DT | 1 | 32 | 5.818490 | 1.5625% | 100% / 100% |

来源：[2026-09-17 固定验证记录](实验记录/案例原始记录/gpu_npu_alignment/2026-09-17-v2-fixed-validation.md)，
commit：`c42fb54`。两端采用相同 checkpoint、输入和 SID registry，每条样本含两个目标商品，
beam width 为 5；SID recall 为该固定验证协议下的召回指标。
BF16 聚合指标对齐，但 beam 指纹存在差异；本表不证明长期推荐质量收益。

### 性能参考（单卡稳态训练，BF16）

| 设备 | 卡数 | Local / Global batch | 吞吐量（samples/s，中位数） | Step 时延（ms，中位数） | 吞吐 CV | 单 epoch 耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| NVIDIA L20 | 1 | 4096 / 4096 | 26,687.41 | 153.48 | 6.63% | 未测 |
| Ascend 950DT | 1 | 4096 / 4096 | 36,859.30 | 111.13 | 0.75% | 未测 |

来源：[2026-09-21 长窗口复核](实验记录/案例原始记录/performance_tuning/2026-09-21-v2-bf16-batch-sweep.md)，
两端 commit：`fbe005a`，关键源码哈希一致；预热 100 步、测量 100 步、重复 3 次，开启样本循环。
该轮 NPU/GPU 吞吐比约 **1.38×**，仅代表这组硬件、软件栈和工作负载；GPU 波动较大，
不能将该倍率视为稳定保证，也不能用循环样本的短窗口吞吐推算完整 epoch 耗时。

### 精度验证

采用相同源码版本、checkpoint、SID registry 和固定输入，检查 loss、梯度、checkpoint
重载、生成合法性及推荐指标。训练功能通过、固定小样本对齐和长期推荐质量是不同结论。
证据见[迁移与验证记录](实验记录/案例原始记录/gpu_npu_alignment/)。

### 性能验证

当前主线为单卡 BF16 训练性能分析，不是 8 卡分布式吞吐测试。
记录吞吐量、step 时延、峰值 allocated memory、重复次数和 CV，并明确计时边界。
CV 是标准差与均值的比值，反映重复运行的相对波动；重复一次得到的零值不能证明稳定。

- 稳态性能测试与 Profile 分开执行，采集开销不能计作真实训练退化。
- 算子设备耗时占比不等于端到端训练时间占比。
- 单变量 A/B 保持设备、软件栈、输入、模型及其他训练配置可比。
- 融合 AdamW、私有格式及延后 loss 读取未达到相应采纳条件；旧 ByteMask Dropout 在目标机不支持，不能作为默认优化。
- TorchAir 和 FA 定向替换须经目标机验证，不能因入口存在就宣称加速。
- GPU/NPU 对比是特定硬件与软件栈的工作负载观测，不是通用硬件性能结论。

本首页仅列已有记录的聚合参考表，不附服务器原始日志、数据或环境明细；上库前仍须按组织要求审查。
详细结果按权限在服务器内查看，实验使用
[性能调优记录模板](实验记录/性能调优记录模板.md)，无收益和失败尝试同样保留。

## 数据安全与交付边界

服务器数据、终端输出、日志、JSON/CSV、Profile trace、样例及模型权重仅在服务器内查看、
分析和留存，不得传至个人电脑、外部聊天工具或未经授权的系统。
GPU/NPU 间的参考包流转也必须在经批准的内部环境和权限范围内进行。

匿名化或仅保留聚合指标不等于获得外发授权。历史实验文档与环境快照须单独审查后才能进入
目标代码库；忽略规则不代替交付检查，不直接打包整个工作目录或未经审查迁入完整 Git 历史。
代码规模及交付注意事项见[上库评审说明](docs/2026-10-22-code-intake-review.md)。

## 参考资料

- 方法与工程假设：[实现假设](configs/assumptions.yaml)、[复用调研](docs/reference_reuse.md)。
