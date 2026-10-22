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

详细证据见 `实验记录/` 下的案例原始记录和案例分析。

## 代码结构说明

上库内容以核心源码和实验分析记录为主：

```text
src/oxygenrec/   # 模型、数据处理、SID、解码、检索、奖励及设备适配
实验记录/       # 经审查允许留存的实验过程、结果分析和结论
```

主线数据流：

```text
行为日志 → 时间切分 → SID registry → 历史 SID 与指令条件
        → Encoder-Decoder → 受约束 SID 生成 → 商品候选与推荐指标
```

## 环境与输入要求

- GPU：已有验证设备为 NVIDIA L20，使用与 CUDA 兼容的 PyTorch。
- NPU：已有验证设备为 Ascend 950DT，使用配套的驱动、固件、CANN、PyTorch 和 torch_npu。
- Qwen 可选路径：需匹配的 Transformers 环境及模型权重，用于指令生成和特征缓存。
- 输入资产：RetailRocket 事件数据、与模型词表一致的 SID registry，以及相应 checkpoint；均需在服务器内按批准流程准备，不随代码提供。

不同实验轮次的软件环境和配置可能不同，复测须按 `实验记录/` 中的对应记录
核对版本、输入及计时协议。

## 快速运行

### 1. 保留运行入口并准备环境

目录介绍只突出核心内容。为保留数据重建、训练、验证和性能测试能力，交付时还须保留
`README.md`、`.gitignore` 及以下入口（相对路径不变）：

| 用途 | 必留文件 |
| --- | --- |
| 模型自检 | `scripts/validate_toy_model.py` |
| SID 与预训练资产重建 | `scripts/fit_retailrocket_sid.py`、`scripts/train_v2_pretraining_ablation_retailrocket.py`、`scripts/train_v2_listwise_retailrocket.py`（被预训练入口导入） |
| 短训练 | `run_v2_training.sh`、`scripts/train_v2_full_migration_smoke.py`（也被续训、性能入口导入） |
| 固定验证 | `run_v2_validation.sh`、`scripts/evaluate_v2_fixed_validation.py` |
| 全 epoch 续训 | `run_v2_full_training.sh`、`scripts/train_v2_full_epochs.py` |
| 稳态性能与比较 | `run_v2_performance_baseline.sh`、`scripts/benchmark_v2_training.py`、`scripts/compare_v2_performance_baselines.py` |
| NPU Profile 与汇总 | `run_v2_npu_profile.sh`、`scripts/summarize_v2_npu_profile.py` |

`src/oxygenrec/` 整包保留，测试目录建议保留用于回归。训练、验证和性能入口要求已知 Git
提交且已跟踪文件无修改；整理后应在目标代码库提交，再从干净工作树运行，无需迁入旧 Git 历史。

先进入仓库根目录，激活已配置好 PyTorch 的服务器 Python 环境。NPU 还需加载 CANN 环境；
以下为常见安装路径，若服务器路径不同，应按实际安装位置修改，不要重新安装或覆盖系统软件栈。

```bash
# 仅 NPU 环境执行
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/cann/set_env.sh
python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); assert torch.npu.is_available()"
```

GPU 环境执行：

```bash
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.version.cuda); assert torch.cuda.is_available()"
```

项目使用源码脚本运行，无需 `pip install -e .`。GPU 和 NPU 的 PyTorch 安装包不能混用；
NPU 的 PyTorch、torch_npu、CANN 必须使用配套版本。基础模型运行无需加载 Qwen，
只有指令生成/特征缓存路径需要额外的 Transformers 和 Qwen 权重。

### 2. 无数据模型自检（可选）

尚未准备真实数据时，可先用合成输入检查模型前向、反向、checkpoint 和解码：

```bash
python scripts/validate_toy_model.py --device cpu --steps 200
```

该命令不需要 RetailRocket 数据或已有 checkpoint，不代表 v2 Full 或 NPU 迁移验证通过。

### 3. 数据下载、处理与 checkpoint 准备

**数据不需要上库。** 如已删除原始文件，在经批准的服务器内从
[RetailRocket 官方数据页](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
重新下载并解压。数据页标注为 CC BY-NC-SA 4.0，企业使用前应确认用途符合数据许可和组织要求。
使用已安装并完成授权的 Kaggle CLI 时，可执行以下命令；凭据不得写入仓库：

```bash
mkdir -p data/raw/retailrocket
kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw/retailrocket --unzip
```

无法联网的服务器须使用批准的数据导入流程。处理入口需要 `events.csv`、
`item_properties_part1.csv` 和 `item_properties_part2.csv`；下载包中的 `category_tree.csv`
不是本轮 SID 拟合的必需输入。保留 CSV 原始列结构，无需手工随机切分。

**如果只删除原始数据、仍保留匹配的 registry 和 checkpoint，恢复 CSV 后直接使用下方路径配置，
不要重新拟合 SID。** 如果连处理结果和 checkpoint 都没有了，按以下顺序重建一套功能验证资产
（示例在 GPU 上运行；仅有 NPU 时可用 CPU 拟合 SID，但耗时更长）：

```bash
# NumPy 用于属性向量处理；需与已有 PyTorch 环境配套安装。
python -c "import numpy, torch"

python scripts/fit_retailrocket_sid.py --device cuda:0 --max-items 5000 --width 256 --initialization kmeans++ --output-dir data/processed/rebuilt_sid
python scripts/train_v2_pretraining_ablation_retailrocket.py --device cuda:0 --events data/raw/retailrocket/events.csv --sid-registry data/processed/rebuilt_sid/sid_registry.json --output-dir checkpoints/rebuilt_v2_pretraining --epochs 1 --max-train-samples 5000
```

SID 入口按全局时间跨度划分 80%/10%/10%，只选训练期商品，并使用训练截止时间前的属性构建
哈希向量和三层残差 K-means 编码；训练入口自动构建历史及每日列表样本。属性哈希是公开代理，
不等于论文工业语义特征。预训练入口会运行 `base`、`ib`、`full` 三组，并生成
`checkpoints/rebuilt_v2_pretraining/full-epoch-1.pt`。
重建是小规模功能流程，不保证复现历史结果或达到充分训练质量。不要混用新 registry 与旧 checkpoint，
GPU/NPU 比较必须使用同一套批准的资产。重复重建时换一个输出目录，保留旧结果。

重建后配置：

```bash
export PYTHON_BIN=python
export EVENTS_PATH=data/raw/retailrocket/events.csv
export SID_REGISTRY_PATH=data/processed/rebuilt_sid/sid_registry.json
export V2_FULL_CHECKPOINT=checkpoints/rebuilt_v2_pretraining/full-epoch-1.pt
export V2_FULL_TRAIN_CHECKPOINT="$V2_FULL_CHECKPOINT"
```

仅有 NPU 时，SID 拟合命令使用 `--device cpu`，预训练命令使用 `--device npu:0`；这条组合需在
目标环境验证。本节没有执行下载、处理或训练。

**已有历史资产时，使用下面这组配置替代上面的重建路径，二选一：**

以下默认路径对应现有 v2 Full 实验资产，请修改为服务器内的实际位置。
这些文件不随代码提供；缺少资产时，须先完成 SID 拟合和 v2 预训练，不能用合成自检的
checkpoint 替代。下面的训练命令是从已有 v2 Full checkpoint 恢复，而不是从零预训练。

```bash
export PYTHON_BIN=python
export EVENTS_PATH=data/raw/retailrocket/events.csv
export SID_REGISTRY_PATH=data/processed/rq_comparison/w256_kmeanspp/sid_registry.json
export V2_FULL_CHECKPOINT=checkpoints/retailrocket_v2_pretraining_ablation_smoke/full-epoch-1.pt
export V2_FULL_TRAIN_CHECKPOINT="$V2_FULL_CHECKPOINT"

# 三项均应检查通过；缺失时先准备资产，不继续执行训练。
test -f "$EVENTS_PATH" && test -f "$SID_REGISTRY_PATH" && test -f "$V2_FULL_CHECKPOINT"
```

### 4. 运行短训练和固定验证

下面以单卡 NPU、BF16 为例。短训练默认执行 20 步并检查 checkpoint 重载；
固定验证评测上一步配置的输入 checkpoint，**不是自动评测短训练输出的 checkpoint**。
所有代码块在同一终端按顺序执行。

```bash
export NPU_DEVICE=npu:0
export OXYGENREC_RUN_ROOT="checkpoints/quickstart/$(date +%Y%m%d_%H%M%S)"
export V2_TRAINING_ROOT="$OXYGENREC_RUN_ROOT/training"
export V2_VALIDATION_ROOT="$OXYGENREC_RUN_ROOT/validation"

bash run_v2_training.sh npu bf16 && bash run_v2_validation.sh npu bf16
```

GPU 运行时，将上面两条启动命令的 `npu` 改为 `gpu`，并设置 `CUDA_DEVICE=cuda:0`；
需要 FP32 时将 `bf16` 改为 `fp32`。结果仅在服务器内查看：

| 结果 | 路径（相对于本轮 `OXYGENREC_RUN_ROOT`） |
| --- | --- |
| NPU 短训练日志及结果 | `training/npu/bf16/`，终端日志为 `training.log` |
| NPU 固定验证摘要 | `validation/npu/bf16/v2_full_npu_bf16_validation_summary.json` |
| NPU 固定验证日志 | `validation/npu/bf16/validation.log` |

GPU 对应路径中的 `npu` 替换为 `gpu`；FP32 对应路径中的 `bf16` 替换为 `fp32`。

### 5. 继续训练一个 epoch（可选）

短训练和固定验证通过后，按需运行。下面仍从 `V2_FULL_TRAIN_CHECKPOINT` 指定的 checkpoint
开始，默认 batch 为 64；完整数据训练的耗时和资源需求高于短训练。

```bash
export V2_FULL_TRAIN_OUTPUT_ROOT="$OXYGENREC_RUN_ROOT/full_training"
export V2_FULL_TRAIN_EPOCHS=1
export V2_FULL_TRAIN_BATCH_SIZE=64
bash run_v2_full_training.sh npu bf16
```

日志、epoch checkpoint 和摘要保存在本轮目录下的 `full_training/npu/bf16/`。
这些快速运行命令用于功能验证和续训，不等同于下表 batch4096 的性能测试协议。

### 6. 单卡性能测试与 GPU/NPU 比较

先完成第 3 节输入配置，并确认代码已提交、工作树干净。以下公共配置需在 GPU/NPU 两侧分别设置；
两侧使用相同 commit、数据、registry 和 checkpoint，不要各自重新训练一份权重。

```bash
export V2_PERF_OUTPUT_ROOT="checkpoints/performance_baseline/manual_$(date +%Y%m%d_%H%M%S)"
export V2_PERF_BATCH_SIZES=4096
export V2_PERF_MAX_SAMPLES=130000
export V2_PERF_WARMUP_STEPS=100
export V2_PERF_MEASURED_STEPS=100
export V2_PERF_REPEATS=3
export V2_PERF_CYCLE_SAMPLES=1
export V2_PERF_OPTIMIZER=adamw
export V2_PERF_ZERO_GRAD_MODE=set_to_none
export V2_PERF_NPU_INTERNAL_FORMAT=default
export V2_PERF_NPU_DROPOUT_IMPLEMENTATION=standard
export V2_PERF_LOSS_HOST_READ_MODE=per_step
export V2_PERF_EXECUTION_MODE=eager
```

按服务器类型选择一条执行：

```bash
# GPU 服务器
CUDA_DEVICE=cuda:0 bash run_v2_performance_baseline.sh gpu bf16
```

```bash
# NPU 服务器
NPU_DEVICE=npu:0 bash run_v2_performance_baseline.sh npu bf16
```

结果分别为本轮 `V2_PERF_OUTPUT_ROOT` 下的 `gpu_bf16_performance.json` 和
`npu_bf16_performance.json`。该配置循环使用样本，避免预热和测量步数超过可用样本量；
若可用样本不足一个 batch，两侧同时降低 batch 并如实记录，不与历史 batch4096 直接比较。

将两份摘要放在经批准的内部可访问位置后，替换下面路径，在服务器内比较：

```bash
python scripts/compare_v2_performance_baselines.py /approved/path/gpu_bf16_performance.json /approved/path/npu_bf16_performance.json
```

比较结果打印到终端；比较器检查输入、workload、源码哈希和 commit。不满足内部流转权限时，
保留各侧结果，不将文件转至外部系统。新资产产生的是新基线，不保证得到下表历史倍率。

### 7. NPU 算子 Profile（按需）

沿用第 3 节输入配置，单独采集短窗口，不与无 Profile 的吞吐结果混比：

```bash
export V2_PROFILE_OUTPUT_ROOT="checkpoints/performance_profiling/manual_$(date +%Y%m%d_%H%M%S)"
export V2_PROFILE_BATCH_SIZE=4096
export NPU_DEVICE=npu:0
bash run_v2_npu_profile.sh
python scripts/summarize_v2_npu_profile.py "$V2_PROFILE_OUTPUT_ROOT" --output "$V2_PROFILE_OUTPUT_ROOT/profile_hotspots.json" --top-k 30
```

本轮目录下，`npu_bf16_profile_summary.json` 为运行摘要，`trace/` 为采集产物，
`profile_hotspots.json` 为算子与 kernel 热点汇总，终端也打印排名；全部留在服务器内分析。

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

