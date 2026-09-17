# OxygenREC GPU → Agentic Search 研究基线

这是从 `oxygenrec-series-gpu-npu` 中抽出的、可独立复制的 GPU 研究子项目。它保留
OxygenREC-v1/v2 已经验证过的公开方法链路，并把后续工作重心放在可迁移到
Agentic Search 的 reasoning、retrieval、trajectory 与 post-training 能力上。

源快照：父项目 commit `c42fb54a28feddcad2056edcd6fb9ea03685d361`。

## 项目边界

本项目是：

- 基于 RetailRocket 等公开数据的 OxygenREC 论文方法自实现；
- SID、约束生成、Contextual Reasoning、Q2I、IGR、检索计划、SA-GCPO、
  v2 listwise pretraining 与 EA-TOSD 的 GPU 实验基线；
- 为 Agentic Search 新实验准备的可复用代码和已知问题交接。

本项目不是：

- 京东私有数据、特征、Reward Service 或线上系统的严格复现；
- 已经证明推荐质量或工业收益的模型；
- 从父项目复制出来的 NPU 迁移分支；
- 可以直接对外发布的成品包。父仓库目前没有独立 LICENSE，公开发布前需要重新审计
  代码、数据和模型权重许可。

详细结论与后续路线见 [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md)。

## 目录

```text
src/oxygenrec/         核心模型、SID、Reasoning、Retrieval、RL/轨迹模块
scripts/               GPU训练、评测、Qwen缓存与Agentic检索实验入口
tests/                 结构、数据边界、损失、生成和检索计划单元测试
docs/                  数据、SID、模型、训练和Qwen协议
实验记录/案例分析/      v1/v2 GPU方法复现的原始验收结论
```

数据、Qwen 权重、checkpoint 和服务器输出未复制。建议在新项目中继续使用以下目录：

```text
data/raw/              原始公开数据
data/processed/        SID registry、属性向量和固定cohort
models/                本地LLM权重，不提交Git
checkpoints/           训练权重，不提交Git
outputs/               评测、案例和轨迹输出
```

## 环境

先根据目标 GPU 驱动安装匹配的 CUDA 版 PyTorch，不要让普通 `pip install torch`
意外替换服务器现有版本。其余依赖分为：

```bash
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-qwen.txt  # 仅Qwen/LoRA实验需要
```

父项目最近的服务器运行环境是 NVIDIA L20、Python 3.12.3 和 PyTorch 2.11 开发版；
这些是实验来源信息，不是本子项目强制锁定的通用兼容矩阵。复制到新环境后先重新记录
GPU、驱动、CUDA、Python、PyTorch 和 Transformers 版本。

## 第一次验证

MacBook 或无 PyTorch 环境可先跑依赖无关测试；PyTorch 测试会按测试自身规则跳过：

```bash
python3 -m unittest discover -s tests -v
```

GPU 服务器安装好 CUDA 版 PyTorch 后，再执行：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/validate_toy_model.py --device cuda --steps 200
CUDA_VISIBLE_DEVICES=0 python scripts/validate_v2_behavior_instruction.py --device cuda
CUDA_VISIBLE_DEVICES=0 python scripts/validate_v2_listwise_generation.py --device cuda
```

真实数据主线默认需要：

```text
data/raw/retailrocket/events.csv
data/processed/rq_comparison/w256_kmeanspp/sid_registry.json
```

先拟合/确认 SID，再运行 v2 GPU 基线：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/fit_retailrocket_sid.py --help
CUDA_VISIBLE_DEVICES=0 bash run_server_test.sh
```

`run_server_test.sh` 是固定预算的 v2 预训练消融入口，不代表完整质量训练。

## Agentic Search 起点

当前代码已把两条检索路径显式分开：

- `paper_igr`：OxygenREC 论文主线，instruction query 对长历史向量做余弦 Top-K；
- `agentic_plan`：在相同 query 和 semantic scores 上执行行为、时效、重复和多样性约束。

因此第一组新实验不需要重写模型，而应固定数据、Qwen 输出、instruction feature、
OxygenREC checkpoint 和候选集，只切换 retrieval mode，比较召回、最终列表、轨迹合法性、
延迟与调用成本。相关入口：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/validate_retrieval_plan_execution.py --help
CUDA_VISIBLE_DEVICES=0 python scripts/validate_qwen_dual_retrieval_modes.py --help
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_qwen_plan_retailrocket.py --help
```

不要直接把当前固定 32 条 smoke cohort 的结果包装成效果提升；新方向的主实验需要独立
validation/test、多 seed、paired baseline 和可审计轨迹。
