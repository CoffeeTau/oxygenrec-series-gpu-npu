# OxygenREC 系列模型复现项目交接

更新日期：2026-08-17  
适用范围：新建独立项目，复现 OxygenREC-v1，并在此基础上逐步实现 OxygenREC-v2。本文只讨论 OxygenREC 系列。

## 1. 项目目标与准确表述

本项目的目标不是复刻京东生产系统的论文主表和线上收益，而是：

1. 根据论文自行实现 OxygenREC 系列的核心模型结构、训练目标和推理链路；
2. 在公开数据上构造可审计的近似 benchmark，先在 GPU 上建立参考结果；
3. 将同一实现迁移到 NPU，完成精度、性能和扩展效率对比；
4. 先完成可运行的 OxygenREC-v1 核心版本，再逐层升级为 OxygenREC-v2；
5. 条件成熟后，最终目标规模为 3B 总参数、约 1B 激活参数的 MoE。

对外应使用以下表述：

> OxygenREC 论文方法自实现 + 公开数据近似 benchmark + GPU/NPU 迁移与性能评测。

不得称为：

- 京东生产模型严格复现；
- OxygenREC 主表复现；
- 京东线上 UCTCVR/GMV 收益复现。

原因是原始工业训练数据、完整特征体系、线上流量、生产模型代码、checkpoint 和部署环境均未公开。

## 2. 必读论文

- OxygenREC-v1：**OxygenREC: An Instruction-Following Generative Framework for E-commerce Recommendation**  
  https://arxiv.org/abs/2512.22386
- OxygenREC-v2：**OxygenREC-v2: Internalizing Discrimination into Generative Recommendation**  
  https://arxiv.org/abs/2607.24255

新项目开始时应重新下载最新版本论文，并把论文版本号、提交日期和本地 PDF 校验信息记录在 `docs/paper_notes.md`。后续所有规模、表格和超参数结论都应注明论文版本和表号。

## 3. 已确认且必须沿用的事实

### 3.1 模型规模纠错

这是当前最重要的纠错项：

- OxygenREC-v2 的离线研究和主表使用 **3B encoder-decoder backbone**；
- 主表中的 OxygenREC-v1（PT-only）、Proxy-RM 和 OxygenREC-v2 使用相同的 3B backbone；
- OxygenREC-v2 线上部署配置是 **3B 总参数、1B 激活参数的 MoE（3B-A1B）**；
- **0.7B 总参数、0.4B 激活参数**属于 OxygenREC-v1 的 scaling ablation，不是 OxygenREC-v2 离线主表配置。

因此：

> 0.7B 可以作为缩小规模的方法验证，但不能称为 OxygenREC-v2 主表同规模复现。

旧项目 `screening_round4_2026-08-11.md` 中“v2 离线消融主要为 0.7B”的描述已经失效，新项目必须以上述纠正结论为准。

### 3.2 OxygenREC-v1 的角色

OxygenREC-v1 是更合理的起点。其核心思路是 Fast-Slow Thinking：

- Slow 路径使用近线 LLM 生成 Contextual Reasoning Instructions；
- Fast 路径使用高效 encoder-decoder 生成推荐结果；
- Instruction-Guided Retrieval（IGR）筛选与当前意图相关的历史行为；
- Query-to-Item（Q2I）目标用于增强 instruction 与商品之间的语义一致性；
- 多场景信息被组织为可控 instruction；
- 完整论文还包含面向多业务目标的偏好对齐方法。

复现时不应一开始实现所有工业组件。第一版只需保证 SID 生成推荐闭环正确。

### 3.3 OxygenREC-v2 相对 v1 的增量

OxygenREC-v2 保持统一推荐 backbone，重点把点击、加购、下单等行为信号内化到生成过程：

1. **Behavior instruction**：目标行为作为 decoder prefix，从第一个生成步骤开始改变候选分布；
2. **Behavior-aware pre-training**：根据日志中的真实行为构造、筛选和加权训练目标；
3. **Verifiable trajectory optimization**：使用可验证的生成轨迹信号进行后训练；
4. **Privileged self-distillation**：使用用户未来交互行为作为训练期特权信息；
5. **Entropy-aware routing/distillation**：控制特权知识蒸馏带来的偏差；
6. 推理阶段仍保持单一统一 backbone，不额外依赖外部排序 reward model。

v2 论文主表离线指标包括 HR@1、HR@512、Recall@512、NDCG@512、MRR@512 和 GAUC。公开数据复现可以采用同名或近似定义，但必须重新声明候选集、数据切分、SID 码本、beam size、过滤规则和行为标签，绝不能直接与私有主表绝对数值横向比较。

## 4. 推荐实施路线

### Phase 0：论文规格与最小设计冻结

交付物：

- 两篇论文的逐表、逐公式笔记；
- `paper_version / experiment_table / total_params / active_params` 参数证据表；
- 数据字段、SID、训练阶段、推理和指标的数据流图；
- 未披露信息与工程假设清单。

原则：论文没有披露的配置不得伪装成论文事实。自选配置统一放入 `configs/assumptions.yaml` 或等价文件，并记录理由。

### Phase 1：最小 OxygenREC-v1 核心闭环

先实现：

```text
公开行为日志
  -> 时间切分
  -> 商品 SID tokenizer/codebook
  -> 用户历史 SID 序列
  -> encoder-decoder 生成模型
  -> 自回归或约束解码
  -> SID 还原为商品
  -> HR@K / Recall@K
```

建议先使用 100M～300M dense 模型完成闭环，不要直接启动 3B MoE。

最低验收条件：

- 数据切分无未来信息泄漏；
- SID 码本可保存、加载和稳定复用；
- 模型 loss 能稳定下降；
- 生成结果的合法 SID 比例可统计；
- HR/Recall 明显优于随机基线；
- checkpoint 可恢复训练并复现实验；
- 单 batch 训练和评测均有确定性/误差记录。

### Phase 2：补充 OxygenREC-v1 关键机制

建议按以下顺序增加：

1. Contextual Reasoning Instruction 的离线构造或模板化替代；
2. Instruction-Guided Retrieval；
3. Q2I alignment loss；
4. 多场景/多行为 instruction；
5. 论文中的偏好对齐阶段。

每加入一个模块都保留独立配置和消融结果。不要把多个机制一次性合入，否则出现 loss 或指标异常时无法定位原因。

### Phase 3：升级到 OxygenREC-v2 预训练

在 v1 PT-only checkpoint 上增加：

- click / cart / order 等行为词表；
- behavior instruction decoder prefix；
- 行为感知的目标扩展、过滤和价值加权；
- 分行为及总体 HR/Recall/NDCG/MRR/GAUC 评测。

优先验证：同一用户上下文在不同 behavior instruction 下是否产生不同且合理的候选分布。

### Phase 4：升级到 OxygenREC-v2 后训练

依次实现：

1. 可验证 trajectory reward；
2. rollout 与轨迹采样；
3. 使用未来行为的 privileged teacher；
4. on-policy self-distillation；
5. entropy-aware token 选择或路由；
6. 与 Proxy-RM 型外部奖励基线比较。

这一阶段必须单独核算 actor、reference/teacher、rollout engine、KV cache、optimizer state 和 activation 的同时驻留开销。不能用普通 SFT 的显存估算代替。

### Phase 5：规模化与 MoE

推荐规模顺序：

1. 100M～300M dense：验证数据和算法闭环；
2. 0.7B 左右：验证扩大规模、分布式训练和主要机制；
3. 3B-A1B MoE：作为最终对齐论文目标规模的工程实验。

只有第三档可称为“与 v2 主表目标规模一致”，但数据和工业协议不同，因此仍不是主表严格复现。

## 5. 公开数据方案

原论文使用京东私有工业日志。当前建议优先评估：

- **RetailRocket**：天然包含 view、add-to-cart、transaction，适合构造行为 instruction；
- **天池淘宝用户行为数据**：可补充大规模点击、收藏、加购、购买行为；
- 必要时再引入其他公开电商序列数据，但第一版不要混合多个数据源。

推荐第一版只选一个数据集，完成：

- 按时间划分 train/validation/test；
- 用户历史截断长度，例如短历史 20、长历史 256；
- 商品冷启动和低频过滤规则；
- 多行为目标定义；
- SID tokenizer 训练集边界；
- 候选全集和评测过滤规则。

SID 可以采用层级聚类、RQ-VAE/RQ-KMeans 或其他可实现方案，但必须把它标注为公开复现选型，而不是声称为论文未披露的官方实现。

## 6. 指标与可比性边界

至少记录：

- HR@1、HR@K；
- Recall@K；
- 合法 SID 比例；
- 必要时记录 NDCG@K、MRR@K、GAUC；
- 分 click/cart/order 的指标；
- loss、吞吐、时延、峰值显存和扩展效率。

同名指标不等于数值可比。报告中必须同时列出：

- 数据集和时间范围；
- 数据切分；
- 候选池；
- SID 码本与碰撞处理；
- beam size；
- 重复商品和非法 SID 过滤；
- ground-truth 定义；
- 是否按用户或曝光计算。

公开数据结果只与本项目自己的 baseline 和消融比较。论文主表只作为方法背景，不作为目标数值。

## 7. GPU -> NPU 迁移路线

设备路线：

1. 在 8×NVIDIA L20 上建立 PyTorch/CUDA 参考；
2. 单卡 GPU 固定数据、随机种子、loss 和输出样例；
3. 单卡 NPU 对齐 forward、backward、loss 和生成结果；
4. 再扩展到多卡 NPU；
5. 最后比较 8 卡吞吐、时延、HBM 和扩展效率。

重点风险：

- encoder-decoder attention、causal/cross mask；
- 变长序列和高效 attention；
- beam search、约束解码和 KV cache；
- 大规模 embedding/SID 词表；
- MoE expert routing、load balance 和 all-to-all；
- BF16/FP16 数值误差；
- rollout 和训练引擎之间的数据交换；
- checkpoint 在 dense/MoE、GPU/NPU 之间的格式一致性。

精度对齐顺序应为：固定输入的 logits -> loss -> 单步梯度 -> 短训练曲线 -> 离线指标。只比较最终 HR@K 无法定位迁移误差。

## 8. 建议复用的组内经验

优先联系：

- **OneRec 负责人**：SID、NTP、约束解码、beam search、HR/Recall 和偏好对齐；
- **GR vLLM 负责人**：生成推理、KV cache、吞吐和解码引擎；
- **已确认来源后的 GR 3B 负责人**：3B 训练、并行、checkpoint 和混合精度。当前“GR 3B”只是内部清单名称，正式论文和配置仍需向负责人确认；
- **MMOE/PLE 负责人**：MoE routing、专家负载和通信经验，但传统 MMOE/PLE 不等同于 LLM MoE；
- **HLLM/长序列模型负责人**：长序列 attention、mask、分布式训练；
- **MTGR 负责人**：HSTU、特征 token 化、序列压缩和大规模训练引擎。MTGR 是判别式 CTR/CTCVR 排序，不是 OxygenREC 式 SID 自回归生成器。

向负责人询问时，应索取具体仓库、配置和复现报告，而不是只问抽象经验。

## 9. 新项目建议目录

```text
oxygenrec-reproduction/
├── README.md
├── configs/
│   ├── data/
│   ├── model/
│   ├── train/
│   └── assumptions.yaml
├── docs/
│   ├── paper_notes.md
│   ├── reproduction_scope.md
│   ├── metric_protocol.md
│   └── gpu_npu_alignment.md
├── src/
│   ├── data/
│   ├── tokenizer/
│   ├── models/
│   ├── decoding/
│   ├── training/
│   ├── post_training/
│   └── evaluation/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── parity/
├── scripts/
└── artifacts/
```

代码边界建议：SID tokenizer、推荐 backbone、解码器、指标和后训练独立成模块，避免后续 v2 机制全部耦合进一个训练脚本。

## 10. 第一周执行清单

1. 固定两篇论文版本并完成逐表笔记；
2. 建立 `paper fact / engineering assumption / unknown` 三列表；
3. 选定 RetailRocket 或天池淘宝之一，不混用；
4. 明确时间切分、行为标签、历史长度和候选池；
5. 实现并测试 SID 训练、保存、加载、编码和解码；
6. 用小型 dense encoder-decoder 跑通单 batch overfit；
7. 实现 HR@K、Recall@K 和合法 SID 比例；
8. 保存固定 batch 的 GPU logits/loss，作为后续 NPU 对齐基准；
9. 联系 OneRec/GR 相关负责人，获取可复用的生成训练与解码基础设施；
10. 在上述闭环通过前，不启动 3B MoE 或 v2 自蒸馏。

## 11. 完成定义

项目不以“跑完脚本”为完成。建议分级验收：

- **M1：v1 核心可用**——公开数据、SID、生成、解码、HR/Recall 闭环通过；
- **M2：v1 方法完整度提高**——instruction、IGR、Q2I 和至少一个偏好对齐机制完成消融；
- **M3：v2 预训练完成**——行为条件生成相对 PT-only 有稳定收益；
- **M4：v2 后训练完成**——trajectory optimization、privileged self-distillation 和 entropy-aware 策略可独立验证；
- **M5：双平台完成**——GPU/NPU 精度对齐并有可复现的性能报告；
- **M6：规模化完成**——3B-A1B MoE 在目标设备上达到稳定训练或推理状态。

如果最终只完成 M1～M3，也应如实报告为缩小规模方法复现，不把未完成的 3B MoE 和工业后训练写入成果。

## 12. 当前待确认问题

- OxygenREC-v1/v2 是否出现官方代码、补充材料或更新版本；
- 论文各阶段完整模型超参数和 MoE 配置是否足够重建；
- SID tokenizer 的官方细节能否从论文附录完全确定；
- Slow LLM 生成 instruction 的提示模板、过滤和缓存协议；
- v2 privileged teacher 的部署级计算开销；
- 公开数据上行为价值权重如何设定；
- 3B-A1B 在目标训练框架中的 expert parallel 和并行组合；
- 组内 GR 3B 是否能提供可直接复用的代码、配置或 checkpoint。

这些问题应进入 issue tracker，不得通过无依据猜测静默填补。
