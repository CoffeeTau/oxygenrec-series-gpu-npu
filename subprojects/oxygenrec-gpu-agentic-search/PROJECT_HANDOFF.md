# OxygenREC GPU → Agentic Search 项目交接

交接日期：2026-09-17
父项目来源：`oxygenrec-series-gpu-npu`
代码快照：`c42fb54a28feddcad2056edcd6fb9ea03685d361`

## 1. 当前最可靠的结论

OxygenREC-v1/v2 已在公开数据代理范围内完成方法级 GPU 链路验证。这里的“完成”是指
数据边界、模型结构、前向/反向、checkpoint、约束生成、检索/轨迹控制与代表案例均有
服务器运行结果，不表示论文私有主表、线上收益或可部署推荐质量已经复现。

可继承的两条方法链如下：

```text
v1:
历史行为 -> Qwen Contextual Reasoning -> hidden state/instruction feature
        -> Q2I + IGR -> Encoder-Decoder GR -> constrained beam
        -> public-proxy SA-GCPO

v2:
daily listwise behavior -> I_s/I_r/I_b -> behavior-weighted NTP
        -> constrained list generation -> on-policy candidate trajectory
        -> verifiable reward + privileged Teacher -> EA-TOSD/SFT control
```

详细验收证据：

- [OxygenREC-v1 GPU方法复现验收报告](实验记录/案例分析/OxygenREC-v1%20GPU方法复现验收报告.md)
- [OxygenREC-v2 GPU方法复现验收报告](实验记录/案例分析/OxygenREC-v2%20GPU方法复现验收报告.md)
- [v2预训练消融过程](实验记录/案例分析/v2_pretraining_ablation_progress.md)
- [v2 EA-TOSD过程](实验记录/案例分析/v2_ea_tosd_progress.md)

## 2. 2026-09-17 固定验证结果及其含义

父项目用同一 Full checkpoint、事件文件、SID registry、固定 32 条 validation 列表和
相同源码 commit，在 NVIDIA L20 与 Ascend 950DT 上分别执行 FP32/BF16 正确性验证。
此处记录是为了交代 GPU baseline 的可信边界，不把 NPU 代码带进本子项目。

| 对比项 | GPU | NPU | 判断 |
|---|---:|---:|---|
| FP32 mean loss | 5.8189945221 | 5.8189883232 | 绝对差 0.0000061989，相对差约 0.0001065% |
| BF16 mean loss | 5.8189778328 | 5.8184895515 | 绝对差 0.0004882813，相对差约 0.0083912% |
| SID recall / position accuracy | 0.015625 / 0.015625 | 相同 | 1/64 个目标位置命中 |
| exact-list / beam hit@5 | 0 / 0 | 相同 | 当前 checkpoint 质量很弱 |
| greedy/beam legal item rate | 1.0 / 1.0 | 相同 | 约束合法性通过 |

输入哈希、target 指纹和源码文件哈希一致；所有 logits 有限，FP32 的 greedy 与 beam
指纹均一致。BF16 的 greedy 指纹一致，但 beam 指纹不同。根据父项目预先冻结的验收规则，
这不是直接失败：聚合指标仍在阈值内，但必须定位发生变化的固定案例，不能宣称 BF16
离散路径已经逐样本完全一致。

对新项目最重要的含义有两点：

1. GPU checkpoint 和固定 cohort 可以作为后续 paired 实验的工程 baseline；
2. `1/64` recall 与 `0` exact-list 说明它不能作为高质量推荐 baseline。Agentic Search
   实验必须报告“控制链变化”和“任务效果”两层指标，不能用可运行性替代效果。

另外，固定验证明确关闭 Transformer MHA fastpath，只用于跨设备正确性比较；其耗时不是
性能基准。NPU 端软件栈版本也与较早环境记录存在开发版日期差异，不能把这四次运行耗时
用于 GPU/NPU 性能结论。

## 3. 本子项目保留了什么

| 能力 | 主要代码 | 对 Agentic Search 的价值 |
|---|---|---|
| SID与约束解码 | `sid.py`、`quantization*.py`、`model.py` | 结构化动作空间、合法动作约束 |
| Reasoning Instruction | `llm_reasoning.py`、`llm_features.py` | 把行为证据转为可审计意图与策略 |
| Q2I与IGR | `alignment.py`、`model.py` | query-item语义对齐、长历史检索 |
| Retrieval Plan | `retrieval_planning.py` | 显式计划、规则编译、确定性执行 |
| SA-GCPO | `rewards.py`、`alignment.py`、`sa_review.py` | 可验证奖励、组相对优势、轨迹审阅 |
| v2轨迹优化 | `ea_tosd.py` | Teacher/Student、熵门控、SFT anchor |
| 数据与评测 | `data/`、`evaluation.py`、review脚本 | 时间边界、paired cohort、匿名案例 |

`device.py` 随核心依赖保留，因为部分 GPU 脚本通过它统一解析 `cuda` 和随机种子；新项目
没有复制 NPU 运行入口、迁移探针和 NPU 环境文档。后续可以将它简化为 GPU-only，但这不是
启动新研究的前置条件。

## 4. 已知问题，不要重复踩坑

1. 当前 Qwen prompt 主要包含行为统计、最近行为和重复程度，缺少商品标题、类目、属性等
   真实语义。模型可能正确描述“高购买意图”，却无法据此定位具体商品。
2. v1 的关键案例已经出现“IGR 找到目标、生成器仍未命中”。这说明检索和生成必须分层
   评估，不能只看最终 Recall 后猜测故障位置。
3. 当前公开代理 checkpoint 欠训练；反复在 32 条 smoke cohort 上调权不构成可信提升。
4. SID 碰撞意味着“重复 SID”不必然等于“重复原始商品”。去重分析要同时看 item 与 SID。
5. SA-GCPO/EA-TOSD 的公开 reward 是论文私有服务的代理。应强调可验证训练机制，不能声称
   复现工业 reward 或线上收益。
6. Qwen 权重、RetailRocket 数据、SID registry、instruction cache 和 checkpoint 都未打包。
   复制项目后要重新登记来源、许可证、哈希和生成参数。

## 5. 建议的新项目主线

不要把 Agentic Search 理解为给现有脚本加一个 LLM 调用。建议按可证伪实验逐级推进：

### AS-0：冻结可复现实验基线

- 固定 data split、SID registry、checkpoint、Qwen 输出、候选集和 seed；
- 建立独立 validation/test，至少 3 个 seed；
- 保存每个案例的 retrieval evidence、plan、selected indices、final list 和 reward components。

验收：相同输入重复运行结果一致；baseline 指标和轨迹文件可由一条命令重建。

### AS-1：Paper IGR 与 Agentic Plan 的严格配对

- 同一 instruction query 与 semantic scores；
- control 仅切换 `paper_igr` / `agentic_plan`；
- 指标分为 retrieval recall、generation/list quality、合法性、延迟和 token/API cost。

验收：不仅说明“输出变了”，还要说明哪些 plan 操作在何种案例上帮助或伤害结果。

### AS-2：引入公开商品语义与多源检索

- 给 Qwen 和 item encoder 增加公开类目/属性/文本；
- 区分行为记忆、商品语义索引与约束规则；
- 做行为-only、semantic-only、hybrid 的 matched ablation。

验收：先证明检索层目标召回改善，再检查是否被生成层转化为最终命中。

### AS-3：Planner–Retriever–Verifier 闭环

- Planner 产生结构化 retrieval plan；
- Retriever 执行多步检索；
- Verifier 根据证据覆盖、合法性和候选一致性决定接受、改写或停止；
- 限制最大步数和预算，避免无限搜索。

验收：报告成功、无效重试、错误修正、预算耗尽四类轨迹，而不仅展示成功案例。

### AS-4：Search RL / trajectory optimization

- 先使用可离线验证的 reward：目标覆盖、排序、证据一致、合法性、成本；
- 与 SFT-only、无 verifier、无 cost penalty 做配对；
- 复用 SA-GCPO 的 group-relative 与阈值门控思想，但不要直接沿用旧权重。

验收：独立 test、多 seed、reward hacking 审计和代表失败轨迹全部通过后，再讨论提升。

这条路线形成的经历链是：

```text
Generative Recommendation
  -> Reasoning-enhanced Retrieval
  -> Planned Multi-step Search
  -> Verifiable Trajectory Optimization
  -> Agentic Search
```

## 6. 复制后第一周的建议任务

1. 将本目录复制为新仓库并创建全新 Git 历史；不要把父项目的 NPU 迁移分支合并进来。
2. 运行单元测试和三个 GPU toy validation，记录新环境版本。
3. 只恢复公开数据、SID registry 和一个 GPU baseline checkpoint，先校验 SHA-256。
4. 为 `paper_igr` 与 `agentic_plan` 写一个统一 paired experiment manifest。
5. 先完成 AS-1，再决定商品语义、Verifier 或 RL 哪个方向进入第二阶段。

建议的第一周交付物：一个可重跑的 paired 命令、一份聚合表、10 个固定案例和一份失败分类。
这比先扩模型规模更能形成清晰、可信的项目经历。
