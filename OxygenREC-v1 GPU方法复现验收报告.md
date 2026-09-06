# OxygenREC-v1 GPU 方法复现验收报告

验收时间：2026-09-06 11:00:52 CST  
验收范围：论文公开方法的 GPU 自实现、公开数据代理实验和代表性案例检查  
不在范围：论文私有数据/特征/服务的主表复现、线上收益、NPU 对齐、Agentic Search 扩展效果

## 1. 验收结论

**OxygenREC-v1 的真实Qwen监督主线已经通过GPU验收；SA-GCPO模块曾在旧`igr_q2i`主线上独立通过，但当前Qwen checkpoint的统一后训练仍需最后一次CUDA验收。**

当前已完成并有数值、控制流和案例证据的监督链路是：

```text
target 前历史
  -> Qwen Contextual Reasoning Instruction
  -> 冻结 Qwen hidden state / instruction feature
  -> instruction query
  -> paper IGR 长历史 Top-K
  -> Fast Encoder-Decoder GR
  -> NTP + Q2I
  -> PrefixTrie 约束 beam
  -> SID/item 指标与案例报告
```

最终待验链路是在上述epoch-3 checkpoint之后继续执行：

```text
冻结 old policy -> constrained beam候选组 -> 公开代理Reward Mapping
  -> SA-GCPO advantage/threshold/ratio -> current policy更新
  -> held-out前后评测与匿名轨迹review
```

它**不表示当前 32 条训练样本的 Fast 模型已经得到可用推荐质量**。本轮 32 条 validation 中，IGR 目标 SID 命中 2 条，但 beam 目标商品命中为 0；案例表明约束生成全部合法，个性化生成仍明显欠训练。

## 2. 聚合结果

服务器实际输出：

| 项目 | 结果 | 解释 |
|---|---:|---|
| validation 样本 | 32 | 固定真实 Qwen feature cohort |
| 代表案例 | 7 | 固定规则选择，不是人工挑选最好结果 |
| repeat-eligible | 4 | 可检查 IGR 重复目标召回的样本很少 |
| IGR 命中 | 2 | IGR 控制流确实能够取回目标 SID，但不能据此统计外推 |
| beam 命中 | 0 | 生成质量未建立；不是合法性问题 |
| Q2I cosine 范围 | -0.063942 ～ 0.406374 | 对齐程度存在明显个例差异 |

覆盖角色如下：

- `v1-review-001`：IGR 命中；
- `v1-review-002`：IGR 漏召回；
- `v1-review-003`：transaction 目标；
- `v1-review-004`：addtocart 目标；
- `v1-review-005`：纯 view 历史；
- `v1-review-006`：本 cohort 的 Q2I 最高值；
- `v1-review-007`：本 cohort 的 Q2I 最低值；
- `beam_hit=null`：本 cohort 没有 beam 命中案例，因此未伪造正例。

本次对话已人工查看前 5 个案例的 Markdown 截图；006/007 已由固定选择器保存到服务器产物，但没有在本次对话中逐字段人工复核。

## 3. 五个代表案例结论

### 3.1 `v1-review-001`：检索成功、生成失败的分界案例

- 历史：109 条，`view=107`、`addtocart=1`、`transaction=1`，重复商品种类 2；
- 目标 SID：`[121, 210, 120]`；Q2I cosine：`0.209320`；
- IGR Top-10 全部为目标 SID，说明 paper IGR 在该例准确抓住了重复兴趣；
- beam Top-10 全部合法，但没有目标商品。

判断：这是最重要的阶段定位证据。**检索成功已经发生，但检索证据没有被训练不足的生成器稳定转化成目标输出**。因此该例的主要问题在 Encoder-Decoder 融合/生成阶段，而不是 IGR。

### 3.2 `v1-review-002`：纯浏览、IGR 漏召回

- 历史：32 条且全部为 view，重复商品种类 9；
- 目标 SID：`[197, 36, 232]`；Q2I cosine：`0.086009`；
- IGR 与 beam 均未命中目标。

判断：Qwen 对“长期浏览兴趣”的概括与聚合输入一致，但 query-item 对齐较弱，属于检索阶段失败；不能把它归因于 beam。

### 3.3 `v1-review-003`：高意图行为推理正确、目标仍漏召回

- 历史：120 条，`view=69`、`addtocart=26`、`transaction=25`，重复商品种类 29；
- 最近行为含连续 4 次 addtocart；目标行为为 transaction；
- 目标 SID：`[81, 52, 192]`；Q2I cosine：`0.084495`；
- Qwen 正确识别高意图行为、购买转化和重复兴趣，但 IGR/beam 均未命中。

判断：自然语言 Reasoning 的行为忠实性较好，但“理解出高意图”不等于 query 已学会对应的商品几何位置。

### 3.4 `v1-review-004`：addtocart 目标与混合行为案例

- 历史：120 条，`view=84`、`addtocart=20`、`transaction=16`，重复商品种类 25；
- 目标 SID：`[15, 95, 185]`；Q2I cosine：`0.168605`；
- Qwen 对长期兴趣和高意图混合模式的描述与输入一致；
- IGR 候选中能看到同一历史商品的 view/cart/transaction 事件，但目标仍未命中。

判断：Reasoning 已能表达行为模式，当前瓶颈仍是这些表达映射到具体 item/SID 的能力，而不是 JSON 格式或控制流。

### 3.5 `v1-review-005`：低意图纯浏览与负 Q2I

- 历史：120 条且全部为 view，重复商品种类 16；
- 目标 SID：`[50, 170, 48]`；Q2I cosine：`-0.058298`；
- Qwen 正确识别无购物车、无交易和长期浏览；IGR/beam 均未命中。

判断：负 Q2I 与该例较弱的检索表现一致。Qwen 文本中的“品类兴趣”只能算行为代理描述，因为当前输入没有真实商品名称、类目文本或属性语义。

## 4. 跨案例发现

### 4.1 已经通过的部分

1. Qwen 能根据聚合行为生成结构化且大体忠实的 Contextual Reasoning Instruction；
2. Qwen hidden state 确实进入 instruction adapter/query，而不是只把文本打印出来；
3. paper IGR 能返回 Top-K 长历史位置，并存在真实目标 SID 命中案例；
4. Q2I 能逐样本输出 cosine，联合损失方向及反向传播已经验证；
5. Encoder-Decoder、teacher forcing、约束 beam 和 SID 合法性闭环正常；
6. 所有展示的 beam 候选均合法，PrefixTrie 约束有效。

### 4.2 当前公开代理的两个实质限制

**第一，Instruction 缺少商品语义输入。** 当前 Qwen 看到的是历史长度、行为计数、最近行为和重复商品种类等聚合证据，没有真实商品标题、类目和属性。因此它能推理“行为意图”，却不能知道用户具体喜欢哪个类目或商品。这限制了论文式语义 IGR/Q2I 的质量上限。

**第二，Fast 生成器严重欠训练。** 当前只有 32 条 train 样本、3 个 epoch；不同案例的 beam 反复出现相似 SID，且即使 `v1-review-001` 的 IGR Top-10 全是目标 SID，beam 仍完全漏掉目标。这更符合共享先验主导、个性化融合不足，而不是解码器语法错误。

### 4.3 一个容易误判、但不是 bug 的现象

同一 SID 在 view、addtocart、transaction 事件上得到完全相同的 IGR 分数。当前 `paper_igr` 的分数是 instruction query 与 item/SID 向量的余弦相似度；行为和时间只用于构造上游 Reasoning 或展示诊断，不是候选打分的直接项。因此这是当前论文主线路径的设计结果。若以后加入显式行为/时间重排，应作为 Agentic/增强实验单独切换，不应悄悄改写 `paper_igr`。

## 5. 模块验收表

| 模块 | 状态 | 验收边界 |
|---|---|---|
| SID / RQ 公开代理 | 完成-代理 | 结构、registry、碰撞和约束生成已验证；非论文私有 tokenizer |
| Encoder-Decoder GR | 完成-实测 | forward/backward、overfit、checkpoint、生成和 beam 均已 CUDA 验证 |
| Contextual Reasoning Instruction | 完成-实测 | 真实 Qwen 文本与 hidden state 已进入 Fast 模型；商品语义输入不足已记录 |
| IGR | 完成-实测 | paper cosine Top-K 与真实命中/漏召回案例均已验证；收益未建立 |
| Q2I | 完成-实测 | 联合目标、逐样本 cosine、反传和方向均已验证；排序收益未建立 |
| SA-GCPO / RL | 模块完成；Qwen统一链待CUDA | 旧`igr_q2i`上的公式、reward、rollout和更新链已验证；当前epoch-3已完成代码接入，尚待服务器实测 |
| Prefix 约束解码 | 完成-实测 | 候选合法率通过；目标命中质量未建立 |
| Agentic Retrieval Plan | 保留扩展、非 v1 主线 | 与 `paper_igr` 已显式分轨，留待后续对比 |
| NPU | 暂缓 | 按用户要求不进入本次 v1 GPU 验收 |

## 6. 收口决定与后续

1. 不再用这 32 条 smoke cohort 反复调 IGR、Q2I 或 beam 权重；
2. 当前标记为 **`[监督主线完成；统一SA-GCPO待CUDA]`**，服务器验收通过后再改为`[完成-方法级GPU复现]`；
3. 当前 checkpoint 和案例报告只作为链路证据，不作为可部署推荐模型；
4. 若未来单独做质量升级，最有价值的两项是：给 Qwen/Instruction 加入公开商品类目或属性语义；扩大经审核训练 cohort 后再训练 Fast 模型；
5. 质量升级应另开实验编号，并与当前方法验收基线对照，不阻塞 v1 收口。
