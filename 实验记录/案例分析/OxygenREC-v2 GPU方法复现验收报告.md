# OxygenREC-v2 GPU方法复现验收报告

> 主日志：[`复现实验日志.md`](../复现实验日志.md)  
> 预训练分析：[`v2_pretraining_ablation_progress.md`](v2_pretraining_ablation_progress.md)  
> 后训练分析：[`v2_ea_tosd_progress.md`](v2_ea_tosd_progress.md)

验收日期：2026-09-11  
验收范围：v2论文公开方法的GPU自实现、RetailRocket公开代理和代表案例检查  
不在范围：私有数据/服务主表、线上收益、稳定质量收益、NPU与多卡性能

## 1. 验收结论

**OxygenREC-v2已在约定范围内完成方法级GPU复现。**

已贯通的预训练链路：

```text
daily同用户/同UTC日/同目标行为列表
  -> 历史SID与历史行为Encoder
  -> [BOS, I_s, I_r, I_b] Decoder前缀
  -> 行为加权NTP
  -> [B,N,3]列表式SID约束生成
```

已贯通的后训练链路：

```text
部署Student输入
  + 同split、严格晚于gold的真实future SID Teacher前缀
  -> on-policy best-of-G候选与几何可验证reward
  -> 熵门控反向/正向KL + SFT anchor
  -> 共享backbone更新
  -> 与同条件SFT-only checkpoint配对比较
```

“方法级完成”表示结构、数据边界、梯度、checkpoint、生成和代表轨迹有CUDA实测证据，
不表示复现论文私有主表或获得可部署推荐质量。

## 2. Behavior-aware Pretraining验收

三组使用同一5,000条训练样本、32条验证列表、79个optimizer step和确定性batch顺序。
`+I_b`与Full初始化完全一致；Base的可对应主干参数逐值复制并通过断言。

| 指标 | Base | `+I_b` | Full |
|---|---:|---:|---:|
| SID recall | 0/64 | 1/64 | 1/64 |
| token命中 | 12/192 | 11/192 | 11/192 |
| 生成SID唯一数 | 47/64 | 45/64 | 44/64 |
| 无重复SID列表 | 15/32 | 13/32 | 12/32 |
| SID合法率 | 1.0 | 1.0 | 1.0 |

组件敏感性复跑确认：`I_b`改变13/32条生成列表，行为加权改变3/32条；
`weighting_gain=null`，即本validation没有任何正的`Full-+I_b token accuracy`案例。
所以两个组件均已实际影响策略，但行为加权没有目标命中收益。

六个固定角色覆盖`I_b`输出变化、`I_b`引入重复、行为加权输出变化、最大正例、最大
负例和transaction。最大正例得到一个完整SID命中，但重复首个目标SID；最大负例
丢失一个局部token命中；transaction代表例三组相同。

一个重要边界是目标SID碰撞：daily数据保证原始目标商品互异，但当前registry允许
不同商品共享SID。故重复SID不必然等于重复原始商品，SID序列唯一率不能脱离目标
SID唯一性单独解释。

## 3. EA-TOSD验收

真实future-prefix smoke使用32条训练/验证列表、G=4，未来行为严格晚于gold且留在
同一split，不加载外部Reward Model。

| 诊断 | 结果 |
|---|---:|
| mean selected reward | 0.050835 |
| 非零reward样本 | 9/32 |
| 全零候选组 | 23/32 |
| low/high entropy gate | 0 / 1 |
| Teacher-Student最大logit差 | 0.706945 |
| 更新前后SID recall | 0 → 0 |
| 更新前后token命中 | 9/192 → 8/192 |

同checkpoint、cohort和训练预算的SFT-only配对得到相同最终离散指标，因此下降不能
归因于EA附加项。只读checkpoint比较进一步确认EA并非无效更新：96个浮点参数张量中
80个变化，validation最大logit差`8.354246616e-03`，teacher-forcing有1/192个argmax
变化，greedy有5/192个token变化并涉及2/32条列表；变化均未新增目标命中。

代表轨迹覆盖非零/零reward、transaction、高熵、最大privilege gap，以及EA相对SFT
的最大gold概率增益/下降和两条greedy变化。它们证明真实future、Teacher/Student、
best-of-G、几何reward、熵门、SFT anchor与策略更新均实际执行。

## 4. 最终模块矩阵

| 模块 | 状态 | 证据边界 |
|---|---|---|
| Decoder Behavior Instruction | 完成-实测 | 独立`I_b`前缀、两层投影及全部生成接口通过CUDA |
| Behavior-aware Pretraining | 完成-方法级GPU复现 | 三组配对、行为权重、列表生成及六例review通过；收益未建立 |
| Listwise SID generation | 完成-方法级GPU复现 | greedy/beam逐商品Trie重置，所有生成SID合法；列表去重未由论文公开规则规定 |
| Verifiable trajectory optimization | 完成-方法级GPU复现 | on-policy候选、几何reward和best-of-G通过；非零reward为部分token命中 |
| Privileged self-distillation | 完成-方法级GPU复现 | 同split真实future Teacher与共享backbone更新通过 |
| Entropy-aware distillation | 完成-方法级GPU复现 | 本轮只覆盖高熵Forward-KL；低熵路径有合成测试、真实样本未触发 |
| 外部Reward Model | 不使用 | v2主线所有终端均为`external_reward_model=False` |

## 5. 质量与迁移边界

当前公开代理仍有训练预算小、SID碰撞、目标命中稀疏、全部真实EA token处于高熵区等
限制。稳定效果判断需要扩大validation、独立test和多seed；这不是本次方法验收的
完成条件，也不能通过继续挑选case替代。

GPU侧方法复现到此冻结。下一阶段按
[`NPU迁移计划`](../../docs/npu_migration_plan.md)执行：先验证Ascend版本栈与单卡基础
计算，再建立v2固定输入的logits/loss/生成/梯度对齐；未完成前不报告NPU通过。
