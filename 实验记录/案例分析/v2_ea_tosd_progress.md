# OxygenREC-v2 EA-TOSD复现进度

> 主日志：[`复现实验日志.md`](../复现实验日志.md)  
> 本轮原始案例：[`ea-tosd-review-005`](../案例原始记录/v2_ea_tosd/ea-tosd-review-005.md)
> 配对终端记录：[`sft-only-paired-smoke`](../案例原始记录/v2_ea_tosd/sft-only-paired-smoke.md)
> checkpoint比较记录：[`checkpoint-comparison-smoke`](../案例原始记录/v2_ea_tosd/checkpoint-comparison-smoke.md)

原始案例索引：

- [`ea-tosd-review-001`](../案例原始记录/v2_ea_tosd/ea-tosd-review-001.md)：最高Teacher熵；
- [`ea-tosd-review-002`](../案例原始记录/v2_ea_tosd/ea-tosd-review-002.md)：全零reward；
- [`ea-tosd-review-003`](../案例原始记录/v2_ea_tosd/ea-tosd-review-003.md)：最大privilege gap；
- [`ea-tosd-review-004`](../案例原始记录/v2_ea_tosd/ea-tosd-review-004.md)：稀有transaction；
- [`ea-tosd-review-005`](../案例原始记录/v2_ea_tosd/ea-tosd-review-005.md)：非零reward。
- [`checkpoint-diff-review-001`](../案例原始记录/v2_ea_tosd/checkpoint-diff-review-001.md)：第一条greedy变化列表；
- [`checkpoint-diff-review-002`](../案例原始记录/v2_ea_tosd/checkpoint-diff-review-002.md)：EA gold log-prob增益最大；
- [`checkpoint-diff-review-003`](../案例原始记录/v2_ea_tosd/checkpoint-diff-review-003.md)：EA gold log-prob下降最大；
- [`checkpoint-diff-review-004`](../案例原始记录/v2_ea_tosd/checkpoint-diff-review-004.md)：最大logit差；
- [`checkpoint-diff-review-005`](../案例原始记录/v2_ea_tosd/checkpoint-diff-review-005.md)：第二条greedy变化列表。

## 2026-09-10：真实future-prefix CUDA smoke

当前真实RetailRocket代理链路已完成一次单卡CUDA结构验收。训练和验证各取32条
future-eligible列表样本，list size为2、每条SID为3个token、best-of-G为4；未来行为
必须严格晚于gold列表且位于同一时间split。整个过程没有加载外部Reward Model。

本次终端聚合结果：

| 诊断 | 数值 |
|---|---:|
| mean selected reward | 0.050835 |
| nonzero reward rate | 0.281250（9/32） |
| all-zero group rate | 0.718750（23/32） |
| low/high entropy gate | 0.000000 / 1.000000 |
| Teacher-Student max logit delta | 0.706945 |
| 更新前/后 exact SID recall | 0.000000 / 0.000000 |
| 更新前/后 SID token accuracy | 0.046875 / 0.041667 |
| 更新前/后 geometric token reward | 0.047030 / 0.041628 |

这证明真实未来前缀能够改变共享模型Teacher分支，on-policy候选、可验证reward、
高熵Forward-KL、SFT anchor和反向更新均已端到端执行。它没有证明质量收益：一次
极小样本更新后，exact SID recall仍为0，token accuracy由9/192降到8/192，几何
token reward也下降。所有token均进入高熵门，说明当前checkpoint仍处于高不确定性
阶段，低熵自蒸馏分支在本轮没有被激活。

## 首轮零reward、高熵与稀有行为案例分析

- `ea-tosd-review-001`（`most_high_entropy`）：future SID与首个gold SID同为
  `[38, 27, 118]`，但4条候选reward全为0；Teacher entropy约为
  `4.64～5.28`，`low/high=0/6`。匿名记录不含原始item ID，只能确认SID重合，
  不能排除SID collision。
- `ea-tosd-review-002`（`zero_reward`）：future连续两次出现`[174, 86, 2]`，
  可能来自同商品重复行为，也可能来自SID collision；候选与gold无token命中，
  reward全为0，六个位置全为高熵。
- `ea-tosd-review-003`（`largest_privilege_gap`）：future中的`[42, 207, 54]`
  与第二个gold SID完全相同，但所有候选reward仍为0；所选错误轨迹首token的
  privilege advantage约为`+0.479`。这说明future改变了候选token概率，不证明
  Teacher已把概率移向gold；低熵门关闭时，该advantage也不会进入低熵SD路径。
- `ea-tosd-review-004`（`transaction`）：future首个SID`[185, 103, 112]`与
  第二个gold SID相同，证明稀有transaction样本、future Teacher与轨迹评分链已贯通；
  但四条候选仍全部为零reward，不是成功推荐案例。

四例的Teacher entropy均高于2.6，与聚合`high_gate_rate=1.0`一致。全零reward时
`argmax`稳定选择索引0，因此`best=0`只是并列处理，不代表第一条候选质量更高。

## 固定代表案例：ea-tosd-review-005

该案例承担`nonzero_reward`角色：目标行为为view，gold列表展平后为
`[217, 136, 66, 8, 0, 218]`，Teacher看到约54秒后发生的真实view，其SID为
`[217, 95, 28]`。未来SID与两个gold SID均不完全相同，但共享第一层token `217`。

四条on-policy候选中只有第一条`[217, 246, 7, 16, 21, 209]`命中第一个token，
其余23个候选token均未命中。采用六步、衰减系数0.9的归一化几何权重后，第一个
位置的权重为：

```text
1 / (1 + 0.9 + 0.9^2 + 0.9^3 + 0.9^4 + 0.9^5) = 0.213420
```

因此该候选reward恰为0.213420并被best-of-G选中，和导出的逐token命中完全一致。
Teacher在第一步给出的privilege advantage为0.284111，和future/gold/candidate共享
粗粒度前缀的现象相符；但这只是SID层级第一token命中，不是完整SID命中，更不能
表述成商品级推荐成功。

## SFT-only配对对照结果

`scripts/train_v2_sft_control_retailrocket.py`已在CUDA服务器运行，训练前一致性断言
`paired_before_match=True`。两个分支使用相同listwise checkpoint、完全相同的
future-eligible训练/验证cohort、seed、batch、epoch和learning rate，差别仅在
SFT-only分支不加入EA-TOSD目标。

| 指标 | 更新前 | SFT-only后 | EA-TOSD后 | EA-SFT |
|---|---:|---:|---:|---:|
| exact SID recall | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| SID token accuracy | 0.046875 | 0.041667 | 0.041667 | 0.000000 |
| geometric token reward | 0.047030 | 0.041628 | 0.041628 | 0.000000 |

因此，E119的轻微下降不能归因于EA-TOSD附加目标：只运行共同的behavior-weighted
SFT也得到同样的离散结果。当前更合理的解释是共享SFT更新与32条验证样本的离散
波动共同作用。

但两组相同的SID指标不表示模型参数相同。EA分支最大共享梯度为`878.134954`，
SFT-only为`872.898237`；口径相同但数值不同，提示EA附加项可能改变了连续更新，
只是没有改变当前greedy输出的命中关系。下一步应直接比较两份checkpoint的参数、
validation logits、token argmax和greedy列表，而不是仅凭相同命中率判断EA“没有作用”。

该配对仍只有32样本、单epoch，不能替代多seed、充分训练和独立test split评测。

## EA与SFT checkpoint连续差异

只读比较确认两条分支并非相同模型：96个浮点参数张量中80个发生变化，参数平均
绝对差为`9.463138810e-07`、最大差为`9.970739484e-05`、相对L2为
`1.008905248e-05`。对应validation logits的平均绝对差为`2.572369801e-04`，
最大差为`8.354246616e-03`，平均对称KL为`1.591029388e-07`。

这些连续变化已经跨过少量决策边界：teacher-forcing的192个token中有1个argmax
不同；constrained greedy的192个token中有5个不同，涉及32条列表中的2条。但
`ea_only_hits=0`、`sft_only_hits=0`，说明这些变化都发生在非目标预测之间，没有
增加或减少目标token命中。EA相对SFT的平均gold log-prob还下降了
`3.282229106e-05`，幅度很小，但方向不是正向收益。

因此当前最准确的结论是：EA-TOSD附加目标已经生效并改变连续策略，也改变了少量
最终列表；只是本轮变化没有向真实目标移动。

## checkpoint差异代表案例分析

五个固定角色案例把连续变化与离散输出的关系进一步拆开：

- `checkpoint-diff-review-001`：EA把第一项从SFT的`[97,214,219]`改成
  `[217,119,241]`，三个SID层级全部变化；第二项不变。两边六个token都没有命中
  gold，EA平均gold log-prob反而下降`2.1843e-4`。SFT原列表的两个商品SID完全
  重复，EA在此例消除了重复，但这不是目标命中收益。
- `checkpoint-diff-review-005`：EA只改变第二个商品的第1/3层token，把SFT的
  `[97,214,219]`改成与首项相同的`[217,214,164]`。六个token仍均未命中，尽管
  EA平均gold log-prob上升`3.3911e-4`。也就是说，连续概率向gold移动并不保证
  greedy列表立刻改善，本例甚至产生了重复商品SID。
- `checkpoint-diff-review-002/003`分别给出最大gold概率增益`+3.6478e-4`和最大
  下降`-2.6981e-4`，但两例EA/SFT greedy输出和teacher-forcing argmax都不变。
  这说明大部分EA更新仍停留在决策边界以内，并且在held-out样本上的方向有正有负。
- `checkpoint-diff-review-004`的单步最大logit差为`8.3542e-3`，仍未改变argmax或
  greedy结果，是“连续策略已变、离散输出未变”的典型样本。

`checkpoint-diff-review-001`只有teacher-forcing第一个argmax变化，但自回归第一项
三个token全部变化；`checkpoint-diff-review-005`则teacher-forcing argmax全不变，
greedy第二项仍变化。这不矛盾：teacher forcing沿真实target prefix比较，greedy沿
模型自己生成的prefix递推，小幅参数变化可能在两种前缀路径上跨过不同决策边界。

五例的`EA/SFT生成SID均合法=True/True`只表示每个三层SID都在PrefixTrie中，不能
推出列表内商品唯一。当前`generate()`按商品重置Trie但不屏蔽前面已经生成的完整SID；
论文公开方法未明确给出列表去重规则，因此不把它补写成论文组件。它应作为公开代理
实现边界保留，并在后续预训练配对中同时报告生成列表唯一率，避免只看token合法率。

综上，EA-TOSD的真实future、Teacher/Student、best-of-G、几何reward、熵门、SFT
anchor、反向更新、配对控制和最终策略变化均已有CUDA及案例证据，可以关闭为
`[完成-方法级GPU复现]`。该状态不表示EA优于SFT：32条smoke没有目标命中增益，
也没有多seed或独立test上的稳定收益证据。
