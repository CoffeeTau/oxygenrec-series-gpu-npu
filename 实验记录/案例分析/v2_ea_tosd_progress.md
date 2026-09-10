# OxygenREC-v2 EA-TOSD复现进度

> 主日志：[`复现实验日志.md`](../复现实验日志.md)  
> 本轮原始案例：[`ea-tosd-review-005`](../案例原始记录/v2_ea_tosd/ea-tosd-review-005.md)

原始案例索引：

- [`ea-tosd-review-001`](../案例原始记录/v2_ea_tosd/ea-tosd-review-001.md)：最高Teacher熵；
- [`ea-tosd-review-002`](../案例原始记录/v2_ea_tosd/ea-tosd-review-002.md)：全零reward；
- [`ea-tosd-review-003`](../案例原始记录/v2_ea_tosd/ea-tosd-review-003.md)：最大privilege gap；
- [`ea-tosd-review-004`](../案例原始记录/v2_ea_tosd/ea-tosd-review-004.md)：稀有transaction；
- [`ea-tosd-review-005`](../案例原始记录/v2_ea_tosd/ea-tosd-review-005.md)：非零reward。

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

## 下一验收：SFT-only配对对照

`scripts/train_v2_sft_control_retailrocket.py`从相同listwise checkpoint重建完全相同
的future-eligible训练/验证cohort，使用相同seed、batch、epoch和learning rate，
但反向目标只保留behavior-weighted SFT。它会先断言训练前的SID recall、token
accuracy和geometric reward与EA结果逐值一致，再输出更新前、SFT-only后、EA后
三点比较。

该对照只能回答当前smoke中的下降主要来自共享SFT续训，还是EA-TOSD附加目标；
由于仍是32样本单epoch，它不能替代后续多seed、足量训练和独立test split评测。
