# OxygenREC-v2预训练配对消融进度

> 主日志：[`复现实验日志.md`](../复现实验日志.md)  
> 当前状态：三组CUDA smoke及四个代表案例已review；案例选择器已纠正，待服务器确定性复跑和缺失角色回传后做最终方法级验收。
> 聚合终端：[`pretraining-ablation-smoke`](../案例原始记录/v2_pretraining_ablation/pretraining-ablation-smoke.md)
> 聚合报告转录：[`pretraining-ablation-report`](../案例原始记录/v2_pretraining_ablation/pretraining-ablation-report.md)

## 目的

论文预训练消融按组件逐步比较三组：Base不使用目标行为指令，`+I_b`加入Decoder
行为指令，Full再加入行为价值加权的预训练损失。本项目使用RetailRocket公开代理，
保留相同daily行为同质列表、模型规模、seed、batch顺序和训练预算，检查两个组件
是否分别产生可观察贡献。

这里的Base准确含义是“没有目标行为指令的生成器”，不是完全删除用户历史中的
行为信息。三组仍共享Encoder侧真实历史行为ID；被消融的是Decoder目标行为`I_b`
以及目标token的行为权重。

## 三组定义

| 变体 | Decoder目标行为`I_b` | NTP token权重 |
|---|---|---|
| Base | 无 | 统一为1 |
| `+I_b` | 真实view/cart/transaction目标行为 | 统一为1 |
| Full | 真实view/cart/transaction目标行为 | `1.2/1.5/2.0` |

`+I_b`和Full从完全相同的参数初始化开始；Base因行为token词表、adapter和Decoder
prefix长度不同，无法做到所有张量同形，但可对应的主干参数、SID普通词表行和目标
prefix位置行会逐值复制，并由脚本断言。三组每个epoch使用同一确定性样本顺序。

## 验收指标

- SID recall、逐token准确率、几何token reward；
- 每个生成SID的PrefixTrie合法率；
- 生成SID唯一率与整条列表无重复率；
- 分view/addtocart/transaction指标；
- Base→`+I_b`、`+I_b`→Full、Base→Full的配对差值；
- 最大增益、最大下降、稀有transaction和重复SID固定角色案例。

单seed、5,000条train与32条validation只用于控制流和方向诊断。即使某组更好，
也不能直接表述成稳定质量收益；稳定结论至少需要扩大validation并做多seed复核。

## 产物

- 入口：`scripts/train_v2_pretraining_ablation_retailrocket.py`
- 聚合报告：`checkpoints/retailrocket_v2_pretraining_ablation_smoke/v2_pretraining_ablation.md`
- 机器摘要：`checkpoints/retailrocket_v2_pretraining_ablation_smoke/v2_pretraining_ablation_summary.json`
- 原始案例：`checkpoints/retailrocket_v2_pretraining_ablation_smoke/v2_pretraining_ablation_cases.jsonl`
- 全部配对行：`checkpoints/retailrocket_v2_pretraining_ablation_smoke/v2_pretraining_ablation_all_rows.jsonl`

## 2026-09-10：三组CUDA聚合结果

三组实际使用`train=5000`、`validation=32`，每组79个optimizer step；
`common_initialization_match=True`且`ib_full_initialization_match=True`，说明配对初值
断言和训练入口均已通过。三组生成的每个SID均通过PrefixTrie，合法率都是1.0。

| 指标 | Base | `+I_b` | Full |
|---|---:|---:|---:|
| SID recall | 0.000000（0/64） | 0.015625（1/64） | 0.015625（1/64） |
| SID token accuracy | 0.062500（12/192） | 0.057292（11/192） | 0.057292（11/192） |
| 生成SID唯一率 | 0.734375（47/64） | 0.703125（45/64） | 0.687500（44/64） |
| 整列表无重复率 | 0.468750（15/32） | 0.406250（13/32） | 0.375000（12/32） |
| SID合法率 | 1.000000 | 1.000000 | 1.000000 |

Base→`+I_b`得到一个完整SID命中，但总token命中从12降至11。因此`I_b`在当前
训练预算下已经改变最终候选并产生一个稀疏完整命中，不能据此说它整体优于Base。
完整SID recall与token accuracy关注的粒度不同，出现一升一降并不矛盾。

`+I_b`→Full的SID recall和token accuracy完全相同，说明行为权重在这32条验证上
没有产生可见的目标命中增益；但生成SID唯一数从45降至44、无重复列表从13降至12，
表明权重更新确实可能改变列表组成，只是变化没有反映到当前目标指标。

重复列表占比很高：Base、`+I_b`、Full分别有17、19、20条列表包含重复SID。
所有SID又都合法，进一步确认这是“逐商品PrefixTrie合法、列表级不去重”的解码
边界，并可能叠加5,000样本单epoch欠训练造成的模式集中。

## 代表案例review

| 案例 | 固定角色 | 可确认变化 | 结论 |
|---|---|---|---|
| [`001`](../案例原始记录/v2_pretraining_ablation/v2-pretrain-ablation-review-001.md) | 旧`weighting_gain` | 三组生成、命中和reward完全相同，`Full-+I_b=0` | 不是加权收益；暴露选择器命名缺陷 |
| [`002`](../案例原始记录/v2_pretraining_ablation/v2-pretrain-ablation-review-002.md) | `generated_duplicate` | Base为两个不同错误SID，`+I_b`/Full为同一错误SID重复两次 | 目标行为指令可加剧列表级模式收缩；SID仍逐个合法 |
| [`003`](../案例原始记录/v2_pretraining_ablation/v2-pretrain-ablation-review-003.md) | `largest_ib_gain`、`largest_full_gain` | token命中`2/6→4/6`，SID recall`0→0.5` | 唯一完整SID正例来自`view`；首个目标被重复，第二目标未命中 |
| [`004`](../案例原始记录/v2_pretraining_ablation/v2-pretrain-ablation-review-004.md) | `largest_full_drop` | Base的一个中层token命中在`+I_b`/Full中消失，`1/6→0/6` | `I_b`变化并非单向改善，三组均无完整SID命中 |

`review-003`是聚合`1/64`完整SID命中的来源，但它不是稀有cart/order行为，而是
历史仅含一次view的view目标。生成结果把第一条正确SID复制到第二个列表位置，因此
这个正例同时暴露了列表去重缺口，不能把`0.5` recall解读成整条列表正确。

训练前后换算也提供了一个重要边界：Base从`3/192`升到`12/192`，`+I_b`和Full
从`2/192`升到`11/192`，三组都净增加9个目标token命中。训练后的`12`对`11`
不等于“Base比`I_b`多学会一个token”；这一票差距在训练前已经存在，并可能来自
不同Decoder前缀结构及初始化映射。当前能确认的是`I_b`改变了预测分布和部分列表，
而不是稳定提高或降低整体目标质量。

## 选择器缺陷与修正

旧规则无条件对`Full-+I_b token accuracy`取最大值。当全体样本的最大值为0时，
仍会把一条零差值案例命名为`weighting_gain`。这与“角色不存在则写`null`”的实验
记录规则冲突，也会误导人工review。

现已修正为：

- `largest_ib_gain`、`largest_full_gain`、`weighting_gain`只从严格正差值中选择；
- `largest_full_drop`只从严格负差值中选择；
- 新增`ib_output_changed`，即使目标命中不变，也能定位Base与`+I_b`生成不同的案例；
- 新增`weighting_output_changed`，专门定位Full与`+I_b`生成不同的案例，用来解释两组
  聚合目标指标相同但唯一率不同；
- 新增`ib_duplicate_introduced`与`weighting_duplicate_introduced`，直接选出某阶段
  从无重复变为有重复的列表，而不是用任意重复案例替代因果边界；
- 新增`v2_pretraining_ablation_all_rows.jsonl`，保存全部validation配对行，避免固定
  角色选择丢失重要的非目标输出变化。

当前截图还缺覆盖表中的`transaction=review-005`，且旧报告没有导出真正的
`weighting_output_changed`及`weighting_duplicate_introduced`案例。这些内容不能从聚合
表反推或补造。因此现阶段判断为：
三组训练与指标计算可解释，案例层面已发现并修正报告缺陷；服务器用同一命令确定性
复跑、确认`weighting_gain=null`并回传加权输出变化与transaction案例后，才能关闭
v2预训练方法级GPU验收。稳定质量收益仍需更大validation和多seed，不由本smoke承担。
