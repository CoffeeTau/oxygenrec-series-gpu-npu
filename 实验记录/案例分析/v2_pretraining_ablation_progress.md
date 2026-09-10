# OxygenREC-v2预训练配对消融进度

> 主日志：[`复现实验日志.md`](../复现实验日志.md)  
> 当前状态：代码与服务器入口完成，待CUDA实测和代表案例回传。

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
