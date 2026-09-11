# OxygenREC-v2预训练三组消融报告聚合

> 记录类型：服务器生成Markdown的截图转录版  
> 实验编号：`E125`  
> 登记日期：2026-09-10  
> 服务器产物路径：`checkpoints/retailrocket_v2_pretraining_ablation_smoke/v2_pretraining_ablation.md`  
> 完整性说明：以下字段按用户提供截图逐项转录，不是服务器文件的字节级副本。  
> 主日志：[`E125`](../../复现实验日志.md)  
> 案例分析：[`v2_pretraining_ablation_progress.md`](../../案例分析/v2_pretraining_ablation_progress.md)

## 聚合指标

| 阶段/变体 | SID recall | token accuracy | geometric reward | generated SID unique | all-unique lists | legal items |
|---|---:|---:|---:|---:|---:|---:|
| before/base | 0.000000 | 0.015625 | 0.015780 | 0.921875 | 0.843750 | 1.000000 |
| before/ib | 0.000000 | 0.010417 | 0.011405 | 0.953125 | 0.906250 | 1.000000 |
| before/full | 0.000000 | 0.010417 | 0.011405 | 0.953125 | 0.906250 | 1.000000 |
| after/base | 0.000000 | 0.062500 | 0.065962 | 0.734375 | 0.468750 | 1.000000 |
| after/ib | 0.015625 | 0.057292 | 0.061100 | 0.703125 | 0.406250 | 1.000000 |
| after/full | 0.015625 | 0.057292 | 0.061100 | 0.687500 | 0.375000 | 1.000000 |

## 训练后差值

- `+Ib - Base`：`{'sid_recall': 0.015625, 'sid_token_accuracy': -0.005208333333333336, 'geometric_token_reward': -0.004861981095230283, 'generated_sid_unique_rate': -0.03125, 'all_unique_list_rate': -0.0625}`
- `Full - +Ib`：`{'sid_recall': 0.0, 'sid_token_accuracy': 0.0, 'geometric_token_reward': 0.0, 'generated_sid_unique_rate': -0.015625, 'all_unique_list_rate': -0.03125}`
- `Full - Base`：`{'sid_recall': 0.015625, 'sid_token_accuracy': -0.005208333333333336, 'geometric_token_reward': -0.004861981095230283, 'generated_sid_unique_rate': -0.046875, 'all_unique_list_rate': -0.09375}`

## 固定案例覆盖

```json
{
  "generated_duplicate": "v2-pretrain-ablation-review-002",
  "largest_full_drop": "v2-pretrain-ablation-review-004",
  "largest_full_gain": "v2-pretrain-ablation-review-003",
  "largest_ib_gain": "v2-pretrain-ablation-review-003",
  "transaction": "v2-pretrain-ablation-review-005",
  "weighting_gain": "v2-pretrain-ablation-review-001"
}
```

截图只包含`review-001`至`review-004`；`review-005`尚未收到，不补造。
