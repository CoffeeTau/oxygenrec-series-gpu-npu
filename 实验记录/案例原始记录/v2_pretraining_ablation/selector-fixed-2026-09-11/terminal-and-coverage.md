# v2预训练三组消融选择器修正后复跑

> 记录类型：服务器终端及生成Markdown截图转录版  
> 实验编号：`E126`  
> 登记日期：2026-09-11  
> 服务器报告：`checkpoints/retailrocket_v2_pretraining_ablation_smoke/v2_pretraining_ablation.md`  
> 完整性说明：以下字段按用户提供截图逐项转录，不是服务器文件的字节级副本。  
> 主日志：[`E126`](../../../复现实验日志.md)  
> 案例分析：[`v2_pretraining_ablation_progress.md`](../../../案例分析/v2_pretraining_ablation_progress.md)

## 终端末行

```text
OK device=cuda variant=v2_pretraining_ablation train=5000 validation=32 steps=79 common_initialization_match=True ib_full_initialization_match=True base_token_accuracy=0.062500 ib_token_accuracy=0.057292 full_token_accuracy=0.057292 base_sid_recall=0.000000 ib_sid_recall=0.015625 full_sid_recall=0.015625 base_unique=0.734375 ib_unique=0.703125 full_unique=0.687500 base_all_unique=0.468750 ib_all_unique=0.406250 full_all_unique=0.375000 base_legal=1.000000 ib_legal=1.000000 full_legal=1.000000 ib_vs_base_changed_lists=13 full_vs_ib_changed_lists=3 review_coverage={...} report=checkpoints/retailrocket_v2_pretraining_ablation_smoke/v2_pretraining_ablation.md external_reward_model=False
```

## 训练后差值

- `+Ib - Base`：`{'sid_recall': 0.015625, 'sid_token_accuracy': -0.005208333333333336, 'geometric_token_reward': -0.004861981095230283, 'generated_sid_unique_rate': -0.03125, 'all_unique_list_rate': -0.0625}`
- `Full - +Ib`：`{'sid_recall': 0.0, 'sid_token_accuracy': 0.0, 'geometric_token_reward': 0.0, 'generated_sid_unique_rate': -0.015625, 'all_unique_list_rate': -0.03125}`
- `Full - Base`：`{'sid_recall': 0.015625, 'sid_token_accuracy': -0.005208333333333336, 'geometric_token_reward': -0.004861981095230283, 'generated_sid_unique_rate': -0.046875, 'all_unique_list_rate': -0.09375}`
- `+Ib`相对Base生成变化列表数：`13`
- Full相对`+Ib`生成变化列表数：`3`

## 固定案例覆盖

```json
{
  "generated_duplicate": "v2-pretrain-ablation-review-002",
  "ib_duplicate_introduced": "v2-pretrain-ablation-review-002",
  "ib_output_changed": "v2-pretrain-ablation-review-001",
  "largest_full_drop": "v2-pretrain-ablation-review-005",
  "largest_full_gain": "v2-pretrain-ablation-review-004",
  "largest_ib_gain": "v2-pretrain-ablation-review-004",
  "transaction": "v2-pretrain-ablation-review-006",
  "weighting_duplicate_introduced": "v2-pretrain-ablation-review-003",
  "weighting_gain": null,
  "weighting_output_changed": "v2-pretrain-ablation-review-003"
}
```

`weighting_gain=null`是服务器复跑的原始输出，不补造正收益案例。
