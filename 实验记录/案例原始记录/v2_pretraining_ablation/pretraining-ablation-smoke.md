# OxygenREC-v2预训练三组配对smoke终端记录

> 记录类型：服务器终端截图转录版  
> 实验编号：`E124`  
> 登记日期：2026-09-10  
> 服务器产物：`checkpoints/retailrocket_v2_pretraining_ablation_smoke/v2_pretraining_ablation.md`  
> 完整性说明：以下转录截图中可辨认的完整聚合行，不是服务器Markdown的字节级副本。  
> 主日志：[`E124`](../../复现实验日志.md)  
> 案例分析：[`v2_pretraining_ablation_progress.md`](../../案例分析/v2_pretraining_ablation_progress.md)

## 终端原始转录

```text
stage=load_events_and_build_paired_daily_cohort
stage=initialize_three_paired_variants
stage=train_three_paired_variants
OK device=cuda variant=v2_pretraining_ablation train=5000 validation=32 steps=79 common_initialization_match=True ib_full_initialization_match=True base_token_accuracy=0.062500 ib_token_accuracy=0.057292 full_token_accuracy=0.057292 base_sid_recall=0.000000 ib_sid_recall=0.015625 full_sid_recall=0.015625 base_unique=0.734375 ib_unique=0.703125 full_unique=0.687500 base_all_unique=0.468750 ib_all_unique=0.406250 full_all_unique=0.375000 base_legal=1.000000 ib_legal=1.000000 full_legal=1.000000 report=checkpoints/retailrocket_v2_pretraining_ablation_smoke/v2_pretraining_ablation.md external_reward_model=False
```

## 截图中的非错误提示

```text
UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
```

该提示只表示未启用nested-tensor性能优化路径。

## 当前转录缺口

- 截图未包含报告中的训练前指标、分行为指标和三段配对delta。
- 截图未包含`v2_pretraining_ablation_cases.jsonl`导出的代表案例。
