# v2 SFT-only / EA-TOSD配对smoke终端记录

> 记录类型：服务器终端截图转录版  
> 实验编号：`E120`  
> 登记日期：2026-09-10  
> 服务器产物：`checkpoints/retailrocket_v2_sft_control_smoke/ea_tosd_vs_sft_control.md`  
> 完整性说明：以下仅转录截图中可辨认的阶段与最终聚合行，不是服务器产物的字节级副本。  
> 主日志：[`E120`](../../复现实验日志.md)  
> 案例分析：[`v2_ea_tosd_progress.md`](../../案例分析/v2_ea_tosd_progress.md)

## 终端原始转录

```text
stage=load_checkpoint_events_and_ea_summary
stage=build_paired_future_eligible_cohort
stage=paired_cohort train=32 validation=32 future_counts={1: 4, 2: 28} behaviors={'transaction': 2, 'view': 30} objective=behavior_weighted_sft_only external_reward_model=False
OK device=cuda variant=v2_sft_control train=32 validation=32 steps=8 checkpoint_positions=9+6 mean_sft=5.851713 shared_grad=872.898237 paired_before_match=True before_sid_recall=0.000000 sft_after_sid_recall=0.000000 ea_after_sid_recall=0.000000 before_token_accuracy=0.046875 sft_after_token_accuracy=0.041667 ea_after_token_accuracy=0.041667 before_geo_reward=0.047030 sft_after_geo_reward=0.041628 ea_after_geo_reward=0.041628 comparison=checkpoints/retailrocket_v2_sft_control_smoke/ea_tosd_vs_sft_control.md external_reward_model=False
```

## 截图中的非错误提示

```text
UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
```

该提示表示当前Transformer没有使用nested-tensor性能路径，不是训练正确性错误。

## 转录缺口

- 截图未展示服务器生成的`ea_tosd_vs_sft_control.md`全文。
- 本轮为聚合配对实验，没有生成新的逐样本代表案例。
