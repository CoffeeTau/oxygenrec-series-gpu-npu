# EA-TOSD / SFT-only checkpoint只读比较终端记录

> 记录类型：服务器终端截图转录版  
> 实验编号：`E121`  
> 登记日期：2026-09-10  
> 服务器产物：`outputs/review/v2_ea_tosd_checkpoint_comparison/checkpoint_comparison.md`  
> 完整性说明：以下转录截图中可辨认的完整聚合行，不是服务器Markdown的字节级副本。  
> 主日志：[`E121`](../../复现实验日志.md)  
> 案例分析：[`v2_ea_tosd_progress.md`](../../案例分析/v2_ea_tosd_progress.md)

## 终端原始转录

```text
stage=load_paired_checkpoints
stage=build_paired_validation_cohort
OK device=cuda variant=v2_ea_vs_sft_checkpoint_readonly validation=32 changed_tensors=80/96 parameter_mean_abs=9.463138810e-07 parameter_max_abs=9.970739484e-05 parameter_relative_l2=1.008905248e-05 logit_mean_abs=2.572369801e-04 logit_max_abs=8.354246616e-03 symmetric_kl=1.591029388e-07 gold_logprob_delta=-3.282229106e-05 teacher_argmax_change=0.005208 greedy_token_change=0.026042 greedy_list_change=0.062500 ea_only_hits=0 sft_only_hits=0 review_cases=5 report=outputs/review/v2_ea_tosd_checkpoint_comparison/checkpoint_comparison.md read_only=True external_reward_model=False
```

## 截图中的非错误提示

```text
UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
```

该提示只表示未启用nested-tensor性能优化路径。

## 代表案例源记录

- 截图未展示`checkpoint_comparison.md`中的最大差异参数列表。
- [`checkpoint-diff-review-001`](checkpoint-diff-review-001.md)：第一条greedy变化列表。
- [`checkpoint-diff-review-002`](checkpoint-diff-review-002.md)：EA gold log-prob增益最大。
- [`checkpoint-diff-review-003`](checkpoint-diff-review-003.md)：EA gold log-prob下降最大。
- [`checkpoint-diff-review-004`](checkpoint-diff-review-004.md)：最大logit差。
- [`checkpoint-diff-review-005`](checkpoint-diff-review-005.md)：第二条greedy变化列表。
