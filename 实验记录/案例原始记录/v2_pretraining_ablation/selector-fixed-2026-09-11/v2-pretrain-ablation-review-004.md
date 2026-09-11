# v2-pretrain-ablation-review-004

> 记录类型：服务器生成Markdown的截图转录版  
> 登记日期：2026-09-11  
> 主日志：[`E126`](../../../复现实验日志.md)  
> 案例分析：[`v2_pretraining_ablation_progress.md`](../../../案例分析/v2_pretraining_ablation_progress.md)

- 代表角色：`['largest_ib_gain', 'largest_full_gain']`
- 目标行为：`view`
- UTC日编号：`16679`
- 历史长度/行为：`1 / {'view': 1}`
- 目标SID列表：`[[97, 22, 147], [97, 100, 47]]`
- Base：`{'generated_sids': [[97, 214, 219], [97, 214, 219]], 'sid_token_hits': [True, False, False, True, False, False], 'sid_token_accuracy': 0.3333333333333333, 'geometric_token_reward': 0.36900369003690037, 'sid_recall': 0.0, 'all_generated_items_legal': True, 'generated_items_unique': False}`
- `+Ib`：`{'generated_sids': [[97, 22, 147], [97, 22, 147]], 'sid_token_hits': [True, True, True, True, False, False], 'sid_token_accuracy': 0.6666666666666666, 'geometric_token_reward': 0.7339523944689996, 'sid_recall': 0.5, 'all_generated_items_legal': True, 'generated_items_unique': False}`
- Full：`{'generated_sids': [[97, 22, 147], [97, 22, 147]], 'sid_token_hits': [True, True, True, True, False, False], 'sid_token_accuracy': 0.6666666666666666, 'geometric_token_reward': 0.7339523944689996, 'sid_recall': 0.5, 'all_generated_items_legal': True, 'generated_items_unique': False}`
- `+Ib-Base` token accuracy：`+0.333333`
- `Full-+Ib` token accuracy：`+0.000000`
- `Full-Base` token accuracy：`+0.333333`
- `+Ib/Base`生成token变化数：`4`
- `Full/+Ib`生成token变化数：`0`
