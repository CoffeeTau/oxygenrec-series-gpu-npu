# v2-pretrain-ablation-review-002

> 记录类型：服务器生成Markdown的截图转录版  
> 登记日期：2026-09-11  
> 主日志：[`E126`](../../../复现实验日志.md)  
> 案例分析：[`v2_pretraining_ablation_progress.md`](../../../案例分析/v2_pretraining_ablation_progress.md)

- 代表角色：`['ib_duplicate_introduced', 'generated_duplicate']`
- 目标行为：`view`
- UTC日编号：`16676`
- 历史长度/行为：`20 / {'view': 20}`
- 目标SID列表：`[[106, 116, 2], [252, 76, 23]]`
- Base：`{'generated_sids': [[52, 76, 147], [97, 214, 219]], 'sid_token_hits': [False, False, False, False, False, False], 'sid_token_accuracy': 0.0, 'geometric_token_reward': 0.0, 'sid_recall': 0.0, 'all_generated_items_legal': True, 'generated_items_unique': True}`
- `+Ib`：`{'generated_sids': [[97, 214, 219], [97, 214, 219]], 'sid_token_hits': [False, False, False, False, False, False], 'sid_token_accuracy': 0.0, 'geometric_token_reward': 0.0, 'sid_recall': 0.0, 'all_generated_items_legal': True, 'generated_items_unique': False}`
- Full：`{'generated_sids': [[97, 214, 219], [97, 214, 219]], 'sid_token_hits': [False, False, False, False, False, False], 'sid_token_accuracy': 0.0, 'geometric_token_reward': 0.0, 'sid_recall': 0.0, 'all_generated_items_legal': True, 'generated_items_unique': False}`
- `+Ib-Base` token accuracy：`+0.000000`
- `Full-+Ib` token accuracy：`+0.000000`
- `Full-Base` token accuracy：`+0.000000`
- `+Ib/Base`生成token变化数：`3`
- `Full/+Ib`生成token变化数：`0`
