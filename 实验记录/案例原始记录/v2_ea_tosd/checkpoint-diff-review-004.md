# checkpoint-diff-review-004

> 记录类型：服务器生成Markdown的截图转录版  
> 登记日期：2026-09-10  
> 服务器产物路径：`outputs/review/v2_ea_tosd_checkpoint_comparison/checkpoint_comparison.md`  
> 完整性说明：以下字段按用户提供截图逐项转录，不是服务器文件的字节级副本。  
> 主日志：[`E122`](../../复现实验日志.md)  
> 案例分析：[`v2_ea_tosd_progress.md`](../../案例分析/v2_ea_tosd_progress.md)

- 代表角色：`['largest_logit_delta']`
- 目标行为：`view`
- 历史长度/行为：`20 / {'addtocart': 2, 'transaction': 2, 'view': 16}`
- 目标SID列表：`[[227, 72, 112], [65, 22, 90]]`
- EA生成：`[[97, 214, 219], [97, 214, 219]]`
- SFT生成：`[[97, 214, 219], [97, 214, 219]]`
- greedy token变化mask：`[False, False, False, False, False, False]`
- EA目标token命中：`[False, False, False, False, False, False]`
- SFT目标token命中：`[False, False, False, False, False, False]`
- teacher-forcing argmax变化mask：`[False, False, False, False, False, False]`
- 每步最大logit差：`[0.008354246616363525, 0.000970005989074707, 0.0013915300369262695, 0.007602781057357788, 0.0009675025939941406, 0.0018091201782226562]`
- mean/max logit差：`2.592827368e-04 / 8.354246616e-03`
- mean symmetric KL：`1.593657970e-07`
- 每步EA-SFT gold log-prob：`[-0.0003008842468261719, -0.0007262229919433594, -2.5987625122070312e-05, -9.918212890625e-05, 0.0008542537689208984, -1.9073486328125e-05]`
- mean EA-SFT gold log-prob：`-5.284945291e-05`
- EA/SFT生成SID均合法：`True / True`
