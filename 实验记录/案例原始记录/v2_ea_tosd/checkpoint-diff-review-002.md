# checkpoint-diff-review-002

> 记录类型：服务器生成Markdown的截图转录版  
> 登记日期：2026-09-10  
> 服务器产物路径：`outputs/review/v2_ea_tosd_checkpoint_comparison/checkpoint_comparison.md`  
> 完整性说明：以下字段按用户提供截图逐项转录，不是服务器文件的字节级副本。  
> 主日志：[`E122`](../../复现实验日志.md)  
> 案例分析：[`v2_ea_tosd_progress.md`](../../案例分析/v2_ea_tosd_progress.md)

- 代表角色：`['largest_ea_gold_gain']`
- 目标行为：`view`
- 历史长度/行为：`5 / {'view': 5}`
- 目标SID列表：`[[91, 214, 122], [109, 130, 156]]`
- EA生成：`[[97, 214, 219], [97, 214, 219]]`
- SFT生成：`[[97, 214, 219], [97, 214, 219]]`
- greedy token变化mask：`[False, False, False, False, False, False]`
- EA目标token命中：`[False, True, False, False, False, False]`
- SFT目标token命中：`[False, True, False, False, False, False]`
- teacher-forcing argmax变化mask：`[False, False, False, False, False, False]`
- 每步最大logit差：`[0.00787276029586792, 0.0008478164672851562, 0.0011612305641174316, 0.007820338010787964, 0.0012146234512329102, 0.0015190839767456055]`
- mean/max logit差：`2.512794163e-04 / 7.872760296e-03`
- mean symmetric KL：`1.524899744e-07`
- 每步EA-SFT gold log-prob：`[0.0002608299255371094, -2.384185791015625e-06, 0.001270294189453125, 0.0002484321594238281, 0.0003261566162109375, 8.535385131835938e-05]`
- mean EA-SFT gold log-prob：`+3.647804260e-04`
- EA/SFT生成SID均合法：`True / True`
