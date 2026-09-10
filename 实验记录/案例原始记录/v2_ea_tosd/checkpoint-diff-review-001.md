# checkpoint-diff-review-001

> 记录类型：服务器生成Markdown的截图转录版  
> 登记日期：2026-09-10  
> 服务器产物路径：`outputs/review/v2_ea_tosd_checkpoint_comparison/checkpoint_comparison.md`  
> 完整性说明：以下字段按用户提供截图逐项转录，不是服务器文件的字节级副本。  
> 主日志：[`E122`](../../复现实验日志.md)  
> 案例分析：[`v2_ea_tosd_progress.md`](../../案例分析/v2_ea_tosd_progress.md)

- 代表角色：`['greedy_changed']`
- 目标行为：`view`
- 历史长度/行为：`1 / {'view': 1}`
- 目标SID列表：`[[119, 167, 3], [119, 119, 185]]`
- EA生成：`[[217, 119, 241], [97, 214, 219]]`
- SFT生成：`[[97, 214, 219], [97, 214, 219]]`
- greedy token变化mask：`[True, True, True, False, False, False]`
- EA目标token命中：`[False, False, False, False, False, False]`
- SFT目标token命中：`[False, False, False, False, False, False]`
- teacher-forcing argmax变化mask：`[True, False, False, False, False, False]`
- 每步最大logit差：`[0.008095353841781616, 0.0011435747146606445, 0.0013604164123535156, 0.007518798112869263, 0.00110160207748413086, 0.0013962090015411377]`
- mean/max logit差：`2.595232800e-04 / 8.095353842e-03`
- mean symmetric KL：`1.663518105e-07`
- 每步EA-SFT gold log-prob：`[-0.0004591941833496094, -0.000270843505859375, 0.00016069412231445312, -0.0009083747863769531, 0.0002739429473876953, -0.0001068115234375]`
- mean EA-SFT gold log-prob：`-2.184311597e-04`
- EA/SFT生成SID均合法：`True / True`
