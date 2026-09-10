# checkpoint-diff-review-003

> 记录类型：服务器生成Markdown的截图转录版  
> 登记日期：2026-09-10  
> 服务器产物路径：`outputs/review/v2_ea_tosd_checkpoint_comparison/checkpoint_comparison.md`  
> 完整性说明：以下字段按用户提供截图逐项转录，不是服务器文件的字节级副本。  
> 主日志：[`E122`](../../复现实验日志.md)  
> 案例分析：[`v2_ea_tosd_progress.md`](../../案例分析/v2_ea_tosd_progress.md)

- 代表角色：`['largest_ea_gold_drop']`
- 目标行为：`view`
- 历史长度/行为：`10 / {'addtocart': 2, 'transaction': 1, 'view': 7}`
- 目标SID列表：`[[38, 109, 143], [38, 86, 3]]`
- EA生成：`[[217, 214, 164], [254, 214, 146]]`
- SFT生成：`[[217, 214, 164], [254, 214, 146]]`
- greedy token变化mask：`[False, False, False, False, False, False]`
- EA目标token命中：`[False, False, False, False, False, False]`
- SFT目标token命中：`[False, False, False, False, False, False]`
- teacher-forcing argmax变化mask：`[False, False, False, False, False, False]`
- 每步最大logit差：`[0.008062779903411865, 0.0010352134704589844, 0.0013477802276611328, 0.007831454277038574, 0.0010285377502441406, 0.0014705657958984375]`
- mean/max logit差：`2.473236818e-04 / 8.062779903e-03`
- mean symmetric KL：`1.504901519e-07`
- 每步EA-SFT gold log-prob：`[-0.0007390975952148438, -0.0006108283996582031, -6.0558319091796875e-05, -0.000514984130859375, 0.00014781951904296875, 0.00015878677368164062]`
- mean EA-SFT gold log-prob：`-2.698103490e-04`
- EA/SFT生成SID均合法：`True / True`
