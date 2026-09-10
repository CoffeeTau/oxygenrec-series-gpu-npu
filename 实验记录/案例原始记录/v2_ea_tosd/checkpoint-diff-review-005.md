# checkpoint-diff-review-005

> 记录类型：服务器生成Markdown的截图转录版  
> 登记日期：2026-09-10  
> 服务器产物路径：`outputs/review/v2_ea_tosd_checkpoint_comparison/checkpoint_comparison.md`  
> 完整性说明：以下字段按用户提供截图逐项转录，不是服务器文件的字节级副本。  
> 主日志：[`E122`](../../复现实验日志.md)  
> 案例分析：[`v2_ea_tosd_progress.md`](../../案例分析/v2_ea_tosd_progress.md)

- 代表角色：`['greedy_changed']`
- 目标行为：`view`
- 历史长度/行为：`1 / {'view': 1}`
- 目标SID列表：`[[130, 143, 219], [130, 130, 28]]`
- EA生成：`[[217, 214, 164], [217, 214, 164]]`
- SFT生成：`[[217, 214, 164], [97, 214, 219]]`
- greedy token变化mask：`[False, False, False, True, False, True]`
- EA目标token命中：`[False, False, False, False, False, False]`
- SFT目标token命中：`[False, False, False, False, False, False]`
- teacher-forcing argmax变化mask：`[False, False, False, False, False, False]`
- 每步最大logit差：`[0.007867306470870972, 0.0012753009796142578, 0.0015834569931030273, 0.007635787129402161, 0.0009772777557373047, 0.0017495155334472656]`
- mean/max logit差：`2.607761708e-04 / 7.867306471e-03`
- mean symmetric KL：`1.727660219e-07`
- 每步EA-SFT gold log-prob：`[0.0002551078796386719, -0.0006766319274902344, 0.0014553070068359375, 3.0040740966796875e-05, 0.0008525848388671875, 0.000118255615234375]`
- mean EA-SFT gold log-prob：`+3.391107020e-04`
- EA/SFT生成SID均合法：`True / True`
