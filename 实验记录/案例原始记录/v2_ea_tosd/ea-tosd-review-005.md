# ea-tosd-review-005

> 记录类型：服务器生成Markdown的截图转录版  
> 登记日期：2026-09-10  
> 服务器产物路径：`checkpoints/retailrocket_v2_ea_tosd_smoke/ea_tosd_representative_trajectories.md`  
> 完整性说明：以下字段按用户提供截图逐项转录，不是服务器文件的字节级副本；未在截图中出现的字段不补造。  
> 主日志：[`E119`](../../复现实验日志.md)  
> 案例分析：[`v2_ea_tosd_progress.md`](../../案例分析/v2_ea_tosd_progress.md)

- 代表角色：`['nonzero_reward']`
- 目标行为：`view`
- gold SID列表：`[217, 136, 66, 8, 0, 218]`
- Teacher未来SID：`[[217, 95, 28]]`
- Teacher未来行为：`['view']`
- 相对gold末尾的小时差：`[0.015003]`
- future是否与gold SID重合：`[False]`
- G条候选：`[[217, 246, 7, 16, 21, 209], [78, 214, 146, 221, 76, 203], [119, 150, 56, 86, 2, 52], [180, 109, 236, 176, 55, 240]]`
- 候选逐token命中：`[[True, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]`
- 候选reward：`[0.2134203016757965, 0.0, 0.0, 0.0]`
- best索引：`0`
- best SID：`[217, 246, 7, 16, 21, 209]`
- best reward：`0.213420`
- Teacher entropy：`[5.290178298950195, 4.70051383972168, 4.713688850402832, 5.2935872077941895, 4.692622184753418, 4.699352264404297]`
- privilege advantage：`[0.28411126136779785, 0.0004439353942871094, 0.007111072540283203, 0.0005321502685546875, -0.00685882568359375, -0.017114639282226562]`
- low/high gate token：`0/6`

## 截图中的人工Review清单

- [ ] future SID均能在该用户gold列表之后的真实行为中找到
- [ ] reward只由候选SID与gold SID逐token命中解释
- [ ] best-of-G确实选择当前组最高reward轨迹
- [ ] 低熵与高熵门和Teacher entropy一致
- [ ] 未把SID代理命中误写成商品级工业指标
