# NPU延后loss主机读取A/B结果

> 记录类型：服务器运行结果摘录
> 实验编号：`E153`
> 登记日期：`2026-09-27`
> 服务器产物根路径：`checkpoints/performance_optimization/npu_deferred_loss_read_20260927/`
> 主日志：[`E153`](../../复现实验日志.md)
> 案例分析：[`v2 BF16单卡性能调优分析`](../../案例分析/v2_bf16_single_card_performance_tuning.md)

## 1. 可比性

- commit：`eec6b27987829d2818269b6023a2b15a9690659b`；tracked dirty为`false`；
- `npu:0` / `Ascend950DT_9572`，BF16，batch 4096；
- 两组均为AdamW、`set_to_none`、标准Dropout、internal format关闭；
- 100步warmup、100步测量、3 repeats，输入哈希与关键源码哈希一致；
- 唯一变量：control在每个计时step内读取loss，treatment在计时结束后统一读取；两组都保存并检查
  全部测量loss。

## 2. 聚合结果

| 指标 | control：逐step读取 | treatment：延后读取 | 变化 |
|---|---:|---:|---:|
| 中位吞吐（samples/s） | 27356.6887 | 27417.9353 | +0.2239% |
| 中位step（ms） | 149.7257 | 149.3913 | -0.3345 ms |
| 吞吐CV | 0.7381% | 0.8982% | +0.1601个百分点 |
| 最小吞吐（samples/s） | 26943.4880 | 26935.6511 | treatment略低 |
| 最大吞吐（samples/s） | 27381.7801 | 27486.1826 | 区间高度重叠 |
| allocated memory（bytes） | 2030623232 | 2030623232 | 相同 |

三个repeat吞吐：

```text
control:   26943.4880, 27356.6887, 27381.7801
treatment: 26935.6511, 27486.1826, 27417.9353
```

first/last/mean loss中位数分别为`5.1928925514 / 4.5640978813 / 4.8420161057`，两组完全一致，
所有loss有限。

## 3. 判断

`+0.2239%`远低于预设2%门槛，且两组范围高度重叠；约`0.3345 ms/step`的差异不足以区分于本轮
运行波动。该结果说明当前每步一次显式loss读取不是端到端主瓶颈，不能用E143中全部
`_local_scalar_dense`调用次数来归因这一处Python转换。

结论：`[运行通过-无稳定收益-未采纳]`。不重复实验，也不继续搜索更多读取间隔。
