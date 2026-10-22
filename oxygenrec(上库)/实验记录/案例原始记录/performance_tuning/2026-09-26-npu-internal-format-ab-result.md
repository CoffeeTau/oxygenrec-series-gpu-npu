# NPU私有格式关闭/开启A/B结果

> 记录类型：服务器运行结果摘录  
> 实验编号：`E149`  
> 服务器结果日期：`2026-09-26`  
> 产物根目录：`checkpoints/performance_optimization/npu_internal_format_20260926/`  
> source commit：`82ac5c3ec89206857971ecc98fea065481f05dba`

## 1. 可比性

- 平台/设备：`npu:0` / `Ascend950DT_9572`；
- precision：BF16；batch：4096；
- warmup：100步；measured：100步；repeats：3；
- 两组均使用`AdamW`、`zero_grad_mode=set_to_none`；
- 输入哈希、关键源码哈希和除`npu_internal_format`外的workload一致；
- control显式`disable`，treatment显式`enable`。

## 2. 聚合结果

| 指标 | Control disable | Treatment enable | 变化 |
|---|---:|---:|---:|
| 中位吞吐（samples/s） | 27206.6611 | 27075.9782 | -0.4803% |
| 中位step时延（ms） | 150.5514 | 151.2780 | +0.7266 ms |
| 吞吐CV | 0.9525% | 1.4046% | treatment波动更高 |
| 最大allocated memory（bytes） | 2030521344 | 2030521344 | 0 |

control三次吞吐范围为`26728.8681–27326.0187 samples/s`，treatment范围为
`26656.0779–27587.1728 samples/s`，两组范围高度重叠。control/treatment平均吞吐分别约为
`27087.1826/27106.4096 samples/s`，均值只差约`+0.071%`，且方向与中位数相反。

## 3. Loss证据

两组所有loss均有限，对应repeat的loss值以及下列中位数完全一致：

| 测量窗口字段 | Control | Treatment | 绝对差 |
|---|---:|---:|---:|
| first loss | 5.192893 | 5.192893 | 0 |
| last loss | 4.564098 | 4.564098 | 0 |
| mean loss | 4.842016 | 4.842016 | 0 |

这说明本轮开关没有改变已记录的loss轨迹，但这里只是性能窗口证据，不升级为完整质量结论。

## 4. 产物路径

```text
checkpoints/performance_optimization/npu_internal_format_20260926/comparison.json
checkpoints/performance_optimization/npu_internal_format_20260926/control_internal_format_disabled/npu_bf16_performance.json
checkpoints/performance_optimization/npu_internal_format_20260926/treatment_internal_format_enabled/npu_bf16_performance.json
```

完整终端日志、设备前后状态和文件清单继续保存在同一根目录。

## 5. 结论

`[运行通过-无性能收益未采纳]`：开启NPU私有格式后中位吞吐下降约0.48%，平均吞吐近乎不变，
显存完全相同，且两组repeat范围高度重叠。该开关对当前OxygenREC训练链没有可确认的端到端收益，
不做第二次确认，也不修改正式训练默认值。此前`masked_fill_`私有格式warning不能据此升级为主要瓶颈。

