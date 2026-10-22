# NPU融合AdamW匹配zero-grad A/B结果

> 记录类型：服务器运行结果摘录  
> 实验编号：`E147`  
> 服务器结果日期：`2026-09-25`  
> 产物根目录：`checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925/`  
> source commit：`c67c593639f35a282f2497ad823a75e32d7c4b5b`

## 1. 可比性

- 平台/设备：`npu:0` / `Ascend950DT_9572`；
- precision：BF16；batch：4096；
- warmup：100步；measured：100步；repeats：3；
- control/treatment均为`zero_grad_mode=zero`；
- 输入checkpoint、events、SID registry以及记录的关键源码SHA-256一致；
- 唯一实验变量：`AdamW`与`NpuFusedAdamW`。

## 2. 聚合结果

| 指标 | Control AdamW | Treatment NpuFusedAdamW | 变化 |
|---|---:|---:|---:|
| 中位吞吐（samples/s） | 27150.7734 | 27879.6196 | +2.6844% |
| 中位step时延（ms） | 150.8613 | 146.9174 | -2.6143% |
| 吞吐CV | 0.9184% | 0.9593% | 均低于1% |
| 最大allocated memory（bytes） | 2035599872 | 2034164224 | -0.0705% |

control范围为`26640.9226–27181.4078 samples/s`，treatment范围为
`27394.5004–28014.8340 samples/s`。三次repeat范围没有重叠，因此本轮约2.68%的小幅收益具有一致性，
但仍低于预先设定的5%工程门槛。

## 3. Loss结果与口径纠正

两组所有loss均有限。中位数为：

| 测量窗口字段 | Control | Treatment | Treatment-Control |
|---|---:|---:|---:|
| first loss | 5.192893 | 5.057745 | -0.135148 |
| last loss | 4.564098 | 4.392895 | -0.171203 |
| mean loss | 4.842016 | 4.680082 | -0.161934 |

`first_loss`不是第一次optimizer更新前的loss，而是各repeat完成100步warmup后的第一个测量loss。
因此它不应被要求完全一致；差异说明两种优化器经过warmup后已形成不同数值轨迹。当前scope明确为
`optimizer_performance_ab_not_quality_training`，这些loss不能解释为推荐质量提升。

## 4. 产物路径

需要分析的原始结果：

```text
checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925/comparison.json
checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925/control_adamw/npu_bf16_performance.json
checkpoints/performance_optimization/npu_fused_adamw_zero_grad_20260925/treatment_npu_fused_adamw/npu_bf16_performance.json
```

服务器日志和设备状态继续保存在同一根目录。

## 5. 结论

`[运行通过-收益不足未采纳]`：融合AdamW有稳定但较小的性能收益，未达到5%门槛；同时数值轨迹
发生变化。停止该路线，不做第二次确认，也不替换正式训练默认优化器。下一轮使用现有Profile定位
layout/copy调用来源，避免重复采集同类Profile。
