# NPU batch 4096 Profile热点结果

> 记录类型：服务器运行结果摘录  
> 实验编号：`E143`  
> 登记日期：`2026-09-24`  
> 服务器产物路径：`checkpoints/performance_profiling/npu_bs4096_short_regenerated_20260923/`  
> 主日志：[`E143`](../../复现实验日志.md)  
> 案例分析：[`v2 BF16单卡性能调优分析`](../../案例分析/v2_bf16_single_card_performance_tuning.md)

## 1. 运行身份与输入

- commit：`23b1d8cbae97d8b4938a1a81525eac6e1c6c0206`；
- tracked dirty：`false`；untracked count：`0`；
- device：`npu:0` / `Ascend950DT_9572`；
- Python：`3.11.0`；PyTorch：`2.7.1+cpu`；TorchNPU：`2.7.1.post4`；
- BF16、batch 4096、AdamW、optimizer state已加载；
- 外部warmup 100步；Profiler内warmup 1步、active 3步；
- checkpoint、events和SID registry三个SHA-256与E141一致。

## 2. Profile摘要

```text
status=passed
measured_steps=4
elapsed_seconds=28.680230789992493
mean_step_seconds=7.170057697498123
samples_per_second=571.2645801203571
peak_memory_allocated_bytes=2030521344
first_loss=5.404853820800781
last_loss=5.36884880065918
mean_loss=5.363986968994141
all_losses_finite=true
```

该计时包含Profiler开销，只用于定位算子，不与无Profiler吞吐基线比较。

## 3. Top operator（Device Self Duration）

| 排名 | operator | calls | 设备自耗时占比 |
|---:|---|---:|---:|
| 1 | `aclnnDropoutV3` | 63 | 24.13% |
| 2 | `aclnnInplaceCopy` | 864 | 15.93% |
| 3 | `aclnnFlashAttentionScoreGradV4` | 6 | 7.44% |
| 4 | `aclnnMm` | 144 | 6.23% |
| 5 | `aclnnMatmul` | 126 | 6.15% |
| 6 | `aclnnFlashAttentionScoreV4` | 6 | 4.99% |
| 7 | `aclnnEmbeddingDenseBackward` | 42 | 4.39% |
| 8 | `aclnnLayerNormBackward` | 36 | 4.32% |
| 9 | `aclnnInplaceAdd` | 105 | 3.46% |
| 10 | `aclnnAddmm` | 78 | 2.73% |

关键词汇总还记录到`_local_scalar_dense`共534次。其设备自耗时为0不代表没有主机同步代价；
按3个active step折算约178次/step，是需要通过A/B验证的host-bound候选。

## 4. Top kernel族

- Dropout：`DropoutV3` 24.02% + `DropoutDoMask` 1.72% + Dropout transpose 0.31%，
  合计约26.05%；
- copy/layout：InplaceCopy transpose 8.27% + cast 6.16% + TensorMove 1.34%，
  合计约15.77%；
- FlashAttention正反向及其transpose约12.53%；
- 单个最大kernel仍是`DropoutV3`，63次、24.02%。

## 5. 证据边界

- dropout占比最高，但关闭dropout会改变训练语义，不能作为本轮公平性能优化；
- copy/layout占比是明确的第二优化方向，但需要先区分其中有多少来自原生AdamW状态更新；
- `_local_scalar_dense`的534次只证明高频调用，当前汇总未给出其Host Total Duration，不能直接声称它占用多少总时间；
- Profile摘要的571.26 samples/s不代表真实稳态吞吐回退。

## 6. 决策

第一项受控优化为原生`torch.optim.AdamW`对比
`torch_npu.optim.NpuFusedAdamW`。它不改变模型、dropout、batch或训练目标，并同时针对optimizer
标量状态、原地更新和copy调用。若没有至少5%的稳定收益，下一轮转向padding mask/layout copy，
不围绕Profiler warning继续做无边界验证。
