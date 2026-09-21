# OxygenREC-v2 BF16 单卡性能调优分析

## 1. 证据入口

- [batch 64→4096 服务器结果摘录](../案例原始记录/performance_tuning/2026-09-21-v2-bf16-batch-sweep.md)
- [NPU Profiler API 与设备状态预检查](../案例原始记录/performance_tuning/2026-09-21-npu-profiler-precheck.md)
- 服务器原始 JSON 仍保存在 `checkpoints/performance_baseline/` 对应实验目录。

## 2. 判断演变与试错价值

### 初始 64/128/256 结果不能直接归因于模型或 NPU 算子

最初 NPU 吞吐只有 GPU 的约 `43.8%–47.6%`。当时的协议只有20步 warmup，且 batch 64
的 NPU step 时延约66 ms。受控复核把 warmup 提高到100步、测量提高到300步后，NPU batch 64
变为 `3459.94 samples/s`，反而是 GPU 的 `1.506×`。这说明旧结果主要暴露了预热、运行状态
或短窗口问题，不能据此修改模型结构。

### 扩大 batch 是有效的基础摸底步骤

从 batch 128 到1024，两端吞吐持续增长，allocated memory 仍远低于设备容量。NPU/GPU 吞吐比
从约 `1.35×` 收敛到约 `1.24×`，说明固定下发/同步开销随 batch 增大被摊薄。这个过程证明
此前先摸 batch 曲线、再开 Profiler 的顺序是合理的。

### batch 2048 的第一次结果是有价值的失败实验

30步测量窗口下 GPU CV 达 `19.92%`，导致表面 NPU/GPU 比值达到 `1.589×`。该点不是“最好
结果”，而是测量协议失稳。它直接推动了 `V2_PERF_CYCLE_SAMPLES=1` 和100步长窗口复核；复核后
GPU batch 2048 CV 降为 `0.329%`，比值回到 `1.405×`。失败点应保留，避免以后重复把短窗口
抖动解释成硬件收益。

### 吞吐膝点位于 2048–4096 区间

- GPU：1024→2048 吞吐约增加 `4.0%`，2048→4096 约增加 `14.4%`；
- NPU：1024→2048 约增加 `18.0%`，2048→4096 约增加 `12.5%`；
- batch 4096 获得当前最高吞吐，但 GPU CV 为 `6.63%`，高于 batch 2048；
- batch 2048 时延更低且两端更稳定，batch 4096 更适合观察峰值吞吐路径。

因此不再继续盲目扩大 batch。下一阶段在 batch 4096 上采集少量稳态 step 的 Profiler，定位
主机下发、H2D/Memcpy、format/TransData、CPU fallback、loss 主机读取同步、masked_fill 与 AdamW
等候选瓶颈；batch 2048保留为稳定对照点。

## 3. 当前环境风险

Profiler API 已具备，但 `npu-smi` 显示所有设备均非 `OK`，设备6为 `Critical`，设备0为
`Warning` 且在 Util=0 时仍占用约48.4 GB HBM。官方定义中 Warning/Critical 分别代表一般告警
和紧急告警，因此在取得健康 Error Code、usages 和进程信息前，不能把该环境标记为健康。

这不自动否定此前结果：此前稳定 repeat 的 CV 较低，且设备0并非 Critical。但正式 Profile
必须同时归档采集前后设备状态；若设备0健康状态变化、存在其他活跃进程或吞吐显著漂移，
该次 Profile 只作为诊断样本，不进入优化前后量化比较。

## 4. 后续每轮记录要求

每轮性能实验必须同时保存：

1. 带日期的服务器命令文档；
2. Git commit、tracked worktree 状态及关键源码/input SHA-256；
3. 完整原始 JSON 或 Profiler 输出目录，不只保存终端最后一行；
4. 开始前与结束后的 `npu-smi info`；
5. batch、warmup、measured steps、repeat、精度和设备；
6. 吞吐、step 时延、CV、allocated memory 以及失败/中断信息；
7. 本轮发现、判断、采取的处理、结论边界和下一步；
8. 若废弃某个结果，保留结果并明确废弃原因，不删除失败记录。

