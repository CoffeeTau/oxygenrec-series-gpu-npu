# v2 Full 固定验证：L20 与 Ascend 950DT

记录日期：2026-09-17
阶段：FP32/BF16 短训练通过后的固定 checkpoint、固定 cohort 正确性验证
证据来源：GPU 与 NPU 服务器运行结果及各自原始 `validation_summary.json`

## 输入与执行条件

- 四次运行的源码 commit 均为 `c42fb54a28feddcad2056edcd6fb9ea03685d361`，
  `tracked_dirty=false`，`untracked_file_count=0`；验证用源码文件 SHA-256 对应一致。
- 事件、Full checkpoint、SID registry 三个输入 SHA-256 对应一致；target 输出指纹一致。
- GPU 为 NVIDIA L20，NPU 为 Ascend950DT_9572；两端分别运行 FP32、BF16。
- 固定 validation 为 32 条、每条 2 个目标商品、beam width 5、seed 17；
  `train_end_ms=1440160551108`，`validation_end_ms=1441352869448`。
- 四次均报告 `status=passed`、`all_logits_finite=true`、greedy/beam legal item rate 为 1.0。
- 为跨设备正确性检查，两端都关闭 Transformer MHA fastpath；`execution.scope` 为
  `correctness_only_not_performance_benchmark`。

服务器原始文件路径：

```text
checkpoints/device_validation/v2_full/gpu/fp32/v2_full_gpu_fp32_validation_summary.json
checkpoints/device_validation/v2_full/gpu/bf16/v2_full_gpu_bf16_validation_summary.json
checkpoints/device_validation/v2_full/npu/fp32/v2_full_npu_fp32_validation_summary.json
checkpoints/device_validation/v2_full/npu/bf16/v2_full_npu_bf16_validation_summary.json
```

## 配对结果

| 指标 | GPU FP32 | NPU FP32 | GPU BF16 | NPU BF16 |
|---|---:|---:|---:|---:|
| mean loss | 5.8189945221 | 5.8189883232 | 5.8189778328 | 5.8184895515 |
| SID recall | 0.015625 | 0.015625 | 0.015625 | 0.015625 |
| position accuracy | 0.015625 | 0.015625 | 0.015625 | 0.015625 |
| SID token accuracy | 0.057291667 | 0.057291667 | 0.0625 | 0.0625 |
| exact-list rate | 0 | 0 | 0 | 0 |
| beam exact-list hit@5 | 0 | 0 | 0 | 0 |
| greedy/beam legal item rate | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |

相同精度的 mean loss：FP32 GPU/NPU 绝对差 `6.198883e-6`、相对差约
`0.0001065%`；BF16 绝对差 `0.00048828125`、相对差约 `0.0083912%`。
同一设备 BF16 相对 FP32 的 mean-loss 差也远低于预设 `1%` 门槛。所有聚合
离散指标完全一致，且 greedy 与 beam 的合法性均通过。

指纹提供比聚合指标更强的判断：

- FP32：GPU/NPU 的 target、greedy、beam 指纹对应相同，离散结果一致。
- BF16：target 与 greedy 指纹对应相同，但 beam 指纹不同。两端 beam 的 aggregate
  hit/MRR/NDCG 虽然同为零，不能据此说逐样本候选列表一致。

## 判断与边界

按服务器运行前冻结的门槛，四次运行通过硬门槛、连续量和离散聚合门槛；FP32
固定验证可以标为通过。BF16 不直接标为失败，但必须按既定规则下钻**beam 发生变化的
固定案例**，在确认变化范围和原因前不宣称 BF16 生成路径完全对齐。

`SID recall=1/64`、`exact-list=0/32` 体现的是当前公开代理 checkpoint 的质量限制，
不是 GPU/NPU 差距；它不能被写成论文效果复现。终端的 nested-tensor warning 仅说明
该优化路径没有启用，本次结果已经完成，不能单凭它断定数值异常。

四次运行耗时只属于 correctness-only 协议，不可用来报告吞吐、时延加速比或性能调优
收益。NPU validation 记录的 TorchNPU 开发版日期与此前训练/环境记录不完全相同；
正式性能测试前还需重新采集运行环境和算子执行位置。

## 下一步

1. 已在后续提交中为固定验证增加小型逐案例文件，并新增
   `scripts/compare_v2_validation_cases.py`，比较时只输出发生变化的案例，不上传全量
   logits 或大型探针 JSON。
2. 用更新后的相同 commit、输入与精度在 GPU/NPU 各重跑一次，先确定变化案例数、候选集合
   与顺序的差异；如只是近似并列排序，记录 margin；如涉及不稳定或非法路径，再深入
   到 beam 步级数值。
3. 完成 BF16 差异分类和 NPU CPU fallback/算子位置核实后，另开正式性能协议：
   warmup、同步计时、重复运行、显存口径、Profiler 和同一有效 batch。当前日志不能
   代替该阶段。

更新代码后，两端只需重跑 BF16；`run_v2_validation.sh` 会同时生成 summary 和小型
cases 文件：

```bash
# GPU服务器
CUDA_VISIBLE_DEVICES=0 bash run_v2_validation.sh gpu bf16

# NPU服务器（环境脚本仍需按迁移计划加载）
NPU_DEVICE=npu:0 bash run_v2_validation.sh npu bf16
```

把两份小型 cases 文件放到同一台机器后执行：

```bash
python3 scripts/compare_v2_validation_cases.py \
  --gpu /path/to/v2_full_gpu_bf16_validation_cases.json \
  --npu /path/to/v2_full_npu_bf16_validation_cases.json \
  --output /path/to/v2_full_bf16_changed_cases.json
```

最终只需要查看 `greedy_changed_cases`、`beam_changed_cases` 和 `changed_cases`，无需传递
全量 logits。
