---
number headings: off
---
# GPU侧Oxygenrec-v1复现

**数据流**
用户历史 X → Encoder  
场景指令 Is + 推理指令 Ir → Decoder → 生成目标 SID

IGR基座：Qwen3-4B-Instruct-2507

| 模块                               | 当前状态         | 已验证内容                                                                              | 问题                                                          |
| -------------------------------- | ------------ | ---------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Encoder-Decoder GR               | 已完成 GPU 方法复现 | weighted NTP、teacher forcing、toy overfit、10万样本训练、checkpoint、greedy、PrefixTrie beam | 是小型 Dense 自实现，不是论文私有规模与参数                                   |
| Contextual Reasoning Instruction | 已完成生成与结构接入   | Qwen3-4B 真实生成 JSON Reasoning/Plan；schema、证据边界和人工 review 通过                         | 文本仍有商品/类目混淆；尚未 SFT                                          |
| Q2I semantic alignment           | 已完成结构与训练验证   | query/item adapter、cosine alignment、variance/decorrelation、与 NTP 联合 loss           | 公开代理数据效果不稳定；不能外推论文收益                                        |
| IGR                              | v1论文主线GPU完成   | Qwen Instruction query检索冻结长历史、top-k拼入Encoder，Plan扩展已和论文分支分轨并透传全部生成入口 | 公开代理上的检索收益未建立；Agentic Plan对比暂缓 |
| SA-GCPO                          | v1代理GPU完成并冻结  | 真实Qwen主线checkpoint上的rollout、公开reward、group objective、更新与代表轨迹均通过 | 私有reward service不可得，32条smoke未产生held-out指标改善 |
# GPU侧Oxygenrec-v2复现

**数据流**
用户历史 X → Encoder  
场景指令 Is + 推理指令 Ir + ==行为指令 Ib==  
↓  
Decoder → 生成多个商品 SID

==后训练阶段：==  
==同一个模型 + 未来目标前缀 F → 特权教师==  
==↓==  
==通过熵门控和自蒸馏训练普通模型==

| 模块                                      | 当前状态 | 已验证内容 | 问题  |
| --------------------------------------- | ---- | ----- | --- |
| Behavior Instruction（Decoder 端行为指令）     | 已完成CUDA方法验证 | 保留行为token、两层投影、`[BOS,I_s,I_r,I_b]`及全部解码接口 | 具体首次CUDA数值未留存；真实列表已接通 |
| Behavior-aware Pretraining（行为感知的列表式预训练） | 已完成方法级GPU复现 | 目标行为ID、`1.2/1.5/2.0`权重、`[B,N,3]→[B,3N]`解码、真实daily列表；同cohort三组配对、13/32与3/32输出变化及六例review均已CUDA核验 | `I_b`唯一完整命中同时复制首目标；`weighting_gain=null`；SID碰撞使重复SID不必然等于重复原始商品，尚无稳定收益证据 |
| EA-TOSD（熵感知的轨迹优化自蒸馏）                    | 已完成方法级GPU复现 | 几何reward、best-of-G、未来SID共享Teacher、双熵蒸馏、SFT anchor、同split真实future、checkpoint迁移、配对控制及差异案例 | EA相对SFT改变80/96个参数张量及2/32条greedy列表，但未新增目标token命中，尚无收益结论 |

# NPU侧Oxygenrec-series迁移与调优

GPU侧v1/v2方法验收已经收口，NPU迁移按以下顺序执行：

```text
环境版本链与设备可见性
  → 单卡张量/backward/AdamW/checkpoint
  → v2固定输入logits/loss/greedy/beam对齐
  → 单步梯度与短训练
  → HCCL多卡
  → 吞吐、HBM与扩展效率
```

- 当前状态：Stage-0脚本已完成本地静态检查，目标Ascend服务器尚未运行；
- 入口：`bash run_npu_stage0.sh`；
- 计划与验收条件：[`docs/npu_migration_plan.md`](docs/npu_migration_plan.md)；
- GPU冻结证据：[`OxygenREC-v2 GPU方法复现验收报告`](实验记录/案例分析/OxygenREC-v2%20GPU方法复现验收报告.md)。

在Stage-0输出回传前，不把CANN/TorchNPU安装状态推断成模型兼容，不进入多卡或性能调优。
