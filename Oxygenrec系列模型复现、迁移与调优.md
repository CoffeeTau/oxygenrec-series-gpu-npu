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
| Behavior-aware Pretraining（行为感知的列表式预训练） | 真实数据CUDA smoke通过 | 目标行为ID、`1.2/1.5/2.0`权重、`[B,N,3]→[B,3N]`解码、真实daily行为同质列表及代表案例 | 欠充分训练，尚未完成PT-only收益对照 |
| EA-TOSD（熵感知的轨迹优化自蒸馏）                    | 真实公开数据代理CUDA贯通，配对归因中 | 几何reward、best-of-G、未来SID共享Teacher、双熵蒸馏、SFT anchor、同split真实future与checkpoint迁移 | 28.125%样本非零reward；五个正反案例已review，单步token指标略降，SFT-only配对对照待CUDA，尚无收益结论 |

# NPU侧Oxygenrec-series迁移与调优
