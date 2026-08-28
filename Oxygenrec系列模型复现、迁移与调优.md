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
| IGR                              | 基础链完成        | query 检索冻结长历史、top-k 拼入 Encoder、真实长历史 paired 验证                                     | Qwen Plan 已接入 `forward()`，但尚未透传到 `generate()/beam_search()` |
| SA-GCPO                          | 代理结构完成       | reward、rollout、group objective 与 checkpoint 更新通过                                   | 私有 reward service 不可得，真实 Qwen RL 尚未做                        |
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
| Behavior Instruction（Decoder 端行为指令）     |      |       |     |
| Behavior-aware Pretraining（行为感知的列表式预训练） |      |       |     |
| EA-TOSD（熵感知的轨迹优化自蒸馏）                    |      |       |     |

# NPU侧Oxygenrec-series迁移与调优

