# Qwen真实Instruction表示接入方案

## 0. 两种可切换检索模式（2026-09-05）

当前代码把论文主线与Agentic Search扩展明确分开：

| `retrieval_mode` | 输入链路 | 检索行为 | 定位 |
|---|---|---|---|
| `paper_igr` | Qwen生成Reasoning JSON中的`intent/evidence/retrieval_strategy` → 自然语言Instruction → Qwen hidden state → `instruction_features` → query | query与长历史item向量做余弦Top-K | OxygenREC论文主线，默认模式 |
| `agentic_plan` | 与`paper_igr`使用完全相同的Instruction特征和query，另外读取`retrieval_plan` | 在相同semantic scores上执行行为、时效、重复和多样性有界重排 | 本项目的Agentic Search扩展 |

模型公开入口`src/oxygenrec/model.py`中的`OxygenRECModel.forward()`、
`generate()`、`beam_search()`和`candidate_log_probs()`均接受同一个开关：

```python
# 论文主线：不允许传retrieval_plans。
paper = model(..., instruction_features=features, retrieval_mode="paper_igr")

# Agentic扩展：必须显式传入已编译Plan。
agentic = model(
    ...,
    instruction_features=features,
    retrieval_mode="agentic_plan",
    retrieval_plans=plans,
)
```

`src/oxygenrec/llm_reasoning.py`中的`contextual_instruction_text()`故意不读取
`retrieval_plan`。因此最终paired实验可以固定数据、Qwen输出、Instruction特征、
query和OxygenREC checkpoint，只比较是否执行Plan重排。

单卡端到端验收：

```bash
CUDA_VISIBLE_DEVICES=0 bash run_server_test.sh
```

该入口会实际执行：Qwen生成结构化Reasoning → 拼装论文式自然语言Instruction →
同一冻结Qwen编码hidden state → OxygenREC instruction adapter/Q2I/query →
`paper_igr`与`agentic_plan`两次检索，并检查模式互斥契约。

## 1. 第一阶段型号

首个GPU基线选择：`Qwen/Qwen3-4B-Instruct-2507`。

选择依据：

- 官方模型为4B dense instruction-following模型，BF16权重约8.06 GB；
- hidden size为2560、36层、32个Q heads和8个KV heads；
- 官方配置要求Transformers 4.51.0，服务器已有4.57.6；
- 单卡L20实际约44.40 GiB，冻结BF16特征提取有充分余量，并为后续LoRA、OxygenREC adapter和短序列activation预留空间；
- 相比1.7B，4B更适合作为首次真实语义质量基线；相比8B，4B更便于后续在不依赖bitsandbytes/DeepSpeed的环境里做单卡LoRA控制实验。

官方来源：

- <https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507>
- <https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/config.json>
- <https://github.com/QwenLM/Qwen3>
- <https://modelscope.cn/qwen/Qwen3-4B-Instruct-2507>

## 2. 当前阶段边界

第一阶段只做：

1. 从本地目录加载官方权重，不在训练脚本中隐式联网；
2. 将严格早于目标的行为证据构造成无标签泄漏prompt；
3. 使用官方chat template格式化prompt，冻结Qwen，对比最后层hidden state的mean pooling与last-token pooling并L2归一化；
4. 将2560维特征输入现有OxygenREC `instruction_feature_adapter`；
5. 验证不同prompt产生不同特征、重复编码确定、logits变化和adapter梯度；
6. 记录峰值GPU显存。

本阶段不做SFT、RL、长文本生成、4-bit量化或多卡训练。只有冻结特征链稳定后，才进入真实RetailRocket缓存和Q2I/IGR消融。

## 3. 权重目录

默认路径：

```text
models/Qwen3-4B-Instruct-2507/
```

服务器默认使用ModelScope官方镜像下载，具体操作见项目根目录
`Qwen服务器接入操作指南.md`；Hugging Face仅保留为备用后端。

也可以通过环境变量覆盖：

```bash
QWEN_MODEL_PATH=/实际模型目录 bash run_server_test.sh
```

目录中至少应包含官方模型的`config.json`、tokenizer文件、safetensors index和全部权重分片。项目脚本设置`local_files_only=True`和`trust_remote_code=False`，缺少本地权重时明确失败，不会后台联网下载。

## 4. 验收输出

```text
OK device=cuda hidden_size=2560 tokens=(...) \
mean_delta=... mean_cosine=... last_delta=... last_cosine=... determinism_error=... \
logit_delta=... adapter_grad=... peak_allocated_gib=...
```

验收条件：

- `hidden_size=2560`；
- `mean_delta>0`且`last_delta>0`；
- `determinism_error<=1e-6`；
- `logit_delta>0`；
- `adapter_grad>0`；
- 峰值显存低于单卡可用容量，并保留后续adapter训练空间。

两个cosine只用于比较池化区分度，不预设必须低于某个主观阈值；正式缓存前依据实测选择，不凭经验决定。

## 5. 后续阶段

冻结特征通过后按顺序推进：

1. 批量缓存Qwen特征，避免每个OxygenREC epoch重复运行4B模型；
2. 与Generic和Hash变体做matched Q2I/IGR消融；
3. 导出真实Qwen特征驱动的review case；
4. 构造并人工审核少量结构化instruction SFT数据；
5. 单卡BF16 LoRA SFT；
6. 在SFT效果成立后再设计偏好优化/SA-GCPO轨迹，不提前进入RL。
