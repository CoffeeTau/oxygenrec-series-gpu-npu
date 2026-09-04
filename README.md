# OxygenREC paper-method reimplementation

This repository is a clean-room PyTorch-oriented reimplementation of
**OxygenREC: An Instruction-Following Generative Framework for E-commerce
Recommendation**.

The target is not a strict reproduction of JD.com's production system. The
paper's private data, feature definitions, checkpoints, reward service, and
serving stack are unavailable. Results produced here must be described as:

> OxygenREC paper-method reimplementation on a public-data approximate benchmark.

This is also a transition project toward general LLM and Agentic Search work.
Implementation and review prioritize transferable reasoning, retrieval,
semantic alignment, policy optimization, and trajectory methods. Recommendation-
specific SID tuning and industrial serving receive only enough work to validate
their GPU control flow.

## Current milestone

中文代码审阅建议先看 [代码阅读与数据流指南](代码阅读与数据流指南.md)。该指南按
真实调用链说明数据切分、SID、Encoder-Decoder、Contextual Reasoning、Q2I、IGR
以及当前尚未完成的边界。

Phase 1 starts with the smallest auditable loop:

```text
behavior log -> temporal split -> item SID codebook -> history SIDs
             -> encoder-decoder -> constrained SID decoding
             -> item IDs -> HR/Recall
```

Implemented now:

- an immutable three-level Semantic ID value object;
- a versioned item-to-SID registry with collision reporting;
- a prefix trie for legal constrained decoding;
- a dataset-neutral interaction schema and streaming RetailRocket adapter;
- global temporal boundaries and leak-resistant next-item sample construction;
- a deterministic residual K-means reference and SID quality diagnostics;
- a small dense Transformer encoder-decoder with level-aware SID embeddings;
- three level-specific prediction heads and weighted next-token loss;
- greedy PrefixTrie-constrained Semantic-ID decoding;
- deterministic reference beam search and HR/Recall/MRR/NDCG evaluation;
- contextual instruction fusion and Q2I semantic alignment;
- long-history IGR and bounded Qwen Retrieval Plan controls;
- local Qwen3-4B structured reasoning generation on GPU;
- public-proxy SA-GCPO objectives and rollout validation;
- dependency-free unit tests for these invariants.

Review/future work:

1. validate on CUDA that executable Qwen plans reach ``forward``, ``generate``,
   ``beam_search``, and rollout candidate log-probabilities;
2. design the real-Qwen SFT data protocol and training interface;
3. review the completed GPU-side v1 method chain and its public-data limits;
4. keep NPU migration and MoE deferred until the GPU learning objectives close.

See [the reuse survey](docs/reference_reuse.md) and
[explicit implementation assumptions](configs/assumptions.yaml). The current
model shapes, masks, loss, and validation commands are in
[the Phase-1 model protocol](docs/model_protocol.md). The bounded real-data run
is documented in [the training protocol](docs/training_protocol.md).
中文总体进度见 [复现进度](复现进度.md)，逐次实验判断与修正过程见
[复现实验日志](复现实验日志.md)。
服务器直接下载并接入Qwen的逐步操作见
[Qwen服务器接入操作指南](Qwen服务器接入操作指南.md)。

## Run the current tests

The initial SID layer uses only the Python standard library:

```bash
python3 -m unittest discover -s tests -v
```
