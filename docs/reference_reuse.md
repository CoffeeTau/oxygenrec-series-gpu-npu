# Public-code reuse survey

Survey date: 2026-08-21

Paper checked into this workspace:

- `Hao 等 - 2025 - OxygenREC An Instruction-Following Generative Framework for E-commerce Recommendation.pdf`
- SHA-256: `ea9b888860e562bc54e8fe8d787c13a79ee8ff0ca5c9019b0fbe193257b9e003`

No JD-authored public repository containing OxygenREC's model-training code was
found. The following projects are the closest reusable references. A project is
not treated as an OxygenREC implementation merely because it generates SIDs.

## Decision matrix

| Project | Inspected revision | License | Useful modules | Decision |
|---|---|---|---|---|
| [MiniOneRec](https://github.com/AkaliKong/MiniOneRec) | `0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed` | Apache-2.0 | RQ-KMeans variants, SID construction, constrained logits, HR/NDCG, SFT/GRPO workflow | Primary reference for SID and decoding; adapt interfaces, do not inherit its decoder-only prompt format |
| [OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec) | `a969edcadd579a06c1966ae1db5984e02f48beff` | Apache-2.0 for code; weights have separate terms | public instruction benchmark, pretraining, veRL-based distillation/RL | Revisit for later instruction/RL stages; too large and architecturally different for the first closed loop |
| [GRID](https://github.com/snap-research/GRID) | `2fe3475b2d369580234093f35d52b1a2f54d0472` | non-commercial research only | modular residual quantization, T5 encoder-decoder, per-level heads, prefix-pruned beam search, metrics | Algorithm/reference reading only; no source copying into this project without a separate license review |
| [LIGER](https://github.com/facebookresearch/liger) | web review only | mostly CC-BY-NC; RQ-VAE subtree Apache-2.0 | RQ-VAE and generative+dense evaluation | Optional comparison; licensing and hybrid reranking make it a poor base |
| [TIGER external implementation](https://github.com/NonameUntitled/tiger) | web review only | MIT | small seq2seq SID baseline and processed Amazon example | Useful smoke-test comparison, but the repository documents a parameter-count discrepancy from the paper |

## Module-by-module migration boundary

### Reuse or adapt early

- Residual quantization algorithm and codebook diagnostics.
- Level-aware SID vocabulary rather than treating all code levels as identical.
- Prefix-trie constrained beam decoding.
- Legal-SID ratio, HR@K, Recall@K, and NDCG@K tests.
- Dataset converters as format references, after independently verifying temporal
  splitting and leakage boundaries.

### Implement specifically for OxygenREC

- Encoder-decoder fast backbone with the instruction token immediately after BOS.
- Scenario ID plus optional trigger-item fusion.
- Query and item adapters, Q2I alignment loss, and IGR over long-term history.
- Weighted NTP for click/cart/purchase signals.
- Near-line instruction-store interface and default-instruction fallback.
- OxygenREC's multi-reward mapping and SA-GCPO objective.

### Defer

- Multimodal Qwen3+CLIP+Q-Former item encoder. Start with auditable precomputed
  text embeddings, then replace the encoder without changing the SID API.
- MoE and 3B scaling. First establish a correct dense 100M-300M reference.
- xGR/xLLM serving optimizations. First validate decoding semantics and metrics.

## Why this repository starts with a clean SID API

The reviewed projects disagree on collision handling, whether hierarchy tokens
share one vocabulary, and whether an extra deduplication digit is appended. The
OxygenREC paper specifies three RQ-KMeans levels with width 8192, but does not
fully specify collision resolution. Therefore the registry records collisions
explicitly instead of silently inventing a fourth token.

