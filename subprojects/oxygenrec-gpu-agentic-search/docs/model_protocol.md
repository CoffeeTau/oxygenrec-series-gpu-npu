# Phase-1 dense model protocol

This module is an engineering baseline, not a claim about undisclosed OxygenREC
hyperparameters.

## Inputs and shapes

```text
history item IDs
  -> one versioned SIDRegistry
  -> history_sids [B, H, 3]
  -> sum of three level-specific code embeddings [B, H, D]
  -> Transformer encoder memory [B, H, D]

default instruction [B]
  -> instruction token + BOS + shifted target SID prefix
  -> causal Transformer decoder with encoder cross-attention
  -> three independent logits tensors, each [B, W]
```

- `B`: batch size;
- `H`: padded history length;
- `D`: hidden size;
- `W`: SID codebook width.

`history_padding_mask[b, h] == True` marks padding. A sample containing only
padding is rejected. Decoder causal masking ensures level `l` can see only the
instruction, BOS, and target codes before `l`. Encoder cross-attention cannot
attend to padded history positions.

The batch builder drops history items absent from the train-fitted registry and
fails on an unknown target. Cold-start targets remain a separate evaluation set.

## Loss and decoding

Each SID level has an independent linear prediction head. The training loss is
the weight-normalized sum of the three cross-entropies. Equal weights are the
default; other values are explicit experiment assumptions.

Greedy and reference beam-search validation apply
`PrefixTrie.allowed_next(prefix)` before every selection, so a generated
three-code path must correspond to at least one item. The reference beam search
prioritizes correctness and deterministic ranking over throughput.

Ranking evaluation maps every generated SID back through the same registry.
For collisions, a candidate is a hit if the target belongs to the registry's
explicit item set. The current next-item protocol has one relevant target, so
Recall@K equals HR@K; MRR, NDCG, and legal-SID rate are also reported.

## Validation

Run all tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

Run the executable PyTorch acceptance check on CPU or one L20:

```bash
python scripts/validate_toy_model.py --device cpu
python scripts/validate_toy_model.py --device cuda
```

It checks forward/backward, single-batch overfit, an exact checkpoint logits
round-trip, and PrefixTrie-constrained generation. GPU results should record the
PyTorch build and device alongside the printed loss values.

## L20 validation record

Validated on the 8 x NVIDIA L20 server environment collected on 2026-08-21.
The acceptance script was run on one CUDA device and reported:

```text
OK device=cuda loss=2.312395->0.002499 generated=[[1, 2, 3], [7, 8, 9]]
```

This confirms the toy-model CUDA forward/backward path, single-batch overfit,
checkpoint logits round-trip, and constrained greedy decoding. It does not yet
validate multi-GPU NCCL, real-data metrics, throughput, or GPU/NPU parity.

The updated acceptance script also validated constrained beam search:

```text
beams=[[[1, 2, 3], [7, 8, 9]], [[7, 8, 9], [1, 2, 3]]]
```

Every returned candidate is a complete path in the toy `PrefixTrie`. This
validates the reference beam-search control flow on CUDA, but not ranking
quality on held-out interactions.
