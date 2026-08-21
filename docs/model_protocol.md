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

Greedy validation applies `PrefixTrie.allowed_next(prefix)` before every
selection, so a generated three-code path must correspond to at least one item.
Beam search is intentionally deferred until greedy correctness is validated.

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
