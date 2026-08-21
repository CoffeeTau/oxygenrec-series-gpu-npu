# OxygenREC paper-method reimplementation

This repository is a clean-room PyTorch-oriented reimplementation of
**OxygenREC: An Instruction-Following Generative Framework for E-commerce
Recommendation**.

The target is not a strict reproduction of JD.com's production system. The
paper's private data, feature definitions, checkpoints, reward service, and
serving stack are unavailable. Results produced here must be described as:

> OxygenREC paper-method reimplementation on a public-data approximate benchmark.

## Current milestone

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
- dependency-free unit tests for these invariants.

Planned next:

1. approve and acquire a public benchmark dataset;
2. item embedding import and residual K-means;
3. a small dense encoder-decoder and weighted next-token prediction;
4. constrained beam search and HR/Recall evaluation;
5. instruction fusion, Q2I, and IGR;
6. multi-scenario post-training and, only after dense validation, MoE.

See [the reuse survey](docs/reference_reuse.md) and
[explicit implementation assumptions](configs/assumptions.yaml).

## Run the current tests

The initial SID layer uses only the Python standard library:

```bash
python3 -m unittest discover -s tests -v
```
