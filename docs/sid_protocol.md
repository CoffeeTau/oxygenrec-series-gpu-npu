# Semantic-ID construction protocol

## Paper facts

The checked OxygenREC paper uses a 256-dimensional multimodal item embedding,
three residual-quantization levels, and codebook width 8192 at every level.

## Reference implementation

`ReferenceResidualKMeans` provides deterministic, dependency-free semantics:

1. run seeded K-means++ and Lloyd updates on the original item embeddings;
2. subtract the assigned centroid from every item;
3. fit the next codebook on those residuals;
4. repeat for three levels;
5. preserve all item-to-SID collisions in `SIDRegistry`.

This implementation is not intended for hundreds of thousands of items. Its
purpose is to define expected codes, reconstruction behavior, validation errors,
and metrics on small fixtures. The scalable FAISS/PyTorch implementation must
match it on controlled fixtures within a documented tolerance.

## Required diagnostics

- Prefix coverage at depth 1, 2, and 3:
  `unique SID prefixes / width ** depth`.
- SID collision distribution: item count per complete SID, including P90, P99,
  and P99.9.
- Colliding-item rate: fraction of items assigned to a non-unique complete SID.
- Codebook load balance at each level: cluster size divided by the ideal
  `number_of_items / codebook_width`, including unused zero-count codes.
- Reconstruction error before approving a new codebook version.

Cluster/category purity requires taxonomy labels and will be added with the
dataset-specific item metadata adapter.

## Versioning rule

Every exported `SIDRegistry` must include a version tied to:

- source dataset and train cutoff;
- item-embedding model/checkpoint;
- embedding preprocessing;
- RQ algorithm and seed;
- levels and width.

A training or evaluation checkpoint must never silently load a registry with a
different version.

