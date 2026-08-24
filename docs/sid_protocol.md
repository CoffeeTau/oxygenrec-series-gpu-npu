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

## Scalable PyTorch backend

`TorchResidualKMeans` implements chunked accelerator distance assignment,
Lloyd centroid updates, residual fitting, versioned tensor persistence, and
`SIDRegistry` generation. It supports seeded random train-vector and seeded
K-Means++ initialization. Both are engineering choices and neither is claimed
as the paper's undisclosed initializer.

Before fitting real item vectors, run:

```bash
bash run_server_test.sh
```

This synthetic CUDA smoke test validates three residual levels, chunked encode,
reconstruction improvement, registry generation, and codebook save/load. CUDA
`index_add_` may not be bitwise deterministic across devices even with a fixed
seed; CPU determinism is covered separately by tests.

## L20 scalable-backend validation

Validated on one L20 on 2026-08-24:

```text
OK device=cuda shape=(3, 8, 16) mse=0.927709->0.006996 collision_rate=0.884766
```

The reconstruction improvement validates the residual fitting path. The high
collision rate is expected for this deliberately clustered synthetic fixture
with width 8; it is not an acceptable target for the real item codebook. Real
item fitting must report collision percentiles, colliding-item rate, prefix
coverage, and load balance before its registry is approved.

## RetailRocket public proxy representation

`fit_retailrocket_sid.py` constructs a 256-dimensional public-data proxy:

1. derive global train/validation cutoffs from `events.csv`;
2. select frequent items visible strictly in the training interval;
3. retain the latest property value for every `(item, property)` pair strictly
   before the training cutoff across both property files;
4. map `property=value` features into 256 dimensions using stable signed
   BLAKE2b feature hashing;
5. L2-normalize item vectors;
6. fit and persist three residual codebooks and the version-matched registry;
7. save reconstruction and SID diagnostics in `metadata.json`.

This representation is deterministic and leak-resistant, but it is not the
paper's multimodal Qwen3/CLIP/Q-Former representation. It validates the public
benchmark and RQ interface before a stronger item encoder is integrated.

Generated artifacts under `data/processed/retailrocket_sid/` are:

- `item_embeddings.npy` and aligned `item_ids.txt`;
- `rq_codebooks.pt`;
- `sid_registry.json`;
- `metadata.json` containing assumptions and aggregate diagnostics.

### First property-proxy run

The first bounded L20 run on 2026-08-24 completed tokenizer fitting and model
training:

```text
represented_items=4808
mse=0.003906->0.002105
collision_rate=0.911606
```

The model epoch and loss completed normally, validating the fitted-registry
integration. The 91.16% colliding-item rate is not acceptable for a promoted
registry. The next controlled run increases selected items and codebook width,
and records exact input-vector uniqueness. If collisions remain high while
input uniqueness is high, RQ configuration/initialization is the main suspect;
if input uniqueness is low, the public proxy representation needs behavioral
co-occurrence features.

The width-256 controlled run produced:

```text
represented_items=18733
unique_vector_rate=1.000000
mse=0.003906->0.001671
collision_rate=0.211071
```

Input vectors are completely unique and reconstruction improved, so remaining
collisions are attributed primarily to RQ capacity/optimization rather than
exact duplicate proxy vectors. The next controlled run holds the item set near
20K, increases width to 512, and changes initialization from random training
vectors to seeded K-Means++.

The width-512 K-Means++ run produced:

```text
represented_items=18733
unique_vector_rate=1.000000
mse=0.003906->0.001596
collision_rate=0.444243
```

K-Means++ improved reconstruction slightly but worsened SID collisions. Because
both width and initialization changed relative to the width-256 random run, the
final SID decision uses a 2x2 controlled comparison. The two missing cells are
width-256 K-Means++ and width-512 random, fitted from the already persisted,
identical embedding matrix without rescanning data or retraining the recommender.
