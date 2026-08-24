# RetailRocket Phase-1 training protocol

The first real-data run uses a bounded dense model and a temporary structural
SID registry. Its purpose is to validate the complete data-to-metric pipeline.
The frequency bootstrap is **not semantic** and must not be reported as the
paper's RQ-KMeans tokenizer.

Run on one L20 from the project root:

```bash
python scripts/train_retailrocket.py \
  --events data/raw/retailrocket/events.csv \
  --device cuda \
  --max-items 50000 \
  --max-train-samples 100000 \
  --max-validation-samples 100 \
  --epochs 3
```

The script:

1. streams and normalizes `events.csv`;
2. proposes global 80/10/10 boundaries over observed time duration;
3. selects train-visible frequent items only;
4. creates a deterministic collision-free three-level bootstrap registry;
5. builds leak-resistant samples using deterministic reservoir caps;
6. trains the dense encoder-decoder;
7. evaluates constrained beam HR/MRR/legal-SID rate;
8. saves the registry and resumable epoch checkpoints under `checkpoints/`.

Terminal output contains stages and aggregate metrics only. Checkpoints remain
local and may contain model parameters, optimizer state, paths, and experiment
configuration, so they should not be shared without review.

The next tokenizer milestone replaces the bootstrap registry with item vectors
and scalable residual K-Means while keeping the same model/training interface.

The training entrypoint accepts a fitted registry:

```bash
python scripts/train_retailrocket.py \
  --events data/raw/retailrocket/events.csv \
  --sid-registry data/processed/retailrocket_sid/sid_registry.json \
  --device cuda
```

When `--sid-registry` is present, frequency bootstrap is bypassed completely.
The registry controls model SID width/levels, event filtering, constrained
decoding, checkpoint version metadata, and item-level evaluation.

## L20 bounded-run validation

The 100,000-sample configuration completed three epochs on the L20 server on
2026-08-24. All three epoch summaries were produced and aggregate training loss
decreased normally. Exact loss and metric values were not exported because of
the server's information-security policy.

This is qualitative evidence that the real-data loading, bounded temporal
sampling, CUDA optimization, epoch validation, and checkpoint path execute end
to end. It is not yet evidence for semantic SID quality, final recommendation
quality, multi-GPU scaling, or comparison with the OxygenREC paper tables.
# RetailRocket Phase-2 ablations

All variants use the same temporal split, SID registry, reservoir seed, model
size, optimizer settings and evaluation code. Only the named method switches
change. Formal comparisons must pass `--matched-igr-cohort`; this restricts
Base and Instruction to the same IGR-eligible sample universe instead of giving
them a different short-history population.

| `--variant` | Scenario instruction | IGR | Q2I |
|---|---:|---:|---:|
| `base` | no | no | no |
| `instruction` | target behavior scenario proxy | no | no |
| `igr` | target behavior scenario proxy | yes | no |
| `igr_q2i` | target behavior scenario proxy | yes | yes |

For IGR variants, the most recent `--max-history` interactions form the short
history. Up to `--long-history` immediately preceding interactions form a
disjoint retrieval pool. A sample is eligible only when it contains at least
`--igr-top-k` older known items, preventing padded entries from being retrieved.

RetailRocket has no real query or generated contextual reasoning text. Its
view/cart/transaction target type is therefore an explicit public-data scenario
proxy, not a reproduction of the paper's private scenario instructions. Report
the four variants as within-project ablations only.

Example full commands share these flags and vary only `--variant` and output:

```bash
python scripts/train_retailrocket.py \
  --events data/raw/retailrocket/events.csv \
  --sid-registry data/processed/rq_comparison/w256_kmeanspp/sid_registry.json \
  --device cuda --max-history 20 --long-history 100 --igr-top-k 10 \
  --matched-igr-cohort \
  --max-train-samples 100000 --max-validation-samples 100 \
  --batch-size 128 --epochs 3 --beam-width 10 \
  --variant igr_q2i --output-dir checkpoints/retailrocket_igr_q2i
```
