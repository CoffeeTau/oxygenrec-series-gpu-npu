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
