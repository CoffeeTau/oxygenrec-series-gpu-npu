#!/usr/bin/env python3
"""Download one pinned Hugging Face model snapshot into the project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--local-dir", type=Path,
        default=Path("models/Qwen3-4B-Instruct-2507"),
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is missing; install a version compatible with the "
            "existing Transformers environment"
        ) from exc

    args = parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(args.repo_id, revision=args.revision)
    commit = info.sha
    print(
        f"stage=download repo={args.repo_id} revision={args.revision} "
        f"commit={commit} local_dir={args.local_dir}"
    )
    snapshot_download(
        repo_id=args.repo_id,
        revision=commit,
        local_dir=args.local_dir,
        max_workers=args.max_workers,
    )
    required = (
        "config.json", "tokenizer_config.json", "tokenizer.json",
        "model.safetensors.index.json",
    )
    missing = [name for name in required if not (args.local_dir / name).is_file()]
    shards = sorted(args.local_dir.glob("model-*.safetensors"))
    if missing or not shards:
        raise RuntimeError(
            f"download incomplete: missing={missing}, safetensor_shards={len(shards)}"
        )
    provenance = {
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_commit": commit,
        "safetensor_shards": [path.name for path in shards],
    }
    (args.local_dir / "MODEL_SOURCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    total_bytes = sum(
        path.stat().st_size for path in args.local_dir.rglob("*") if path.is_file()
    )
    print(
        f"OK model_dir={args.local_dir} commit={commit} "
        f"shards={len(shards)} size_gib={total_bytes / 1024**3:.3f}"
    )


if __name__ == "__main__":
    main()
