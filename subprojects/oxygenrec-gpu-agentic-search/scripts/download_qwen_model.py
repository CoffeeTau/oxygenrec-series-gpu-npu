#!/usr/bin/env python3
"""Download the selected Qwen snapshot into the project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", choices=("modelscope", "huggingface"), default="modelscope",
    )
    parser.add_argument("--repo-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--local-dir", type=Path,
        default=Path("models/Qwen3-4B-Instruct-2507"),
    )
    parser.add_argument(
        "--revision", default=None,
        help="Defaults to master for ModelScope and main for Hugging Face.",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--insecure-skip-tls-verify", action="store_true",
        help="Disable TLS certificate verification only inside this process.",
    )
    return parser.parse_args()


def disable_tls_verification_for_process() -> None:
    """Force Requests sessions to skip verification in this process only."""

    import requests
    import urllib3

    original = requests.sessions.Session.request

    def insecure_request(session, method, url, **kwargs):
        kwargs["verify"] = False
        return original(session, method, url, **kwargs)

    requests.sessions.Session.request = insecure_request
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def main() -> None:
    args = parse_args()
    if args.insecure_skip_tls_verify:
        print(
            "WARNING tls_verification=false scope=current_process "
            "download_source_authenticity_not_guaranteed"
        )
        disable_tls_verification_for_process()
    args.local_dir.mkdir(parents=True, exist_ok=True)
    revision = args.revision or ("master" if args.provider == "modelscope" else "main")
    resolved_commit = None
    if args.provider == "modelscope":
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "modelscope is missing; install it from the company-approved PyPI mirror"
            ) from exc
    else:
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is missing") from exc
        resolved_commit = HfApi().model_info(args.repo_id, revision=revision).sha
        revision = resolved_commit
    print(
        f"stage=download provider={args.provider} repo={args.repo_id} "
        f"revision={revision} local_dir={args.local_dir}"
    )
    download_kwargs = {
        "revision": revision,
        "local_dir": str(args.local_dir),
        "max_workers": args.max_workers,
    }
    if args.provider == "modelscope":
        snapshot_download(model_id=args.repo_id, **download_kwargs)
    else:
        snapshot_download(repo_id=args.repo_id, **download_kwargs)
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
        "provider": args.provider,
        "tls_verification": not args.insecure_skip_tls_verify,
        "repo_id": args.repo_id,
        "requested_revision": revision,
        "resolved_commit": resolved_commit,
        "safetensor_shards": [path.name for path in shards],
        "safetensor_sizes": {path.name: path.stat().st_size for path in shards},
    }
    (args.local_dir / "MODEL_SOURCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    total_bytes = sum(
        path.stat().st_size for path in args.local_dir.rglob("*") if path.is_file()
    )
    print(
        f"OK provider={args.provider} model_dir={args.local_dir} "
        f"revision={revision} "
        f"shards={len(shards)} size_gib={total_bytes / 1024**3:.3f}"
    )


if __name__ == "__main__":
    main()
