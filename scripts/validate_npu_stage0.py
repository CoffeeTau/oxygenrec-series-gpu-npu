#!/usr/bin/env python3
"""Validate one Ascend card with tensor, autograd, optimizer and checkpoint I/O."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    snapshot_date = datetime.now(timezone.utc).date().isoformat()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "checkpoints/npu_stage0/"
            f"npu_single_card_validation_{snapshot_date}.json"
        ),
    )
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    args = parse_args()
    if not args.device.startswith("npu"):
        raise ValueError("Stage-0 requires an npu device such as npu:0")

    import torch
    import torch_npu  # noqa: F401

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch_npu imported but torch.npu.is_available() is false")
    if torch.npu.device_count() < 1:
        raise RuntimeError("no Ascend NPU is visible to PyTorch")

    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.manual_seed(17)
    torch.npu.manual_seed_all(17)

    features = torch.arange(1, 17, dtype=torch.float32, device=device).reshape(4, 4)
    weight = torch.nn.Parameter(torch.eye(4, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW([weight], lr=1e-2)
    before = weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    prediction = features @ weight
    loss = prediction.square().mean()
    loss.backward()
    if weight.grad is None or not torch.isfinite(weight.grad).all():
        raise RuntimeError("NPU backward did not produce finite gradients")
    gradient_l1 = float(weight.grad.detach().abs().sum().cpu())
    optimizer.step()
    torch.npu.synchronize()
    parameter_delta = float((weight.detach() - before).abs().max().cpu())
    if parameter_delta <= 0.0:
        raise RuntimeError("AdamW step did not change the NPU parameter")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output.with_suffix(".pt")
    torch.save({"weight": weight.detach().cpu()}, checkpoint_path)
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_match = torch.equal(restored["weight"], weight.detach().cpu())
    if not checkpoint_match:
        raise RuntimeError("checkpoint save/restore changed the tensor")

    try:
        device_name = str(torch.npu.get_device_name(device))
    except Exception:
        device_name = str(torch.npu.get_device_name(0))
    report = {
        "stage": "npu_single_card",
        "device": str(device),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "torch_npu_version": package_version("torch-npu"),
        "loss": float(loss.detach().cpu()),
        "gradient_l1": gradient_l1,
        "parameter_max_abs_delta": parameter_delta,
        "checkpoint_match": checkpoint_match,
        "checkpoint": str(checkpoint_path),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "OK stage=npu_single_card "
        f"device={device} device_name={device_name!r} "
        f"loss={report['loss']:.6f} gradient_l1={gradient_l1:.6f} "
        f"parameter_max_abs_delta={parameter_delta:.6e} "
        f"checkpoint_match={checkpoint_match} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
