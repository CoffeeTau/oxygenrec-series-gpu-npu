#!/usr/bin/env python3
"""Collect a privacy-bounded Ascend/PyTorch environment report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


REPORT_SCHEMA_VERSION = 1
COMMAND_TIMEOUT_SECONDS = 20

PACKAGE_NAMES = (
    "torch",
    "torch-npu",
    "torchvision",
    "torchaudio",
    "transformers",
    "accelerate",
    "deepspeed",
    "numpy",
    "scipy",
    "scikit-learn",
    "datasets",
    "sentencepiece",
    "safetensors",
)

SAFE_ENVIRONMENT_VARIABLES = (
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_AICPU_PATH",
    "ASCEND_TOOLKIT_HOME",
    "ASCEND_RT_VISIBLE_DEVICES",
    "NPU_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def run_command(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": command, "error": "not found"}
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": True, "command": command, "error": repr(error)}
    return {
        "available": True,
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def package_versions() -> dict[str, str | None]:
    versions = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def safe_environment() -> dict[str, str]:
    selected = {
        name: os.environ[name]
        for name in SAFE_ENVIRONMENT_VARIABLES
        if name in os.environ
    }
    for name, value in os.environ.items():
        if name.startswith("HCCL_"):
            selected[name] = value
    return dict(sorted(selected.items()))


def system_memory() -> dict[str, str]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return {}
    wanted = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    result = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator and key in wanted:
            result[key] = value.strip()
    return result


def collect_torch_npu() -> dict[str, Any]:
    try:
        import torch
    except Exception as error:
        return {"torch_importable": False, "torch_error": repr(error)}

    report: dict[str, Any] = {
        "torch_importable": True,
        "torch_version": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "distributed_available": torch.distributed.is_available(),
    }
    try:
        import torch_npu  # noqa: F401
    except Exception as error:
        report.update({"torch_npu_importable": False, "torch_npu_error": repr(error)})
        return report

    report["torch_npu_importable"] = True
    npu = getattr(torch, "npu", None)
    if npu is None:
        report.update({"npu_available": False, "npu_error": "torch.npu is absent"})
        return report

    try:
        report["npu_available"] = bool(npu.is_available())
        report["npu_device_count"] = int(npu.device_count())
    except Exception as error:
        report.update({"npu_available": False, "npu_error": repr(error)})
        return report

    hccl_available = getattr(torch.distributed, "is_hccl_available", None)
    report["distributed_hccl_available"] = (
        bool(hccl_available()) if callable(hccl_available) else None
    )
    devices = []
    for index in range(report["npu_device_count"]):
        entry: dict[str, Any] = {"index": index}
        try:
            entry["name"] = str(npu.get_device_name(index))
        except Exception as error:
            entry["name_error"] = repr(error)
        try:
            properties = npu.get_device_properties(index)
            entry["properties"] = str(properties)
        except Exception as error:
            entry["properties_error"] = repr(error)
        devices.append(entry)
    report["devices"] = devices
    return report


def collect_report() -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "memory": system_memory(),
        },
        "python": {
            "version": sys.version,
            "version_info": list(sys.version_info[:5]),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
        "packages": package_versions(),
        "torch_npu": collect_torch_npu(),
        "environment": safe_environment(),
        "commands": {
            "npu_smi_info": run_command(["npu-smi", "info"]),
            "atc_version": run_command(["atc", "--version"]),
            "gcc_version": run_command(["gcc", "--version"]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    snapshot_date = datetime.now(timezone.utc).date().isoformat()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "checkpoints/npu_stage0/"
            f"npu_server_environment_snapshot_{snapshot_date}.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = collect_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    runtime = report["torch_npu"]
    print(
        "OK stage=npu_environment_collection "
        f"torch_importable={runtime.get('torch_importable', False)} "
        f"torch_npu_importable={runtime.get('torch_npu_importable', False)} "
        f"npu_available={runtime.get('npu_available', False)} "
        f"npu_device_count={runtime.get('npu_device_count', 0)} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
