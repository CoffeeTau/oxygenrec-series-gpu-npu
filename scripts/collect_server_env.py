#!/usr/bin/env python3
"""Collect the L20 server environment needed by the OxygenREC implementation.

The script is read-only and uses only the Python standard library. PyTorch and
other packages are imported only when available. It intentionally avoids a full
environment dump and never reads variables whose names suggest credentials.

Usage:
    python collect_server_env.py --output oxygenrec_server_env.json
"""

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
    "torchvision",
    "torchaudio",
    "transformers",
    "accelerate",
    "deepspeed",
    "flash-attn",
    "xformers",
    "triton",
    "bitsandbytes",
    "apex",
    "faiss-cpu",
    "faiss-gpu",
    "numpy",
    "scipy",
    "scikit-learn",
    "datasets",
    "sentencepiece",
    "safetensors",
)

SAFE_ENVIRONMENT_VARIABLES = (
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TORCH_HOME",
    "TORCH_EXTENSIONS_DIR",
)


def run_command(command: list[str]) -> dict[str, Any]:
    """Run a diagnostic command without failing the complete report."""

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
    versions: dict[str, str | None] = {}
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
        if name.startswith("NCCL_"):
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


def collect_torch() -> dict[str, Any]:
    try:
        import torch
    except Exception as error:  # Import can fail because of missing shared libraries.
        return {"importable": False, "error": repr(error)}

    report: dict[str, Any] = {
        "importable": True,
        "version": torch.__version__,
        "git_version": getattr(torch.version, "git_version", None),
        "debug_build": bool(torch.version.debug),
        "cuda_runtime": torch.version.cuda,
        "hip_runtime": torch.version.hip,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "distributed_available": torch.distributed.is_available(),
        "distributed_nccl_available": (
            torch.distributed.is_available() and torch.distributed.is_nccl_available()
        ),
        "distributed_gloo_available": (
            torch.distributed.is_available() and torch.distributed.is_gloo_available()
        ),
    }

    try:
        report["parallel_config"] = torch.__config__.show()
    except Exception as error:
        report["parallel_config_error"] = repr(error)

    if not torch.cuda.is_available():
        return report

    try:
        report["cudnn_version"] = torch.backends.cudnn.version()
        report["cudnn_available"] = torch.backends.cudnn.is_available()
    except Exception as error:
        report["cudnn_error"] = repr(error)
    try:
        report["nccl_version"] = torch.cuda.nccl.version()
    except Exception as error:
        report["nccl_version_error"] = repr(error)
    try:
        report["compiled_cuda_arch_list"] = torch.cuda.get_arch_list()
    except Exception as error:
        report["compiled_cuda_arch_list_error"] = repr(error)

    devices = []
    for index in range(torch.cuda.device_count()):
        try:
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "compute_capability": [properties.major, properties.minor],
                    "total_memory_bytes": properties.total_memory,
                    "multi_processor_count": properties.multi_processor_count,
                }
            )
        except Exception as error:
            devices.append({"index": index, "error": repr(error)})
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
        "torch": collect_torch(),
        "environment": safe_environment(),
        "commands": {
            "nvidia_smi_summary": run_command(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,driver_version,memory.total,compute_cap",
                    "--format=csv,noheader",
                ]
            ),
            "nvidia_smi_topology": run_command(["nvidia-smi", "topo", "-m"]),
            "nvidia_smi_nvlink": run_command(["nvidia-smi", "nvlink", "--status"]),
            "nvcc": run_command(["nvcc", "--version"]),
            "gcc": run_command(["gcc", "--version"]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("oxygenrec_server_env.json"),
        help="JSON report path (default: oxygenrec_server_env.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = collect_report()
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    torch_report = report["torch"]
    print(f"Wrote environment report: {args.output.resolve()}")
    print(f"Python: {report['python']['version_info'][:3]}")
    if torch_report.get("importable"):
        print(f"PyTorch: {torch_report.get('version')}")
        print(f"CUDA runtime: {torch_report.get('cuda_runtime')}")
        print(f"CUDA available: {torch_report.get('cuda_available')}")
        print(f"CUDA device count: {torch_report.get('cuda_device_count')}")
    else:
        print(f"PyTorch import failed: {torch_report.get('error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

