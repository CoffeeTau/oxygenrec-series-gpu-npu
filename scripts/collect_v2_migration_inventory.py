#!/usr/bin/env python3
"""Collect a source-level v2 migration inventory without claiming operator support."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
from typing import Any


DEFAULT_FILES = (
    "scripts/train_v2_pretraining_ablation_retailrocket.py",
    "scripts/export_v2_gpu_reference.py",
    "scripts/compare_device_reference.py",
    "src/oxygenrec/device.py",
    "src/oxygenrec/migration_alignment.py",
    "src/oxygenrec/model.py",
    "src/oxygenrec/data/model_inputs.py",
    "src/oxygenrec/data/temporal.py",
    "src/oxygenrec/sid.py",
)

PACKAGE_NAMES = (
    "torch",
    "torch-npu",
    "numpy",
    "scipy",
    "scikit-learn",
    "transformers",
    "msprobe",
)

PATTERNS = {
    "platform_specific": re.compile(
        r"torch\.(?:cuda|npu)|torch_npu|CUDA_VISIBLE_DEVICES|"
        r"ASCEND_RT_VISIBLE_DEVICES|\bnccl\b|\bhccl\b|DataParallel"
    ),
    "host_synchronization": re.compile(r"\.item\(\)|\.tolist\(\)|\.cpu\(\)"),
    "dynamic_shape_review": re.compile(
        r"\.shape\b|\.reshape\(|\.view\(|\.expand\(|repeat_interleave\(|"
        r"torch\.empty\(|torch\.cat\(|torch\.stack\("
    ),
    "precision_sensitive": re.compile(
        r"softmax\(|log_softmax\(|cross_entropy\(|multinomial\(|"
        r"masked_fill\(|isfinite\(|AdamW\("
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def imported_names(tree: ast.AST) -> list[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return sorted(names)


def called_apis(tree: ast.AST) -> list[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = dotted_name(node.func)
            if name:
                names.add(name)
    return sorted(names)


def matching_lines(lines: list[str], pattern: re.Pattern[str]) -> list[dict[str, Any]]:
    return [
        {"line": number, "text": line.strip()}
        for number, line in enumerate(lines, start=1)
        if pattern.search(line)
    ]


def inspect_file(project_root: Path, relative_path: str) -> dict[str, Any]:
    path = project_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"migration target does not exist: {relative_path}")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=relative_path)
    lines = source.splitlines()
    return {
        "path": relative_path,
        "sha256": sha256(path),
        "line_count": len(lines),
        "imports": imported_names(tree),
        "called_apis": called_apis(tree),
        "review_points": {
            name: matching_lines(lines, pattern) for name, pattern in PATTERNS.items()
        },
    }


def package_versions() -> dict[str, str | None]:
    versions = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def tool_candidates() -> dict[str, str | None]:
    names = ("msprobe", "msFmkTransPlt", "msfmktransplt", "pytorch_analyse")
    return {name: shutil.which(name) for name in names}


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# OxygenREC-v2迁移支持度预检查",
        "",
        f"> 生成时间：`{report['collected_at_utc']}`  ",
        "> 性质：项目内静态清单，不替代华为PyTorch Analyse或目标NPU实跑。",
        "",
        "## 结论边界",
        "",
        "- 本报告只枚举迁移入口、依赖、设备专用调用、主机同步点和精度敏感API。",
        "- `dynamic_shape_review`只表示需要人工检查，不表示该行不受NPU支持。",
        "- 算子/API最终支持度必须以当前CANN/TorchNPU配套文档、官方工具及实跑为准。",
        "",
        "## 环境",
        "",
        f"- Python包：`{report['packages']}`",
        f"- PATH中的官方工具候选：`{report['tool_candidates']}`",
        "",
        "## 源文件清单",
        "",
        "| 文件 | SHA-256 | 平台专用 | 主机同步 | 动态shape复核 | 精度敏感 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in report["files"]:
        points = item["review_points"]
        lines.append(
            f"| `{item['path']}` | `{item['sha256']}` | "
            f"{len(points['platform_specific'])} | "
            f"{len(points['host_synchronization'])} | "
            f"{len(points['dynamic_shape_review'])} | "
            f"{len(points['precision_sensitive'])} |"
        )
    lines.extend([
        "",
        "## 下一步",
        "",
        "1. 在目标Ascend环境确认官方迁移分析工具入口并保存原始报告；",
        "2. 优先处理平台专用调用，保持模型公式与GPU路径一致；",
        "3. 用冻结batch完成FP32 forward/loss/generation/backward对齐；",
        "4. 只有出现差异时，再用msProbe下钻到Module或API级数据。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("files", nargs="*", default=list(DEFAULT_FILES))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    files = [inspect_file(project_root, item) for item in args.files]
    report = {
        "schema_version": 1,
        "protocol": "oxygenrec_v2_migration_static_inventory",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "support_claim": False,
        "official_tool_required": True,
        "packages": package_versions(),
        "tool_candidates": tool_candidates(),
        "files": files,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "v2_migration_inventory.json"
    markdown_path = args.output_dir / "v2_migration_inventory.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(report, markdown_path)
    platform_refs = sum(
        len(item["review_points"]["platform_specific"]) for item in files
    )
    print(
        "OK stage=v2_migration_static_inventory "
        f"files={len(files)} platform_specific_refs={platform_refs} "
        f"official_tool_required=True output={json_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
