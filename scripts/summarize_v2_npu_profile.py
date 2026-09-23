#!/usr/bin/env python3
"""Summarize Ascend PyTorch Profiler CSV files without third-party packages."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Iterable


KEYWORDS = (
    "_local_scalar_dense",
    "memcpy",
    "transdata",
    "masked_fill",
    "adam",
    "optimizer",
    "cpu",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def choose_column(
    fieldnames: Iterable[str],
    exact_candidates: Iterable[str],
    contains_candidates: Iterable[tuple[str, ...]] = (),
) -> str | None:
    fields = list(fieldnames)
    by_normalized = {normalized(field): field for field in fields}
    for candidate in exact_candidates:
        if normalized(candidate) in by_normalized:
            return by_normalized[normalized(candidate)]
    for required_parts in contains_candidates:
        for field in fields:
            compact = normalized(field)
            if all(normalized(part) in compact for part in required_parts):
                return field
    return None


def numeric(value: str | None) -> float:
    if value is None:
        return 0.0
    stripped = value.strip().replace(",", "")
    if not stripped or stripped.lower() in {"na", "n/a", "none", "nan"}:
        return 0.0
    try:
        return float(stripped)
    except ValueError:
        return 0.0


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def summarize_operator(path: Path, top_k: int) -> dict:
    fields, rows = read_csv(path)
    name_column = choose_column(
        fields,
        ("Name", "Operator Name", "Op Name", "OP Type", "Type"),
    )
    if name_column is None:
        raise ValueError(f"cannot find operator name column in {path}: {fields}")

    duration_columns = [
        column for column in (
            choose_column(fields, ("Device Self Duration(us)",)),
            choose_column(fields, ("Device Total Duration(us)",)),
            choose_column(fields, ("Host Self Duration(us)",)),
            choose_column(fields, ("Host Total Duration(us)",)),
        ) if column is not None
    ]
    if not duration_columns:
        fallback = choose_column(
            fields,
            ("Duration(us)", "Duration"),
            (("duration",),),
        )
        if fallback is not None:
            duration_columns.append(fallback)
    if not duration_columns:
        raise ValueError(f"cannot find duration column in {path}: {fields}")

    call_column = choose_column(fields, ("Call Count", "Calls", "Count"))
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        name = (row.get(name_column) or "<empty>").strip()
        calls = numeric(row.get(call_column)) if call_column else 1.0
        totals[name]["calls"] += calls or 1.0
        for column in duration_columns:
            totals[name][column] += numeric(row.get(column))

    primary = duration_columns[0]
    primary_total = sum(values[primary] for values in totals.values())
    ranked = []
    for name, values in totals.items():
        item = {
            "name": name,
            "calls": int(values["calls"]),
            **{column: values[column] for column in duration_columns},
        }
        item["primary_duration_share"] = (
            values[primary] / primary_total if primary_total else 0.0
        )
        ranked.append(item)
    ranked.sort(key=lambda item: float(item[primary]), reverse=True)

    keyword_rows = []
    for keyword in KEYWORDS:
        matches = [item for item in ranked if keyword in str(item["name"]).lower()]
        keyword_rows.append({
            "keyword": keyword,
            "matched_operators": len(matches),
            "calls": sum(int(item["calls"]) for item in matches),
            "primary_duration": sum(float(item[primary]) for item in matches),
            "primary_duration_share": sum(
                float(item["primary_duration_share"]) for item in matches
            ),
        })

    return {
        "path": str(path),
        "columns": fields,
        "rows": len(rows),
        "name_column": name_column,
        "duration_columns": duration_columns,
        "primary_duration_column": primary,
        "primary_duration_total": primary_total,
        "top": ranked[:top_k],
        "keyword_summary": keyword_rows,
    }


def summarize_kernel(path: Path, top_k: int) -> dict:
    fields, rows = read_csv(path)
    name_column = choose_column(fields, ("Name", "Kernel Name", "Task Name", "Type"))
    type_column = choose_column(fields, ("Type", "Kernel Type", "Task Type"))
    duration_column = choose_column(
        fields,
        ("Duration(us)", "Duration"),
        (("duration",),),
    )
    if name_column is None or duration_column is None:
        raise ValueError(f"cannot find kernel name/duration columns in {path}: {fields}")

    totals: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in rows:
        name = (row.get(name_column) or "<empty>").strip()
        kernel_type = (row.get(type_column) or "") if type_column else ""
        key = (name, kernel_type.strip())
        duration = numeric(row.get(duration_column))
        totals[key]["calls"] += 1
        totals[key]["duration"] += duration
        totals[key]["max_duration"] = max(totals[key]["max_duration"], duration)

    duration_total = sum(values["duration"] for values in totals.values())
    ranked = []
    for (name, kernel_type), values in totals.items():
        calls = int(values["calls"])
        ranked.append({
            "name": name,
            "type": kernel_type,
            "calls": calls,
            "duration": values["duration"],
            "mean_duration": values["duration"] / calls if calls else 0.0,
            "max_duration": values["max_duration"],
            "duration_share": values["duration"] / duration_total if duration_total else 0.0,
        })
    ranked.sort(key=lambda item: float(item["duration"]), reverse=True)

    keyword_rows = []
    for keyword in KEYWORDS:
        matches = [
            item for item in ranked
            if keyword in f"{item['name']} {item['type']}".lower()
        ]
        keyword_rows.append({
            "keyword": keyword,
            "matched_kernels": len(matches),
            "calls": sum(int(item["calls"]) for item in matches),
            "duration": sum(float(item["duration"]) for item in matches),
            "duration_share": sum(float(item["duration_share"]) for item in matches),
        })

    return {
        "path": str(path),
        "columns": fields,
        "rows": len(rows),
        "name_column": name_column,
        "type_column": type_column,
        "duration_column": duration_column,
        "duration_total": duration_total,
        "top": ranked[:top_k],
        "keyword_summary": keyword_rows,
    }


def print_table(title: str, rows: list[dict], duration_key: str, share_key: str) -> None:
    print(title)
    for index, row in enumerate(rows, start=1):
        name = str(row["name"])
        calls = int(row["calls"])
        duration = float(row[duration_key])
        share = 100.0 * float(row[share_key])
        suffix = f" type={row['type']}" if row.get("type") else ""
        print(
            f"{index:02d} duration={duration:.3f} share={share:.2f}% "
            f"calls={calls} name={name}{suffix}"
        )


def print_keywords(title: str, rows: list[dict], duration_key: str, share_key: str) -> None:
    print(title)
    for row in rows:
        print(
            f"keyword={row['keyword']} duration={float(row[duration_key]):.3f} "
            f"share={100.0 * float(row[share_key]):.2f}% calls={int(row['calls'])}"
        )


def main() -> int:
    args = parse_args()
    if not args.profile_root.is_dir():
        raise NotADirectoryError(args.profile_root)
    if args.top_k < 1:
        raise ValueError("top-k must be positive")

    operator_files = sorted(args.profile_root.rglob("operator_details*.csv"))
    kernel_files = sorted(args.profile_root.rglob("kernel_details*.csv"))
    if len(operator_files) != 1 or len(kernel_files) != 1:
        raise ValueError(
            "expected exactly one operator_details CSV and one kernel_details CSV; "
            f"found operators={len(operator_files)} kernels={len(kernel_files)}"
        )

    operator = summarize_operator(operator_files[0], args.top_k)
    kernel = summarize_kernel(kernel_files[0], args.top_k)
    payload = {
        "schema_version": 1,
        "protocol": "oxygenrec_v2_npu_profile_csv_summary_v1",
        "profile_root": str(args.profile_root),
        "operator": operator,
        "kernel": kernel,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    primary = str(operator["primary_duration_column"])
    print(f"operator_columns={operator['columns']}")
    print_table(
        f"TOP_OPERATORS primary={primary}",
        list(operator["top"]),
        primary,
        "primary_duration_share",
    )
    print_keywords(
        "OPERATOR_KEYWORDS",
        list(operator["keyword_summary"]),
        "primary_duration",
        "primary_duration_share",
    )
    print(f"kernel_columns={kernel['columns']}")
    print_table(
        f"TOP_KERNELS duration={kernel['duration_column']}",
        list(kernel["top"]),
        "duration",
        "duration_share",
    )
    print_keywords(
        "KERNEL_KEYWORDS",
        list(kernel["keyword_summary"]),
        "duration",
        "duration_share",
    )
    print(f"OK profile_summary output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
