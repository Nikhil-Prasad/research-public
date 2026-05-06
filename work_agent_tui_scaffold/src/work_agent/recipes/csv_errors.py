from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pick_column(headers: list[str], aliases: list[str]) -> str | None:
    normalized = {h.lower().strip(): h for h in headers}
    for alias in aliases:
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    return None


def summarize_error_csv(path: Path, aliases: dict[str, list[str]]) -> dict[str, Any]:
    rows = load_csv_rows(path)
    headers = list(rows[0].keys()) if rows else []

    colmap = {logical: pick_column(headers, names) for logical, names in aliases.items()}

    def val(row: dict[str, str], logical: str) -> str:
        col = colmap.get(logical)
        if not col:
            return ""
        return row.get(col, "").strip()

    doc_ids = sorted({val(r, "doc_id") for r in rows if val(r, "doc_id")})
    by_use_case = Counter(val(r, "use_case") or "unknown" for r in rows)
    by_field = Counter(val(r, "field") or "unknown" for r in rows)
    by_error_type = Counter(val(r, "error_type") or "unknown" for r in rows)

    clusters = defaultdict(list)
    for r in rows:
        key = (val(r, "use_case") or "unknown", val(r, "field") or "unknown", val(r, "error_type") or "unknown")
        clusters[key].append(r)

    top_clusters = []
    for idx, (key, group) in enumerate(sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True), start=1):
        use_case, field, error_type = key
        top_clusters.append(
            {
                "cluster_id": f"cluster_{idx:03d}",
                "use_case": use_case,
                "field": field,
                "error_type": error_type,
                "count": len(group),
                "examples": group[:5],
            }
        )

    return {
        "csv_path": str(path),
        "row_count": len(rows),
        "headers": headers,
        "column_mapping": colmap,
        "doc_ids": doc_ids,
        "by_use_case": dict(by_use_case.most_common(20)),
        "by_field": dict(by_field.most_common(20)),
        "by_error_type": dict(by_error_type.most_common(20)),
        "top_clusters": top_clusters[:10],
    }
