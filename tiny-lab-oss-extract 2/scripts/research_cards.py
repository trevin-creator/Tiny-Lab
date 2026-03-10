#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def latest_rows(rows: list[dict]) -> list[dict]:
    latest = {}
    for row in rows:
        experiment_id = row.get("id")
        if not experiment_id:
            continue
        latest[experiment_id] = row
    return list(latest.values())


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "experiment"


def classify_source(row: dict) -> str:
    if row.get("surface"):
        return f"surface:{row['surface']}"
    if row.get("machine"):
        return f"machine:{row['machine']}"
    if row.get("family"):
        return f"family:{row['family']}"
    return "unknown"


def changed_variable(row: dict) -> str:
    return row.get("changed_variable") or row.get("changed") or "none"


def conclusion(row: dict) -> str:
    return row.get("decision") or row.get("class") or row.get("status") or "undocumented"


def render_card(row: dict) -> str:
    experiment_id = row.get("id", "UNKNOWN")
    result = row.get("class", row.get("status", "UNKNOWN"))
    return "\n".join(
        [
            f"# {experiment_id}",
            "",
            f"- source: {classify_source(row)}",
            f"- question: {row.get('question', '')}",
            f"- control: {row.get('control') or row.get('parent') or 'none'}",
            f"- changed variable: {changed_variable(row)}",
            f"- result: {result}",
            f"- conclusion: {conclusion(row)}",
            f"- experiment id: {experiment_id}",
            "",
            "## Metrics",
            "",
            "```json",
            json.dumps(row.get("primary_metric", {}), indent=2, sort_keys=True),
            "```",
            "",
            "## Notes",
            "",
            row.get("notes", "").strip() or "No notes recorded.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate benchmark cards from research ledger rows")
    parser.add_argument("--ledger", default="research/ledger.jsonl")
    parser.add_argument("--out-dir", default="research/cards")
    parser.add_argument("--id", dest="experiment_id")
    parser.add_argument("--all", action="store_true", help="render latest row for every experiment id")
    args = parser.parse_args()

    ledger = Path(args.ledger)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(ledger)
    rows = latest_rows(rows)
    if args.experiment_id:
        rows = [row for row in rows if row.get("id") == args.experiment_id]
    elif not args.all:
        rows = rows[-1:] if rows else []

    for row in rows:
        path = out_dir / f"{safe_slug(row.get('id', 'unknown'))}.md"
        path.write_text(render_card(row))
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
