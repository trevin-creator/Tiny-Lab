#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize backend heartbeat / state sanity from the board")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    root_override = os.environ.get("TINY_LAB_ROOT")
    root = Path(root_override).expanduser() if root_override else Path(__file__).resolve().parents[1]
    board_cmd = [sys.executable, str(root / "bin/surface"), "board", "--json"]
    result = subprocess.run(
        board_cmd,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "TINY_LAB_ROOT": str(root)},
    )
    board = json.loads(result.stdout)

    research = [item for item in board.get("surfaces", []) if item.get("backend") in {"ane", "mlx"}]
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "lane_count": len(research),
        "running": [item["id"] for item in research if item.get("status") == "running"],
        "untracked": [item["id"] for item in research if item.get("activity") == "untracked"],
        "stale_or_broken": board.get("broken_gates", []),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.write:
        out_dir = root / "probes/results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"backend-heartbeat-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
