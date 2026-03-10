#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


def sysctl_value(name: str) -> str:
    result = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=False)
    return result.stdout.strip()


def vm_pages() -> dict[str, int]:
    result = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False)
    pages = {}
    for raw in result.stdout.splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        value = value.strip().rstrip(".").replace(".", "")
        if value.isdigit():
            pages[key.strip()] = int(value)
    return pages


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate memory ceiling / batch feasibility")
    parser.add_argument("--required-gb", type=float, default=0.0)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    page_size = int(sysctl_value("hw.pagesize") or "4096")
    total_bytes = int(sysctl_value("hw.memsize") or "0")
    pages = vm_pages()
    free_pages = pages.get("Pages free", 0) + pages.get("Pages speculative", 0)
    free_bytes = free_pages * page_size

    payload = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "memory_gb_total": round(total_bytes / (1024**3), 2),
        "memory_gb_free_estimate": round(free_bytes / (1024**3), 2),
        "required_gb": args.required_gb,
        "fits_required_working_set": free_bytes >= args.required_gb * (1024**3),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.write:
        out_dir = Path("probes/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"memory-ceiling-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
