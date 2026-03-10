#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from datetime import datetime
from pathlib import Path


def module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def run_python(code: str) -> str:
    result = subprocess.run(["python3", "-c", code], capture_output=True, text=True, check=False)
    return (result.stdout or result.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Check MPS / MLX / ANE environment sanity")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    payload = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mlx_importable": module_available("mlx"),
        "torch_importable": module_available("torch"),
        "mps_available": "True" in run_python("import torch; print(torch.backends.mps.is_available())"),
        "ane_train_binary_present": Path("ane/primitives/train_tiny").exists(),
        "ane_train_source_present": Path("ane/primitives/train_tiny.m").exists(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.write:
        out_dir = Path("probes/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"env-sanity-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
