from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SURFACE = REPO_ROOT / "bin/surface"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture()
def lab_root(tmp_path: Path) -> Path:
    root = tmp_path / "lab"
    ane = root / "surfaces/ane"
    gpu = root / "surfaces/gpu"
    ane.mkdir(parents=True)
    gpu.mkdir(parents=True)
    (root / "config").mkdir(parents=True)
    (root / "research").mkdir(parents=True)
    write(
        root / "config/tiny_lab_lanes.tsv",
        "\n".join(
            [
                "# target|machine|protocol|host|workdir|backend|lane|mode|capabilities",
                f"tiny-ane|tiny|file|local|{ane}|ane|ane|search|fixture ANE lane",
                f"tiny-gpu|tiny|file|local|{gpu}|mlx|gpu|control|fixture MLX lane",
            ]
        )
        + "\n",
    )
    write(root / "surfaces.txt", "")
    write(
        ane / "runner.py",
        textwrap.dedent(
            """
            from pathlib import Path
            import os
            import sys
            import time

            steps = int(os.environ.get("RUNNER_STEPS", "200"))
            delay = float(os.environ.get("RUNNER_SLEEP", "0.05"))
            for step in range(steps):
                Path("steering_status.txt").write_text(
                    "\\n".join(
                        [
                            f"step={step}",
                            f"loss={1.0 / (step + 1):.6f}",
                            "val_accuracy=0.9",
                            "lr=0.001",
                            "ms_step=5.0",
                            f"best_val={0.9 + step / 1000:.3f}",
                            f"best_step={step}",
                        ]
                    )
                    + "\\n"
                )
                steering = Path("steering.txt")
                if steering.exists() and "early_stop=1" in steering.read_text():
                    Path("run_stopped.txt").write_text(str(step))
                    sys.exit(0)
                time.sleep(delay)
            """
        ),
    )
    write(
        gpu / "train.py",
        textwrap.dedent(
            """
            import time
            for _ in range(1000):
                time.sleep(0.1)
            """
        ),
    )
    return root


def surface(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["TINY_LAB_ROOT"] = str(root)
    return subprocess.run(
        [sys.executable, str(SURFACE), *args],
        text=True,
        capture_output=True,
        env=env,
        check=check,
    )


def surface_popen(root: Path, *args: str) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["TINY_LAB_ROOT"] = str(root)
    return subprocess.Popen(
        [sys.executable, str(SURFACE), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def wait_for(predicate, timeout: float = 10.0, interval: float = 0.1) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("condition not met before timeout")


def db_rows(root: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(root / "tiny_lab.db")
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute("SELECT * FROM runs ORDER BY id"))
    finally:
        conn.close()


def test_surface_run_managed_short_run_launches(lab_root: Path) -> None:
    result = surface(lab_root, "run", "tiny-ane", "RUNNER_STEPS=5 RUNNER_SLEEP=0.01 python3 runner.py")
    assert "Run complete:" in result.stdout
    rows = db_rows(lab_root)
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    status = surface(lab_root, "status", "tiny-ane", "--verbose").stdout
    assert "truth_source=state_json" in status
    assert "manifest_owner=" in status


def test_surface_stop_status_and_rerun(lab_root: Path) -> None:
    run_proc = surface_popen(lab_root, "run", "tiny-ane", "RUNNER_STEPS=1200 RUNNER_SLEEP=0.02 python3 runner.py", "--name", "stop-case")
    wait_for(lambda: "truth_source=live_process" in surface(lab_root, "status", "tiny-ane", "--verbose").stdout)
    verbose = surface(lab_root, "status", "tiny-ane", "--verbose").stdout
    assert "activity=managed" in verbose
    assert "stop_path=steering+kill" in verbose
    assert "manifest_owner=" in verbose and "manifest_owner=\n" not in verbose

    stop = surface(lab_root, "stop", "tiny-ane")
    assert "stopped stop-case" in stop.stdout
    run_stdout, run_stderr = run_proc.communicate(timeout=20)
    assert run_proc.returncode == 0, run_stderr
    assert "Status: stopped" in run_stdout

    board = json.loads(surface(lab_root, "board", "--json").stdout)
    lane = next(item for item in board["surfaces"] if item["id"] == "tiny-ane")
    assert lane["status"] != "running"
    assert not board["active_runs"]

    rerun = surface(lab_root, "run", "tiny-ane", "RUNNER_STEPS=3 RUNNER_SLEEP=0.01 python3 runner.py", "--name", "rerun")
    assert "Run complete: rerun-" in rerun.stdout
    board_after = json.loads(surface(lab_root, "board", "--json").stdout)
    lane_after = next(item for item in board_after["surfaces"] if item["id"] == "tiny-ane")
    assert lane_after["status"] in {"completed", "stopped", "idle"}


def test_surface_doctor_catches_and_fixes_stale_state(lab_root: Path) -> None:
    surface(lab_root, "list")
    run_dir = lab_root / "surfaces/ane/runs/ghost-run"
    run_dir.mkdir(parents=True)
    write(
        run_dir / "run.json",
        json.dumps(
            {
                "run_name": "ghost-run",
                "target": "tiny-ane",
                "machine": "tiny",
                "backend": "ane",
                "lane": "ane",
                "status": "running",
                "pid": 999999,
                "session_id": 999999,
                "owner": "fixture",
                "linked_paths": {
                    "steering_status.txt": str(lab_root / "surfaces/ane/steering_status.txt"),
                },
            }
        ),
    )
    write(run_dir / "state.json", json.dumps({"status": "running", "live": True}))
    write(run_dir / "run_state.txt", "status=running\n")
    (lab_root / "surfaces/ane/steering_status.txt").symlink_to(Path("runs/ghost-run/steering_status.txt"))
    write(run_dir / "steering_status.txt", "step=9\nloss=0.9\n")
    stale_status = surface(lab_root, "status", "tiny-ane", "--verbose").stdout
    assert "lane_state=stale" in stale_status
    assert "truth_source=state_json" in stale_status
    conn = sqlite3.connect(lab_root / "tiny_lab.db")
    try:
        conn.execute(
            "INSERT INTO runs (surface, run_name, started, command, status) VALUES (?, ?, ?, ?, ?)",
            ("tiny-ane", "ghost-run", "2026-03-10T00:00:00", "python3 runner.py", "running"),
        )
        conn.commit()
    finally:
        conn.close()
    write(lab_root / "research/.research-lock", "123456")

    doctor = surface(lab_root, "doctor", "tiny-ane", check=False)
    assert "STALE_DB_ROW" in doctor.stdout
    assert "DEAD_RUN" in doctor.stdout
    assert "ORPHAN_STATUS_FILE" in doctor.stdout
    assert "STALE_LOCK" in doctor.stdout

    fixed = surface(lab_root, "doctor", "tiny-ane", "--fix")
    assert "FIXED:" in fixed.stdout
    doctor_after = surface(lab_root, "doctor", "tiny-ane")
    assert doctor_after.stdout.strip() == "OK"
    assert not (lab_root / "research/.research-lock").exists()


def test_lane_separation_keeps_ane_state_out_of_gpu_status(lab_root: Path) -> None:
    write(lab_root / "surfaces/ane/steering_status.txt", "step=42\nloss=0.42\n")
    ane_status = surface(lab_root, "status", "tiny-ane", "--verbose").stdout
    gpu_status = surface(lab_root, "status", "tiny-gpu", "--verbose").stdout
    assert "step=42" in ane_status
    assert "step=42" not in gpu_status
    assert "backend=mlx" in gpu_status


def test_untracked_detection_is_explicit(lab_root: Path) -> None:
    proc = subprocess.Popen(
        ["bash", "-lc", f"cd {shlex_quote(str(lab_root / 'surfaces/gpu'))} && python3 train.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for(lambda: "activity=untracked" in surface(lab_root, "status", "tiny-gpu", "--verbose").stdout)
        verbose = surface(lab_root, "status", "tiny-gpu", "--verbose").stdout
        assert "activity=untracked" in verbose
        assert "untracked_count=1" in verbose
        board = json.loads(surface(lab_root, "board", "--json").stdout)
        lane = next(item for item in board["surfaces"] if item["id"] == "tiny-gpu")
        assert lane["activity"] == "untracked"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)
