#!/bin/bash
# research-trigger.sh — optional tiny-lab MLX hypothesis runner

set -euo pipefail

ROOT="${TINY_LAB_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SURFACE="$ROOT/bin/surface"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LANES="$ROOT/config/tiny_lab_lanes.tsv"
CAL_MLX="$ROOT/calibration/mlx-5min.yaml"
LOCK="$ROOT/research/.research-lock"
QUEUE="$ROOT/research/hypothesis_queue.md"
LEDGER="$ROOT/research/ledger.jsonl"
LOG="$ROOT/research/trigger.log"
PROMPT="$ROOT/research/.current_cycle_prompt.md"
AGENT_PROMPT="$ROOT/research/agent.md"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
CLAUDE_ALLOWED_TOOLS="${CLAUDE_ALLOWED_TOOLS:-Bash,Read,Write,Edit}"
CLAUDE_MAX_TURNS="${CLAUDE_MAX_TURNS:-200}"
MAX_EXPERIMENTS_PER_CYCLE="${MAX_EXPERIMENTS_PER_CYCLE:-10}"
MAX_LOCK_AGE_MINUTES="${MAX_LOCK_AGE_MINUTES:-120}"

mkdir -p "$ROOT/research"

log() { printf "%s %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$LOG"; }
say() { printf "%s\n" "$*"; log "$*"; }

if [ ! -f "$AGENT_PROMPT" ]; then
    say "SKIP: missing agent prompt ($AGENT_PROMPT)"
    exit 0
fi

if ! command -v "$CLAUDE_BIN" >/dev/null 2>&1; then
    say "SKIP: claude CLI not available ($CLAUDE_BIN)"
    exit 0
fi

yaml_get() {
    local file="$1" key="$2"
    [ -f "$file" ] || return 0
    awk -F': ' -v key="$key" '$1 == key {print $2; exit}' "$file"
}

if [ -f "$LOCK" ]; then
    lock_pid=$(cat "$LOCK" 2>/dev/null || echo "")
    lock_age=$(( ( $(date +%s) - $(stat -f%m "$LOCK" 2>/dev/null || echo "0") ) / 60 ))
    if [ -n "$lock_pid" ] && kill -0 "$lock_pid" 2>/dev/null; then
        if [ "$lock_age" -lt "$MAX_LOCK_AGE_MINUTES" ]; then
            say "SKIP: research session still active (PID $lock_pid, ${lock_age}m old)"
            exit 0
        fi
        say "STALE: lock held by PID $lock_pid for ${lock_age}m. Removing."
        rm -f "$LOCK"
    else
        say "ORPHAN: lock for dead PID $lock_pid. Removing."
        rm -f "$LOCK"
    fi
fi

pending=$(grep -c '^\[ \]' "$QUEUE" 2>/dev/null || echo "0")
if [ "$pending" -eq 0 ]; then
    say "SKIP: no pending hypotheses in queue ($QUEUE)"
    exit 0
fi

readarray -t idle_lines < <(
    "$PYTHON_BIN" "$SURFACE" board --json 2>/dev/null | "$PYTHON_BIN" - "$LANES" "$CAL_MLX" <<'PY'
import json
import sys
from pathlib import Path

lanes_path = Path(sys.argv[1])
mlx_path = Path(sys.argv[2])
priority = {"search": 0, "validate": 1, "control": 2, "": 3}
calibration = {
    "mlx": mlx_path.name if mlx_path.exists() else "",
}

board = json.load(sys.stdin)
allowed = {}
if lanes_path.exists():
    for raw in lanes_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in raw.split("|")]
        if len(parts) < 9:
            continue
        allowed[parts[0]] = {
            "backend": parts[5],
            "mode": parts[7],
            "calibration": calibration.get(parts[5], ""),
        }

idle = []
for item in board.get("surfaces", []):
    info = allowed.get(item.get("id"))
    if not info:
        continue
    status = item.get("status", "")
    activity = item.get("activity", "")
    if status not in {"idle", "completed", "stopped"}:
        continue
    if activity == "untracked":
        continue
    idle.append(
        (
            priority.get(info["mode"], 3),
            item["id"],
            info["backend"],
            info["mode"],
            info["calibration"],
        )
    )

for _, target, backend, mode, cal in sorted(idle):
    print(f"{target}|{backend}|{mode}|{cal}")
PY
)

if [ "${#idle_lines[@]}" -eq 0 ]; then
    say "SKIP: no idle research lanes"
    exit 0
fi

idle_targets=""
idle_summary=""
for entry in "${idle_lines[@]}"; do
    IFS='|' read -r target backend mode calibration <<<"$entry"
    idle_targets="${idle_targets}${idle_targets:+ }$target"
    idle_summary="${idle_summary}- $target backend=$backend mode=$mode calibration=${calibration:-none}\n"
done

already_tested=0
while IFS= read -r line; do
    [[ "$line" != "[ ]"* ]] && continue
    hyp_key=$(echo "$line" | sed 's/^\[ \] //' | cut -c1-50)
    if grep -q "$hyp_key" "$LEDGER" 2>/dev/null; then
        already_tested=$((already_tested + 1))
    fi
done < "$QUEUE"

new_pending=$((pending - already_tested))
if [ "$new_pending" -le 0 ]; then
    say "SKIP: $pending hypotheses in queue but all $already_tested already in ledger"
    exit 0
fi

recent_invalids=$(tail -20 "$LEDGER" 2>/dev/null | grep -c '"INVALID"' || echo "0")
if [ "$recent_invalids" -ge 5 ]; then
    say "CIRCUIT BREAKER: $recent_invalids INVALID results in last 20 experiments. Needs human review."
    exit 0
fi

mlx_minutes=$(yaml_get "$CAL_MLX" wall_clock_minutes)
mlx_metric=$(yaml_get "$CAL_MLX" metric)

say "TRIGGER: $new_pending pending hypotheses, idle lanes: $idle_targets"
echo $$ >"$LOCK"

cat >"$PROMPT" <<PROMPT
You are running the optional tiny-lab hypothesis queue on the shipped MLX example.

Read these files first:
- research/ledger.jsonl
- research/questions.yaml
- research/hypothesis_queue.md
- research/agent.md
- examples/mlx/train.py
- calibration/mlx-5min.yaml

Run these before you launch anything:
- surface board --json
- surface doctor [TARGET]
- surface status [TARGET] --verbose

IDLE LANES AVAILABLE:
$(printf "%b" "$idle_summary")

CALIBRATION DEFAULTS:
- MLX lanes default to calibration/mlx-5min.yaml (${mlx_minutes:-5} minute wall clock, metric ${mlx_metric:-val_bpb})

SCHEDULER RULES:
1. Work only on the shipped MLX trainer in \`examples/mlx/train.py\`.
2. Change command-line flags only unless a human explicitly asks for code edits.
3. Keep each run to one changed variable relative to the baseline.
4. Skip any lane that surface doctor marks unhealthy or untracked.

YOUR TASK:
1. Read the hypothesis queue. Pick the top unchecked [ ] item.
2. Choose one idle MLX lane from the list above.
3. Launch one single-variable run with: surface run [TARGET] "python3 train.py ..." --name [EXP-ID] --eval-on-checkpoint ane/eval_tiny.py
4. Wait for completion.
5. Read \`examples/mlx/results.tsv\` and the eval output for the checkpoint.
6. Record the result in ledger.jsonl (WIN/LOSS/INVALID/INCONCLUSIVE).
7. Mark the hypothesis [x] when resolved.
8. If idle lanes remain and queue has items, repeat up to $MAX_EXPERIMENTS_PER_CYCLE experiments this cycle.

RULES:
- One variable per experiment.
- Use the public baseline from \`research/ledger.jsonl\` as the control.
- INVALID is not LOSS.
- If a hypothesis requires code edits, installs, or cloning: mark it [?] and skip it.
- Maximum $MAX_EXPERIMENTS_PER_CYCLE experiments this cycle, then exit.
PROMPT

say "LAUNCHING: research cycle with max $MAX_EXPERIMENTS_PER_CYCLE experiments"
cd "$ROOT"
"$CLAUDE_BIN" -p "$(cat "$PROMPT")" \
  --allowedTools "$CLAUDE_ALLOWED_TOOLS" \
  --max-turns "$CLAUDE_MAX_TURNS" \
  --output-format text \
  >>"$LOG" 2>&1
cycle_exit=$?

rm -f "$LOCK"
say "CYCLE COMPLETE (exit=$cycle_exit)"
