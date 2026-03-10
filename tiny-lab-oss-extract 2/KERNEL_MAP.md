# Kernel Map

Source analyzed:

- export repo at `tiny-lab-oss-extract`
- private v2 hardening worktree used as the ops source of truth

Real research slices preserved in the export:

- evaluator-trust chain: `EXP-005 -> EXP-006 -> EXP-007`
- transfer/control slice: `EXP-018 -> EXP-021 -> EXP-022 -> EXP-024 -> EXP-025`

## Must Export

| File | Purpose | Why it ships |
| --- | --- | --- |
| `bin/surface` | control plane | Real run manager, truth resolver, stop path, doctor, and board |
| `config/tiny_lab_lanes.tsv` | lane map | Real lane separation without private fleet config |
| `tests/tiny_lab/test_surface.py` | control-plane proof | Covers stale-state repair, stop, rerun, lane separation, and untracked detection |
| `bin/research-trigger.sh` | optional scheduler | Small real trigger path, now scoped as an example |
| `ane/eval_tiny.py` | primary evaluator | Real NumPy scoring path |
| `ane/eval_tiny_mlx.py` | cross-check evaluator | Real MLX validation path |
| `research/agent.md` | research prompt | Minimal decision protocol for the queue-driven loop |
| `research/ledger.jsonl` | ledger slice | Honest experiment history slice |
| `research/questions.yaml` | question graph | Dependency map for the retained ledger slice |
| `research/hypothesis_queue.md` | active queue snapshot | Current public snapshot of the queue |
| `research/levers.yaml` | lever list | Small real mutation space description |
| `calibration/*.yaml` | run budgets | Lightweight backend calibration presets |
| `probes/*.py` | local sanity checks | Small useful runtime probes |
| `scripts/research_cards.py` | result formatter | Lightweight card generator for ledger rows |

## Optional Support

| File | Purpose | Why it is optional |
| --- | --- | --- |
| `surfaces.txt` | legacy registry | Kept only for extra manual targets users may add |
| `runtime/*/.gitkeep` | scratch roots | Keeps sample local lane roots in git without shipping generated state |
| `gates/.gitkeep` | placeholder directory | Preserves the runtime path without shipping private or dead gate files |

## Exclude

| Category | Why it stays out |
| --- | --- |
| trainer binaries, source, checkpoints, corpora | too large, too private, and not required for the exported kernel |
| remote SSH lane config | contains private topology and is not needed for the public local control plane |
| generated DBs, board snapshots, logs, caches, run directories | runtime artifacts only |
