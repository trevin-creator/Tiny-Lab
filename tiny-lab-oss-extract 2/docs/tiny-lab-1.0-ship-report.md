# Tiny-Lab 1.0 Ship Report

Date: 2026-03-10

## Release Scope

`tiny-lab` 1.0 is a small Apple Silicon public tool:

- one local `surface` control plane
- one shipped MLX backend example
- one public eval bundle
- one optional hypothesis trigger

Remote SSH lanes, public ANE training support, and multi-machine orchestration are not part of this release.

## What Closed The Last Blockers

- added `examples/mlx/train.py` as the real shipped backend path
- tightened `config/tiny_lab_lanes.tsv` to one obvious local lane
- documented `ane/eval_bundle/heldout.txt` provenance in `ane/eval_bundle/README.md`
- kept `bin/research-trigger.sh` as an optional advanced example instead of core behavior
- rewrote the root README around a clean single-machine quickstart

## Quickstart Smoke

Executed from a fresh copy on one Apple Silicon machine:

1. `python3 -m venv .venv`
   - pass

2. `source .venv/bin/activate`
   - pass

3. `python3 -m pip install -r requirements.txt`
   - pass
   - installed `mlx`, `numpy`, and `pytest` into a fresh venv

4. `python3 bin/surface list`
   - pass
   - rendered `tiny-gpu`

5. `python3 bin/surface status tiny-gpu --verbose`
   - pass
   - lane reported truth and status cleanly

6. `python3 bin/surface run tiny-gpu "python3 train.py --steps 120 --eval-every 20 --save-every 20 --checkpoint-out tiny_weights.bin" --name quickstart --eval-on-checkpoint ane/eval_tiny.py`
   - pass
   - produced a real checkpoint and completed cleanly
   - best checkpoint landed at step `40` with `best_val_bpb=4.427482`

7. `python3 bin/surface doctor tiny-gpu`
   - pass
   - healthy lane reported `OK`

8. `python3 ane/eval_tiny.py --checkpoint examples/mlx/tiny_weights.bin --tokenizer ane/primitives/bpe_vocab_512.bin --eval-bundle ane/eval_bundle --out examples/mlx/quickstart_eval.json`
   - pass
   - reported `bpb=5.2819`

9. `python3 ane/eval_tiny_mlx.py --checkpoint examples/mlx/tiny_weights.bin --tokenizer ane/primitives/bpe_vocab_512.bin --eval-bundle ane/eval_bundle --compare`
   - pass
   - MLX and NumPy matched exactly with `DELTA: 0.0000% [PASS]`

## Control-Plane Smoke

- `surface run -> status --verbose -> stop -> doctor -> rerun`
  - pass on `tiny-gpu`
- `pytest -q tests/tiny_lab/test_surface.py`
  - pass
  - 5 tests passed
- stale-state repair via `doctor --fix`
  - pass
- untracked-process reporting
  - pass

## Trigger Decision

`bin/research-trigger.sh` ships as an optional advanced example.

- it is not required for the core quickstart
- it requires an external `claude` CLI
- `CLAUDE_BIN=claude-does-not-exist bash bin/research-trigger.sh`
  - pass
  - printed `SKIP: claude CLI not available (claude-does-not-exist)`
- it prints a clear skip reason when that dependency is absent or the queue is empty

## ANE Ship Gate

Result: did not pass for the public 1.0 release.

- Build: fail
  - the public repo ships no ANE build instructions, no ANE runner, and no exported ANE vendor substrate
- Run: fail
  - the public repo ships no ANE lane or supported `surface` launch path
- Stop: fail
  - no public ANE managed run exists to stop
- Eval: not applicable as a public ANE path
  - the shipped evaluators can score the shared checkpoint format, but they do not make ANE training publicly runnable
- Docs: fail for a public ANE claim
  - the real ANE path in the private repo depends on lower-level primitives, vendor code, and private-framework build steps that are not part of this release

Positioning for 1.0:

- `ane/` in the public repo is evaluation tooling
- MLX is the only supported training path
- ANE is background context, not a shipped public lane

## Scrub Summary

- removed private hostnames, usernames, SSH targets, and home paths from tracked files
- changed managed-run owner metadata to default to `local-session` instead of `user@hostname`
- removed generated DBs, run artifacts, board snapshots, logs, caches, and Finder metadata
- kept one public lane, one public trainer, and one public eval path

## Verdict

READY TO OPEN SOURCE
