# Tightness Audit

## What Feels Direct

- `bin/surface` is still one file that does the real work: lane lookup, launch, truth resolution, stop, doctor, and board output.
- The evaluator path is minimal: `ane/eval_tiny.py`, `ane/eval_tiny_mlx.py`, one tokenizer loader, one vocab file, one eval bundle.
- The research loop is still obvious: queue, questions, levers, ledger, trigger, and one agent protocol file.

## What Feels Bloated

- The heldout eval bundle is larger than the rest of the repo and still lacks provenance notes.
- The research snapshot shows two adjacent eras of the live system: the original TinyLM question graph and the newer transfer queue.

## What Should Be Deleted

- Nothing else from the live monorepo should come over by default.
- Generated state such as `tiny_lab.db`, `protocol/board.json`, trigger logs, run directories, and lockfiles should never ship.

## What Must Be Rewritten

- README quickstart must stay kernel-only. It cannot pretend this repo includes a trainer or a turnkey cluster.
- If the eval bundle cannot be cleared for redistribution, it must be replaced with a public bundle or removed.

## What Still Is Not Honest Enough For OSS

- The trigger still depends on the Claude CLI for autonomous cycles.
- The MLX evaluator is real, but the MLX training backend that motivated the current queue is intentionally not bundled here.
- The repo is reviewable and direct, but it is not yet a full runnable public reproduction of the live lab.
