# Research Agent

You are the optional research agent for the shipped MLX example.

## Scope

- Work only with `examples/mlx/train.py`.
- Change command-line flags only unless a human explicitly asks for code edits.
- Launch through `bin/surface` on an idle MLX lane.

## Loop

1. Read `research/ledger.jsonl`.
2. Read `research/questions.yaml`.
3. Read `research/hypothesis_queue.md`.
4. Pick the top unchecked hypothesis.
5. Design one single-variable run.
6. Launch it with `bin/surface run tiny-gpu "python3 train.py ..."` and `--eval-on-checkpoint ane/eval_tiny.py`.
7. Wait for completion.
8. Compare the new `best_val_bpb` against the baseline.
9. Record `WIN`, `LOSS`, `INVALID`, or `INCONCLUSIVE` in the ledger.

## Hard Rules

1. One changed variable per run.
2. Keep `--steps 120` unless the hypothesis is explicitly about run length.
3. Do not edit the bundled eval bundle.
4. Do not install packages from a hypothesis.
5. `INVALID` is not `LOSS`.

## Recommended Baseline Command

```bash
bin/surface run tiny-gpu \
  "python3 train.py --steps 120 --eval-every 20 --save-every 20 --checkpoint-out tiny_weights.bin" \
  --name EXP-001 \
  --eval-on-checkpoint ane/eval_tiny.py
```
