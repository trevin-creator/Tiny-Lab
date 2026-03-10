# MLX Example Backend

This directory is the shipped 1.0 backend path.

- `train.py` trains a tiny 0-layer tied-embedding language model with MLX.
- `train_corpus.txt` is the small local corpus used by the example trainer.
- The trainer writes `tiny_weights.bin`, `results.tsv`, and `last_run.json` in this directory.
- When launched through `surface`, it also writes run-scoped status and checkpoint artifacts under `examples/mlx/runs/`.
- The checkpoint format matches `ane/eval_tiny.py` and `ane/eval_tiny_mlx.py`.

Example:

```bash
python3 train.py --steps 120 --eval-every 20 --save-every 20 --checkpoint-out tiny_weights.bin
```
