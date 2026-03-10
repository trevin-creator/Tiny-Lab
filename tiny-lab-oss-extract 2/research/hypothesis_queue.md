# Hypotheses — public starter queue

These are example hypotheses for the shipped MLX trainer in `examples/mlx/train.py`.
Keep each run to one flag change relative to the current baseline.

- [ ] Lower `--learning-rate` from `0.02` to `0.01` and compare `best_val_bpb` after `120` steps.
- [ ] Lower `--dim` from `64` to `32` and compare `best_val_bpb` after `120` steps.
- [ ] Raise `--batch-size` from `128` to `256` and compare `best_val_bpb` after `120` steps.
