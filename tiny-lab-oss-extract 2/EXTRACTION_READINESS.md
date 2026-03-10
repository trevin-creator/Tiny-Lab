# Extraction Readiness

Verdict: READY TO OPEN SOURCE

What 1.0 now contains:

- hardened Python `bin/surface` with `run`, `status --verbose`, `stop`, `doctor`, and `board --json`
- one shipped local lane in `config/tiny_lab_lanes.tsv`
- one real public backend example in `examples/mlx/train.py`
- real NumPy and MLX evaluators
- a documented public eval bundle in `ane/eval_bundle/`
- a small public research queue, ledger, questions, levers, and agent prompt
- an optional `claude`-driven trigger in `bin/research-trigger.sh`

What changed from the earlier experimental-kernel export:

- the repo now ships a real MLX training path instead of asking users to bring their own trainer
- the quickstart is single-machine runnable from a clean clone on Apple Silicon
- the eval bundle now has explicit provenance and license notes in `ane/eval_bundle/README.md`
- the optional trigger is scoped as an advanced example instead of part of the guaranteed core tool
- the public lane config now ships only one obvious local example lane

What is explicitly out of scope for 1.0:

- remote SSH lanes
- public ANE training support
- multi-machine orchestration
- public checkpoint hosting
