# tiny-lab

`tiny-lab` is a small Apple Silicon experiment tool. It ships one real MLX trainer, one hardened `surface` control plane, one real eval bundle, and one optional research trigger. The public 1.0 path is single-machine only: configure one local lane, launch a run, stop it safely, score the checkpoint, and keep a small hypothesis queue.

## Who It Is For

Use `tiny-lab` if you want a direct local workflow for tiny language-model experiments on one Apple Silicon machine. It is not a cluster manager, not a general scheduler, and not a full reproduction of the private lab it came from.

## What Ships

- `bin/surface` — the control plane: `list`, `run`, `status --verbose`, `stop`, `doctor`, and `board --json`
- `examples/mlx/train.py` — the shipped MLX backend example
- `ane/eval_tiny.py` — NumPy evaluator
- `ane/eval_tiny_mlx.py` — MLX cross-check evaluator
- `ane/eval_bundle/` — public eval bundle, including the TinyStories held-out slice used for bpb scoring
- `config/tiny_lab_lanes.tsv` — the single shipped local lane
- `research/` — a small public queue, ledger, questions, levers, and agent prompt
- `bin/research-trigger.sh` — optional advanced trigger for a `claude`-driven queue cycle

## What `surface` Does

`surface` manages one or more lanes. In 1.0, the shipped lane is local and file-based. `surface` launches commands in the lane workdir, records `run.json` and `state.json`, keeps a small SQLite index, reports the current truth source, stops runs with an early-stop token plus process kill fallback, and repairs stale state with `doctor --fix`.

## What `eval` Does

`ane/eval_tiny.py` is the canonical scorer. It loads the binary checkpoint format used by the shipped trainer, computes bits per byte on a fixed held-out bundle, emits a word score, and writes a JSON report. `ane/eval_tiny_mlx.py` runs the same checkpoint through MLX and can compare MLX against the NumPy scorer.

## ANE Status

`tiny-lab` 1.0 does not ship a public ANE training path.

- the public repo ships `ane/` evaluators and tokenizer assets only
- the default and only supported training quickstart is the MLX path in `examples/mlx/`
- the private working ANE system depends on lower-level primitives, vendor code, and build steps that are not part of this public repo

If you see `ane/` in this repo, read it as evaluation tooling for the shared checkpoint format, not as a supported ANE trainer.

## Quickstart

Run these from the repo root on Apple Silicon:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 bin/surface list
python3 bin/surface status tiny-gpu --verbose
```

The shipped lane is defined in `config/tiny_lab_lanes.tsv` and points at `examples/mlx/`. You can use it as-is, or edit that one row if you want a different lane name or workdir.

Launch one real run:

```bash
python3 bin/surface run tiny-gpu \
  "python3 train.py --steps 120 --eval-every 20 --save-every 20 --checkpoint-out tiny_weights.bin" \
  --name quickstart \
  --eval-on-checkpoint ane/eval_tiny.py
```

Inspect what happened:

```bash
python3 bin/surface status tiny-gpu --verbose
python3 bin/surface board --json
python3 bin/surface doctor tiny-gpu
```

Evaluate the checkpoint directly:

```bash
python3 ane/eval_tiny.py \
  --checkpoint examples/mlx/tiny_weights.bin \
  --tokenizer ane/primitives/bpe_vocab_512.bin \
  --eval-bundle ane/eval_bundle \
  --out examples/mlx/quickstart_eval.json

python3 ane/eval_tiny_mlx.py \
  --checkpoint examples/mlx/tiny_weights.bin \
  --tokenizer ane/primitives/bpe_vocab_512.bin \
  --eval-bundle ane/eval_bundle \
  --compare
```

Test safe stop in a second terminal:

```bash
python3 bin/surface run tiny-gpu \
  "python3 train.py --steps 1000 --eval-every 25 --save-every 25 --sleep 0.1 --checkpoint-out tiny_weights.bin" \
  --name stop-demo \
  --eval-on-checkpoint ane/eval_tiny.py
```

Then stop it:

```bash
python3 bin/surface stop tiny-gpu
python3 bin/surface doctor tiny-gpu
```

## Hypotheses

The optional research path is file-based. Add a new line to `research/hypothesis_queue.md` with the format `- [ ] your hypothesis here`. The shipped queue and ledger are built around the MLX example in `examples/mlx/train.py`.

If you have a working `claude` CLI, you can ask the optional trigger to process one queue cycle:

```bash
bash bin/research-trigger.sh
```

If `claude` is missing or no hypotheses are pending, the script prints a skip reason and exits without launching anything.

## Config

`config/tiny_lab_lanes.tsv` is the one obvious place to edit local lane settings. The columns are:

`target|machine|protocol|host|workdir|backend|lane|mode|capabilities`

1.0 supports only local `file` lanes in the shipped config. Remote SSH lanes are out of scope for this release.

## Eval Bundle Provenance

`ane/eval_bundle/heldout.txt` is a 100,574-byte held-out slice from the public TinyStories dataset. The dataset card lists the license as `cdla-sharing-1.0`. The repo preserves the attribution and license link in `ane/eval_bundle/README.md`.

## Optional Vs Core

Core 1.0 path:

- `bin/surface`
- `examples/mlx/train.py`
- `ane/eval_tiny.py`
- `ane/eval_tiny_mlx.py`
- `config/tiny_lab_lanes.tsv`

Optional:

- `bin/research-trigger.sh`
- `research/` queue automation
- editing the shipped lane map for your own trainer/workdir

Experimental or advanced notes:

- private ANE training exists in the source system, but it is not a shipped 1.0 path here

## Out Of Scope

- remote SSH lanes
- public ANE training support
- multi-machine orchestration
- public checkpoint hosting
- general scheduler abstractions

## Credits

- Trevin Peterson — author of the extraction, tiny-lab system design, and related `autoresearch-mlx` work
- Andrej Karpathy — `autoresearch`, `nanochat`, and the broader research-loop framing this repo builds on
- `danpacary` / `ncdrone`, `maderix`, `miolini`, and the Apple MLX team — key upstream and adjacent implementation work

Full attribution lives in `CREDITS.md`.
