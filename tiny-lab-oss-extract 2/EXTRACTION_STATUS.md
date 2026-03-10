# Extraction Status

- Date: 2026-03-10
- Source repo: live source working tree
- Export repo: this repo
- Source baseline: current on-disk tree in the live repo
- Safety mode: read-only inspection of the source repo; all edits happen only in this export repo
- Status: export candidate assembled and audited

## Scope

Target kernel under extraction:

- `bin/surface`
- `bin/research-trigger.sh`
- `ane/eval_tiny.py`
- `ane/eval_tiny_mlx.py`
- `research/`

Adjacent dependencies pulled in:

- `ane/primitives/bpe_tokenizer.py`
- `ane/primitives/bpe_vocab_512.bin`
- `ane/eval_bundle/`
- `surfaces.txt`
- `research/agent.md`

## Outcome

- Export size: about `352K`
- Source repo size at extraction time: about `38G`
- Current verdict: see `SHIP_DECISION.md`
