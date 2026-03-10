# Exported Files

| Export path | Source role | Why it made the cut |
| --- | --- | --- |
| `bin/surface` | hardened control plane | smallest real manager that still keeps truth, stop, doctor, and board behavior |
| `bin/research-trigger.sh` | optional trigger | real scheduler entrypoint, now tightened for public local use |
| `config/tiny_lab_lanes.tsv` | lane map | public replacement for the private fleet registry |
| `tests/tiny_lab/test_surface.py` | smoke suite | proves the local control plane without touching live runs |
| `ane/eval_tiny.py` | NumPy evaluator | primary scoring path |
| `ane/eval_tiny_mlx.py` | MLX evaluator | cross-validation path |
| `ane/primitives/bpe_tokenizer.py` | tokenizer loader | required by both evaluators |
| `ane/primitives/bpe_vocab_512.bin` | tokenizer artifact | required by the default eval path |
| `ane/eval_bundle/*` | eval inputs | real heldout/prompts/word list used by the evaluators |
| `research/agent.md` | agent prompt | minimal protocol for the queue-driven loop |
| `research/ledger.jsonl` | ledger slice | honest retained experiment history |
| `research/questions.yaml` | question graph | companion dependency map for the ledger slice |
| `research/hypothesis_queue.md` | queue snapshot | active public queue snapshot |
| `research/levers.yaml` | lever list | compact list of real mutation axes |
| `calibration/*.yaml` | calibration presets | tiny backend budget helpers from v2 |
| `probes/*.py` | local probes | useful health checks worth shipping |
| `scripts/research_cards.py` | card generator | lightweight formatter for ledger rows |
| `surfaces.txt` | legacy registry | retained for user-added manual targets only |
