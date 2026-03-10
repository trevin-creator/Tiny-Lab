# Redaction Audit

## Completed

- Lane-map redaction: the private multi-machine lane registry was replaced with two local sample lanes under `runtime/`.
- Registry redaction: `surfaces.txt` is now a legacy-only comment file instead of a shipped fleet map.
- Trigger redaction: the trigger no longer points at a personal repo path, now reads `research/agent.md`, and skips cleanly if `claude` is absent.
- Ledger redaction: absolute home paths, private lane names, and internal recovery notes were removed from the exported subset.
- Queue redaction: lane names were normalized to public roles.
- Naming cleanup: internal root and DB naming in the manager was replaced with `TINY_LAB_ROOT` and `tiny_lab.db`.

## Preserved On Purpose

- The actual surface manager logic is still one direct script plus sqlite and subprocess/ssh. No framework layer was added.
- The evaluator math is still the real working code path from the live repo.
- The research snapshot still shows that the queue and question graph are slightly out of sync in the live system.

## Still Needs Human Review

- `ane/eval_bundle/heldout.txt`: provenance and redistribution rights were not established from the local repo alone.
- `LICENSE`: the repo now has a candidate permissive license, but the owner should confirm that choice before publishing.
- `research/ledger.jsonl`: the metrics are real and public-safe after redaction, but the owner should confirm there is no strategic sensitivity in publishing them.
