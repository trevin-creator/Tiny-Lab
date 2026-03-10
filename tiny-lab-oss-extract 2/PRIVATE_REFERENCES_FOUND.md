# Private References Found

| File | Pattern / line | Why it is private or machine-specific | What changed | Human review needed |
| --- | --- | --- | --- | --- |
| `bin/research-trigger.sh` | hardcoded repo root, prompt path, and CLI assumptions | Tied the script to a private local layout | Replaced with `TINY_LAB_ROOT`, `research/agent.md`, and optional `claude` detection | No |
| `config/tiny_lab_lanes.tsv` | private lane registry from the v2 worktree | Usernames, hostnames, SSH targets, and absolute workdirs | Replaced with two local sample lanes rooted at `runtime/` | No |
| `surfaces.txt` | legacy private registry entries | Usernames, hostnames, Bonjour names, absolute workdirs, unrelated internal services | Reduced to a legacy-only comment file | No |
| `research/ledger.jsonl` | live recovery and transfer entries | Absolute home paths, machine names, internal lane names, commit references, reclaimed-run notes | Trimmed to a public-safe subset and normalized lane names | Yes |
| `research/hypothesis_queue.md` | live machine-state and priority sections | Private lane names and operational notes from the live cluster | Rewrote as a sanitized snapshot with normalized lane names | Yes |
| `bin/surface` | root env name, DB name, lane workdirs, and owner truth handling | Export should not ship internal naming or absolute local assumptions | Renamed to `TINY_LAB_ROOT`, `tiny_lab.db`, repo-relative file lanes, and explicit stale truth handling | No |
| `ane/eval_tiny.py` | result JSON path fields | The live file recorded resolved absolute paths in result JSONs | Export keeps user-supplied paths instead of resolved absolute paths | No |
