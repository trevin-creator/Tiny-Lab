# Probes

Small direct sanity checks for tiny-lab 2.0.

Scripts:

- `memory_ceiling.py` — reports total memory, a rough free-memory estimate, and whether a requested working set fits.
- `backend_heartbeat.py` — summarizes lane health from `surface board --json`.
- `env_sanity.py` — checks MLX, MPS, and ANE-adjacent local prerequisites.

Each probe prints JSON to stdout. Add `--write` to drop a timestamped record in `probes/results/`.

Examples:

```bash
python3 probes/memory_ceiling.py --required-gb 24 --write
python3 probes/backend_heartbeat.py --write
python3 probes/env_sanity.py --write
```
