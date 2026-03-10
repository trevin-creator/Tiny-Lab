# Release Gate

1. Ships one real runnable trainer/backend example
   - pass
   - `examples/mlx/train.py` is the shipped MLX backend path and runs through `surface`

2. Ships one real public eval path with publishable data handling
   - pass
   - `ane/eval_tiny.py`, `ane/eval_tiny_mlx.py`, and `ane/eval_bundle/README.md` now document the TinyStories held-out slice and license path

3. Supports a clean single-machine quickstart end to end
   - pass
   - the README quickstart was run from a fresh copy with one Apple Silicon machine and the shipped `tiny-gpu` lane

4. Remote setup is either supported or explicitly scoped out
   - pass
   - remote SSH lanes are explicitly out of scope for 1.0

5. Scheduler/trigger is either generic enough to ship or clearly optional
   - pass
   - `bin/research-trigger.sh` is kept as an optional advanced example and prints skip reasons when `claude` is absent or the queue is empty

6. No private machine identity remains in tracked files
   - pass
   - tracked config and docs use local sample names and repo-relative paths only

7. README works line by line from a clean clone
   - pass
   - install, run, status, stop, eval, and optional trigger behavior were rechecked from a fresh copy

8. Truth model works correctly
   - pass
   - the shipped control plane reports live process truth correctly, stale state does not outrank process truth, and `doctor --fix` repairs stale state

9. ANE is either fully shipped or clearly out of scope
   - pass
   - the public repo ships no ANE trainer, lane, or build path; README and release docs now position `ane/` as eval tooling only

10. No generated junk ships in the repo
   - pass
   - local DBs, run artifacts, caches, logs, and board snapshots were removed and ignored

11. Release verdict can honestly say READY TO OPEN SOURCE
   - pass
   - the public 1.0 scope is now small, real, single-machine runnable, and documented without hidden private dependencies
