# Release Checklist

- Confirm the owner wants MIT for this export.
- Confirm `ane/eval_bundle/*` can be redistributed publicly.
- Run the smoke path on a fresh clone: `bash -n bin/research-trigger.sh`, `python3 bin/surface list`, `python3 bin/surface status tiny-ane --verbose`, `pytest -q tests/tiny_lab/test_surface.py`.
- Verify no runtime artifacts are present before publish: `tiny_lab.db`, `protocol/board.json`, `research/.research-lock`, `research/.current_cycle_prompt.md`, `research/trigger.log`, `runtime/*/runs/`.
- Decide whether to publish the research snapshot exactly as-is or trim it further.
- If a real trainer will not ship, keep the README kernel-only and do not add fake gate files.
- If the trigger will publish, confirm the Claude CLI dependency is acceptable to state plainly.
