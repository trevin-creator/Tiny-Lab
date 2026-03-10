# Ship Decision

This repo now clears the 1.0 bar for the scope it claims. It ships one real local MLX trainer, one hardened control plane, one public eval bundle with documented provenance, one optional queue trigger, and one line-by-line single-machine README path that was re-smoked from a fresh copy. Remote lanes and public ANE training are explicitly out of scope instead of half-promised.

The remaining limitations are product scope limits, not release blockers: the tool is Apple Silicon only, the optional trigger depends on an external `claude` CLI, and the repo intentionally does not include remote orchestration or the private ANE substrate. Within the public 1.0 claim, the behavior and docs now match.

READY TO OPEN SOURCE
