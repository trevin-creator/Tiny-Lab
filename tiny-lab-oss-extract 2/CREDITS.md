# Credits

This repo is a small extraction of a larger working system. It should not read as a claim of sole invention. The kernel here sits on top of direct upstream influence, related implementations, and community research.

## Core Contributors

- Trevin Peterson — author of the extraction, tiny-lab system design, and related `autoresearch-mlx` work
- Andrej Karpathy — `autoresearch`, `nanochat`, and the broader research-loop framing this repo builds on

## Key Upstream And Referenced Work

- `danpacary` / `ncdrone` — ANE training implementation, dynamic weights via IOSurface, ANE vs MLX benchmarking, and gossip / cross-pollination ideas that informed this line of work
- `maderix` — reverse-engineered ANE private APIs and demonstrated backprop on Apple Neural Engine
- `miolini` — macOS / MPS port of `autoresearch`
- Apple MLX team — the MLX framework used by the cross-check evaluator path

## Community Acknowledgments

- `Anemll`
- `thebasedcapital`
- `Vipul Divyanshu`
- `HyperspaceAI`

Those names are included because they were part of the research thread and implementation context this extraction grew out of.
