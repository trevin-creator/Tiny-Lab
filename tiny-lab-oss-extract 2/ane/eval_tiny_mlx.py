#!/usr/bin/env python3
"""eval_tiny_mlx.py — EXP-007: MLX/GPU forward pass for TinyLM.

Loads the same binary checkpoint as eval_tiny.py, runs the forward pass on MLX,
and computes bits-per-byte.  If the MLX and numpy evals agree within 1%, we
trust the substrate.

Usage:
    python3 ane/eval_tiny_mlx.py --checkpoint ane/primitives/tiny_weights.bin \
        --tokenizer ane/primitives/bpe_vocab_512.bin \
        --eval-bundle ane/eval_bundle --max-eval-tokens 4096
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent / "primitives"))
from bpe_tokenizer import BPETokenizer  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_MAGIC = 0x54594E32
CHECKPOINT_VERSION = 2
HEADER_FMT = "<9I4xQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
LOGIT_CAP = 15.0
ALIBI_SLOPES = [0.5, 0.25, 0.125, 0.0625]


# ---------------------------------------------------------------------------
# Checkpoint loader (binary → mx.array)
# ---------------------------------------------------------------------------
def load_checkpoint(path: str) -> tuple[dict, dict]:
    import numpy as np
    data = Path(path).read_bytes()
    fields = struct.unpack_from(HEADER_FMT, data, 0)
    magic, version, step, layers, vocab, dim, hidden, heads, seq, cls_count = fields
    assert magic == CHECKPOINT_MAGIC, f"bad magic: 0x{magic:08X}"
    assert version == CHECKPOINT_VERSION
    assert cls_count == vocab * dim
    cfg = dict(step=step, layers=layers, vocab=vocab, dim=dim,
               hidden=hidden, heads=heads, seq=seq)

    off = HEADER_SIZE
    def read(n, shape):
        nonlocal off
        arr = np.frombuffer(data, dtype=np.float32, count=n, offset=off).reshape(shape)
        off += n * 4
        return mx.array(arr)
    def skip(n):
        nonlocal off
        off += n * 4

    classifier = read(cls_count, (vocab, dim))
    skip(cls_count); skip(cls_count)  # adam_m, adam_v

    attn_n, ffn_in, ffn_out = dim * dim, dim * hidden, hidden * dim
    layer_weights = []
    for _ in range(layers):
        lw = dict(
            wq=read(attn_n, (dim, dim)), wk=read(attn_n, (dim, dim)),
            wv=read(attn_n, (dim, dim)), wo=read(attn_n, (dim, dim)),
            w1=read(ffn_in, (dim, hidden)), w2=read(ffn_out, (hidden, dim)),
            w3=read(ffn_in, (dim, hidden)),
        )
        for i in range(14):
            n = attn_n if i % 7 < 4 else (ffn_in if i % 7 != 5 else ffn_out)
            skip(n)
        layer_weights.append(lw)
    return cfg, dict(classifier=classifier, layers=layer_weights)


# ---------------------------------------------------------------------------
# MLX forward pass
# ---------------------------------------------------------------------------
def forward(tokens: list[int], cfg: dict, weights: dict) -> mx.array:
    vocab, dim, seq = cfg["vocab"], cfg["dim"], cfg["seq"]
    heads, n_layers = cfg["heads"], cfg["layers"]
    head_dim = dim // heads
    classifier = weights["classifier"]

    # Embed
    n = min(len(tokens), seq)
    tok_arr = mx.array(tokens[:n], dtype=mx.int32)
    x = mx.zeros((seq, dim))
    x = x.at[:n].add(classifier[tok_arr])

    # Pre-compute ALiBi bias + causal mask
    pos = mx.arange(seq)
    dist = pos[None, :] - pos[:, None]                    # [seq, seq]
    slopes = mx.array(ALIBI_SLOPES[:heads]).reshape(heads, 1, 1)
    alibi = slopes * dist[None, :, :]                      # [heads, seq, seq]
    mask = mx.where(dist > 0, mx.array(-1e9), mx.array(0.0))  # [seq, seq]
    bias = alibi + mask[None, :, :]                        # [heads, seq, seq]
    scale = 1.0 / math.sqrt(head_dim)

    for layer in weights["layers"]:
        # --- attention ---
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        x_norm = x / rms

        Q = (x_norm @ layer["wq"]).reshape(seq, heads, head_dim).transpose(1, 0, 2)
        K = (x_norm @ layer["wk"]).reshape(seq, heads, head_dim).transpose(1, 0, 2)
        V = (x_norm @ layer["wv"]).reshape(seq, heads, head_dim).transpose(1, 0, 2)

        scores = (Q @ K.transpose(0, 2, 1)) * scale + bias
        scores = scores - mx.max(scores, axis=-1, keepdims=True)
        exp_s = mx.exp(scores)
        attn_w = exp_s / mx.sum(exp_s, axis=-1, keepdims=True)
        attn_out = (attn_w @ V).transpose(1, 0, 2).reshape(seq, dim)
        x = x + attn_out @ layer["wo"]

        # --- ffn ---
        rms2 = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        x_norm2 = x / rms2
        h1 = x_norm2 @ layer["w1"]
        gate = mx.sigmoid(h1) * h1 * (x_norm2 @ layer["w3"])  # silu(h1)*h3
        x = x + gate @ layer["w2"]

    # Logits + softcap
    logits = x @ classifier.T
    logits = LOGIT_CAP * mx.tanh(logits / LOGIT_CAP)
    return logits


# ---------------------------------------------------------------------------
# NLL / bpb
# ---------------------------------------------------------------------------
def compute_nll(tokens: list[int], cfg: dict, weights: dict) -> tuple[float, int]:
    seq, vocab = cfg["seq"], cfg["vocab"]
    total_nll, count = 0.0, 0
    for start in range(0, len(tokens) - 1, seq):
        chunk = tokens[start: start + seq]
        if len(chunk) < 2:
            break
        logits = forward(chunk, cfg, weights)
        mx.eval(logits)
        logits_np = logits.tolist()
        n = len(chunk)
        for t in range(1, n):
            target = chunk[t]
            if target < 0 or target >= vocab:
                continue
            row = logits_np[t - 1]
            max_l = max(row)
            exp_l = [math.exp(v - max_l) for v in row]
            s = sum(exp_l)
            prob = exp_l[target] / s
            total_nll += -math.log(max(prob, 1e-12))
            count += 1
    return total_nll, count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="eval_tiny_mlx — EXP-007 substrate check")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--eval-bundle", required=True)
    p.add_argument("--max-eval-tokens", type=int, default=4096)
    p.add_argument("--compare", action="store_true",
                   help="also run numpy eval and report delta")
    args = p.parse_args()

    bundle = Path(args.eval_bundle)
    heldout_path = bundle / "heldout.txt"
    assert heldout_path.exists(), f"missing: {heldout_path}"

    print(f"Loading checkpoint: {args.checkpoint}", file=sys.stderr)
    cfg, weights = load_checkpoint(args.checkpoint)
    print(f"  config: {cfg['layers']}L d={cfg['dim']} h={cfg['hidden']} "
          f"heads={cfg['heads']} vocab={cfg['vocab']} seq={cfg['seq']} "
          f"step={cfg['step']}", file=sys.stderr)

    tok = BPETokenizer.load(args.tokenizer)
    heldout_tokens = tok.encode(heldout_path.read_bytes())
    heldout_tokens = heldout_tokens[:args.max_eval_tokens]
    total_bytes = sum(tok.piece_length(t) for t in heldout_tokens)
    print(f"  eval tokens: {len(heldout_tokens)}, eval bytes: {total_bytes}", file=sys.stderr)

    print("Computing NLL (MLX)...", file=sys.stderr)
    total_nll, nll_count = compute_nll(heldout_tokens, cfg, weights)
    total_bits = total_nll / math.log(2)
    bpb = total_bits / max(total_bytes, 1)

    print(f"MLX_EVAL step={cfg['step']} bpb={bpb:.4f}")

    if args.compare:
        print("Running numpy eval for comparison...", file=sys.stderr)
        # Import the numpy version
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from eval_tiny import load_checkpoint as np_load, compute_nll as np_nll, bits_per_byte
        np_cfg, np_w = np_load(args.checkpoint)
        np_total_nll, np_count = np_nll(heldout_tokens, np_cfg, np_w)
        np_bpb = bits_per_byte(np_total_nll, total_bytes)
        delta_pct = abs(bpb - np_bpb) / np_bpb * 100 if np_bpb > 0 else 0
        match = "PASS" if delta_pct < 1.0 else "FAIL"
        print(f"NUMPY_EVAL step={np_cfg['step']} bpb={np_bpb:.4f}")
        print(f"DELTA: {delta_pct:.4f}% [{match}]")


if __name__ == "__main__":
    main()
