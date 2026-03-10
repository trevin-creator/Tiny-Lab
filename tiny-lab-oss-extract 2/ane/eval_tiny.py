#!/usr/bin/env python3
"""eval_tiny.py — the ruler.

Loads a TinyLM checkpoint, scores a fixed eval bundle, emits bits-per-byte +
word formation + samples.  Deterministic CPU forward pass in pure numpy.
Works for any tokenizer (byte-level or BPE-N).

Usage:
    python3 ane/eval_tiny.py --checkpoint path/to/tiny_weights.bin \
                             --tokenizer path/to/bpe_vocab_512.bin \
                             --eval-bundle ane/eval_bundle \
                             --out eval.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# BPE tokenizer loader (reuse the repo's bpe_tokenizer.py)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent / "primitives"))
from bpe_tokenizer import BPETokenizer  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_MAGIC = 0x54594E32
CHECKPOINT_VERSION = 2
HEADER_FMT = "<9I4xQ"          # 9 uint32 + 4-byte pad + 1 uint64 = 48 bytes
HEADER_SIZE = struct.calcsize(HEADER_FMT)
LOGIT_CAP = 15.0
ALIBI_SLOPES_4 = [0.5, 0.25, 0.125, 0.0625]


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------
def load_checkpoint(path: str) -> tuple[dict, dict]:
    """Load a TinyLM v2 checkpoint.  Returns (config, weights)."""
    data = Path(path).read_bytes()
    fields = struct.unpack_from(HEADER_FMT, data, 0)
    magic, version, step, layers, vocab, dim, hidden, heads, seq, cls_count = fields
    assert magic == CHECKPOINT_MAGIC, f"bad magic: 0x{magic:08X}"
    assert version == CHECKPOINT_VERSION, f"bad version: {version}"
    assert cls_count == vocab * dim

    cfg = dict(step=step, layers=layers, vocab=vocab, dim=dim,
               hidden=hidden, heads=heads, seq=seq)
    head_dim = dim // heads

    off = HEADER_SIZE
    def read_floats(n):
        nonlocal off
        arr = np.frombuffer(data, dtype=np.float32, count=n, offset=off).copy()
        off += n * 4
        return arr

    # Classifier embedding [vocab, dim]
    classifier = read_floats(cls_count).reshape(vocab, dim)
    read_floats(cls_count)  # adam_m — skip
    read_floats(cls_count)  # adam_v — skip

    attn_n = dim * dim
    ffn_in_n = dim * hidden
    ffn_out_n = hidden * dim

    layer_weights = []
    for _ in range(layers):
        wq = read_floats(attn_n).reshape(dim, dim)
        wk = read_floats(attn_n).reshape(dim, dim)
        wv = read_floats(attn_n).reshape(dim, dim)
        wo = read_floats(attn_n).reshape(dim, dim)
        w1 = read_floats(ffn_in_n).reshape(dim, hidden)
        w2 = read_floats(ffn_out_n).reshape(hidden, dim)
        w3 = read_floats(ffn_in_n).reshape(dim, hidden)
        # Skip adam m and v (7 matrices each)
        for _ in range(14):
            skip_n = attn_n if _ % 7 < 4 else (ffn_in_n if _ % 7 != 5 else ffn_out_n)
            read_floats(skip_n)
        layer_weights.append(dict(wq=wq, wk=wk, wv=wv, wo=wo, w1=w1, w2=w2, w3=w3))

    weights = dict(classifier=classifier, layers=layer_weights)
    return cfg, weights


# ---------------------------------------------------------------------------
# CPU forward pass — matches ane_transformer_layer_forward exactly
# ---------------------------------------------------------------------------
def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMS normalization along last axis. Weights are implicitly 1.0."""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def attention(x_norm: np.ndarray, wq, wk, wv, wo, heads: int, seq: int) -> np.ndarray:
    """Multi-head attention with ALiBi.  x_norm: [seq, dim]."""
    dim = x_norm.shape[-1]
    head_dim = dim // heads

    # Project Q, K, V: checkpoint stores W, ANE bakes W^T for conv, so
    # the forward pass is x @ W (not x @ W^T).
    Q = x_norm @ wq   # [seq, dim]
    K = x_norm @ wk
    V = x_norm @ wv

    # Reshape to [heads, seq, head_dim]
    Q = Q.reshape(seq, heads, head_dim).transpose(1, 0, 2)
    K = K.reshape(seq, heads, head_dim).transpose(1, 0, 2)
    V = V.reshape(seq, heads, head_dim).transpose(1, 0, 2)

    # Scaled dot-product attention scores: [heads, seq, seq]
    scale = 1.0 / math.sqrt(head_dim)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) * scale

    # ALiBi bias + causal mask (vectorized)
    pos = np.arange(seq)
    dist = pos[None, :] - pos[:, None]            # dist[i,j] = j - i
    causal_mask = dist > 0                         # future positions
    slopes = np.array(ALIBI_SLOPES_4[:heads]).reshape(heads, 1, 1)
    alibi = slopes * dist[None, :, :]              # [heads, seq, seq]
    scores += alibi
    scores[:, causal_mask] = -1e9

    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Weighted sum of values: [heads, seq, head_dim]
    attn_out = np.matmul(attn_weights, V)

    # Concatenate heads → [seq, dim]
    attn_out = attn_out.transpose(1, 0, 2).reshape(seq, dim)

    # Output projection
    proj = attn_out @ wo   # [seq, dim]
    return proj


def transformer_layer(x: np.ndarray, layer: dict, heads: int, seq: int) -> np.ndarray:
    """One transformer layer.  Pre-norm (LLaMA-style)."""
    # Attention block
    x_norm = rms_norm(x)
    attn_out = attention(x_norm, layer["wq"], layer["wk"], layer["wv"],
                         layer["wo"], heads, seq)
    x2 = x + attn_out

    # FFN block
    x2_norm = rms_norm(x2)
    h1 = x2_norm @ layer["w1"]    # [seq, hidden]
    h3 = x2_norm @ layer["w3"]    # [seq, hidden]
    gate = silu(h1) * h3
    ffn_out = gate @ layer["w2"]   # [seq, dim]

    return x2 + ffn_out


def forward(tokens: list[int], cfg: dict, weights: dict) -> np.ndarray:
    """Full forward pass.  Returns logits [seq, vocab] (softcapped)."""
    vocab, dim, seq = cfg["vocab"], cfg["dim"], cfg["seq"]
    heads, n_layers = cfg["heads"], cfg["layers"]
    classifier = weights["classifier"]

    # Embed: classifier[tok] for each position
    n = min(len(tokens), seq)
    x = np.zeros((seq, dim), dtype=np.float32)
    for t in range(n):
        tok = tokens[t]
        if 0 <= tok < vocab:
            x[t] = classifier[tok]

    # Transformer layers
    for layer in weights["layers"]:
        x = transformer_layer(x, layer, heads, seq)

    # Logits: x @ classifier^T → [seq, vocab]
    logits = x @ classifier.T

    # Softcap
    logits = LOGIT_CAP * np.tanh(logits / LOGIT_CAP)
    return logits


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def compute_nll(tokens: list[int], cfg: dict, weights: dict) -> tuple[float, int]:
    """Compute total NLL (nats) over a token sequence.  Returns (total_nll, count).

    Processes in non-overlapping windows of cfg['seq'] tokens.
    For each window, the loss is computed at positions 1..seq-1
    (predicting token[i] from token[i-1]).
    """
    seq = cfg["seq"]
    total_nll = 0.0
    count = 0
    vocab = cfg["vocab"]

    for start in range(0, len(tokens) - 1, seq):
        chunk = tokens[start: start + seq]
        if len(chunk) < 2:
            break
        n = len(chunk)

        logits = forward(chunk, cfg, weights)  # [seq, vocab]

        # Cross-entropy for positions 1..n-1 (predict chunk[t] from position t-1)
        for t in range(1, n):
            target = chunk[t]
            if target < 0 or target >= vocab:
                continue
            row = logits[t - 1]
            max_l = row.max()
            exp_l = np.exp(row - max_l)
            prob = exp_l[target] / exp_l.sum()
            total_nll += -math.log(max(prob, 1e-12))
            count += 1

    return total_nll, count


def bits_per_byte(total_nll_nats: float, total_bytes: int) -> float:
    """Convert total NLL in nats to bits per byte."""
    total_bits = total_nll_nats / math.log(2)
    return total_bits / max(total_bytes, 1)


def generate(prompt_tokens: list[int], cfg: dict, weights: dict,
             max_tokens: int = 100) -> list[int]:
    """Greedy-ish generation (top-k=5, temp=0.8) for sample quality."""
    vocab, seq = cfg["vocab"], cfg["seq"]
    context = list(prompt_tokens[:seq])
    generated = []
    rng = np.random.RandomState(42)

    for _ in range(max_tokens):
        logits = forward(context, cfg, weights)
        last_logits = logits[len(context) - 1]

        # Top-k=5, temp=0.8
        top_k = min(5, vocab)
        indices = np.argpartition(last_logits, -top_k)[-top_k:]
        top_logits = last_logits[indices] / 0.8
        top_logits -= top_logits.max()
        probs = np.exp(top_logits)
        probs /= probs.sum()
        chosen = indices[rng.choice(top_k, p=probs)]

        generated.append(int(chosen))
        if len(context) < seq:
            context.append(int(chosen))
        else:
            context = context[1:] + [int(chosen)]

    return generated


def word_score(text: str, wordlist: set[str]) -> float:
    """Fraction of whitespace-delimited tokens that are real English words."""
    words = text.lower().split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if w.strip(".,!?;:'\"") in wordlist)
    return hits / len(words)


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------
def file_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def code_hash() -> str:
    return file_hash(__file__)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="eval_tiny — the ruler")
    p.add_argument("--checkpoint", required=True, help="path to tiny_weights.bin")
    p.add_argument("--tokenizer", required=True, help="path to bpe_vocab_N.bin")
    p.add_argument("--eval-bundle", required=True, help="path to eval_bundle dir")
    p.add_argument("--out", default=None, help="output JSON path (default: stdout)")
    p.add_argument("--max-eval-tokens", type=int, default=4096,
                   help="max tokens to evaluate for NLL (default: 4096)")
    args = p.parse_args()

    bundle = Path(args.eval_bundle)
    heldout_path = bundle / "heldout.txt"
    prompts_path = bundle / "prompts.txt"
    wordlist_path = bundle / "wordlist.txt"

    for f in [heldout_path, prompts_path, wordlist_path]:
        assert f.exists(), f"missing eval bundle file: {f}"

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}", file=sys.stderr)
    cfg, weights = load_checkpoint(args.checkpoint)
    print(f"  config: {cfg['layers']}L d={cfg['dim']} h={cfg['hidden']} "
          f"heads={cfg['heads']} vocab={cfg['vocab']} seq={cfg['seq']} "
          f"step={cfg['step']}", file=sys.stderr)

    # Load tokenizer
    tok = BPETokenizer.load(args.tokenizer)
    print(f"  tokenizer: vocab={tok.vocab_size}", file=sys.stderr)

    # Load eval bundle
    heldout_text = heldout_path.read_bytes()
    prompts = [line.strip() for line in prompts_path.read_text().splitlines() if line.strip()]
    wordlist = set(wordlist_path.read_text().lower().split())

    # Tokenize heldout text
    heldout_tokens = tok.encode(heldout_text)
    heldout_tokens = heldout_tokens[:args.max_eval_tokens]
    total_bytes = sum(tok.piece_length(t) for t in heldout_tokens)

    print(f"  eval tokens: {len(heldout_tokens)}, eval bytes: {total_bytes}", file=sys.stderr)

    # Compute NLL
    print("Computing NLL...", file=sys.stderr)
    total_nll, nll_count = compute_nll(heldout_tokens, cfg, weights)
    bpb = bits_per_byte(total_nll, total_bytes)
    token_nll = total_nll / max(nll_count, 1)
    print(f"  NLL: {token_nll:.4f} nats/token, bpb: {bpb:.4f}", file=sys.stderr)

    # Generate samples
    print("Generating samples...", file=sys.stderr)
    samples = {}
    all_generated_text = ""
    for prompt_text in prompts[:10]:
        prompt_tokens = tok.encode(prompt_text.encode("utf-8"))
        gen_tokens = generate(prompt_tokens, cfg, weights, max_tokens=50)
        gen_text = tok.decode(gen_tokens).decode("utf-8", errors="replace")
        samples[prompt_text] = gen_text
        all_generated_text += " " + gen_text

    # Word score
    ws = word_score(all_generated_text, wordlist)
    print(f"  word_score: {ws:.4f}", file=sys.stderr)

    # Build result
    result = {
        "checkpoint": args.checkpoint,
        "tokenizer": args.tokenizer,
        "step": cfg["step"],
        "config": {
            "layers": cfg["layers"],
            "dim": cfg["dim"],
            "hidden": cfg["hidden"],
            "heads": cfg["heads"],
            "vocab": cfg["vocab"],
            "seq": cfg["seq"],
        },
        "bits_per_byte": round(bpb, 6),
        "token_nll": round(token_nll, 6),
        "eval_tokens": len(heldout_tokens),
        "eval_bytes": total_bytes,
        "word_score": round(ws, 4),
        "samples": samples,
        "data_hash": file_hash(str(heldout_path)),
        "tokenizer_hash": file_hash(args.tokenizer),
        "code_hash": code_hash(),
        "deterministic": True,
    }

    # One-line summary
    summary = (f"EVAL step={cfg['step']} bpb={bpb:.4f} words={ws:.4f} "
               f"config={cfg['layers']}L-{cfg['dim']}-{cfg['hidden']}"
               f"-bpe{cfg['vocab']}-alibi")
    print(summary)

    out_text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(out_text + "\n")
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(out_text)


if __name__ == "__main__":
    main()
