#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ane" / "primitives"))
from bpe_tokenizer import BPETokenizer  # noqa: E402

CHECKPOINT_MAGIC = 0x54594E32
CHECKPOINT_VERSION = 2
HEADER_FMT = "<9I4xQ"
LOGIT_CAP = 15.0


class TinyBigram(nn.Module):
    def __init__(self, vocab_size: int, dim: int, scale: float = 0.02) -> None:
        super().__init__()
        self.classifier = mx.random.normal((vocab_size, dim)) * scale

    def __call__(self, input_ids: mx.array) -> mx.array:
        embeddings = self.classifier[input_ids]
        logits = embeddings @ self.classifier.T
        return LOGIT_CAP * mx.tanh(logits / LOGIT_CAP)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny MLX example checkpoint for tiny-lab")
    parser.add_argument("--corpus", default="train_corpus.txt", help="training corpus relative to examples/mlx/")
    parser.add_argument("--tokenizer", default=str(REPO_ROOT / "ane/primitives/bpe_vocab_512.bin"))
    parser.add_argument("--checkpoint-out", default="tiny_weights.bin", help="final checkpoint output path")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.0, help="extra sleep per step for stop-smoke demos")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_work_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def load_tokens(corpus_path: Path, tokenizer_path: Path) -> tuple[BPETokenizer, list[int]]:
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokens = tokenizer.encode(corpus_path.read_bytes())
    if len(tokens) < 32:
        raise SystemExit(f"corpus too small for training: {corpus_path}")
    return tokenizer, tokens


def split_pairs(tokens: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inputs = np.asarray(tokens[:-1], dtype=np.int32)
    targets = np.asarray(tokens[1:], dtype=np.int32)
    split = max(int(len(inputs) * 0.9), 16)
    split = min(split, len(inputs) - 16)
    if split <= 0:
        raise SystemExit("token split failed; corpus is too small")
    return inputs[:split], targets[:split], inputs[split:], targets[split:]


def stop_requested(steering_path: Path) -> bool:
    return steering_path.exists() and "early_stop=1" in steering_path.read_text()


def evaluate_bpb(model: TinyBigram, tokenizer: BPETokenizer, inputs: np.ndarray, targets: np.ndarray, batch_size: int) -> float:
    total_nll = 0.0
    total_bytes = sum(tokenizer.piece_length(int(token)) for token in targets)
    for start in range(0, len(inputs), batch_size):
        end = start + batch_size
        batch_inputs = mx.array(inputs[start:end], dtype=mx.int32)
        batch_targets = mx.array(targets[start:end], dtype=mx.int32)
        logits = model(batch_inputs)
        losses = nn.losses.cross_entropy(logits, batch_targets, reduction="none")
        total_nll += float(mx.sum(losses))
    return (total_nll / math.log(2)) / max(total_bytes, 1)


def write_status(
    status_path: Path,
    step: int,
    train_loss: float,
    val_bpb: float,
    lr: float,
    ms_step: float,
    best_val: float,
    best_step: int,
    checkpoint_path: Path,
) -> None:
    status_path.write_text(
        "\n".join(
            [
                f"step={step}",
                f"loss={train_loss:.6f}",
                f"val_bpb={val_bpb:.6f}",
                f"lr={lr:.6f}",
                f"ms_step={ms_step:.2f}",
                f"best_val={best_val:.6f}",
                f"best_step={best_step}",
                f"checkpoint={checkpoint_path}",
            ]
        )
        + "\n"
    )


def append_results(results_path: Path, step: int, train_loss: float, val_bpb: float, checkpoint_path: Path) -> None:
    if not results_path.exists():
        results_path.write_text("timestamp\tstep\ttrain_loss\tval_bpb\tcheckpoint\n")
    with results_path.open("a") as handle:
        handle.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{step}\t{train_loss:.6f}\t{val_bpb:.6f}\t{checkpoint_path}\n"
        )


def save_checkpoint(
    output_path: Path,
    run_dir: Path,
    classifier: mx.array,
    step: int,
    vocab_size: int,
    dim: int,
    hidden: int,
    heads: int,
    seq: int,
) -> Path:
    mx.eval(classifier)
    classifier_np = np.asarray(classifier, dtype=np.float32)
    zeros = np.zeros_like(classifier_np)
    payload = struct.pack(
        HEADER_FMT,
        CHECKPOINT_MAGIC,
        CHECKPOINT_VERSION,
        step,
        0,
        vocab_size,
        dim,
        hidden,
        heads,
        seq,
        vocab_size * dim,
    )
    payload += classifier_np.tobytes()
    payload += zeros.tobytes()
    payload += zeros.tobytes()

    run_checkpoint = (run_dir / "tiny_weights.bin").resolve()
    run_checkpoint.write_bytes(payload)
    output_path.write_bytes(payload)
    return output_path


def main() -> int:
    args = parse_args()
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    workdir = Path.cwd().resolve()
    run_dir = Path(os.environ.get("TINY_LAB_RUN_DIR", workdir / ".local_run")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    steering_path = Path(os.environ.get("TINY_LAB_STEERING_PATH", run_dir / "steering.txt"))
    status_path = Path(os.environ.get("TINY_LAB_STATUS_PATH", run_dir / "steering_status.txt"))
    results_path = workdir / "results.tsv"
    summary_path = workdir / "last_run.json"

    corpus_path = resolve_work_path(args.corpus)
    tokenizer_path = Path(args.tokenizer).resolve()
    checkpoint_path = resolve_work_path(args.checkpoint_out)

    tokenizer, tokens = load_tokens(corpus_path, tokenizer_path)
    train_inputs, train_targets, val_inputs, val_targets = split_pairs(tokens)
    model = TinyBigram(tokenizer.vocab_size, args.dim)
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    loss_and_grad = nn.value_and_grad(
        model,
        lambda current, batch_inputs, batch_targets: nn.losses.cross_entropy(
            current(batch_inputs), batch_targets, reduction="mean"
        ),
    )

    best_val = float("inf")
    best_step = 0
    last_val = float("inf")
    last_checkpoint = checkpoint_path
    final_status = "completed"
    start_time = time.time()

    for step in range(1, args.steps + 1):
        step_start = time.time()
        batch_indices = rng.integers(0, len(train_inputs), size=args.batch_size)
        batch_inputs = mx.array(train_inputs[batch_indices], dtype=mx.int32)
        batch_targets = mx.array(train_targets[batch_indices], dtype=mx.int32)

        loss, grads = loss_and_grad(model, batch_inputs, batch_targets)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())

        train_loss = float(loss)
        ms_step = (time.time() - step_start) * 1000.0
        should_eval = step == 1 or step == args.steps or step % args.eval_every == 0
        should_save = step == 1 or step == args.steps or step % args.save_every == 0

        if should_eval:
            last_val = evaluate_bpb(model, tokenizer, val_inputs, val_targets, args.batch_size)
            if last_val < best_val:
                best_val = last_val
                best_step = step
                last_checkpoint = save_checkpoint(
                    checkpoint_path,
                    run_dir,
                    model.classifier,
                    step,
                    tokenizer.vocab_size,
                    args.dim,
                    args.hidden,
                    args.heads,
                    args.seq,
                )
        elif should_save:
            last_checkpoint = save_checkpoint(
                checkpoint_path,
                run_dir,
                model.classifier,
                step,
                tokenizer.vocab_size,
                args.dim,
                args.hidden,
                args.heads,
                args.seq,
            )

        write_status(status_path, step, train_loss, last_val, args.learning_rate, ms_step, best_val, best_step, last_checkpoint)
        if should_eval:
            append_results(results_path, step, train_loss, last_val, last_checkpoint)
            print(f"step={step} loss={train_loss:.4f} val_bpb={last_val:.4f} checkpoint={last_checkpoint}")

        if stop_requested(steering_path):
            final_status = "stopped"
            save_checkpoint(
                checkpoint_path,
                run_dir,
                model.classifier,
                step,
                tokenizer.vocab_size,
                args.dim,
                args.hidden,
                args.heads,
                args.seq,
            )
            write_status(status_path, step, train_loss, last_val, args.learning_rate, ms_step, best_val, best_step, checkpoint_path)
            break

        if args.sleep:
            time.sleep(args.sleep)

    elapsed = time.time() - start_time
    summary = {
        "status": final_status,
        "steps_requested": args.steps,
        "best_step": best_step,
        "best_val_bpb": round(best_val, 6),
        "final_val_bpb": round(last_val, 6),
        "checkpoint": str(checkpoint_path),
        "run_checkpoint": str(run_dir / "tiny_weights.bin"),
        "corpus": str(corpus_path),
        "tokenizer": str(tokenizer_path),
        "elapsed_seconds": round(elapsed, 3),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
