#!/usr/bin/env python3
from __future__ import annotations

import struct
from array import array
from collections import Counter
from pathlib import Path

MAGIC = 0x31504542
VERSION = 1
BASE = 256
HEADER = struct.Struct("<4I")


def _bytes(data):
    if isinstance(data, (str, Path)):
        p = Path(data).expanduser()
        return p.read_bytes() if p.exists() else str(data).encode("utf-8")
    return bytes(data)


class BPETokenizer:
    def __init__(self, merges=None):
        self.merges = []
        self.rank = {}
        self.pieces = [bytes([i]) for i in range(BASE)]
        self.lens = [1] * BASE
        self.by_first = [[] for _ in range(BASE)]
        if merges:
            self._build(list(merges))

    @property
    def vocab_size(self):
        return len(self.pieces)

    @property
    def max_piece_length(self):
        return max(self.lens, default=1)

    def piece_length(self, token_id):
        return self.lens[token_id]

    def train(self, text_path, num_merges=256):
        ids = list(_bytes(text_path))
        merges, pieces, lens = [], [bytes([i]) for i in range(BASE)], [1] * BASE
        for _ in range(num_merges):
            counts = Counter(zip(ids, ids[1:]))
            if not counts:
                break
            pair, _ = counts.most_common(1)[0]
            new_id, out, i = BASE + len(merges), [], 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
            merges.append(pair)
            piece = pieces[pair[0]] + pieces[pair[1]]
            pieces.append(piece)
            lens.append(len(piece))
        self.merges, self.rank, self.pieces, self.lens = merges, {p: i for i, p in enumerate(merges)}, pieces, lens
        self._index()
        return self

    def encode(self, text):
        raw = _bytes(text)
        if not raw or not self.merges:
            return list(raw)
        out, i, n = [], 0, len(raw)
        while i < n:
            token = raw[i]
            for piece, cand, plen in self.by_first[token]:
                if raw.startswith(piece, i):
                    token = cand
                    i += plen
                    break
            else:
                i += 1
            out.append(token)
        return out

    def decode(self, tokens):
        return b"".join(self.pieces[t] for t in tokens)

    def save(self, path):
        left = array("H", (a for a, _ in self.merges))
        right = array("H", (b for _, b in self.merges))
        if struct.pack("=H", 1) != struct.pack("<H", 1):
            left.byteswap()
            right.byteswap()
        with Path(path).expanduser().resolve().open("wb") as f:
            f.write(HEADER.pack(MAGIC, VERSION, self.vocab_size, len(self.merges)))
            left.tofile(f)
            right.tofile(f)

    @classmethod
    def load(cls, path):
        data = Path(path).expanduser().resolve().read_bytes()
        magic, version, vocab, merges = HEADER.unpack_from(data, 0)
        if magic != MAGIC or version != VERSION or vocab != BASE + merges:
            raise ValueError(f"bad tokenizer file: {path}")
        off = HEADER.size
        left = struct.unpack_from(f"<{merges}H", data, off) if merges else ()
        off += merges * 2
        right = struct.unpack_from(f"<{merges}H", data, off) if merges else ()
        return cls(zip(left, right))

    def _build(self, merges):
        self.merges = merges
        self.rank = {p: i for i, p in enumerate(merges)}
        self.pieces = [bytes([i]) for i in range(BASE)]
        self.lens = [1] * BASE
        for left, right in merges:
            piece = self.pieces[left] + self.pieces[right]
            self.pieces.append(piece)
            self.lens.append(len(piece))
        self._index()

    def _index(self):
        self.by_first = [[] for _ in range(BASE)]
        for token_id, piece in enumerate(self.pieces[BASE:], start=BASE):
            self.by_first[piece[0]].append((piece, token_id, len(piece)))
        for bucket in self.by_first:
            bucket.sort(key=lambda item: item[2], reverse=True)
