"""Embedders. A deterministic hashing embedder (no dependencies, always available) and an optional local ONNX model.

Both return unit-length float32 vectors so a dot product is cosine similarity. Which one is used is a measured
decision (data/MEMORY_DESIGN.md), not a hope: the hashing embedder is the offline/test fallback.
"""

from __future__ import annotations

import re
import zlib
from typing import Protocol

import numpy as np

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)
_STOP = frozenset(
    "a an and are as at be but by do does for from had has have he her his i in is it its me my of on or our she so than that the their them "
    "then there these they this to was we were what when where which who why will with would you your am been being can could should "
    "did before previously earlier".split()
)


def tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall((text or "").lower())]


def content_tokens(text: str) -> list[str]:
    return [_stem(w) for w in tokens(text) if w not in _STOP and len(w) > 1]


def _stem(w: str) -> str:
    w = w.replace("'s", "")
    for suf in ("ingly", "ings", "ing", "edly", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


class Embedder(Protocol):
    name: str
    dim: int
    floor: float  # cosine below this is not "related" for this embedder (calibrated in scripts/memory_eval.py)
    dup: float  # cosine at/above this is the same memory said twice

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Feature hashing over stemmed words, word bigrams and character trigrams. Lexical, deterministic, ~20k texts/s."""

    name = "hash-v1"
    floor = 0.22
    dup = 0.85

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _features(self, text: str) -> dict[int, float]:
        feats: dict[int, float] = {}
        toks = content_tokens(text)

        def put(key: str, w: float) -> None:
            h = zlib.crc32(key.encode("utf-8"))
            idx = h % self.dim
            sign = 1.0 if (h >> 16) & 1 else -1.0
            feats[idx] = feats.get(idx, 0.0) + sign * w

        for t in toks:
            put("w:" + t, 1.0)
            if len(t) >= 5:
                padded = f"#{t}#"
                for i in range(len(padded) - 2):
                    put("c:" + padded[i : i + 3], 0.25)
        for a, b in zip(toks, toks[1:]):
            put(f"b:{a}_{b}", 0.6)
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for idx, v in self._features(t).items():
                out[i, idx] = v
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class FastEmbedder:
    """A small local ONNX embedding model through fastembed (optional dependency; model downloads once)."""

    floor = 0.52  # calibrated by scripts/memory_eval.py (data/MEMORY_RESULTS.md): recall 0.81 with 88% of unrelated personal questions left alone
    dup = 0.93

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", floor: float | None = None) -> None:
        from fastembed import TextEmbedding  # noqa: PLC0415 -- optional

        self.name = model
        self._model = TextEmbedding(model_name=model)
        self.dim = int(len(next(iter(self._model.embed(["probe"])))))
        if floor is not None:
            self.floor = floor

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        arr = np.asarray(list(self._model.embed(list(texts))), dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def get_embedder(kind: str = "auto", model: str = "BAAI/bge-small-en-v1.5") -> Embedder:
    """`hash` always works; `fastembed` needs the optional package; `auto` uses fastembed when it loads, else hash."""
    kind = (kind or "auto").lower()
    if kind == "hash":
        return HashEmbedder()
    try:
        return FastEmbedder(model)
    except Exception:  # noqa: BLE001 -- not installed / offline / model missing
        if kind == "fastembed":
            raise
        return HashEmbedder()