"""GT-free bag-of-tokens hashing embeddings (no sklearn / network).

Uses path / instruction / task text only — **not** ``failure_type`` — so
cluster purity vs type is an honest label-consistency check.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, Sequence

from failure_bank.case import FailureCase

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
DEFAULT_DIM = 64


def tokenize(*parts: str | None) -> list[str]:
    text = " ".join(p for p in parts if p)
    return _TOKEN_RE.findall(text.lower())


def case_text(case: FailureCase) -> str:
    """GT-free text used for embedding."""
    return " ".join(
        p
        for p in (case.path_text, case.instruction, case.task, case.episode_id)
        if p
    )


def _stable_hash(token: str) -> int:
    """Deterministic 32-bit hash (avoids PYTHONHASHSEED nondeterminism)."""
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def hash_embed(
    texts: Sequence[str],
    *,
    dim: int = DEFAULT_DIM,
) -> list[list[float]]:
    """Hashing-trick bag-of-tokens → L2-normalized vectors."""
    if dim < 8:
        raise ValueError("dim must be >= 8")
    out: list[list[float]] = []
    for text in texts:
        vec = [0.0] * dim
        toks = tokenize(text)
        if not toks:
            out.append(vec)
            continue
        for tok in toks:
            h = _stable_hash(tok)
            idx = h % dim
            sign = 1.0 if (h & 1) == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        out.append(vec)
    return out


def embed_cases(
    cases: Iterable[FailureCase],
    *,
    dim: int = DEFAULT_DIM,
) -> list[list[float]]:
    rows = list(cases)
    return hash_embed([case_text(c) for c in rows], dim=dim)


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """1 - cosine similarity; both assumed L2-normalized (or zero)."""
    if len(a) != len(b):
        raise ValueError("vector length mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    sim = max(-1.0, min(1.0, dot))
    return 1.0 - sim
