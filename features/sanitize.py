"""Strip success/fail path-directory cues (label leakage ablation).

RoboFAC (and fixtures) put GT in folder names: ``dataset_success_cleaned``,
``stack_ok``, ``fail_*``, etc. Baseline / stub / llm_proxy read those via
``path_text``. Ablating them yields a fairer detection ceiling.

Enable with ``LEAK_ABLATE=1``, ``DecisionConfig.ablate_path_leak``, or an
explicit ``ablate_path_leak=True`` on extract.
"""

from __future__ import annotations

import os
import re
from typing import Iterable

# Extra RoboFAC / fixture folder cues beyond DecisionConfig path tokens.
# Applied longest-first inside sanitize_path_text.
EXTRA_LEAK_TOKENS: tuple[str, ...] = (
    "dataset_success_cleaned",
    "dataset_success",
    "success_cleaned",
    "stack_ok",
    "/success/",
    "/fail/",
    "/failure/",
    "_success_",
    "_fail_",
    "_failure_",
    "success/",
    "fail/",
    "failure/",
)

# Defaults mirroring configs/decision.json (kept here to avoid import cycles).
_DEFAULT_FAIL = ("fail", "failure")
_DEFAULT_SUCCESS = ("success", "success_cleaned", "stack_ok")


def path_leak_ablate_enabled(explicit: bool | None = None) -> bool:
    """Resolve ablation flag: explicit arg > env ``LEAK_ABLATE``."""
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get("LEAK_ABLATE", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def collect_leak_tokens(
    fail_path_tokens: Iterable[str] | None = None,
    success_path_tokens: Iterable[str] | None = None,
    *,
    extra: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Union of fail/success path tokens + common folder cues (longest first)."""
    fail = tuple(fail_path_tokens) if fail_path_tokens is not None else _DEFAULT_FAIL
    ok = (
        tuple(success_path_tokens)
        if success_path_tokens is not None
        else _DEFAULT_SUCCESS
    )
    extras = tuple(extra) if extra is not None else EXTRA_LEAK_TOKENS
    seen: set[str] = set()
    ordered: list[str] = []
    for tok in (*fail, *ok, *extras):
        t = (tok or "").strip().lower()
        if t and t not in seen:
            seen.add(t)
            ordered.append(t)
    ordered.sort(key=len, reverse=True)
    return tuple(ordered)


def sanitize_path_text(
    text: str,
    tokens: Iterable[str] | None = None,
) -> str:
    """Remove leak tokens from path_text (case-insensitive substring wipe)."""
    cleaned = (text or "").lower()
    toks = (
        tuple(tokens)
        if tokens is not None
        else collect_leak_tokens()
    )
    # Ensure longest-first even if caller passed an unsorted list.
    toks = tuple(sorted({t.lower() for t in toks if t}, key=len, reverse=True))
    for tok in toks:
        cleaned = cleaned.replace(tok, " ")
    cleaned = re.sub(r"[/_\s]+", " ", cleaned)
    return cleaned.strip()
