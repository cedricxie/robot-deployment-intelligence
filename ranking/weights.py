"""Configurable ranking weights (A–D set + E/F bonuses)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "ranking_weights.json"
)


@dataclass(frozen=True)
class RankingWeights:
    """Score contributions. E/F never admit; A required for non-zero rank when gated."""

    a: float = 1.0
    b: float = 1.0
    c: float = 1.0
    d: float = 1.5
    e: float = 0.5
    f: float = 0.5
    require_a: bool = True
    prefer_proxy_important: bool = True
    proxy_important_boost: float = 2.0

    def __post_init__(self) -> None:
        for name in ("a", "b", "c", "d", "e", "f", "proxy_important_boost"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")


def load_weights(path: str | Path | None = None) -> RankingWeights:
    cfg_path = Path(path) if path is not None else DEFAULT_WEIGHTS_PATH
    if not cfg_path.is_file():
        return RankingWeights()
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    known = {f.name for f in RankingWeights.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {k: v for k, v in raw.items() if k in known and not k.startswith("_")}
    return RankingWeights(**kwargs)


def weights_to_dict(weights: RankingWeights) -> dict:
    return asdict(weights)
