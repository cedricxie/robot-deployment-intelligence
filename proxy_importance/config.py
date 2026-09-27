"""Tunable thresholds for proxy review-priority (no magic numbers in rules)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "proxy_importance.json"
)


@dataclass(frozen=True)
class ProxyImportanceConfig:
    """Defaults match configs/proxy_importance.json.

    - rare_freq_quantile: B — type count ≤ empirical quantile of per-type counts
    - high_freq_min_count: C — type absolute count among reference fails
    - novelty_min_score: D — novelty_score ≥ this (unseen type → 1.0)
    - top_k: report Top-K proxy-important ids
    """

    rare_freq_quantile: float = 0.25
    high_freq_min_count: int = 3
    novelty_min_score: float = 1.0
    top_k: int = 10

    def __post_init__(self) -> None:
        if not 0.0 <= self.rare_freq_quantile <= 1.0:
            raise ValueError("rare_freq_quantile must be in [0, 1]")
        if self.high_freq_min_count < 1:
            raise ValueError("high_freq_min_count must be >= 1")
        if self.novelty_min_score < 0.0:
            raise ValueError("novelty_min_score must be >= 0")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")


def load_config(path: str | Path | None = None) -> ProxyImportanceConfig:
    """Load config from JSON; missing file → defaults."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        return ProxyImportanceConfig()
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    known = {f.name for f in ProxyImportanceConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {k: v for k, v in raw.items() if k in known}
    return ProxyImportanceConfig(**kwargs)


def config_to_dict(cfg: ProxyImportanceConfig) -> dict:
    return asdict(cfg)
