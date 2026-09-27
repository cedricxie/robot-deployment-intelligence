"""Ranker.score(case_or_cluster, weights) — honors A–D set + E/F bonuses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from failure_bank.case import FailureCase
from ranking.weights import RankingWeights, load_weights


@dataclass(frozen=True)
class RankedItem:
    case_id: str
    score: float
    proxy_important: bool
    rule_hits: tuple[str, ...]


class Ranker:
    """Review-priority ranking over FailureCase rows (not business importance)."""

    def __init__(self, weights: RankingWeights | None = None) -> None:
        self.weights = weights or load_weights()

    def score(self, case: FailureCase, weights: RankingWeights | None = None) -> float:
        """Weighted A–D hits + E/F bonuses. Non-A → 0 when ``require_a``."""
        w = weights or self.weights
        hits = set(case.rule_hits)
        if w.require_a and "A" not in hits:
            return 0.0
        s = 0.0
        if "A" in hits:
            s += w.a
        if "B" in hits:
            s += w.b
        if "C" in hits:
            s += w.c
        if "D" in hits:
            s += w.d
        # E/F ranking bonuses only (do not admit).
        s += w.e * float(case.bonus_e)
        s += w.f * float(case.bonus_f)
        if w.prefer_proxy_important and case.proxy_important:
            s += w.proxy_important_boost
        return s

    def score_cluster(
        self,
        cases: Sequence[FailureCase],
        weights: RankingWeights | None = None,
    ) -> float:
        """Cluster score = max member score (representative priority)."""
        if not cases:
            return 0.0
        return max(self.score(c, weights) for c in cases)

    def rank(
        self,
        cases: Iterable[FailureCase],
        weights: RankingWeights | None = None,
        *,
        proxy_important_only: bool = False,
    ) -> list[RankedItem]:
        rows = list(cases)
        if proxy_important_only:
            rows = [c for c in rows if c.proxy_important]
        scored = [
            RankedItem(
                case_id=c.case_id,
                score=self.score(c, weights),
                proxy_important=c.proxy_important,
                rule_hits=c.rule_hits,
            )
            for c in rows
        ]
        scored.sort(key=lambda r: (-r.score, r.case_id))
        return scored
