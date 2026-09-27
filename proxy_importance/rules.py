"""Rule engine: proxy_important ⇔ A ∧ (B ∨ C ∨ D).

E/F ranking bonuses are stubs (out of scope for admission in PR-impl-3).
Scores attach to ProxyReviewPriorityResult — Episode stays lean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from data.schemas.episode import Episode, FrequencyBucket
from proxy_importance.config import ProxyImportanceConfig
from proxy_importance.failure_bank import FailureBank
from proxy_importance.frequency import FrequencyTable, build_frequency_table


@dataclass(frozen=True)
class ProxyReviewPriorityResult:
    """Side-channel score for one episode (do not bloat Episode)."""

    episode_id: str
    proxy_important: bool
    rule_hits: tuple[str, ...]
    frequency_bucket: FrequencyBucket | None
    novelty_score: float
    type_count: int
    type_rel_freq: float
    ranking_bonus: float = 0.0  # E/F stubs; not used for admission


def rule_a_is_failure(episode: Episode) -> bool:
    """A: episode is a fail (success=False)."""
    return episode.ground_truth.success is False


def rule_b_is_rare(failure_type: str | None, freq: FrequencyTable) -> bool:
    """B: rare failure_type vs reference corpus frequency."""
    return freq.is_rare(failure_type)


def rule_c_is_high_freq(failure_type: str | None, freq: FrequencyTable) -> bool:
    """C: high-frequency recurrence of same failure_type."""
    return freq.is_high_freq(failure_type)


def rule_d_is_novel(
    failure_type: str | None,
    bank: FailureBank,
    config: ProxyImportanceConfig,
) -> bool:
    """D: novelty vs Failure Bank."""
    return bank.is_novel(failure_type, min_score=config.novelty_min_score)


def ranking_bonus_e_actionable_phrases(_episode: Episode) -> float:
    """E stub — diagnosis actionable phrases; ranking only; not built yet."""
    return 0.0


def ranking_bonus_f_cascade_uncertainty(_episode: Episode) -> float:
    """F stub — cascade disagree/low-conf; ranking only; not built yet."""
    return 0.0


def score_episode(
    episode: Episode,
    freq: FrequencyTable,
    bank: FailureBank,
    config: ProxyImportanceConfig | None = None,
) -> ProxyReviewPriorityResult:
    """Compute A–D hits and proxy_important for one episode.

    B/C/D are only evaluated when A holds (successes are never review-priority).
    """
    cfg = config or ProxyImportanceConfig()
    ft = episode.ground_truth.failure_type
    hits: list[str] = []

    a = rule_a_is_failure(episode)
    novelty = bank.novelty_score(ft) if a else 0.0
    b = c = d = False

    if a:
        hits.append("A")
        b = rule_b_is_rare(ft, freq)
        if b:
            hits.append("B")
        c = rule_c_is_high_freq(ft, freq)
        if c:
            hits.append("C")
        d = rule_d_is_novel(ft, bank, cfg)
        if d:
            hits.append("D")

    important = a and (b or c or d)
    bonus = 0.0
    if a:
        bonus = ranking_bonus_e_actionable_phrases(episode) + ranking_bonus_f_cascade_uncertainty(
            episode
        )

    return ProxyReviewPriorityResult(
        episode_id=episode.episode_id,
        proxy_important=important,
        rule_hits=tuple(hits),
        frequency_bucket=freq.bucket(ft) if a else None,
        novelty_score=novelty,
        type_count=freq.count_of(ft) if a else 0,
        type_rel_freq=freq.rel_freq(ft) if a else 0.0,
        ranking_bonus=bonus,
    )


def score_episodes(
    episodes: Iterable[Episode],
    freq: FrequencyTable | None = None,
    bank: FailureBank | None = None,
    config: ProxyImportanceConfig | None = None,
    *,
    reference_episodes: Iterable[Episode] | None = None,
) -> list[ProxyReviewPriorityResult]:
    """Score many episodes. Frequency/bank default from ``reference_episodes`` or ``episodes``."""
    cfg = config or ProxyImportanceConfig()
    eps = list(episodes)
    ref = list(reference_episodes) if reference_episodes is not None else eps
    table = freq if freq is not None else build_frequency_table(ref, cfg)
    fb = bank if bank is not None else FailureBank.from_episodes(ref)
    return [score_episode(ep, table, fb, cfg) for ep in eps]


def apply_proxy_to_ground_truth(
    episode: Episode,
    result: ProxyReviewPriorityResult,
) -> Episode:
    """Optional: copy computed proxy fields onto Episode.ground_truth (new object)."""
    gt = episode.ground_truth.model_copy(
        update={
            "proxy_important": result.proxy_important,
            "proxy_rule_hits": list(result.rule_hits),
            "frequency_bucket": result.frequency_bucket,
            "novelty_score": result.novelty_score,
        }
    )
    return episode.model_copy(update={"ground_truth": gt})
