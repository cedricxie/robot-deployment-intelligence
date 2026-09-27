"""Configurable cascade routing policy (JSON)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from decision.base import DecisionResult
from proxy_importance.rules import ProxyReviewPriorityResult

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "cascade_routing.json"
)


@dataclass(frozen=True)
class RoutingPolicy:
    """When to stop at cheap/fast vs escalate to deep.

    Candidates matching escalate flags are ranked; at most
    ``max_deep_fraction`` of the slice (or ``max_deep_calls``) receive deep.
    """

    cheap_stop_confidence: float = 0.8
    fast_stop_confidence: float = 0.8
    escalate_if_uncertain: bool = True
    escalate_if_needs_deep_review: bool = False
    escalate_if_proxy_important: bool = False
    # Treat prior-only / abstain as uncertain even if numeric conf looks high.
    uncertain_if_no_path_cue: bool = True
    max_deep_fraction: float = 0.20
    max_deep_calls: int | None = None
    prefer_proxy_important_in_budget: bool = True
    cost_units_cheap: float = 0.1
    cost_units_fast_baseline: float = 1.0
    cost_units_fast_jev: float = 10.0
    cost_units_deep: float = 50.0

    def __post_init__(self) -> None:
        if not 0.0 < self.max_deep_fraction <= 1.0:
            raise ValueError("max_deep_fraction must be in (0, 1]")
        if self.max_deep_calls is not None and self.max_deep_calls < 0:
            raise ValueError("max_deep_calls must be >= 0")


def load_routing_policy(path: str | Path | None = None) -> RoutingPolicy:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        return RoutingPolicy()
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    known = {f.name for f in RoutingPolicy.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs: dict[str, Any] = {
        k: v for k, v in raw.items() if k in known and not str(k).startswith("_")
    }
    return RoutingPolicy(**kwargs)


def policy_to_dict(policy: RoutingPolicy) -> dict:
    return asdict(policy)


def has_path_cue(result: DecisionResult) -> bool:
    ev = " ".join(result.evidence)
    return any(
        tag in ev
        for tag in (
            "success_path_cue",
            "fail_path_cue",
            "cheap_fail_path_cue",
            "cheap_success_path_cue",
            "cheap_mixed_path_cue",
        )
    )


def is_uncertain(result: DecisionResult, policy: RoutingPolicy) -> bool:
    if policy.uncertain_if_no_path_cue and not has_path_cue(result):
        return True
    return result.confidence < policy.fast_stop_confidence


def cheap_can_stop(result: DecisionResult, policy: RoutingPolicy) -> bool:
    return has_path_cue(result) and result.confidence >= policy.cheap_stop_confidence


def should_escalate_candidate(
    result: DecisionResult,
    *,
    proxy_important: bool,
    policy: RoutingPolicy,
) -> bool:
    """Whether this episode is a deep *candidate* (before budget trim)."""
    if policy.escalate_if_uncertain and is_uncertain(result, policy):
        return True
    if policy.escalate_if_needs_deep_review and result.needs_deep_review:
        return True
    if policy.escalate_if_proxy_important and proxy_important:
        return True
    return False


def uncertainty_score(result: DecisionResult) -> float:
    """Higher = more uncertain (for ranking who gets the deep budget)."""
    base = 1.0 - float(result.confidence)
    if not has_path_cue(result):
        return base + 0.5
    return base


def select_deep_ids(
    results: Sequence[DecisionResult],
    proxy_by_id: dict[str, ProxyReviewPriorityResult],
    policy: RoutingPolicy,
    n_episodes: int,
) -> set[str]:
    """Pick up to budget deep targets from escalate candidates."""
    candidates: list[DecisionResult] = []
    for r in results:
        proxy = proxy_by_id.get(r.episode_id)
        important = bool(proxy and proxy.proxy_important)
        if should_escalate_candidate(r, proxy_important=important, policy=policy):
            candidates.append(r)

    budget = int(n_episodes * policy.max_deep_fraction)
    if policy.max_deep_calls is not None:
        budget = (
            min(budget, policy.max_deep_calls)
            if budget > 0
            else policy.max_deep_calls
        )
    budget = max(0, budget)

    def sort_key(r: DecisionResult) -> tuple:
        important = bool(
            (p := proxy_by_id.get(r.episode_id)) and p.proxy_important
        )
        pref = 0 if (policy.prefer_proxy_important_in_budget and important) else 1
        return (pref, -uncertainty_score(r), r.episode_id)

    ranked = sorted(candidates, key=sort_key)
    return {r.episode_id for r in ranked[:budget]}
