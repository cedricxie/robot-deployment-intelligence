"""Cascade runner: strategy A (all-deep) vs B (cheap→fast→deep)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from data.schemas.episode import Episode
from decision.base import DecisionConfig, DecisionResult, FastDecisionEngine
from decision.baseline import BaselineFastDecisionEngine
from decision.jev import CostSnapshot, JevDecisionEngine, build_jev_engine
from evaluation.harness import detection_prf1, failure_type_agreement
from features.extract import EpisodeFeatures, FeatureExtractor
from proxy_importance.rules import ProxyReviewPriorityResult
from reasoning.cheap import CheapPathEngine
from reasoning.deep import DeepReasoner, StructuredDiagnosis
from reasoning.routing import (
    RoutingPolicy,
    cheap_can_stop,
    load_routing_policy,
    select_deep_ids,
)

Strategy = Literal["all_deep", "cascade"]
FastBackend = Literal["baseline", "jev"]


@dataclass
class CascadeCost:
    cheap_calls: int = 0
    fast_calls: int = 0
    deep_calls: int = 0
    cheap_units: float = 0.0
    fast_units: float = 0.0
    deep_units: float = 0.0
    fast_tokens: int = 0
    deep_tokens: int = 0

    @property
    def total_units(self) -> float:
        return self.cheap_units + self.fast_units + self.deep_units


@dataclass
class StrategyResult:
    strategy: Strategy
    fast_backend: str
    results: list[DecisionResult]
    diagnoses: dict[str, StructuredDiagnosis] = field(default_factory=dict)
    deep_ids: set[str] = field(default_factory=set)
    stopped_at_cheap: set[str] = field(default_factory=set)
    stopped_at_fast: set[str] = field(default_factory=set)
    cost: CascadeCost = field(default_factory=CascadeCost)
    detection_f1: float = 0.0
    type_agreement: float = 0.0
    proxy_important_recall: float = 0.0
    n_proxy_important: int = 0
    n_episodes: int = 0


def _proxy_important_detection_recall(
    episodes: Sequence[Episode],
    results: Sequence[DecisionResult],
    proxy_results: Sequence[ProxyReviewPriorityResult],
) -> tuple[float, int]:
    """Recall of fail-detection on the proxy-important subset."""
    important = {p.episode_id for p in proxy_results if p.proxy_important}
    by_id = {r.episode_id: r for r in results}
    tp = fn = 0
    for ep in episodes:
        if ep.episode_id not in important:
            continue
        gt = ep.ground_truth.success
        if gt is not False:
            continue
        pred = by_id.get(ep.episode_id)
        if pred is None or pred.predicted_success is not False:
            fn += 1
        else:
            tp += 1
    n = tp + fn
    return (tp / n if n else 1.0), n


def _fast_cost(
    engine: FastDecisionEngine,
    policy: RoutingPolicy,
    fast_backend: str,
    n_calls_override: int | None = None,
) -> CostSnapshot:
    snap_fn = getattr(engine, "cost_snapshot", None)
    base = snap_fn() if callable(snap_fn) else CostSnapshot()
    calls = n_calls_override if n_calls_override is not None else base.calls
    if fast_backend == "baseline":
        units = policy.cost_units_fast_baseline * calls
        return CostSnapshot(calls=calls, tokens=base.tokens, stub_units=units)
    if base.stub_units > 0:
        return CostSnapshot(calls=calls, tokens=base.tokens, stub_units=base.stub_units)
    units = policy.cost_units_fast_jev * calls
    return CostSnapshot(calls=calls, tokens=base.tokens, stub_units=units)


class CascadeRunner:
    """Compare all-deep (A) vs cascade (B) on one fast-ladder backend."""

    def __init__(
        self,
        *,
        policy: RoutingPolicy | None = None,
        decision_config: DecisionConfig | None = None,
        deep: DeepReasoner | None = None,
        cheap: CheapPathEngine | None = None,
    ) -> None:
        self.policy = policy or load_routing_policy()
        self.decision_config = decision_config or DecisionConfig()
        self.deep = deep or DeepReasoner(
            decision_config=self.decision_config,
            cost_units_per_call=self.policy.cost_units_deep,
        )
        self.cheap = cheap or CheapPathEngine(
            self.decision_config,
            cost_units_per_call=self.policy.cost_units_cheap,
        )

    def _make_fast(self, backend: FastBackend) -> FastDecisionEngine:
        if backend == "baseline":
            return BaselineFastDecisionEngine(self.decision_config)
        eng = build_jev_engine(self.decision_config)
        if eng is None:
            return JevDecisionEngine(self.decision_config, mode="stub")
        return eng

    def run_strategy(
        self,
        strategy: Strategy,
        episodes: Sequence[Episode],
        features: Sequence[EpisodeFeatures],
        *,
        reference: Sequence[Episode],
        proxy_results: Sequence[ProxyReviewPriorityResult],
        fast_backend: FastBackend = "baseline",
    ) -> StrategyResult:
        policy = self.policy
        feats_by_id = {f.episode_id: f for f in features}
        proxy_by_id = {p.episode_id: p for p in proxy_results}
        fast = self._make_fast(fast_backend)
        fit = getattr(fast, "fit", None)
        if callable(fit):
            fit(reference)
        reset = getattr(fast, "reset_cost", None)
        if callable(reset):
            reset()
        self.cheap.reset_cost()
        self.deep.reset_cost()

        final: dict[str, DecisionResult] = {}
        diagnoses: dict[str, StructuredDiagnosis] = {}
        deep_ids: set[str] = set()
        stopped_cheap: set[str] = set()
        stopped_fast: set[str] = set()

        if strategy == "all_deep":
            # A: run fast on all (cost column) + deep on all.
            all_feats = list(features)
            fast_results = {
                r.episode_id: r
                for r in fast.evaluate_many(all_feats, self.decision_config)
            }
            fast_cost = _fast_cost(
                fast, policy, fast_backend, n_calls_override=len(all_feats)
            )
            deep_ids = {ep.episode_id for ep in episodes}
            for ep in episodes:
                feat = feats_by_id[ep.episode_id]
                diag = self.deep.reason(feat, prior=fast_results[ep.episode_id])
                diagnoses[ep.episode_id] = diag
                final[ep.episode_id] = self.deep.to_decision(diag, backend=fast_backend)
            cheap_calls = 0
            cheap_units = 0.0
        else:
            # B: cheap → fast (non-stopped) → budgeted deep
            cheap_map: dict[str, DecisionResult] = {}
            for feat in features:
                cheap_map[feat.episode_id] = self.cheap.evaluate(
                    feat, self.decision_config
                )
            cheap_snap = self.cheap.cost_snapshot()
            cheap_calls = cheap_snap.calls
            cheap_units = cheap_snap.stub_units

            need_fast: list[EpisodeFeatures] = []
            for ep in episodes:
                eid = ep.episode_id
                cr = cheap_map[eid]
                if cheap_can_stop(cr, policy):
                    final[eid] = cr
                    stopped_cheap.add(eid)
                else:
                    need_fast.append(feats_by_id[eid])

            fast_results: dict[str, DecisionResult] = {}
            if need_fast:
                for r in fast.evaluate_many(need_fast, self.decision_config):
                    fast_results[r.episode_id] = r
            fast_cost = _fast_cost(
                fast, policy, fast_backend, n_calls_override=len(need_fast)
            )

            routing_view = list(fast_results.values())
            deep_ids = select_deep_ids(
                routing_view, proxy_by_id, policy, n_episodes=len(episodes)
            )
            for feat in need_fast:
                eid = feat.episode_id
                fr = fast_results[eid]
                if eid in deep_ids:
                    diag = self.deep.reason(feat, prior=fr)
                    diagnoses[eid] = diag
                    final[eid] = self.deep.to_decision(diag, backend=fast_backend)
                else:
                    final[eid] = fr
                    stopped_fast.add(eid)

        deep_snap = self.deep.cost_snapshot()
        ordered = [final[ep.episode_id] for ep in episodes]
        prf = detection_prf1(episodes, ordered)
        agree = failure_type_agreement(episodes, ordered)
        proxy_recall, n_proxy = _proxy_important_detection_recall(
            episodes, ordered, proxy_results
        )
        cost = CascadeCost(
            cheap_calls=cheap_calls,
            fast_calls=fast_cost.calls,
            deep_calls=deep_snap.calls,
            cheap_units=cheap_units,
            fast_units=fast_cost.stub_units,
            deep_units=deep_snap.stub_units,
            fast_tokens=fast_cost.tokens,
            deep_tokens=deep_snap.tokens,
        )
        return StrategyResult(
            strategy=strategy,
            fast_backend=fast_backend,
            results=ordered,
            diagnoses=diagnoses,
            deep_ids=deep_ids,
            stopped_at_cheap=stopped_cheap,
            stopped_at_fast=stopped_fast,
            cost=cost,
            detection_f1=prf.f1,
            type_agreement=agree,
            proxy_important_recall=proxy_recall,
            n_proxy_important=n_proxy,
            n_episodes=len(episodes),
        )

    def compare(
        self,
        episodes: Sequence[Episode],
        *,
        reference: Sequence[Episode],
        proxy_results: Sequence[ProxyReviewPriorityResult],
        fast_backends: Sequence[FastBackend] = ("baseline", "jev"),
    ) -> list[tuple[StrategyResult, StrategyResult]]:
        feats = FeatureExtractor().extract_many(episodes)
        pairs: list[tuple[StrategyResult, StrategyResult]] = []
        for backend in fast_backends:
            a = self.run_strategy(
                "all_deep",
                episodes,
                feats,
                reference=reference,
                proxy_results=proxy_results,
                fast_backend=backend,
            )
            b = self.run_strategy(
                "cascade",
                episodes,
                feats,
                reference=reference,
                proxy_results=proxy_results,
                fast_backend=backend,
            )
            pairs.append((a, b))
        return pairs
