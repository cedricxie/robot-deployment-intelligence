"""Eval harness: Random → Frequency → Baseline → JEV ladder columns.

Emits detection P/R/F1, failure-type agreement, proxy-important retention,
review reduction, and cost proxy (calls / tokens / stub units). Uses
``proxy_importance`` for proxy labels.

Hidden isolation: this module scores whatever Episode list the caller passes.
Improvement-loop helpers must not load hidden_eval — callers that tune should
use ``proxy_importance.access`` / ``data.split_access`` role checks. The harness
itself never opens split manifests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from data.schemas.episode import Episode
from decision.base import DecisionConfig, DecisionResult, FastDecisionEngine, load_decision_config
from decision.baseline import BaselineFastDecisionEngine
from decision.frequency import FrequencyOnlyDecisionEngine
from decision.jev import CostSnapshot, JevDecisionEngine, build_jev_engine
from decision.random import RandomDecisionEngine
from features.extract import FeatureExtractor
from proxy_importance.config import ProxyImportanceConfig, load_config as load_proxy_config
from proxy_importance.failure_bank import FailureBank
from proxy_importance.frequency import build_frequency_table
from proxy_importance.rules import ProxyReviewPriorityResult, score_episodes


@dataclass(frozen=True)
class Prf1:
    precision: float
    recall: float
    f1: float
    support: int  # GT-positive count


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def detection_prf1(
    episodes: Sequence[Episode],
    results: Sequence[DecisionResult],
) -> Prf1:
    """Failure detection: positive class = fail (success=False)."""
    by_id = {r.episode_id: r for r in results}
    tp = fp = fn = 0
    support = 0
    for ep in episodes:
        gt = ep.ground_truth.success
        if gt is None:
            continue
        pred = by_id.get(ep.episode_id)
        if pred is None:
            continue
        gt_fail = gt is False
        pred_fail = pred.predicted_success is False
        if gt_fail:
            support += 1
        if pred_fail and gt_fail:
            tp += 1
        elif pred_fail and not gt_fail:
            fp += 1
        elif (not pred_fail) and gt_fail:
            fn += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return Prf1(precision=precision, recall=recall, f1=f1, support=support)


def failure_type_agreement(
    episodes: Sequence[Episode],
    results: Sequence[DecisionResult],
) -> float:
    """Accuracy where both GT and prediction have a non-null failure_type."""
    by_id = {r.episode_id: r for r in results}
    correct = total = 0
    for ep in episodes:
        gt_t = ep.ground_truth.failure_type
        pred = by_id.get(ep.episode_id)
        if gt_t is None or pred is None or pred.failure_type is None:
            continue
        total += 1
        if gt_t == pred.failure_type:
            correct += 1
    return _safe_div(correct, total)


def select_review_set(
    results: Sequence[DecisionResult],
    top_k: int,
) -> set[str]:
    """Top-K by failure_probability (ties broken by episode_id)."""
    ranked = sorted(
        results,
        key=lambda r: (-r.failure_probability, r.episode_id),
    )
    return {r.episode_id for r in ranked[:top_k]}


def proxy_retention_and_review_reduction(
    proxy_results: Sequence[ProxyReviewPriorityResult],
    review_ids: set[str],
    n_episodes: int,
) -> tuple[float, float, int]:
    """Return (proxy_retention, review_reduction, n_proxy_important)."""
    important_ids = {p.episode_id for p in proxy_results if p.proxy_important}
    n_imp = len(important_ids)
    retained = len(important_ids & review_ids)
    retention = _safe_div(retained, n_imp) if n_imp else 1.0  # vacuous → 1.0
    reduction = 1.0 - _safe_div(len(review_ids), n_episodes) if n_episodes else 0.0
    return retention, reduction, n_imp


def engine_cost(engine: FastDecisionEngine) -> CostSnapshot:
    """Read cost proxy from engines that track it; else zeros (local heuristics)."""
    snap = getattr(engine, "cost_snapshot", None)
    if callable(snap):
        return snap()
    return CostSnapshot()


@dataclass
class LadderColumn:
    backend: str
    detection_precision: float
    detection_recall: float
    detection_f1: float
    failure_type_agreement: float
    proxy_retention: float
    review_reduction: float
    n_proxy_important: int
    review_top_k: int
    n_episodes: int
    cost_calls: int = 0
    cost_tokens: int = 0
    cost_stub_units: float = 0.0


@dataclass
class HarnessReport:
    columns: list[LadderColumn] = field(default_factory=list)

    def column(self, backend: str) -> LadderColumn | None:
        for c in self.columns:
            if c.backend == backend:
                return c
        return None

    def as_rows(self) -> list[dict]:
        return [
            {
                "backend": c.backend,
                "detection_precision": c.detection_precision,
                "detection_recall": c.detection_recall,
                "detection_f1": c.detection_f1,
                "failure_type_agreement": c.failure_type_agreement,
                "proxy_retention": c.proxy_retention,
                "review_reduction": c.review_reduction,
                "n_proxy_important": c.n_proxy_important,
                "review_top_k": c.review_top_k,
                "n_episodes": c.n_episodes,
                "cost_calls": c.cost_calls,
                "cost_tokens": c.cost_tokens,
                "cost_stub_units": c.cost_stub_units,
            }
            for c in self.columns
        ]


def _score_column(
    backend: str,
    episodes: Sequence[Episode],
    results: Sequence[DecisionResult],
    proxy_results: Sequence[ProxyReviewPriorityResult],
    top_k: int,
    cost: CostSnapshot,
) -> LadderColumn:
    prf = detection_prf1(episodes, results)
    agree = failure_type_agreement(episodes, results)
    review_ids = select_review_set(results, top_k)
    retention, reduction, n_imp = proxy_retention_and_review_reduction(
        proxy_results, review_ids, len(episodes)
    )
    return LadderColumn(
        backend=backend,
        detection_precision=prf.precision,
        detection_recall=prf.recall,
        detection_f1=prf.f1,
        failure_type_agreement=agree,
        proxy_retention=retention,
        review_reduction=reduction,
        n_proxy_important=n_imp,
        review_top_k=top_k,
        n_episodes=len(episodes),
        cost_calls=cost.calls,
        cost_tokens=cost.tokens,
        cost_stub_units=cost.stub_units,
    )


def run_ladder(
    episodes: Sequence[Episode],
    engines: Sequence[FastDecisionEngine],
    *,
    reference_episodes: Sequence[Episode] | None = None,
    decision_config: DecisionConfig | None = None,
    proxy_config: ProxyImportanceConfig | None = None,
    top_k: int | None = None,
) -> HarnessReport:
    """Run each engine on ``episodes``; emit side-by-side ladder columns.

    ``reference_episodes`` feeds Frequency/Baseline/JEV fit + proxy frequency/bank.
    Defaults to ``episodes`` when omitted (fine for fixture smoke).
    """
    dcfg = decision_config or load_decision_config()
    pcfg = proxy_config or load_proxy_config()
    k = top_k if top_k is not None else dcfg.review_top_k
    ref = list(reference_episodes) if reference_episodes is not None else list(episodes)
    eps = list(episodes)

    freq_table = build_frequency_table(ref, pcfg)
    bank = FailureBank.from_episodes(ref)
    proxy_results = score_episodes(eps, freq=freq_table, bank=bank, config=pcfg)

    extractor = FeatureExtractor()
    features = extractor.extract_many(eps)

    report = HarnessReport()
    for engine in engines:
        # Fit engines that support it (Frequency / Baseline / JEV) on reference only.
        fit = getattr(engine, "fit", None)
        if callable(fit):
            fit(ref)
        reset = getattr(engine, "reset_cost", None)
        if callable(reset):
            reset()
        results = engine.evaluate_many(features, dcfg)
        report.columns.append(
            _score_column(engine.name, eps, results, proxy_results, k, engine_cost(engine))
        )
    return report


class EvaluationHarness:
    """Thin OO wrapper around ``run_ladder`` with default ladder engines."""

    def __init__(
        self,
        decision_config: DecisionConfig | None = None,
        proxy_config: ProxyImportanceConfig | None = None,
        *,
        include_jev: bool | None = None,
    ) -> None:
        self.decision_config = decision_config or load_decision_config()
        self.proxy_config = proxy_config or load_proxy_config()
        # None → respect JEV_MODE; True force stub; False skip.
        self.include_jev = include_jev

    def default_engines(
        self,
        reference: Iterable[Episode] | None = None,
    ) -> list[FastDecisionEngine]:
        ref = list(reference) if reference is not None else []
        types: list[str] = []
        for ep in ref:
            ft = ep.ground_truth.failure_type
            if ft and ft not in types:
                types.append(ft)
        rnd = RandomDecisionEngine(
            self.decision_config, failure_types=types or None
        )
        freq = FrequencyOnlyDecisionEngine(self.decision_config)
        base = BaselineFastDecisionEngine(self.decision_config)
        engines: list[FastDecisionEngine] = [rnd, freq, base]
        if self.include_jev is False:
            return engines
        if self.include_jev is True:
            engines.append(JevDecisionEngine(self.decision_config, mode="stub"))
            return engines
        jev = build_jev_engine(self.decision_config)
        if jev is not None:
            engines.append(jev)
        return engines

    def evaluate(
        self,
        episodes: Sequence[Episode],
        *,
        reference_episodes: Sequence[Episode] | None = None,
        engines: Sequence[FastDecisionEngine] | None = None,
        top_k: int | None = None,
    ) -> HarnessReport:
        ref = reference_episodes if reference_episodes is not None else episodes
        engs = list(engines) if engines is not None else self.default_engines(ref)
        return run_ladder(
            episodes,
            engs,
            reference_episodes=ref,
            decision_config=self.decision_config,
            proxy_config=self.proxy_config,
            top_k=top_k,
        )
