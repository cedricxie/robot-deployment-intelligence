"""JSONL FailureBank: upsert, frequency / novelty queries.

Distinct from ``proxy_importance.failure_bank.FailureBank`` (type-set only for
rule D). This store persists cases; ``to_proxy_bank()`` bridges novelty API.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from data.schemas.episode import Episode
from failure_bank.case import FailureCase
from features.extract import extract_features
from proxy_importance.config import ProxyImportanceConfig
from proxy_importance.failure_bank import FailureBank as ProxyTypeBank
from proxy_importance.frequency import FrequencyTable, build_frequency_table
from proxy_importance.rules import score_episode
from ranking.bonuses import compute_bonus_e, compute_bonus_f

DEFAULT_ARTIFACT = (
    Path(__file__).resolve().parents[1] / "data" / "artifacts" / "failure_bank.jsonl"
)


@dataclass
class FailureBank:
    """Case store keyed by ``case_id`` (default = episode_id)."""

    path: Path | None = None
    _cases: dict[str, FailureCase] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self._cases)

    def upsert(self, case: FailureCase) -> None:
        self._cases[case.case_id] = case

    def get(self, case_id: str) -> FailureCase | None:
        return self._cases.get(case_id)

    def all_cases(self) -> list[FailureCase]:
        return list(self._cases.values())

    def known_types(self) -> set[str]:
        return {c.failure_type for c in self._cases.values() if c.failure_type}

    def type_frequency(self, failure_type: str | None) -> int:
        if failure_type is None:
            return 0
        return sum(1 for c in self._cases.values() if c.failure_type == failure_type)

    def novelty_score(self, failure_type: str | None) -> float:
        """1.0 if type missing/unknown; else 0.0 (same contract as proxy bank)."""
        if failure_type is None:
            return 1.0
        if failure_type not in self.known_types():
            return 1.0
        return 0.0

    def is_novel(self, failure_type: str | None, min_score: float = 1.0) -> bool:
        return self.novelty_score(failure_type) >= min_score

    def to_proxy_bank(self) -> ProxyTypeBank:
        return ProxyTypeBank(known_types=self.known_types())

    def save(self, path: Path | str | None = None) -> Path:
        out = Path(path) if path is not None else (self.path or DEFAULT_ARTIFACT)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for case in sorted(self._cases.values(), key=lambda c: c.case_id):
                fh.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
        self.path = out
        return out

    def load(self, path: Path | str | None = None) -> FailureBank:
        src = Path(path) if path is not None else (self.path or DEFAULT_ARTIFACT)
        self._cases.clear()
        if not src.is_file():
            self.path = src
            return self
        with src.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                self.upsert(FailureCase.from_dict(json.loads(line)))
        self.path = src
        return self

    @classmethod
    def from_jsonl(cls, path: Path | str) -> FailureBank:
        return cls(path=Path(path)).load()

    @classmethod
    def from_episodes(
        cls,
        episodes: Iterable[Episode],
        *,
        reference: Iterable[Episode] | None = None,
        config: ProxyImportanceConfig | None = None,
        path: Path | None = None,
        seed_only_fails: bool = True,
    ) -> FailureBank:
        """Build bank from fail episodes; attach proxy A–D + E/F bonuses."""
        cfg = config or ProxyImportanceConfig()
        eps = list(episodes)
        ref = list(reference) if reference is not None else eps
        freq = build_frequency_table(ref, cfg)
        proxy_bank = ProxyTypeBank.from_episodes(ref)
        bank = cls(path=path)
        for ep in eps:
            if seed_only_fails and ep.ground_truth.success is not False:
                continue
            bank.upsert(case_from_episode(ep, freq, proxy_bank, cfg))
        return bank


def case_from_episode(
    episode: Episode,
    freq: FrequencyTable,
    proxy_bank: ProxyTypeBank,
    config: ProxyImportanceConfig | None = None,
) -> FailureCase:
    """Convert one episode into a FailureCase with proxy signals."""
    cfg = config or ProxyImportanceConfig()
    feats = extract_features(episode)
    result = score_episode(episode, freq, proxy_bank, cfg)
    bucket = result.frequency_bucket.value if result.frequency_bucket else None
    return FailureCase(
        case_id=episode.episode_id,
        episode_id=episode.episode_id,
        failure_type=episode.ground_truth.failure_type,
        task=episode.task,
        instruction=episode.instruction,
        path_text=feats.path_text,
        diagnosis=episode.ground_truth.diagnosis,
        correction=episode.ground_truth.correction,
        rule_hits=result.rule_hits,
        frequency_bucket=bucket,
        novelty_score=result.novelty_score,
        type_count=result.type_count,
        bonus_e=compute_bonus_e(
            episode.ground_truth.diagnosis,
            episode.ground_truth.correction,
            episode.instruction,
        ),
        bonus_f=compute_bonus_f(episode),
        proxy_important=result.proxy_important,
        metadata={"source": (episode.metadata or {}).get("source")},
    )


def type_counts(bank: FailureBank) -> dict[str, int]:
    return dict(Counter(c.failure_type for c in bank.all_cases() if c.failure_type))
