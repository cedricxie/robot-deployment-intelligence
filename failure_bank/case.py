"""FailureCase — one stored failure for the bank / clustering / ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FailureCase:
    """Persisted failure record (JSONL-friendly)."""

    case_id: str
    episode_id: str
    failure_type: str | None = None
    task: str | None = None
    instruction: str | None = None
    path_text: str = ""
    diagnosis: str | None = None
    correction: str | None = None
    rule_hits: tuple[str, ...] = ()
    frequency_bucket: str | None = None
    novelty_score: float = 0.0
    type_count: int = 0
    bonus_e: float = 0.0
    bonus_f: float = 0.0
    proxy_important: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rule_hits"] = list(self.rule_hits)
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FailureCase:
        hits = raw.get("rule_hits") or ()
        return cls(
            case_id=str(raw["case_id"]),
            episode_id=str(raw["episode_id"]),
            failure_type=raw.get("failure_type"),
            task=raw.get("task"),
            instruction=raw.get("instruction"),
            path_text=str(raw.get("path_text") or ""),
            diagnosis=raw.get("diagnosis"),
            correction=raw.get("correction"),
            rule_hits=tuple(hits),
            frequency_bucket=raw.get("frequency_bucket"),
            novelty_score=float(raw.get("novelty_score") or 0.0),
            type_count=int(raw.get("type_count") or 0),
            bonus_e=float(raw.get("bonus_e") or 0.0),
            bonus_f=float(raw.get("bonus_f") or 0.0),
            proxy_important=bool(raw.get("proxy_important") or False),
            metadata=dict(raw.get("metadata") or {}),
        )
