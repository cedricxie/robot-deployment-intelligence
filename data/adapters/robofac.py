"""RoboFAC → Episode adapter. RoboFAC field names stay inside this module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from data.schemas.episode import Episode, GroundTruth

# RoboFAC QA section keys (adapter-private)
_FAIL_DETECT = "Failure detection"
_FAIL_ID = "Failure identification"
_FAIL_EXPLAIN = "Failure explanation"
_FAIL_LOCATE = "Failure locating"
_TASK_ID = "Task identification"
_HIGH_CORR = "High-level correction"
_LOW_CORR = "Low-level correction"


def _assistant_text(annos: dict[str, Any], section: str) -> str | None:
    turns = annos.get(section) or []
    for turn in turns:
        if turn.get("from") == "assistant":
            value = turn.get("value")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _parse_success(annos: dict[str, Any]) -> bool | None:
    text = _assistant_text(annos, _FAIL_DETECT)
    if text is None:
        return None
    lower = text.lower()
    if lower.startswith("yes"):
        return True
    if lower.startswith("no"):
        return False
    return None


def _normalize_failure_type(text: str | None) -> str | None:
    if not text:
        return None
    return text.rstrip(".").strip() or None


class RoboFACAdapter:
    """Load RoboFAC-style JSON samples into unified Episode records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> Iterator[Episode]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "episodes" in payload:
            for item in payload["episodes"]:
                yield self.map_record(item["episode_id"], item)
            return
        if isinstance(payload, list):
            for item in payload:
                eid = item.get("episode_id") or Path(item.get("video", "unknown")).stem
                yield self.map_record(eid, item)
            return
        # RoboFAC annos_per_video: {episode_id: {video, task, annos}}
        for eid, record in payload.items():
            yield self.map_record(eid, record)

    def map_record(self, episode_id: str, record: dict[str, Any]) -> Episode:
        annos = record.get("annos") or {}
        video = record.get("video")
        instruction = _assistant_text(annos, _TASK_ID)
        diagnosis = _assistant_text(annos, _FAIL_EXPLAIN)
        correction = _assistant_text(annos, _HIGH_CORR) or _assistant_text(
            annos, _LOW_CORR
        )
        failure_type = _normalize_failure_type(_assistant_text(annos, _FAIL_ID))
        success = _parse_success(annos)

        metadata: dict[str, Any] = {
            "source": "robofac",
            "source_path": str(self.path),
        }
        locating = _assistant_text(annos, _FAIL_LOCATE)
        if locating:
            metadata["failure_subtask"] = locating

        return Episode(
            episode_id=str(episode_id),
            task=record.get("task"),
            instruction=instruction,
            video_paths=[video] if video else [],
            metadata=metadata,
            ground_truth=GroundTruth(
                success=success,
                failure_type=failure_type,
                diagnosis=diagnosis,
                correction=correction,
            ),
        )


def load_episodes(paths: Iterable[str | Path]) -> list[Episode]:
    out: list[Episode] = []
    for path in paths:
        out.extend(RoboFACAdapter(path).load())
    return out
