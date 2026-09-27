"""Unified Episode schema v0 (plan §6.1)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0"

DecisionBackend = Literal["random", "frequency", "baseline", "jev"]


class FrequencyBucket(str, Enum):
    rare = "rare"
    mid = "mid"
    common = "common"


class GroundTruth(BaseModel):
    success: bool | None = None
    failure_type: str | None = None
    failure_timestamp: float | None = None
    diagnosis: str | None = None
    correction: str | None = None
    proxy_important: bool | None = None
    proxy_rule_hits: list[str] | None = None
    frequency_bucket: FrequencyBucket | None = None
    novelty_score: float | None = None


class ModelOutput(BaseModel):
    failure_probability: float | None = None
    failure_type: str | None = None
    confidence: float | None = None
    severity: float | None = None
    novelty: float | None = None
    needs_deep_review: bool | None = None
    diagnosis: str | None = None
    evidence: list[str] = Field(default_factory=list)
    recommended_actions: list[dict[str, Any]] = Field(default_factory=list)
    decision_backend: DecisionBackend | None = None


class Episode(BaseModel):
    """Unified episode record. Dataset-specific names stay in adapters."""

    schema_version: str = SCHEMA_VERSION
    episode_id: str
    task: str | None = None
    instruction: str | None = None
    video_paths: list[str] = Field(default_factory=list)
    frames: list[str] | None = None
    robot_state: dict[str, Any] | None = None
    actions: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    ground_truth: GroundTruth = Field(default_factory=GroundTruth)
    model_output: ModelOutput = Field(default_factory=ModelOutput)
