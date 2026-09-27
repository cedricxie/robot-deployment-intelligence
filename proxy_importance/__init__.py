"""Proxy review-priority rules A–D (plan §5.3.1 / PR-impl-3).

proxy_important ⇔ A ∧ (B ∨ C ∨ D). E/F are ranking bonuses only — stubs here.
"""

from proxy_importance.config import ProxyImportanceConfig, load_config
from proxy_importance.failure_bank import FailureBank
from proxy_importance.frequency import FrequencyTable, build_frequency_table
from proxy_importance.rules import (
    ProxyReviewPriorityResult,
    apply_proxy_to_ground_truth,
    score_episode,
    score_episodes,
)

__all__ = [
    "ProxyImportanceConfig",
    "load_config",
    "ProxyReviewPriorityResult",
    "score_episode",
    "score_episodes",
    "apply_proxy_to_ground_truth",
    "FrequencyTable",
    "build_frequency_table",
    "FailureBank",
]
