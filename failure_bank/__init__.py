"""Failure case store (JSONL) — M2 / PR-impl-7.

Novelty for proxy rule D still lives in ``proxy_importance.failure_bank``
(type-set). This package persists full cases and bridges via ``to_proxy_bank``.
"""

from failure_bank.case import FailureCase
from failure_bank.store import FailureBank, case_from_episode, type_counts

__all__ = [
    "FailureCase",
    "FailureBank",
    "case_from_episode",
    "type_counts",
]
