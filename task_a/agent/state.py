"""Shared graph state for Task A."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    persona: str
    product: str

    fingerprint: dict[str, Any]
    similar_users: list[dict[str, Any]]
    similar_users_summary: str
    cohort_pct_5star: float

    nigerian_context: dict[str, Any]
    nigerian_context_applied: bool
    language_region: str

    draft_review: str
    draft_rating: float
    draft_confidence: float

    review: str
    rating: float
    confidence: float
    quality_issues: list[str]
