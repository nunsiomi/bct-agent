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

    # Phase 5 grounding state
    exemplars: list[dict[str, Any]]
    exemplars_block: str
    exemplar_rating_prior: float | None
    exemplar_domain: str | None
    reviewer_id: str | None  # eval-mode override; unset in production

    draft_review: str
    draft_rating: float
    draft_confidence: float

    review: str
    rating: float
    confidence: float
    quality_issues: list[str]
