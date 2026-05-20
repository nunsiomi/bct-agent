"""Shared graph state for Task B."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    persona: str
    domain: str
    niche: str | None

    fingerprint: dict[str, Any]
    similar_users: list[dict[str, Any]]
    similar_users_summary: str
    cohort_pct_5star: float

    nigerian_context: dict[str, Any]
    nigerian_context_applied: bool
    language_region: str

    resolved_domain: str
    resolved_niche: str | None
    domain_valid: bool
    fallback_used: bool
    clarification_question: str | None

    candidates: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
