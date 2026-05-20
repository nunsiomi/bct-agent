"""Quality-checker node — validates the draft review and promotes it to final fields."""

from __future__ import annotations

import math

from task_a.agent.state import AgentState


def score_review(state: AgentState) -> tuple[float, list[str]]:
    """Return (confidence, list_of_issues) after sanity checks."""
    issues: list[str] = []
    review = (state.get("draft_review") or "").strip()
    rating = state.get("draft_rating")
    confidence = float(state.get("draft_confidence", 0.0) or 0.0)

    if not review or len(review) < 20:
        issues.append("review_too_short")
        confidence = 0.0

    try:
        rating_f = float(rating)
        if math.isnan(rating_f):
            raise ValueError("nan")
        if not (1.0 <= rating_f <= 5.0):
            issues.append("rating_out_of_range")
    except (TypeError, ValueError):
        issues.append("rating_invalid")

    return confidence, issues


def quality_checker_node(state: AgentState) -> AgentState:
    """Sanitize draft fields and promote to final `review`, `rating`, `confidence`."""
    confidence, issues = score_review(state)

    review = (state.get("draft_review") or "").strip()
    if "review_too_short" in issues:
        review = "Unable to generate a confident review for this product."

    try:
        rating = float(state.get("draft_rating", 3.0) or 3.0)
        if math.isnan(rating):
            rating = 3.0
    except (TypeError, ValueError):
        rating = 3.0
    rating = round(max(1.0, min(5.0, rating)), 1)

    state["review"] = review
    state["rating"] = rating
    state["confidence"] = round(max(0.0, min(0.99, confidence)), 2)
    state["quality_issues"] = issues
    return state
