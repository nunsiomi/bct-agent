"""Revise node: re-prompt the LLM with the draft + critique feedback.

Runs at most ``state['max_revisions']`` times before the loop bails out.
Critically, the revision prompt instructs the LLM to KEEP the rating
calibrated against the empirical prior -- we only want the text to be
fixed, not the rating to drift.
"""

from __future__ import annotations

import json
import math

from core.json_utils import parse_json_block, warn
from core.llm import call_claude, get_anthropic_client
from task_a.agent.review_generator import build_review_prompt
from task_a.agent.state import AgentState


_REVISE_SYSTEM = (
    "You are revising a Nigerian product review. The previous draft was rejected "
    "for specific issues; address each one. Keep the rating roughly where it was. "
    "Return JSON only with keys: review (string, 60-400 chars), "
    "rating (float in [1.0, 5.0]), rationale (short string). JSON only."
)


def revise_review_node(state: AgentState) -> AgentState:
    """Re-prompt the LLM, then write the result back into the draft slots."""
    base_system, base_user = build_review_prompt(state)
    issues = state.get("critique_issues") or []
    draft_review = (state.get("draft_review") or "").strip()
    draft_rating = state.get("draft_rating")
    try:
        rating_anchor = float(draft_rating) if draft_rating is not None else 3.0
    except (TypeError, ValueError):
        rating_anchor = 3.0

    issues_block = "\n".join(f"  - {i}" for i in issues) or "  - (no specific issue listed)"

    revise_user = (
        f"{base_user}\n\n"
        f"--- PREVIOUS DRAFT (rejected) ---\n"
        f"review: {draft_review!r}\n"
        f"rating: {rating_anchor:.2f}\n\n"
        f"--- ISSUES TO FIX ---\n{issues_block}\n\n"
        "Rewrite the review addressing every issue above. Anchor the rating near "
        f"{rating_anchor:.1f}; only move it if a sentiment/text alignment issue forces it. "
        "Return JSON only with EXACTLY these keys: review (string, 60-400 chars), "
        "rating (float in [1.0, 5.0]), rationale (short string)."
    )

    client = get_anthropic_client()
    raw = call_claude(client, system=_REVISE_SYSTEM, user=revise_user, max_tokens=600, temperature=0.4)

    try:
        parsed = parse_json_block(raw)
    except ValueError as exc:
        warn(f"revise_review: JSON parse failed ({exc}); keeping prior draft")
        parsed = {}

    new_review = str(parsed.get("review", "") or "").strip()
    try:
        new_rating = float(parsed.get("rating", rating_anchor) or rating_anchor)
    except (TypeError, ValueError):
        new_rating = rating_anchor
    if math.isnan(new_rating):
        new_rating = rating_anchor

    # Only accept the revision if it actually produced meaningful text.
    if len(new_review) >= 30:
        state["draft_review"] = new_review
        # Blend the new rating with the exemplar prior, same as the generator.
        prior = state.get("exemplar_rating_prior")
        if prior is not None:
            delta = abs(new_rating - float(prior))
            alpha = 0.25 if delta < 0.5 else (0.40 if delta < 1.5 else 0.55)
            new_rating = (1 - alpha) * new_rating + alpha * float(prior)
        state["draft_rating"] = new_rating

    state["revision_count"] = int(state.get("revision_count", 0)) + 1
    return state
