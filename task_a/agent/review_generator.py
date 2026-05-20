"""Review-generator node — drafts the persona-voiced product review."""

from __future__ import annotations

import json
import math
from typing import Any

from shared.utils import call_claude, get_anthropic_client, parse_json_block, warn
from task_a.agent.state import AgentState


_SYSTEM = (
    "You are simulating a Nigerian product review. Match the persona's tone, "
    "rating tendencies, and cultural register. Return JSON only with keys: "
    "review (string), rating (float 1-5), rationale (short)."
)


def _ng_lists(ctx: dict[str, Any]) -> str:
    return (
        f"Pidgin markers available: {', '.join(ctx.get('pidgin_markers', []))}\n"
        f"Cultural vocab available: {', '.join(ctx.get('cultural_vocab', []))}\n"
        f"Food brands: {', '.join(ctx.get('food_brands', []))}\n"
        f"Currency: {ctx.get('currency_symbol', '₦')}"
    )


def build_review_prompt(state: AgentState) -> tuple[str, str]:
    """Compose (system_prompt, user_prompt) for the review-drafting LLM call."""
    persona = state.get("persona", "")
    product = state.get("product", "")
    fingerprint = state.get("fingerprint", {})
    ctx = state.get("nigerian_context", {}) or {}
    summary = state.get("similar_users_summary", "")
    apply_ctx = state.get("nigerian_context_applied", False)

    ctx_instruction = (
        "Weave in 1-2 Pidgin markers and/or one ₦ price reference naturally — "
        "don't sprinkle for the sake of it.\n"
        if apply_ctx else
        "Keep the language neutral; do not inject Pidgin or Nigerian-specific markers.\n"
    )

    user_prompt = (
        f"Persona description:\n{persona}\n\n"
        f"Product being reviewed:\n{product}\n\n"
        f"Persona fingerprint (JSON):\n{json.dumps(fingerprint, ensure_ascii=False)}\n\n"
        f"Nigerian context palette:\n{_ng_lists(ctx)}\n\n"
        f"Behavioral grounding from similar real users:\n{summary}\n\n"
        f"{ctx_instruction}\n"
        "Write the review in the persona's voice. Be concrete about the product. "
        "Match rating to the persona's typical rating behaviour.\n\n"
        "Return JSON only with EXACTLY these keys: review (string, 60-400 chars), "
        "rating (float in [1.0, 5.0]), rationale (short string)."
    )
    return _SYSTEM, user_prompt


def _compute_confidence(
    review: str,
    rating: float,
    cohort_pct_5star: float,
    ctx: dict[str, Any],
    ctx_applied: bool,
) -> float:
    base = 0.85
    if rating >= 4.75 and cohort_pct_5star > 0.5:
        base = min(base, 0.70)
    if len(review) < 60:
        base -= 0.15
    if ctx_applied:
        haystack = review.lower()
        palette = [m.lower() for m in ctx.get("pidgin_markers", []) + ctx.get("cultural_vocab", [])]
        if palette and not any(m in haystack for m in palette):
            base -= 0.05
    return max(0.0, min(0.99, base))


def review_generator_node(state: AgentState) -> AgentState:
    """Produce `draft_review`, `draft_rating`, and `draft_confidence`."""
    system, user = build_review_prompt(state)
    client = get_anthropic_client()
    raw = call_claude(client, system=system, user=user, max_tokens=600, temperature=0.8)

    try:
        parsed = parse_json_block(raw)
    except ValueError as exc:
        warn(f"review_generator: JSON parse failed ({exc}); raw response={raw[:200]!r}")
        parsed = {}

    review = str(parsed.get("review", "") or "").strip()
    try:
        rating = float(parsed.get("rating", 3.0) or 3.0)
    except (TypeError, ValueError):
        rating = 3.0
    if math.isnan(rating):
        rating = 3.0

    confidence = _compute_confidence(
        review=review,
        rating=rating,
        cohort_pct_5star=float(state.get("cohort_pct_5star", 0.0) or 0.0),
        ctx=state.get("nigerian_context", {}) or {},
        ctx_applied=bool(state.get("nigerian_context_applied", False)),
    )

    state["draft_review"] = review
    state["draft_rating"] = rating
    state["draft_confidence"] = confidence
    return state
