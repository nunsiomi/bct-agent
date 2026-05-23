"""Task A self-critique node.

Inspects the draft review against the persona / product / Nigerian-context
state and emits a structured critique. The critique is *heuristic-only*
(deterministic, zero LLM calls) so the loop is cheap and testable. It
catches the failure modes we see in practice on Llama-3.3 / Claude:

- review under 60 chars or over 400 chars
- review never mentions the actual product
- Nigerian context was supposedly applied but no Pidgin / cultural markers landed
- rating <-> sentiment mismatch (e.g. rating 5 with the word "terrible")
- rating wildly off the empirical exemplar prior (>1.5 stars)

When ``critique_passes`` is False, ``should_revise`` routes the graph back
through ``revise_review`` exactly once (capped by ``max_revisions``). When
it passes, the graph proceeds to ``quality_checker``.
"""

from __future__ import annotations

import math
import re
from typing import Any

from core.nigerian_context import _CULTURAL_VOCAB, _PIDGIN_MARKERS  # type: ignore
from task_a.agent.state import AgentState

_NEG_WORDS = {
    "terrible", "awful", "horrible", "useless", "trash", "garbage",
    "disappointing", "broken", "worst", "bad", "rubbish", "shit",
    "scam", "fake", "regret", "waste",
}
_POS_WORDS = {
    "love", "loved", "amazing", "excellent", "perfect", "great",
    "wonderful", "fantastic", "sweet", "lovely", "brilliant", "best",
    "smooth", "happy",
}

MAX_REVISIONS_DEFAULT = 1


def _mentions_product(review: str, product: str) -> bool:
    """True if the review names the product or any 4+ char token from it."""
    r = (review or "").lower()
    p = (product or "").lower()
    if not r or not p:
        return False
    if p in r:
        return True
    tokens = [t for t in re.split(r"[^a-z0-9]+", p) if len(t) >= 4]
    return any(t in r for t in tokens)


def _has_nigerian_marker(review: str) -> bool:
    """True if any Pidgin marker or cultural vocab item appears in the review."""
    haystack = " " + (review or "").lower() + " "
    for m in _PIDGIN_MARKERS + _CULTURAL_VOCAB:
        if " " in m:
            if m in haystack:
                return True
        else:
            if re.search(rf"\b{re.escape(m)}\b", haystack):
                return True
    return "₦" in (review or "")


def _sentiment_consistent(review: str, rating: float) -> tuple[bool, str | None]:
    """Heuristic sentiment <-> rating consistency check.

    Returns (ok, reason). Negative-leaning text with rating>=4 fails;
    positive-leaning text with rating<=2 fails. Mixed / neutral always passes.
    """
    words = set(re.findall(r"[a-z']+", (review or "").lower()))
    neg = bool(words & _NEG_WORDS)
    pos = bool(words & _POS_WORDS)
    if rating >= 4.0 and neg and not pos:
        return False, "rating>=4 but review reads negative"
    if rating <= 2.0 and pos and not neg:
        return False, "rating<=2 but review reads positive"
    return True, None


def critique_review(state: AgentState) -> tuple[bool, list[str], float]:
    """Return (passes, issues, score in [0,1])."""
    review = (state.get("draft_review") or "").strip()
    rating = state.get("draft_rating")
    try:
        rating_f = float(rating) if rating is not None else 3.0
    except (TypeError, ValueError):
        rating_f = 3.0
    if math.isnan(rating_f):
        rating_f = 3.0

    product = state.get("product", "") or ""
    ng_applied = bool(state.get("nigerian_context_applied", False))
    prior = state.get("exemplar_rating_prior")

    issues: list[str] = []

    # 1. Length
    n = len(review)
    if n < 60:
        issues.append(f"too short ({n} chars; want 60-400)")
    elif n > 400:
        issues.append(f"too long ({n} chars; want 60-400)")

    # 2. Mentions the product
    if not _mentions_product(review, product):
        issues.append("never mentions the product or any product keyword")

    # 3. Nigerian context applied but no markers landed
    if ng_applied and review and not _has_nigerian_marker(review):
        issues.append("ng_applied=True but no Pidgin marker / cultural item / ₦ found")

    # 4. Sentiment <-> rating
    sentiment_ok, sentiment_reason = _sentiment_consistent(review, rating_f)
    if not sentiment_ok and sentiment_reason:
        issues.append(sentiment_reason)

    # 5. Rating wildly off empirical prior
    if prior is not None and abs(rating_f - float(prior)) > 1.5:
        issues.append(
            f"rating {rating_f:.1f} deviates >1.5 from exemplar prior {float(prior):.1f}"
        )

    passes = not issues
    # Score is 1.0 when no issues; each issue knocks 0.2 off.
    score = max(0.0, 1.0 - 0.2 * len(issues))
    return passes, issues, score


def critique_node(state: AgentState) -> AgentState:
    passes, issues, score = critique_review(state)
    state["critique_passes"] = passes
    state["critique_issues"] = issues
    state["critique_score"] = score
    state.setdefault("revision_count", 0)
    state.setdefault("max_revisions", MAX_REVISIONS_DEFAULT)
    return state


def should_revise(state: AgentState) -> str:
    """Conditional edge: 'revise' if critique failed and budget remains, else 'finalize'."""
    if state.get("critique_passes"):
        return "finalize"
    if int(state.get("revision_count", 0)) >= int(state.get("max_revisions", MAX_REVISIONS_DEFAULT)):
        return "finalize"
    return "revise"
