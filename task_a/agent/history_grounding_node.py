"""Task A: retrieve real exemplar reviews from the Jumia user-history corpus.

Runs between ``nigerian_context`` and ``review_generator``. Writes:
- ``state['exemplars']``           list of real reviews (style/voice grounding)
- ``state['exemplars_block']``     prompt-ready string for direct injection
- ``state['exemplar_rating_prior']`` numeric mean rating across exemplars
- ``state['exemplar_domain']``     the resolved domain heuristic for logging

The node degrades gracefully: missing history file or no matches -> empty
exemplars + neutral prior; the generator falls back to the legacy prompt.
"""

from __future__ import annotations

from typing import Any

from core.history_grounding import (
    exemplar_rating_prior,
    format_exemplars_for_prompt,
    get_exemplar_reviews,
)
from core.json_utils import warn
from task_a.agent.state import AgentState


# Heuristic mapping from product keywords to the Task B canonical domain
# (which is what the Jumia histories are labelled with). This lets the node
# bias exemplar retrieval toward the right product family even though the
# Task A API doesn't take a `domain` field.
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "skincare": ["serum", "cream", "lotion", "soap", "sunscreen", "moisturizer", "vitamin c"],
    "tech": ["phone", "iphone", "samsung", "tecno", "infinix", "redmi", "earbud", "laptop",
             "power bank", "smart watch"],
    "food": ["rice", "noodle", "oil", "snack", "drink", "tea", "coffee", "kunu", "zobo"],
    "general lifestyle": ["lunch bag", "kitchen", "pot", "blender", "fan", "iron"],
}


def _infer_domain(product: str) -> str | None:
    p = (product or "").lower()
    for domain, keys in _DOMAIN_KEYWORDS.items():
        if any(k in p for k in keys):
            return domain
    return None


def history_grounding_node(state: AgentState) -> AgentState:
    """Populate state with real exemplar reviews + numeric rating prior."""
    fingerprint = state.get("fingerprint", {}) or {}
    product = state.get("product", "") or ""
    domain = _infer_domain(product)
    # Eval / known-reviewer hook: an upstream caller may set state['reviewer_id'].
    reviewer_id = state.get("reviewer_id")  # type: ignore[assignment]

    try:
        exemplars = get_exemplar_reviews(
            fingerprint=fingerprint,
            domain=domain,
            product_hint=product,
            reviewer_id=reviewer_id,
            k=3,
        )
    except FileNotFoundError as exc:
        warn(f"history_grounding_node: histories unavailable ({exc}); skipping grounding")
        exemplars = []
    except Exception as exc:  # noqa: BLE001
        warn(f"history_grounding_node: grounding failed ({exc}); skipping")
        exemplars = []

    state["exemplars"] = exemplars
    state["exemplars_block"] = format_exemplars_for_prompt(exemplars)
    state["exemplar_rating_prior"] = exemplar_rating_prior(exemplars)
    state["exemplar_domain"] = domain
    return state
