"""Nigerian-context node — layers cultural / linguistic / market context.

Mirrors `task_a/agent/nigerian_context.py`. Keep both in sync.
"""

from __future__ import annotations

from typing import Any

from task_b.agent.state import AgentState


_PIDGIN_MARKERS = [
    "abeg", "sha", "wahala", "no wahala", "e be like", "na", "wey",
    "small small", "chai", "oya", "abi", "biko",
]

_CULTURAL_VOCAB = [
    "jollof", "suya", "danfo", "agbada", "naija", "owambe", "ankara",
    "buka", "amala", "egusi", "puff-puff", "garri", "akara", "moi-moi",
    "fufu", "pepper soup", "ofada", "asaba", "yoruba", "igbo", "hausa",
]

_FOOD_BRANDS = [
    "Chicken Republic", "Mama Put", "Iya Basira", "Sweet Sensation",
    "The Place", "Tantalizers", "Mr Bigg's",
]


def derive_nigerian_context(fingerprint: dict[str, Any]) -> dict[str, Any]:
    """Build the Nigerian-context dict from a fingerprint."""
    tone = str(fingerprint.get("tone", "balanced") or "balanced").lower()
    markers = fingerprint.get("nigerian_markers") or []

    if tone == "pidgin":
        register = "pidgin"
    elif markers or tone == "casual":
        register = "nigerian_english"
    else:
        register = "neutral"

    apply_ctx = (tone in {"pidgin", "casual"}) or bool(markers) or True

    return {
        "language_register": register,
        "currency_symbol": "₦",
        "pidgin_markers": list(_PIDGIN_MARKERS),
        "cultural_vocab": list(_CULTURAL_VOCAB),
        "food_brands": list(_FOOD_BRANDS),
        "apply": apply_ctx,
    }


def nigerian_context_node(state: AgentState) -> AgentState:
    """Read `fingerprint`; write `nigerian_context` and `nigerian_context_applied`."""
    fingerprint = state.get("fingerprint", {}) or {}
    ctx = derive_nigerian_context(fingerprint)
    state["nigerian_context"] = ctx
    state["nigerian_context_applied"] = bool(ctx["apply"])
    return state
