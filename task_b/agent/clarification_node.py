"""Clarification node — surfaces a graceful fallback message when the domain is unknown.

Does not pause the graph: control proceeds to retrieval with the
`general lifestyle` fallback domain set by the validator.
"""

from __future__ import annotations

from task_b.agent.state import AgentState


def craft_clarification_question(state: AgentState) -> str:
    """Generate a short, persona-aware clarification message."""
    raw_domain = (state.get("domain") or "").strip() or "your domain"
    return (
        f"We couldn't pin down '{raw_domain}' to a supported domain — "
        "defaulting to general lifestyle recommendations grounded in your persona."
    )


def clarification_node(state: AgentState) -> AgentState:
    """Write `clarification_question` to state."""
    state["clarification_question"] = craft_clarification_question(state)
    return state
