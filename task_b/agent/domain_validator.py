"""Domain-validator node — decides whether the resolved domain is in-scope."""

from __future__ import annotations

from task_b.agent.state import AgentState


KNOWN_DOMAINS: set[str] = {
    "movies",
    "food",
    "books",
    "music",
    "skincare",
    "hotel",
    "travel",
    "fitness",
    "tech",
    "fashion",
    "sport",
    "drink",
}


def is_supported_domain(domain: str) -> bool:
    """True if `domain` is one of the supported canonical domains."""
    return (domain or "").strip().lower() in KNOWN_DOMAINS


def domain_validator_node(state: AgentState) -> AgentState:
    """Write `domain_valid` and `fallback_used`; switch to general-lifestyle fallback if needed."""
    resolved = (state.get("resolved_domain") or "").strip().lower()
    if resolved in KNOWN_DOMAINS:
        state["domain_valid"] = True
        state["fallback_used"] = False
    else:
        state["domain_valid"] = False
        state["fallback_used"] = True
        state["resolved_domain"] = "general lifestyle"
    return state
