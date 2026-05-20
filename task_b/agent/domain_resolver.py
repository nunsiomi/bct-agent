"""Domain-resolver node — normalises a raw domain/niche pair."""

from __future__ import annotations

from task_b.agent.state import AgentState


_DOMAIN_ALIASES = {
    "films": "movies",
    "film": "movies",
    "movie": "movies",
    "cinema": "movies",
    "eat": "food",
    "restaurants": "food",
    "restaurant": "food",
    "read": "books",
    "book": "books",
    "songs": "music",
    "drinks": "drink",
    "beauty": "skincare",
    "wellness": "fitness",
    "gym": "fitness",
    "gadgets": "tech",
    "gadget": "tech",
    "travelling": "travel",
    "travelling": "travel",
    "hotels": "hotel",
    "sports": "sport",
    "clothes": "fashion",
    "clothing": "fashion",
}


def resolve_domain(domain: str | None, niche: str | None) -> tuple[str, str | None]:
    """Return (canonical_domain, canonical_niche)."""
    d = (domain or "").strip().lower()
    d = _DOMAIN_ALIASES.get(d, d)

    n: str | None
    if niche is None:
        n = None
    else:
        nn = str(niche).strip().lower()
        n = nn if nn else None

    return d, n


def domain_resolver_node(state: AgentState) -> AgentState:
    """Write `resolved_domain` and `resolved_niche`."""
    d, n = resolve_domain(state.get("domain"), state.get("niche"))
    state["resolved_domain"] = d
    state["resolved_niche"] = n
    return state
