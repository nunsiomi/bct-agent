"""Retrieval node — scores candidate items from the catalog against persona signals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared.utils import warn
from task_b.agent.state import AgentState


_CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "data_prep" / "artifacts" / "catalog.json"
)
_CATALOG: dict[str, list[dict[str, Any]]] | None = None


def _load_catalog() -> dict[str, list[dict[str, Any]]]:
    """Load and memoize the JSON catalog from data_prep/artifacts/."""
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG
    try:
        with _CATALOG_PATH.open("r", encoding="utf-8") as fh:
            _CATALOG = json.load(fh)
    except FileNotFoundError:
        warn(f"retrieval_node: catalog.json not found at {_CATALOG_PATH}; using empty catalog")
        _CATALOG = {}
    return _CATALOG


def load_corpus(domain: str) -> list[dict[str, Any]]:
    """Return the catalog slice for `domain`, falling back to `general lifestyle`."""
    catalog = _load_catalog()
    return catalog.get(domain) or catalog.get("general lifestyle") or []


def _overlap(a: list[str], b: list[str]) -> int:
    """Number of elements in `a` that appear as a substring of any element in `b` (case-insensitive)."""
    if not a or not b:
        return 0
    lower_b = [str(x).lower() for x in b]
    hits = 0
    for x in a:
        sx = str(x).lower()
        for y in lower_b:
            if sx == y or sx in y or y in sx:
                hits += 1
                break
    return hits


def _population_priors(state: AgentState) -> list[str]:
    """Flatten the top_categories of the matched cohort into a single list."""
    out: list[str] = []
    for row in state.get("similar_users", []) or []:
        cats = row.get("top_categories") or []
        if isinstance(cats, list):
            out.extend(str(c) for c in cats)
    return out


def _score(item: dict[str, Any], state: AgentState) -> float:
    fingerprint = state.get("fingerprint", {}) or {}
    affinity = [str(x).lower() for x in (fingerprint.get("category_affinity") or [])]
    niche = (state.get("resolved_niche") or "").strip().lower()
    ng_applied = bool(state.get("nigerian_context_applied", False))

    cats = [str(c).lower() for c in (item.get("categories") or [])]
    niches = [str(n).lower() for n in (item.get("niches") or [])]
    tags = [str(t).lower() for t in (item.get("tags") or [])]

    score = 0.0
    score += 2.0 * _overlap(affinity, cats)

    if niche:
        if any(niche in n or n in niche for n in niches):
            score += 3.0

    if ng_applied and "nigerian" in tags:
        score += 1.0

    if not affinity:
        priors = _population_priors(state)
        score += 0.5 * _overlap(priors, cats)

    return score


def retrieve_candidates(state: AgentState) -> list[dict[str, Any]]:
    """Score the corpus and return the top-10 candidates."""
    items = load_corpus(state.get("resolved_domain") or "general lifestyle")
    scored = []
    for item in items:
        s = _score(item, state)
        scored.append({**item, "raw_score": float(s)})
    scored.sort(key=lambda x: x["raw_score"], reverse=True)
    return scored[:10]


def retrieval_node(state: AgentState) -> AgentState:
    """Write `candidates` to state."""
    state["candidates"] = retrieve_candidates(state)
    return state
