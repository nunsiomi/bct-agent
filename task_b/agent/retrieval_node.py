"""Retrieval node -- hybrid TF-IDF + BM25 retrieval against the committed catalog.

Replaces the legacy substring-overlap scorer. The heavy lifting lives in
``core.retrieval.HybridRetriever``; this node is a thin adapter that builds a
query from persona signals and writes candidates into graph state.

Cold-start: when the fingerprint has no category affinity and the cohort has
no signal, falls back to popularity-based retrieval over the domain (the
top-rated items from the catalog).

Cross-domain: when the resolved domain is unknown / fallback, retrieves
across all domains so the persona's interests can pull matches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.json_utils import warn
from core.retrieval import get_retriever
from task_b.agent.state import AgentState


_CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "data_prep" / "artifacts" / "catalog.json"
)
_CATALOG: dict[str, list[dict[str, Any]]] | None = None


def _load_catalog() -> dict[str, list[dict[str, Any]]]:
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG
    try:
        _CATALOG = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        warn(f"retrieval_node: catalog.json not found at {_CATALOG_PATH}; using empty catalog")
        _CATALOG = {}
    return _CATALOG


def load_corpus(domain: str) -> list[dict[str, Any]]:
    """Return the catalog slice for `domain`, falling back to `general lifestyle`."""
    catalog = _load_catalog()
    return catalog.get(domain) or catalog.get("general lifestyle") or []


def _build_query(state: AgentState) -> str:
    """Compose a retrieval query from persona signals, niche, and population priors.

    Order matters for TF-IDF / BM25: niche first (highest specificity), then
    fingerprint affinity tags, then persona free-text, then cohort priors.
    """
    parts: list[str] = []

    niche = (state.get("resolved_niche") or "").strip()
    if niche:
        parts.append(niche)

    fingerprint = state.get("fingerprint", {}) or {}
    affinity = fingerprint.get("category_affinity") or []
    if affinity:
        parts.append(" ".join(str(x) for x in affinity))
    markers = fingerprint.get("nigerian_markers") or []
    if markers:
        parts.append(" ".join(str(x) for x in markers))

    persona = (state.get("persona") or "").strip()
    if persona:
        parts.append(persona)

    # Cohort priors (cold-start backstop).
    for row in state.get("similar_users", []) or []:
        cats = row.get("top_categories") or []
        if isinstance(cats, list):
            parts.extend(str(c) for c in cats[:3])

    return " ".join(parts).strip() or "popular nigerian product"


def _is_cold_start(state: AgentState) -> bool:
    fingerprint = state.get("fingerprint", {}) or {}
    affinity = fingerprint.get("category_affinity") or []
    cohort = state.get("similar_users", []) or []
    return not affinity and not cohort


def _popularity_fallback(domain: str, k: int = 10) -> list[dict[str, Any]]:
    """Top-k items by ``avg_rating * log(total_ratings)`` within a domain.

    Used when retrieval finds nothing OR the persona is fully cold-start.
    """
    import math
    items = load_corpus(domain)
    def pop(it: dict[str, Any]) -> float:
        attr = it.get("attributes") or {}
        return float(attr.get("avg_rating", 0.0)) * math.log(1 + float(attr.get("total_ratings", 0)))
    return sorted(items, key=pop, reverse=True)[:k]


def retrieve_candidates(state: AgentState) -> list[dict[str, Any]]:
    """Retrieve top-10 candidates via hybrid TF-IDF + BM25 + RRF."""
    domain = state.get("resolved_domain") or "general lifestyle"
    fallback_used = bool(state.get("fallback_used", False))

    query = _build_query(state)

    # Cross-domain expansion when the domain validator routed to fallback.
    domain_filter: str | list[str] | None
    if fallback_used or domain == "general lifestyle":
        domain_filter = None  # search the whole catalog; persona signals will steer
    else:
        domain_filter = domain

    try:
        retriever = get_retriever()
        hits = retriever.retrieve(query=query, domain=domain_filter, k=10)
    except FileNotFoundError as exc:
        warn(f"retrieval_node: hybrid index unavailable ({exc}); falling back to popularity")
        hits = []
    except Exception as exc:  # noqa: BLE001
        warn(f"retrieval_node: hybrid retrieval failed ({exc}); falling back to popularity")
        hits = []

    if not hits:
        return _popularity_fallback(domain, k=10)

    # Join back with full catalog item records (retriever returns lightweight rows).
    catalog = _load_catalog()
    by_id: dict[str, dict[str, Any]] = {}
    for items in catalog.values():
        for it in items:
            by_id[it["item_id"]] = it

    out: list[dict[str, Any]] = []
    for h in hits:
        full = by_id.get(h["item_id"], {})
        merged = {**full, **h}
        merged["raw_score"] = float(h.get("score", 0.0))
        out.append(merged)

    if _is_cold_start(state):
        # Blend in a popularity tail so cold-start personas still see canonical items.
        seen = {c["item_id"] for c in out}
        for it in _popularity_fallback(domain, k=10):
            if it["item_id"] not in seen and len(out) < 10:
                out.append({**it, "raw_score": 0.0})

    return out[:10]


def retrieval_node(state: AgentState) -> AgentState:
    """Write `candidates` to state."""
    state["candidates"] = retrieve_candidates(state)
    return state
