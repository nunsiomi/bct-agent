"""Verify node -- post-rank validation gate.

Drops any recommended title not present in the catalog (kills hallucinations),
enforces uniqueness, caps the list to 5, and renumbers ranks. Runs after the
reasoning ranker, before END.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.json_utils import warn
from task_b.agent.state import AgentState

_CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "data_prep" / "artifacts" / "catalog.json"
)
_CATALOG_TITLES: set[str] | None = None


def _catalog_titles() -> set[str]:
    global _CATALOG_TITLES
    if _CATALOG_TITLES is not None:
        return _CATALOG_TITLES
    try:
        cat = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        warn("verify_node: catalog.json missing; skipping verification")
        _CATALOG_TITLES = set()
        return _CATALOG_TITLES
    _CATALOG_TITLES = {
        str(item.get("title", ""))
        for items in cat.values()
        for item in items
        if item.get("title")
    }
    return _CATALOG_TITLES


def verify_recommendations(recs: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Filter ``recs`` to titles present in the catalog and unique.

    Returns (verified, dropped_titles). If verification empties the list, the
    caller is expected to backfill from ``candidates`` (also catalog-real).
    """
    titles = _catalog_titles()
    seen: set[str] = set()
    verified: list[dict[str, Any]] = []
    dropped: list[str] = []

    for r in recs:
        title = str(r.get("title", "")).strip()
        if not title:
            dropped.append("<empty>")
            continue
        if titles and title not in titles:
            dropped.append(title)
            continue
        if title in seen:
            dropped.append(f"{title} (duplicate)")
            continue
        seen.add(title)
        verified.append(r)

    if not verified and candidates:
        # Backfill from real candidates (already catalog-real) so the response
        # is never empty even if the ranker hallucinated wholesale.
        for c in candidates[:5]:
            title = str(c.get("title", ""))
            if title and title not in seen:
                verified.append({
                    "rank": len(verified) + 1,
                    "title": title,
                    "reason": "Strong cohort match against your persona.",
                    "match_score": float(c.get("raw_score", 0.5)) or 0.5,
                })
                seen.add(title)

    for i, r in enumerate(verified[:5], start=1):
        r["rank"] = i
    return verified[:5], dropped


def verify_node(state: AgentState) -> AgentState:
    recs = state.get("recommendations", []) or []
    candidates = state.get("candidates", []) or []
    verified, dropped = verify_recommendations(recs, candidates)
    if dropped:
        warn(f"verify_node: dropped {len(dropped)} hallucinated/duplicate titles: {dropped[:3]}")
    state["recommendations"] = verified
    state["verification_dropped"] = dropped
    return state
