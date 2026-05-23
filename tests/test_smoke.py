"""Phase-0 smoke tests — guard the reproducibility floor.

These would have caught the original "empty catalog on a clean clone" bug:
they assert the committed catalog loads and that retrieval returns real,
on-catalog candidates for every supported domain. The single LLM call in the
Task B ranker is mocked, so the whole suite runs offline with no API key.

Run:
    python -m pytest tests/test_smoke.py -q
    # or, dependency-free:
    python tests/test_smoke.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CATALOG = ROOT / "data_prep" / "artifacts" / "catalog.json"

# Import after path setup; these are pure-python and need no API key.
from task_b.agent.domain_validator import KNOWN_DOMAINS
from task_b.agent.retrieval_node import retrieve_candidates


def _sample_state(domain: str, niche: str | None = None) -> dict:
    return {
        "resolved_domain": domain,
        "resolved_niche": niche,
        "fingerprint": {
            "tone": "casual",
            "price_sensitivity": "medium",
            "category_affinity": ["nigerian", "afrobeats", "food"],
            "nigerian_markers": ["lagos"],
        },
        "similar_users": [{"top_categories": ["food", "restaurants"]}],
        "nigerian_context_applied": True,
    }


def test_catalog_committed_and_loads():
    assert CATALOG.exists(), f"catalog.json missing at {CATALOG} — Task B cannot work"
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert isinstance(catalog, dict) and catalog, "catalog is empty"
    n_items = sum(len(v) for v in catalog.values())
    assert n_items >= 50, f"catalog too thin ({n_items} items)"


def test_every_known_domain_has_items():
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    for domain in KNOWN_DOMAINS:
        assert catalog.get(domain), f"no catalog items for supported domain '{domain}'"


def test_items_well_formed():
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    for domain, items in catalog.items():
        for item in items:
            assert item.get("title"), f"{domain}: item missing title"
            assert isinstance(item.get("categories"), list)
            assert isinstance(item.get("niches"), list)
            assert isinstance(item.get("tags"), list)


def test_retrieval_returns_candidates_for_every_domain():
    for domain in KNOWN_DOMAINS:
        cands = retrieve_candidates(_sample_state(domain))
        assert cands, f"retrieval returned NO candidates for '{domain}'"
        assert all(c.get("title") for c in cands)


def test_ranker_fallback_yields_five_when_llm_returns_nothing(monkeypatch=None):
    """Even if the LLM emits garbage, the node must return 5 on-catalog recs."""
    import task_b.agent.reasoning_ranker as rr

    # Force the LLM to return an empty/garbage payload.
    orig = rr.call_claude
    rr.call_claude = lambda *a, **k: ""  # type: ignore
    try:
        state = _sample_state("music", "afrobeats")
        state["candidates"] = retrieve_candidates(state)
        out = rr.reasoning_ranker_node(state)
    finally:
        rr.call_claude = orig

    recs = out["recommendations"]
    assert len(recs) == 5, f"expected 5 fallback recs, got {len(recs)}"
    titles = {c["title"] for c in state["candidates"]}
    assert all(r["title"] in titles for r in recs), "ranker hallucinated an off-catalog title"


if __name__ == "__main__":
    # Dependency-free runner (no pytest required).
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
