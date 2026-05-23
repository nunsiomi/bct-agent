"""Phase-4 tests: hybrid retriever, verify node, and the full Task B graph.

All LLM-calling nodes are monkeypatched, so this suite runs offline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.retrieval import get_retriever
from task_b.agent.domain_validator import KNOWN_DOMAINS
from task_b.agent.verify_node import verify_recommendations, verify_node


# --------------------------------------------------------------------------- #
# HybridRetriever
# --------------------------------------------------------------------------- #

def test_hybrid_retriever_returns_on_domain_results():
    retr = get_retriever()
    hits = retr.retrieve(query="afrobeats wizkid rema", domain="music", k=5)
    assert hits, "hybrid retriever returned nothing for clear music query"
    assert all(h["domain"] == "music" for h in hits), (
        "hybrid retriever broke its domain filter"
    )


def test_hybrid_retriever_fuses_signals():
    """Mixing a niche term + cultural marker should still return on-domain items."""
    retr = get_retriever()
    hits = retr.retrieve(query="serum nigerian skincare brightening", domain="skincare", k=10)
    assert hits, "hybrid retriever returned nothing"
    # At least one Jumia (real) skincare item should surface near the top.
    top_titles = [h["title"].lower() for h in hits[:5]]
    assert any("serum" in t or "vitamin" in t or "cream" in t for t in top_titles), (
        f"top-5 had no plausible skincare match: {top_titles}"
    )


def test_hybrid_retriever_works_across_every_domain():
    retr = get_retriever()
    for domain in KNOWN_DOMAINS:
        hits = retr.retrieve(query="nigerian", domain=domain, k=3)
        assert hits, f"hybrid retriever empty for domain={domain}"


# --------------------------------------------------------------------------- #
# verify_node
# --------------------------------------------------------------------------- #

def test_verify_drops_hallucinated_titles_and_backfills():
    candidates = [
        {"item_id": "essence-wizkid", "title": "Essence — Wizkid ft. Tems", "raw_score": 0.9},
        {"item_id": "last-last", "title": "Last Last — Burna Boy", "raw_score": 0.8},
    ]
    fake_recs = [
        {"rank": 1, "title": "Made-Up Song That Does Not Exist", "reason": "...", "match_score": 0.9},
        {"rank": 2, "title": "Essence — Wizkid ft. Tems", "reason": "real", "match_score": 0.8},
    ]
    verified, dropped = verify_recommendations(fake_recs, candidates)
    titles = {r["title"] for r in verified}
    assert "Made-Up Song That Does Not Exist" not in titles, "verify let a hallucinated title through"
    assert "Essence — Wizkid ft. Tems" in titles, "verify dropped a real catalog title"
    assert dropped, "verify should report what it dropped"


def test_verify_backfills_when_all_recs_are_garbage():
    candidates = [
        {"item_id": "a", "title": "Things Fall Apart", "raw_score": 0.7},
        {"item_id": "b", "title": "Half of a Yellow Sun", "raw_score": 0.6},
    ]
    garbage = [{"rank": 1, "title": "Nonexistent", "reason": "", "match_score": 0.5}]
    verified, _ = verify_recommendations(garbage, candidates)
    assert verified, "verify should backfill from candidates when all recs are hallucinated"
    assert verified[0]["title"] in {"Things Fall Apart", "Half of a Yellow Sun"}


def test_verify_node_mutates_state():
    state = {
        "recommendations": [{"rank": 1, "title": "ZZ-NotInCatalog", "reason": "x", "match_score": 0.5}],
        "candidates": [{"item_id": "c1", "title": "Jollof Rice", "raw_score": 0.9}],
    }
    out = verify_node(state)
    assert out["recommendations"], "verify_node emptied recommendations"
    assert out["recommendations"][0]["title"] == "Jollof Rice"
    assert "ZZ-NotInCatalog" in (out.get("verification_dropped") or [])


# --------------------------------------------------------------------------- #
# Full graph integration with a mocked LLM
# --------------------------------------------------------------------------- #

def test_full_task_b_graph_end_to_end_with_mocked_llm():
    """Compile the real graph, mock the LLM, ensure the response is well-formed."""
    import core.persona_builder as pb
    import task_b.agent.reasoning_ranker as rr

    def mock_persona(_text: str) -> dict:
        return {
            "rating_bias": 0.0,
            "tone": "casual",
            "price_sensitivity": "medium",
            "category_affinity": ["afrobeats", "nigerian"],
            "nigerian_markers": ["lagos"],
        }

    # Have the ranker emit JSON the parser likes; titles will be re-checked
    # by the verify node, so any hallucinations get scrubbed.
    def mock_call_claude(*_a, **_kw) -> str:
        return json.dumps([
            {"rank": 1, "title": "Essence — Wizkid ft. Tems", "reason": "naija fave", "match_score": 92},
            {"rank": 2, "title": "Last Last — Burna Boy", "reason": "vibey", "match_score": 88},
            {"rank": 3, "title": "Calm Down — Rema", "reason": "global hit", "match_score": 85},
            {"rank": 4, "title": "Unavailable — Davido", "reason": "party", "match_score": 80},
            {"rank": 5, "title": "Rush — Ayra Starr", "reason": "fresh", "match_score": 77},
        ])

    orig_pb = pb.build_persona_profile
    orig_rr = rr.call_claude
    pb.build_persona_profile = mock_persona  # type: ignore
    rr.call_claude = mock_call_claude  # type: ignore
    try:
        from task_b.agent.graph import build_graph
        graph = build_graph()
        result = graph.invoke({
            "persona": "Lagos professional who loves Afrobeats and Nigerian food",
            "domain": "music",
            "niche": "afrobeats",
        })
    finally:
        pb.build_persona_profile = orig_pb  # type: ignore
        rr.call_claude = orig_rr  # type: ignore

    recs = result.get("recommendations", [])
    assert recs, "full graph produced no recommendations"
    assert len(recs) <= 5, "verify node failed to cap at 5"
    # All titles must be on-catalog.
    catalog = json.loads(
        (ROOT / "data_prep" / "artifacts" / "catalog.json").read_text(encoding="utf-8")
    )
    titles = {it["title"] for items in catalog.values() for it in items}
    for r in recs:
        assert r["title"] in titles, f"graph emitted off-catalog title: {r['title']!r}"


if __name__ == "__main__":
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
