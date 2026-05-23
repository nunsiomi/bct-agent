"""Phase-6 tests: Task A self-critique + revise loop.

The critique node is heuristic (deterministic, no LLM call), so most checks
run instantly. The full-graph test mocks the LLM and exercises the back-edge:

    review_generator -> critique --(fails)--> revise -> critique --(passes)--> finalize
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task_a.agent.critique_node import (
    _has_nigerian_marker,
    _mentions_product,
    _sentiment_consistent,
    critique_review,
    should_revise,
)


# --------------------------------------------------------------------------- #
# Heuristic primitives
# --------------------------------------------------------------------------- #

def test_mentions_product_exact():
    assert _mentions_product("This Tecno Spark is fire", "Tecno Spark 20")
    assert not _mentions_product("Nice purchase, would buy again", "Tecno Spark 20")


def test_mentions_product_token():
    assert _mentions_product("The serum cleared my skin", "Roushun Vitamin C Serum")


def test_nigerian_marker_detection():
    assert _has_nigerian_marker("e sweet well well, no wahala")
    assert _has_nigerian_marker("I bought it for ₦2500")
    assert _has_nigerian_marker("Lovely jollof, abeg try it")
    assert not _has_nigerian_marker("Great product, very satisfied with the purchase.")


def test_sentiment_consistent():
    ok, _ = _sentiment_consistent("Absolutely terrible product, never again", 5.0)
    assert ok is False  # high rating + negative text -> fail
    ok, _ = _sentiment_consistent("Love this, perfect for my skin", 1.0)
    assert ok is False  # low rating + positive text -> fail
    ok, _ = _sentiment_consistent("Solid product, did the job", 4.0)
    assert ok is True   # neutral text + reasonable rating -> ok
    ok, _ = _sentiment_consistent("Mixed feelings, some good some bad", 3.0)
    assert ok is True   # mixed -> ok


# --------------------------------------------------------------------------- #
# critique_review aggregator
# --------------------------------------------------------------------------- #

def test_critique_passes_on_good_draft():
    state = {
        "draft_review": "This Tecno Spark 20 phone work well well, the screen sweet, "
                        "battery dey last. I buy am for ₦230,000, e worth am.",
        "draft_rating": 4.0,
        "product": "Tecno Spark 20",
        "nigerian_context_applied": True,
        "exemplar_rating_prior": 4.2,
    }
    passes, issues, score = critique_review(state)
    assert passes, f"expected pass, got issues: {issues}"
    assert score == 1.0


def test_critique_fails_on_short_draft():
    state = {
        "draft_review": "nice",
        "draft_rating": 4.0,
        "product": "Tecno Spark 20",
        "nigerian_context_applied": False,
    }
    passes, issues, _ = critique_review(state)
    assert not passes
    assert any("short" in i.lower() for i in issues)


def test_critique_fails_when_product_missing():
    state = {
        "draft_review": "I really enjoyed using this, abeg buy am, e sweet well well "
                        "and the price was fair. Solid purchase overall, no wahala.",
        "draft_rating": 4.5,
        "product": "Tecno Spark 20",
        "nigerian_context_applied": True,
    }
    passes, issues, _ = critique_review(state)
    assert not passes
    assert any("product" in i.lower() for i in issues)


def test_critique_fails_on_sentiment_mismatch():
    state = {
        "draft_review": "This Tecno Spark phone is absolutely terrible, total trash, "
                        "the worst purchase I ever made, abeg avoid it well well.",
        "draft_rating": 5.0,
        "product": "Tecno Spark 20",
        "nigerian_context_applied": True,
    }
    passes, issues, _ = critique_review(state)
    assert not passes
    assert any("rating" in i.lower() and "negative" in i.lower() for i in issues)


def test_critique_fails_when_prior_far_off():
    state = {
        "draft_review": "This Tecno Spark 20 is decent, abeg, no wahala, e do the job "
                        "small small. Battery dey last, screen alright, ₦230k feels fair.",
        "draft_rating": 5.0,
        "product": "Tecno Spark 20",
        "nigerian_context_applied": True,
        "exemplar_rating_prior": 2.5,
    }
    passes, issues, _ = critique_review(state)
    assert not passes
    assert any("prior" in i.lower() for i in issues)


# --------------------------------------------------------------------------- #
# should_revise (conditional edge)
# --------------------------------------------------------------------------- #

def test_should_revise_routes_to_finalize_when_passing():
    assert should_revise({"critique_passes": True}) == "finalize"


def test_should_revise_routes_to_revise_when_failing_with_budget():
    state = {"critique_passes": False, "revision_count": 0, "max_revisions": 1}
    assert should_revise(state) == "revise"


def test_should_revise_finalizes_when_budget_spent():
    state = {"critique_passes": False, "revision_count": 1, "max_revisions": 1}
    assert should_revise(state) == "finalize"


# --------------------------------------------------------------------------- #
# Full Task A graph with the critique loop (LLM mocked)
# --------------------------------------------------------------------------- #

def test_task_a_critique_loop_fixes_a_bad_first_draft():
    """First draft is too short -> critique fails -> revise emits a good draft."""
    import core.persona_builder as pb
    import task_a.agent.review_generator as rg
    import task_a.agent.revise_node as rv

    def mock_persona(_text: str) -> dict:
        return {
            "rating_bias": 0.0,
            "tone": "casual",
            "price_sensitivity": "medium",
            "category_affinity": ["tech"],
            "nigerian_markers": ["lagos"],
        }

    # First LLM call (generator) emits a too-short review.
    # Second LLM call (revise) emits a proper review that mentions the product.
    calls = {"n": 0}

    def mock_generator_call(*_a, **_kw) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"review": "nice", "rating": 4.0, "rationale": "ok"})
        return json.dumps({
            "review": "This Tecno Spark 20 is solid for the money, abeg. "
                      "Screen sharp, battery dey last small small. "
                      "₦230k feels fair, e be like say I no go regret am.",
            "rating": 4.0,
            "rationale": "addressed length + product mention",
        })

    orig_pb = pb.build_persona_profile
    orig_rg = rg.call_claude
    orig_rv = rv.call_claude
    pb.build_persona_profile = mock_persona  # type: ignore
    rg.call_claude = mock_generator_call  # type: ignore
    rv.call_claude = mock_generator_call  # type: ignore
    try:
        from task_a.agent.graph import build_graph
        graph = build_graph()
        result = graph.invoke({
            "persona": "Lagos tech worker, mid-budget, casual reviewer",
            "product": "Tecno Spark 20",
        })
    finally:
        pb.build_persona_profile = orig_pb  # type: ignore
        rg.call_claude = orig_rg  # type: ignore
        rv.call_claude = orig_rv  # type: ignore

    # The revise branch must have been taken at least once.
    assert result.get("revision_count", 0) == 1, (
        f"expected exactly one revision, got {result.get('revision_count')}"
    )
    # And the final review must be the longer, product-mentioning one.
    review = result.get("review") or ""
    assert "tecno" in review.lower() or "spark" in review.lower(), (
        f"final review still doesn't mention product: {review!r}"
    )
    assert len(review) >= 60


def test_task_a_critique_loop_terminates_after_max_revisions():
    """If revisions never fix the issue, the loop must still bail out."""
    import core.persona_builder as pb
    import task_a.agent.review_generator as rg
    import task_a.agent.revise_node as rv

    def mock_persona(_text: str) -> dict:
        return {
            "rating_bias": 0.0,
            "tone": "casual",
            "price_sensitivity": "medium",
            "category_affinity": ["tech"],
            "nigerian_markers": [],
        }

    # Always emit a too-short review so critique always fails.
    def always_bad(*_a, **_kw) -> str:
        return json.dumps({"review": "ok", "rating": 4.0, "rationale": "x"})

    orig_pb = pb.build_persona_profile
    orig_rg = rg.call_claude
    orig_rv = rv.call_claude
    pb.build_persona_profile = mock_persona  # type: ignore
    rg.call_claude = always_bad  # type: ignore
    rv.call_claude = always_bad  # type: ignore
    try:
        from task_a.agent.graph import build_graph
        graph = build_graph()
        result = graph.invoke({
            "persona": "Lagos shopper, casual",
            "product": "Some Product",
            "max_revisions": 1,  # explicit budget
        })
    finally:
        pb.build_persona_profile = orig_pb  # type: ignore
        rg.call_claude = orig_rg  # type: ignore
        rv.call_claude = orig_rv  # type: ignore

    # Should NOT exceed the budget.
    assert result.get("revision_count", 0) <= 1


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
