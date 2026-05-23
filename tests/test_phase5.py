"""Phase-5 tests: real-history grounding for Task A.

Covers:
- ``core.history_grounding`` returns real exemplars from the Jumia corpus
  for both the known-reviewer path and the cold-start fingerprint-match path.
- The Task A prompt embeds exemplar text when grounding finds something.
- The full Task A graph runs end-to-end with a mocked LLM and produces a
  review whose final rating is pulled toward the grounded prior.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from core.history_grounding import (
    exemplar_rating_prior,
    find_similar_reviewers,
    format_exemplars_for_prompt,
    get_exemplar_reviews,
    load_histories,
    reviewer_signals,
)


# --------------------------------------------------------------------------- #
# Reviewer-signal table
# --------------------------------------------------------------------------- #

def test_reviewer_signals_has_one_row_per_reviewer():
    sig = reviewer_signals()
    hist = load_histories()
    assert len(sig) == hist["reviewer"].nunique(), (
        f"reviewer_signals has {len(sig)} rows; expected {hist['reviewer'].nunique()}"
    )
    # Sanity: ratings are bounded.
    assert sig["avg_rating"].between(1.0, 5.0).all()


def test_find_similar_reviewers_returns_close_matches():
    fp = {
        "rating_bias": 0.5,  # generous reviewer
        "tone": "terse",     # short reviews
        "nigerian_markers": ["lagos", "naija"],
    }
    similar = find_similar_reviewers(fp, k=10)
    assert len(similar) == 10
    # The matched reviewers should skew toward higher ratings, given bias=+0.5.
    assert similar["avg_rating"].mean() >= 3.5


# --------------------------------------------------------------------------- #
# Exemplar selection
# --------------------------------------------------------------------------- #

def test_known_reviewer_path_returns_their_own_reviews():
    """When reviewer_id is provided, exemplars come from that reviewer's history."""
    hist = load_histories()
    # Pick a reviewer with multiple reviews so we can verify the path.
    counts = hist["reviewer"].value_counts()
    repeater = counts[counts >= 3].index[0]

    exemplars = get_exemplar_reviews(
        fingerprint={"tone": "balanced"},
        reviewer_id=repeater,
        k=3,
    )
    assert exemplars, "known-reviewer path returned no exemplars"
    assert all(e["reviewer"] == repeater for e in exemplars), (
        "known-reviewer path leaked in exemplars from other reviewers"
    )


def test_cold_start_path_returns_diverse_exemplars():
    """Without reviewer_id, exemplars come from multiple distinct reviewers."""
    fp = {
        "rating_bias": 0.0,
        "tone": "casual",
        "nigerian_markers": ["lagos"],
    }
    exemplars = get_exemplar_reviews(fingerprint=fp, k=3)
    assert exemplars, "cold-start path returned no exemplars"
    distinct = {e["reviewer"] for e in exemplars}
    assert len(distinct) >= 2, f"cold-start exemplars not diverse: {distinct}"


def test_exemplars_have_real_review_text():
    fp = {"rating_bias": 0.0, "tone": "balanced", "nigerian_markers": []}
    exemplars = get_exemplar_reviews(fingerprint=fp, k=3)
    assert exemplars
    for e in exemplars:
        assert e.get("review_text") and len(e["review_text"]) >= 5
        assert 1 <= int(e["rating"]) <= 5


def test_domain_filter_biases_exemplars():
    fp = {"rating_bias": 0.0, "tone": "balanced", "nigerian_markers": []}
    skincare = get_exemplar_reviews(fingerprint=fp, domain="skincare", k=3)
    tech = get_exemplar_reviews(fingerprint=fp, domain="tech", k=3)
    assert skincare and tech
    # At least one exemplar in each set should match the requested domain.
    assert any(e["domain"] == "skincare" for e in skincare)
    assert any(e["domain"] == "tech" for e in tech)


def test_rating_prior_is_within_bounds():
    fp = {"rating_bias": 0.0, "tone": "balanced", "nigerian_markers": []}
    exemplars = get_exemplar_reviews(fingerprint=fp, k=3)
    prior = exemplar_rating_prior(exemplars)
    assert prior is not None
    assert 1.0 <= prior <= 5.0


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #

def test_review_prompt_includes_exemplar_text():
    """The review_generator prompt must surface real exemplar text when present."""
    from task_a.agent.review_generator import build_review_prompt

    fake_exemplars = [
        {
            "reviewer": "OluwakemiX",
            "item_id": "roushun-vitamin-c-serum",
            "product_name": "Roushun Vitamin C Serum",
            "domain": "skincare",
            "rating": 4,
            "review_title": "my face is glowing",
            "review_text": "I love it, perfectly on my skin no wahala",
        }
    ]
    state = {
        "persona": "Lagos office worker, mid-budget",
        "product": "Roushun Vitamin C Serum 30ml",
        "fingerprint": {"tone": "casual", "nigerian_markers": ["lagos"]},
        "nigerian_context": {"apply": True, "pidgin_markers": [], "cultural_vocab": []},
        "nigerian_context_applied": True,
        "similar_users_summary": "",
        "exemplars": fake_exemplars,
        "exemplars_block": format_exemplars_for_prompt(fake_exemplars),
        "exemplar_rating_prior": 4.0,
    }
    _system, user = build_review_prompt(state)
    assert "Roushun Vitamin C Serum" in user, "product_name from exemplar missing from prompt"
    assert "I love it" in user, "exemplar review_text missing from prompt"
    assert "4.00/5" in user or "Empirical rating prior" in user, "rating prior line missing from prompt"


def test_rating_is_calibrated_toward_prior():
    """A wildly off LLM rating should be pulled toward the prior."""
    from task_a.agent.review_generator import review_generator_node

    import task_a.agent.review_generator as rg
    orig = rg.call_claude
    # Pretend the LLM returned 5.0 when the prior is 2.0; expect blend.
    rg.call_claude = lambda *a, **k: '{"review": "Solid product overall, did the job for me.", "rating": 5.0, "rationale": "tested"}'  # type: ignore
    try:
        state = {
            "persona": "A skeptical Lagos shopper",
            "product": "Some Item",
            "fingerprint": {"tone": "casual", "nigerian_markers": []},
            "nigerian_context": {"apply": False, "pidgin_markers": [], "cultural_vocab": []},
            "nigerian_context_applied": False,
            "similar_users_summary": "",
            "exemplars": [],
            "exemplars_block": "",
            "exemplar_rating_prior": 2.0,
            "cohort_pct_5star": 0.0,
        }
        out = review_generator_node(state)
    finally:
        rg.call_claude = orig

    rating = out["draft_rating"]
    assert rating < 5.0, "rating prior failed to pull rating down"
    assert rating > 2.0, "blending overcorrected -- rating should be between prior and model"


# --------------------------------------------------------------------------- #
# Full Task A graph with mocked LLM
# --------------------------------------------------------------------------- #

def test_task_a_graph_end_to_end_with_grounding_and_mocked_llm():
    import core.persona_builder as pb
    import task_a.agent.review_generator as rg

    def mock_persona(_text: str) -> dict:
        return {
            "rating_bias": 0.0,
            "tone": "casual",
            "price_sensitivity": "medium",
            "category_affinity": ["skincare", "nigerian"],
            "nigerian_markers": ["lagos"],
        }

    def mock_call_claude(*_a, **_kw) -> str:
        return json.dumps({
            "review": "Bought this serum for my Lagos weather skin. e dey work small small, "
                      "no wahala. Worth the price for the size you get.",
            "rating": 4,
            "rationale": "good value, mild result",
        })

    orig_pb = pb.build_persona_profile
    orig_rg = rg.call_claude
    pb.build_persona_profile = mock_persona  # type: ignore
    rg.call_claude = mock_call_claude  # type: ignore
    try:
        from task_a.agent.graph import build_graph
        graph = build_graph()
        result = graph.invoke({
            "persona": "Lagos office worker, mid-budget, values vitamin C serums",
            "product": "Roushun Vitamin C Serum 30ml",
        })
    finally:
        pb.build_persona_profile = orig_pb  # type: ignore
        rg.call_claude = orig_rg  # type: ignore

    # Grounding should have populated exemplars from the real Jumia corpus.
    exemplars = result.get("exemplars") or []
    assert exemplars, "history_grounding_node failed to populate exemplars"
    assert all(e.get("review_text") for e in exemplars), "exemplars missing review_text"
    # The final review/rating should be well-formed.
    assert 1.0 <= result["rating"] <= 5.0
    assert isinstance(result["review"], str) and len(result["review"]) >= 20


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
