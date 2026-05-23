"""Real-history grounding for Task A (Phase 5).

Given a persona fingerprint (and optionally a known reviewer id), retrieve
2-4 *real* exemplar reviews from the 15k Jumia user-history corpus that match
the persona's signals. These exemplars are injected into the Task A prompt as
few-shot grounding -- the single biggest lever for rating accuracy (RMSE) and
text fidelity (ROUGE-L), since the model is now conditioned on how *people
like this persona* actually write.

Selection strategy
------------------
1. **Known-reviewer path**: when ``reviewer_id`` is provided (eval / repeat
   user), pull that reviewer's own training reviews first. This is the canonical
   RAG case and what gives the largest accuracy lift.
2. **Cold-start path**: project each Jumia reviewer into a tiny signal space
   (mean rating, mean review length, naija-marker density, top domain) and
   pick the closest reviewers to the persona fingerprint, then sample 2-4
   diverse reviews from their histories.

We deliberately avoid expensive ranking models here -- the signals + a
domain filter give us strong, deterministic exemplars in milliseconds.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from core.config import ARTIFACTS_DIR
from core.nigerian_context import _CULTURAL_VOCAB, _PIDGIN_MARKERS  # type: ignore

HISTORIES_PATH = ARTIFACTS_DIR / "user_histories.csv"


# --------------------------------------------------------------------------- #
# Reviewer-signal table (built once, cached)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def load_histories() -> pd.DataFrame:
    """Load the Jumia review history table built by Phase 2."""
    if not HISTORIES_PATH.exists():
        raise FileNotFoundError(
            f"user_histories.csv missing at {HISTORIES_PATH}; run "
            f"`python -m data_pipeline.build_user_histories`"
        )
    df = pd.read_csv(HISTORIES_PATH, parse_dates=["date"])
    df["review_text"] = df["review_text"].astype(str)
    return df


def _naija_density(text: str) -> float:
    blob = text.lower()
    return sum(1 for m in (_PIDGIN_MARKERS + _CULTURAL_VOCAB) if m in blob)


@lru_cache(maxsize=1)
def reviewer_signals() -> pd.DataFrame:
    """One row per reviewer with the signals used for fingerprint matching."""
    df = load_histories()
    g = df.groupby("reviewer", sort=False)
    out = pd.DataFrame({
        "reviewer": g.size().index,
        "n_reviews": g.size().values,
        "avg_rating": g["rating"].mean().values,
        "avg_len": g["review_text"].apply(lambda s: float(np.mean([len(t) for t in s])) if len(s) else 0.0).values,
        "naija_score": g["review_text"].apply(
            lambda s: float(np.mean([_naija_density(t) for t in s])) if len(s) else 0.0
        ).values,
        "top_domain": g["domain"].agg(lambda x: x.value_counts().index[0] if len(x) else "").values,
    }).reset_index(drop=True)
    # Pre-compute tone proxy bucket from avg_len.
    out["tone_proxy"] = out["avg_len"].apply(
        lambda v: "detailed" if v >= 120 else ("balanced" if v >= 40 else "terse")
    )
    return out


# --------------------------------------------------------------------------- #
# Fingerprint matching
# --------------------------------------------------------------------------- #

_TONE_TO_LEN = {
    "formal": 120.0,
    "detailed": 150.0,
    "balanced": 70.0,
    "casual": 50.0,
    "pidgin": 50.0,
    "terse": 25.0,
}


def _project_fingerprint(fingerprint: dict[str, Any]) -> dict[str, float]:
    """Project the persona fingerprint into reviewer-signal space."""
    rating_bias = float(fingerprint.get("rating_bias", 0.0) or 0.0)
    tone = str(fingerprint.get("tone", "balanced") or "balanced").lower()
    markers = fingerprint.get("nigerian_markers") or []
    return {
        # Centre on the Jumia mean (~3.7) plus the bias direction.
        "avg_rating": 3.7 + rating_bias,
        "avg_len": _TONE_TO_LEN.get(tone, 70.0),
        "naija_score": float(min(len(markers), 5)),
    }


def find_similar_reviewers(
    fingerprint: dict[str, Any],
    domain: str | None = None,
    k: int = 20,
) -> pd.DataFrame:
    """Return the top-k reviewers closest to the fingerprint in signal space."""
    signals = reviewer_signals()
    if domain:
        in_domain = signals[signals["top_domain"] == domain]
        # Don't strictly require in-domain -- fall back to global if too thin.
        signals = in_domain if len(in_domain) >= k else signals

    target = _project_fingerprint(fingerprint)
    # Min-max-style normalised L2 over the three signal axes.
    diffs = pd.DataFrame({
        "rating": (signals["avg_rating"] - target["avg_rating"]).abs() / 4.0,  # span ~ [1,5]
        "len": (signals["avg_len"] - target["avg_len"]).abs() / 200.0,         # cap
        "naija": (signals["naija_score"] - target["naija_score"]).abs() / 5.0,
    })
    score = 0.5 * diffs["rating"] + 0.3 * diffs["len"] + 0.2 * diffs["naija"]
    out = signals.assign(_score=score.values).nsmallest(k, "_score").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Exemplar selection
# --------------------------------------------------------------------------- #

def _diversify(reviews: pd.DataFrame, k: int) -> pd.DataFrame:
    """Pick k reviews with rating + reviewer diversity (no two from same person)."""
    if reviews.empty:
        return reviews
    picked: list[int] = []
    seen_reviewers: set[str] = set()
    # Sort by review length desc -- longer reviews tend to be more useful exemplars.
    ordered = reviews.assign(_len=reviews["review_text"].str.len()).sort_values("_len", ascending=False)
    for idx, row in ordered.iterrows():
        if row["reviewer"] in seen_reviewers:
            continue
        picked.append(idx)
        seen_reviewers.add(row["reviewer"])
        if len(picked) >= k:
            break
    # If we still don't have k, allow same-reviewer.
    if len(picked) < k:
        for idx, _ in ordered.iterrows():
            if idx not in picked:
                picked.append(idx)
                if len(picked) >= k:
                    break
    return ordered.loc[picked].drop(columns=["_len"])


def get_exemplar_reviews(
    fingerprint: dict[str, Any],
    domain: str | None = None,
    product_hint: str | None = None,
    reviewer_id: str | None = None,
    k: int = 3,
) -> list[dict[str, Any]]:
    """Return up to k real exemplar reviews matching the persona.

    Each exemplar is a dict with: ``reviewer``, ``item_id``, ``product_name``,
    ``domain``, ``rating``, ``review_title``, ``review_text``.
    """
    df = load_histories()

    # Path 1: known reviewer -- their own training reviews.
    if reviewer_id is not None:
        own = df[df["reviewer"] == reviewer_id]
        if domain:
            own_in_domain = own[own["domain"] == domain]
            if not own_in_domain.empty:
                own = own_in_domain
        if not own.empty:
            picks = _diversify(own, k)
            return _to_records(picks)

    # Path 2: cold-start / API path -- fingerprint match against similar reviewers.
    similar = find_similar_reviewers(fingerprint, domain=domain, k=20)
    if similar.empty:
        return []
    candidates = df[df["reviewer"].isin(similar["reviewer"])]
    if domain:
        in_domain = candidates[candidates["domain"] == domain]
        if not in_domain.empty:
            candidates = in_domain
    picks = _diversify(candidates, k)
    return _to_records(picks)


def _to_records(picks: pd.DataFrame) -> list[dict[str, Any]]:
    cols = ["reviewer", "item_id", "product_name", "domain", "rating", "review_title", "review_text"]
    have = [c for c in cols if c in picks.columns]
    return picks[have].to_dict("records")


# --------------------------------------------------------------------------- #
# Prompt formatting
# --------------------------------------------------------------------------- #

def format_exemplars_for_prompt(exemplars: list[dict[str, Any]]) -> str:
    """Render exemplar reviews as a compact few-shot block for the LLM."""
    if not exemplars:
        return "No similar real reviews available."
    lines = ["Real reviews from similar users (use as style + tone reference, do NOT copy):"]
    for i, e in enumerate(exemplars, start=1):
        product = str(e.get("product_name") or e.get("item_id") or "?")
        title = str(e.get("review_title") or "").strip()
        text = str(e.get("review_text") or "").strip().replace("\n", " ")
        if len(text) > 240:
            text = text[:237] + "..."
        rating = e.get("rating")
        rating_str = f"{int(rating)}/5" if rating is not None else "?/5"
        lines.append(
            f"  [{i}] product: {product}\n"
            f"      rating: {rating_str}  title: {title!r}\n"
            f"      review: {text!r}"
        )
    return "\n".join(lines)


def exemplar_rating_prior(exemplars: list[dict[str, Any]]) -> float | None:
    """Return the mean rating of the exemplars (used as a prior for Task A)."""
    if not exemplars:
        return None
    vals = [float(e["rating"]) for e in exemplars if e.get("rating") is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)
