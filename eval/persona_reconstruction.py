"""Reconstruct a free-text persona from a reviewer's training history.

Phase 3 holdout rows are real (reviewer, item, rating, review_text) tuples.
To run a fair end-to-end eval we need to express *who this reviewer is* the
way a user would (in plain text) -- without leaking the holdout review. We
build that text from the reviewer's training-set reviews only.

The reconstructed persona captures:
- typical rating tendency (generous / harsh / balanced)
- average review length (terse / detailed)
- domains they buy in (skincare / tech / food / ...)
- whether they use Nigerian markers in their own writing
- top product categories from their history
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.validation import COMMON_WORDS as _COMMON  # reuse for Naija sniff
from core.nigerian_context import _CULTURAL_VOCAB, _PIDGIN_MARKERS  # type: ignore


def _bias_label(avg_rating: float) -> str:
    if avg_rating >= 4.5:
        return "rates very generously"
    if avg_rating >= 4.0:
        return "rates positively on the whole"
    if avg_rating >= 3.0:
        return "rates pragmatically (not afraid of 3-star)"
    return "rates harshly, complains often"


def _length_label(avg_len: float) -> str:
    if avg_len >= 120:
        return "writes detailed reviews"
    if avg_len >= 40:
        return "writes moderate-length reviews"
    return "writes short, terse reviews"


def _naija_score(texts: list[str]) -> float:
    if not texts:
        return 0.0
    blob = " ".join(t.lower() for t in texts)
    hits = sum(1 for m in (_PIDGIN_MARKERS + _CULTURAL_VOCAB) if m in blob)
    return hits / max(len(_PIDGIN_MARKERS + _CULTURAL_VOCAB), 1)


def reconstruct_persona(reviewer_history: pd.DataFrame) -> dict[str, Any]:
    """Return a dict with a free-text ``persona`` plus structured signals.

    ``reviewer_history`` is the slice of ``user_histories.csv`` (or train)
    belonging to one reviewer. Empty input yields a neutral persona.
    """
    if reviewer_history is None or reviewer_history.empty:
        return {
            "persona": "An anonymous Nigerian Jumia shopper with no review history yet.",
            "avg_rating": 3.0,
            "n_reviews": 0,
            "domains": [],
        }

    avg_rating = float(reviewer_history["rating"].mean())
    texts = reviewer_history["review_text"].dropna().astype(str).tolist()
    avg_len = sum(len(t) for t in texts) / max(len(texts), 1)
    naija = _naija_score(texts)
    domains = sorted(reviewer_history["domain"].dropna().unique().tolist())

    bias_phrase = _bias_label(avg_rating)
    length_phrase = _length_label(avg_len)
    naija_phrase = (
        "writes in Nigerian English / Pidgin"
        if naija >= 0.02
        else "writes in plain English"
    )
    domain_phrase = (
        f"shops mainly in {', '.join(domains)}" if domains else "shops across categories"
    )

    persona = (
        f"Nigerian Jumia shopper who {bias_phrase}, {length_phrase}, "
        f"{naija_phrase}, and {domain_phrase}. "
        f"Past purchases: {min(len(texts), 5)} of {len(texts)} recent reviews seen."
    )
    return {
        "persona": persona,
        "avg_rating": avg_rating,
        "n_reviews": len(texts),
        "domains": domains,
        "avg_review_len": avg_len,
        "naija_score": naija,
    }
