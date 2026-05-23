"""Pure-Python metric implementations.

All metrics here are zero-dep (numpy only) so the eval harness can run in any
environment without torch / rouge_score / bert_score installed.

- ``rmse``                -- root mean squared error
- ``mae``                 -- mean absolute error
- ``rouge_l``             -- ROUGE-L F1 (LCS-based, char-tokenised on whitespace)
- ``ndcg_at_k``           -- normalised DCG, single positive item per query
- ``hit_rate_at_k``       -- whether the true item is in the top-K
- ``reciprocal_rank``     -- 1 / rank of the true item (0 if absent)

For Task A's text fidelity, ROUGE-L is the offline metric. BERTScore is a
strict upgrade but pulls torch; we leave it as an optional path the user
enables explicitly.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import math


# --------------------------------------------------------------------------- #
# Rating metrics
# --------------------------------------------------------------------------- #

def rmse(preds: Iterable[float], truths: Iterable[float]) -> float:
    preds, truths = list(preds), list(truths)
    if not preds:
        return 0.0
    sq = [(float(p) - float(t)) ** 2 for p, t in zip(preds, truths)]
    return math.sqrt(sum(sq) / len(sq))


def mae(preds: Iterable[float], truths: Iterable[float]) -> float:
    preds, truths = list(preds), list(truths)
    if not preds:
        return 0.0
    return sum(abs(float(p) - float(t)) for p, t in zip(preds, truths)) / len(preds)


# --------------------------------------------------------------------------- #
# Text metrics
# --------------------------------------------------------------------------- #

def _tokenise(text: str) -> list[str]:
    return [t for t in (text or "").lower().split() if t]


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence (DP, O(len(a)*len(b)))."""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    # Roll a 1-D DP array to keep memory linear in n.
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 score against a single reference."""
    hyp = _tokenise(hypothesis)
    ref = _tokenise(reference)
    if not hyp or not ref:
        return 0.0
    lcs = _lcs_length(hyp, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def rouge_l_corpus(hypotheses: Iterable[str], references: Iterable[str]) -> float:
    hyps, refs = list(hypotheses), list(references)
    if not hyps:
        return 0.0
    return sum(rouge_l(h, r) for h, r in zip(hyps, refs)) / len(hyps)


# --------------------------------------------------------------------------- #
# Ranking metrics (single relevant item per query)
# --------------------------------------------------------------------------- #

def hit_rate_at_k(ranked_ids: Sequence, true_id, k: int = 10) -> float:
    return 1.0 if true_id in list(ranked_ids)[:k] else 0.0


def reciprocal_rank(ranked_ids: Sequence, true_id) -> float:
    for i, item in enumerate(ranked_ids, start=1):
        if item == true_id:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: Sequence, true_id, k: int = 10) -> float:
    """NDCG@K when a single item is relevant (binary relevance)."""
    for i, item in enumerate(ranked_ids[:k], start=1):
        if item == true_id:
            return 1.0 / math.log2(i + 1)
    return 0.0


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0
