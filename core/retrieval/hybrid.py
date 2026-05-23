"""Hybrid retriever: dense TF-IDF + sparse BM25, fused with Reciprocal Rank Fusion.

Drop-in replacement for the substring-overlap scorer that previously lived
in ``task_b/agent/retrieval_node.py``. Loads its dense side from the prebuilt
``vector_index.npz`` and constructs BM25 in-process from the same catalog.

Use ``get_retriever()`` to obtain a process-wide singleton.

Phase-4 wiring:

    from core.retrieval import get_retriever
    cands = get_retriever().retrieve(query="...", domain="tech", k=10)
"""

from __future__ import annotations

import json
import pickle
from functools import lru_cache
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from core.config import ARTIFACTS_DIR


def _tokenise(text: str) -> list[str]:
    return [t for t in (text or "").lower().split() if t]


class HybridRetriever:
    """Lazy-loaded dense + sparse retriever with RRF fusion."""

    def __init__(self) -> None:
        npz = ARTIFACTS_DIR / "vector_index.npz"
        meta = ARTIFACTS_DIR / "vector_index_meta.json"
        emb = ARTIFACTS_DIR / "vector_index_embedder.pkl"
        if not (npz.exists() and meta.exists() and emb.exists()):
            raise FileNotFoundError(
                "vector index missing; run `python -m data_pipeline.build_vector_index`"
            )
        self._vectors: np.ndarray = np.load(npz)["vectors"]
        meta_obj = json.loads(meta.read_text(encoding="utf-8"))
        self._rows: list[dict[str, Any]] = meta_obj["rows"]
        with emb.open("rb") as fh:
            self._embedder = pickle.load(fh)

        # BM25 over the same corpus, tokenised by whitespace.
        self._tokenised_corpus = [_tokenise(r.get("text_blob") or r["title"]) for r in self._rows]
        self._bm25 = BM25Okapi(self._tokenised_corpus)

        # Per-domain index masks for fast filtering.
        self._domain_index: dict[str, np.ndarray] = {}
        for d in {r["domain"] for r in self._rows}:
            self._domain_index[d] = np.array([r["domain"] == d for r in self._rows])

    # --------------------------------------------------------------------- #
    # Components
    # --------------------------------------------------------------------- #

    def _dense_scores(self, query: str) -> np.ndarray:
        q = self._embedder.encode([query])  # (1, D)
        return (self._vectors @ q.T).ravel()  # cosine via dot (L2-normed)

    def _sparse_scores(self, query: str) -> np.ndarray:
        toks = _tokenise(query)
        if not toks:
            return np.zeros(len(self._rows), dtype=np.float32)
        return np.asarray(self._bm25.get_scores(toks), dtype=np.float32)

    def _domain_mask(self, domains: list[str] | None) -> np.ndarray | None:
        if not domains:
            return None
        mask = np.zeros(len(self._rows), dtype=bool)
        for d in domains:
            m = self._domain_index.get(d)
            if m is not None:
                mask |= m
        return mask if mask.any() else None

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def retrieve(
        self,
        query: str,
        domain: str | list[str] | None = None,
        k: int = 10,
        rrf_k: int = 60,
        per_side: int = 50,
    ) -> list[dict[str, Any]]:
        """Return top-k items fused from dense + sparse rankings.

        ``domain`` may be a single domain string, a list (cross-domain), or
        None (no filter). RRF score = sum_i 1 / (rrf_k + rank_i) across the
        dense and sparse rankings, restricted to the union of their top
        ``per_side`` candidates.
        """
        domains = [domain] if isinstance(domain, str) else (domain or None)
        mask = self._domain_mask(domains)

        dense = self._dense_scores(query)
        sparse = self._sparse_scores(query)

        if mask is not None:
            dense = np.where(mask, dense, -np.inf)
            sparse = np.where(mask, sparse, -np.inf)

        # Pull top-N from each side, then fuse via RRF.
        dense_order = np.argsort(-dense)[:per_side]
        sparse_order = np.argsort(-sparse)[:per_side]

        rrf: dict[int, float] = {}
        for rank, idx in enumerate(dense_order, start=1):
            if dense[idx] == -np.inf:
                break
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (rrf_k + rank)
        for rank, idx in enumerate(sparse_order, start=1):
            if sparse[idx] == -np.inf:
                break
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (rrf_k + rank)

        if not rrf:
            return []

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
        out: list[dict[str, Any]] = []
        for idx, score in ranked:
            row = dict(self._rows[idx])
            row["score"] = float(score)
            row["dense_score"] = float(dense[idx]) if dense[idx] != -np.inf else 0.0
            row["sparse_score"] = float(sparse[idx]) if sparse[idx] != -np.inf else 0.0
            out.append(row)
        return out


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
