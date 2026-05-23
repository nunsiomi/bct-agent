"""Offline TF-IDF retrieval baseline.

Loads the prebuilt vector index from ``data_prep/artifacts/`` and exposes a
single ``retrieve(query, domain, k)`` function used by the Task B eval path
when ``--live`` is off. This is the same backbone the Phase-4 hybrid
retriever will wrap (BM25 + RRF + rerank) once that lands.
"""

from __future__ import annotations

import json
import pickle
from typing import Any

import numpy as np

from core.config import ARTIFACTS_DIR


class _Index:
    def __init__(self) -> None:
        npz_path = ARTIFACTS_DIR / "vector_index.npz"
        meta_path = ARTIFACTS_DIR / "vector_index_meta.json"
        emb_path = ARTIFACTS_DIR / "vector_index_embedder.pkl"
        if not (npz_path.exists() and meta_path.exists() and emb_path.exists()):
            raise FileNotFoundError(
                "vector index missing; run `python -m data_pipeline.build_vector_index`"
            )
        self.vectors = np.load(npz_path)["vectors"]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.rows: list[dict[str, Any]] = meta["rows"]
        with emb_path.open("rb") as fh:
            self.embedder = pickle.load(fh)

    def retrieve(
        self,
        query: str,
        domain: str | None = None,
        k: int = 10,
    ) -> list[dict[str, Any]]:
        q_vec = self.embedder.encode([query])  # (1, D)
        sims = (self.vectors @ q_vec.T).ravel()  # cosine since L2-normed

        if domain:
            mask = np.array([r["domain"] == domain for r in self.rows])
            if mask.any():
                sims = np.where(mask, sims, -1.0)

        order = np.argsort(-sims)[:k]
        return [
            {**self.rows[i], "score": float(sims[i])}
            for i in order
        ]


_INDEX: _Index | None = None


def get_index() -> _Index:
    global _INDEX
    if _INDEX is None:
        _INDEX = _Index()
    return _INDEX
