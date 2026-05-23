"""Embedding backend selection."""

from __future__ import annotations

from functools import lru_cache

from core.config import EMBEDDING_BACKEND
from core.embeddings.base import EmbeddingProvider


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingProvider:
    if EMBEDDING_BACKEND == "tfidf":
        from core.embeddings.tfidf import TfidfEmbedder
        return TfidfEmbedder()
    if EMBEDDING_BACKEND == "sentence_transformer":
        from core.embeddings.sentence_transformer import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder()
    raise ValueError(f"Unknown EMBEDDING_BACKEND={EMBEDDING_BACKEND!r}")
