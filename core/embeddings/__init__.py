"""Embedding provider abstraction.

`get_embedder()` returns the configured backend. Default is TF-IDF (sklearn,
zero-dependency beyond what's already installed). `sentence_transformer` is
an opt-in upgrade gated on env config + the optional dep being installed.
"""

from core.embeddings.base import EmbeddingProvider
from core.embeddings.registry import get_embedder

__all__ = ["EmbeddingProvider", "get_embedder"]
