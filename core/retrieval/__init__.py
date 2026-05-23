"""Retrieval primitives shared across tasks.

- ``HybridRetriever`` -- dense (TF-IDF) + sparse (BM25) + RRF fusion.
- ``get_retriever()`` -- process-wide singleton (lazy build).
"""

from core.retrieval.hybrid import HybridRetriever, get_retriever

__all__ = ["HybridRetriever", "get_retriever"]
