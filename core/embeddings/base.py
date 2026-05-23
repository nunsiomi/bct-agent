"""EmbeddingProvider protocol."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    dim: int

    def fit(self, corpus: Sequence[str]) -> None:
        """Optional: train (e.g. TF-IDF vocabulary). Pretrained backends no-op."""
        ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) L2-normalised matrix of embeddings."""
        ...
