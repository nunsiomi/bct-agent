"""sentence-transformers backend (opt-in).

Heavier dependency footprint (torch + a model download). Import is lazy so
this file is safe to ship without the dep installed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from core.config import EMBEDDING_MODEL


class SentenceTransformerEmbedder:
    name = "sentence_transformer"

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers not installed; pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name or EMBEDDING_MODEL)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, corpus: Sequence[str]) -> None:
        # Pretrained -- no-op
        return None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        X = self._model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
        return X.astype(np.float32)
