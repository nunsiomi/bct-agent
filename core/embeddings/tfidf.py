"""TF-IDF backend.

Lightweight, deterministic, sklearn-only. Suitable for catalog-size corpora
(thousands of items). Produces L2-normalised vectors so cosine == dot product.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self, max_features: int = 20000, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self._vec = TfidfVectorizer(
            lowercase=True,
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self._fitted = False
        self._dim = 0

    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, corpus: Sequence[str]) -> None:
        if not corpus:
            raise ValueError("cannot fit TF-IDF on empty corpus")
        self._vec.fit(list(corpus))
        self._dim = len(self._vec.vocabulary_)
        self._fitted = True

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.encode called before fit()")
        X = self._vec.transform(list(texts)).toarray().astype(np.float32)
        # L2-normalise -> cosine via dot product
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms
