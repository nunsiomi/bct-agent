"""Build the catalog vector index (TF-IDF by default).

Output: ``data_prep/artifacts/vector_index.npz`` (vectors + per-row metadata).
Phase 4's hybrid retriever loads this for dense-side scoring.

Run:
    python -m data_pipeline.build_vector_index
"""

from __future__ import annotations

import json
import pickle

import numpy as np

from core.config import ARTIFACTS_DIR, CATALOG_PATH
from core.embeddings import get_embedder

OUT_NPZ = ARTIFACTS_DIR / "vector_index.npz"
OUT_META = ARTIFACTS_DIR / "vector_index_meta.json"
OUT_EMBEDDER = ARTIFACTS_DIR / "vector_index_embedder.pkl"


def _flatten(catalog: dict) -> list[dict]:
    rows: list[dict] = []
    for domain, items in catalog.items():
        for it in items:
            rows.append({
                "item_id": it.get("item_id"),
                "domain": domain,
                "title": it.get("title"),
                "text_blob": it.get("text_blob") or it.get("title", ""),
            })
    return rows


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    rows = _flatten(catalog)
    if not rows:
        raise SystemExit("catalog is empty; run `python -m data_pipeline.build_catalog` first")

    embedder = get_embedder()
    corpus = [r["text_blob"] for r in rows]
    embedder.fit(corpus)
    vectors = embedder.encode(corpus)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, vectors=vectors)
    OUT_META.write_text(
        json.dumps({"backend": embedder.name, "dim": embedder.dim, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    # Persist the fitted vectorizer so query-time encoding uses the same vocab.
    with OUT_EMBEDDER.open("wb") as fh:
        pickle.dump(embedder, fh)
    print(
        f"Wrote {OUT_NPZ} ({vectors.shape}), {OUT_META.name}, {OUT_EMBEDDER.name} "
        f"-- backend={embedder.name} dim={embedder.dim}"
    )


if __name__ == "__main__":
    main()
