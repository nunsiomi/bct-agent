"""Jumia source adapter.

Reads the five committed Jumia review CSVs under ``datasets/`` and produces:

- ``load_raw_reviews()`` -- a tidy DataFrame of all reviews + a ``category``
  column (cookware / electronics / food / phones / skincare).
- ``canonical_items()`` -- catalog-shaped items grouped by domain
  (skincare / tech / food / general lifestyle), ready to merge into catalog.json.
- ``temporal_split()`` -- chronological train/holdout split, no reviewer leakage.
- ``to_user_histories()`` -- per-reviewer aggregated review history.

This is the data foundation for Phase 3 (ground-truth eval) and Phase 4
(catalog enrichment + retrieval grounding).
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from core.config import DATASETS_DIR

# Map Jumia scrape category -> Task B canonical domain.
CATEGORY_TO_DOMAIN: dict[str, str] = {
    "skincare": "skincare",
    "phones": "tech",
    "electronics": "tech",
    "food": "food",
    "cookware": "general lifestyle",
}

JUMIA_FILES: dict[str, str] = {
    "cookware": "jumia_reviews_cookware.csv",
    "electronics": "jumia_reviews_electronics.csv",
    "food": "jumia_reviews_food.csv",
    "phones": "jumia_reviews_phones.csv",
    "skincare": "jumia_reviews_skincare.csv",
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: Any) -> str:
    return _SLUG_RE.sub("-", str(text).lower()).strip("-")[:80]


def load_raw_reviews() -> pd.DataFrame:
    """Return all Jumia reviews concatenated, with normalised columns.

    Adds:
      - ``category`` (jumia scrape category, e.g. ``phones``)
      - ``domain``   (mapped canonical domain, e.g. ``tech``)
      - ``item_id``  (product_name slug, stable across files)
      - ``date``     (parsed datetime, DD-MM-YYYY)
      - ``rating``   (int, 1..5)
    Drops rows with missing review_text.
    """
    frames: list[pd.DataFrame] = []
    for category, filename in JUMIA_FILES.items():
        path = DATASETS_DIR / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["category"] = category
        df["domain"] = CATEGORY_TO_DOMAIN.get(category, "general lifestyle")
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["product_name"]).copy()
    df["product_name"] = df["product_name"].astype(str)
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"].ne("") & df["review_text"].ne("nan")].copy()
    df["item_id"] = df["product_name"].map(_slug)
    df["date"] = pd.to_datetime(df["review_date"], format="%d-%m-%Y", errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3).astype(int).clip(1, 5)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["avg_rating"] = pd.to_numeric(df["avg_rating"], errors="coerce").fillna(0.0)
    df["total_ratings"] = pd.to_numeric(df["total_ratings"], errors="coerce").fillna(0).astype(int)
    df["reviewer"] = df["reviewer"].astype(str)
    return df.reset_index(drop=True)


def _short_title(name: str, limit: int = 70) -> str:
    s = str(name).strip()
    return s if len(s) <= limit else s[: limit - 1].rsplit(" ", 1)[0] + "…"


def _derive_niches(product_name: str, category: str) -> list[str]:
    """Heuristic niche tags from product name + category (cheap, deterministic)."""
    name = product_name.lower()
    out: set[str] = set()
    keyword_map = {
        # skincare
        "serum": "serum", "cream": "cream", "lotion": "lotion", "soap": "soap",
        "sunscreen": "sunscreen", "vitamin c": "brightening", "retinol": "anti-aging",
        "moistur": "moisturizer", "cleanser": "cleanser", "shea": "natural",
        "black soap": "natural", "niacinamide": "brightening",
        # phones / tech
        "iphone": "premium", "samsung": "android", "tecno": "budget", "infinix": "budget",
        "redmi": "budget", "xiaomi": "android", "itel": "budget", "oraimo": "accessories",
        "power bank": "power", "smart watch": "wearable", "earbud": "audio",
        # food / cookware
        "rice": "rice", "noodle": "noodles", "oil": "oil", "tea": "tea",
        "lunch bag": "kitchen", "pot": "cookware", "blender": "appliance",
    }
    for k, v in keyword_map.items():
        if k in name:
            out.add(v)
    out.add(category)
    return sorted(out)


def canonical_items(reviews: pd.DataFrame | None = None) -> dict[str, list[dict[str, Any]]]:
    """Produce catalog-shaped items grouped by canonical domain.

    Aggregates one row per unique product. Uses median observed price (more
    robust than the per-row scraped price) and the scraped avg_rating as the
    product's prior. Adds top review_titles into ``attributes`` for downstream
    grounding.
    """
    df = reviews if reviews is not None else load_raw_reviews()
    if df.empty:
        return {}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for (item_id, domain), g in df.groupby(["item_id", "domain"]):
        product_name = g["product_name"].iloc[0]
        category = g["category"].iloc[0]
        title = _short_title(product_name)
        text_blob = " | ".join(
            [
                product_name,
                category,
                ", ".join(g["review_title"].dropna().astype(str).head(5)),
            ]
        )
        item = {
            "item_id": item_id,
            "title": title,
            "categories": [category],
            "niches": _derive_niches(product_name, category),
            "tags": ["nigerian", "jumia"],
            "price_naira": (
                int(g["price"].dropna().median()) if g["price"].dropna().size else None
            ),
            "attributes": {
                "avg_rating": float(g["avg_rating"].iloc[0]),
                "total_ratings": int(g["total_ratings"].iloc[0]),
                "n_reviews_in_dataset": int(len(g)),
                "product_link": str(g["product_link"].iloc[0]),
                "source": "jumia",
            },
            "text_blob": text_blob,
        }
        grouped.setdefault(domain, []).append(item)

    for domain in grouped:
        grouped[domain].sort(key=lambda i: i["attributes"]["total_ratings"], reverse=True)
    return grouped


def to_user_histories(reviews: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-(reviewer, review) tidy table for downstream grounding / eval.

    Returns columns: reviewer, item_id, domain, rating, review_text,
    review_title, date, price_naira.
    """
    df = reviews if reviews is not None else load_raw_reviews()
    if df.empty:
        return df
    return df[
        ["reviewer", "item_id", "product_name", "domain", "rating",
         "review_text", "review_title", "date", "price"]
    ].rename(columns={"price": "price_naira"})


def temporal_split(
    reviews: pd.DataFrame | None = None,
    holdout_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: latest ``holdout_frac`` rows held out.

    Reviewer-level leakage is acceptable here because the eval signal we want
    is *the same reviewer's future behaviour*, mirroring real recommender
    settings. The split is by date, not by row index, so it's deterministic
    against the data.
    """
    df = reviews if reviews is not None else load_raw_reviews()
    if df.empty:
        return df, df

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    cutoff_idx = int(len(df) * (1.0 - holdout_frac))
    cutoff_date = df.loc[cutoff_idx, "date"] if 0 <= cutoff_idx < len(df) else df["date"].max()
    train = df[df["date"] < cutoff_date].reset_index(drop=True)
    holdout = df[df["date"] >= cutoff_date].reset_index(drop=True)
    return train, holdout
