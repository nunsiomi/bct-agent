"""Persona-signals data layer.

Loads the offline-built `datasets/persona_signals.csv` (produced by
`data_prep/bct-dataprep.ipynb` from Yelp / Amazon / Goodreads) and finds
the top-n similar rows for a given fingerprint.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import PERSONA_SIGNALS_PATH

_SIGNALS_CACHE: pd.DataFrame | None = None

_TONE_NEIGHBOURS = {
    "formal": {"balanced": 0.5, "detailed": 0.5},
    "balanced": {"formal": 0.5, "casual": 0.5, "detailed": 0.5, "terse": 0.5},
    "casual": {"balanced": 0.5, "pidgin": 0.5, "terse": 0.5},
    "pidgin": {"casual": 0.5},
    "detailed": {"formal": 0.5, "balanced": 0.5},
    "terse": {"balanced": 0.5, "casual": 0.5},
}

_PRICE_LEVELS = {"low": 0, "medium": 1, "high": 2}


def load_persona_signals(path: str | Path | None = None) -> pd.DataFrame:
    """Load and cache the persona signals CSV."""
    global _SIGNALS_CACHE
    if _SIGNALS_CACHE is not None and path is None:
        return _SIGNALS_CACHE

    csv_path = Path(path) if path else PERSONA_SIGNALS_PATH
    df = pd.read_csv(csv_path)

    def _parse_cats(v: Any) -> list[str]:
        if isinstance(v, list):
            return v
        if not isinstance(v, str):
            return []
        try:
            parsed = ast.literal_eval(v)
            return [str(x) for x in parsed] if isinstance(parsed, (list, tuple)) else []
        except (ValueError, SyntaxError):
            return []

    df["top_categories"] = df["top_categories"].apply(_parse_cats)
    if "elite_ever" in df.columns:
        df["elite_ever"] = df["elite_ever"].astype(bool)

    if path is None:
        _SIGNALS_CACHE = df
    return df


def _tone_cost(fp_tone: str, row_tone: str) -> float:
    if not isinstance(row_tone, str) or not isinstance(fp_tone, str):
        return 1.0
    a, b = fp_tone.lower(), row_tone.lower()
    if a == b:
        return 0.0
    return _TONE_NEIGHBOURS.get(a, {}).get(b, 1.0)


def _price_cost(fp_price: str, row_price: str) -> float:
    if not isinstance(row_price, str) or not isinstance(fp_price, str):
        return 1.0
    a = _PRICE_LEVELS.get(fp_price.lower())
    b = _PRICE_LEVELS.get(row_price.lower())
    if a is None or b is None:
        return 1.0
    return abs(a - b) / 2.0


def find_similar_users(fingerprint: dict, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the top-n rows whose tone/price/rating_bias most closely match the fingerprint."""
    fp_tone = str(fingerprint.get("tone", "balanced")).lower()
    fp_price = str(fingerprint.get("price_sensitivity", "medium")).lower()
    fp_bias = float(fingerprint.get("rating_bias", 0.0) or 0.0)

    work = df.copy()
    work["_tone_cost"] = work["tone_proxy"].apply(lambda t: _tone_cost(fp_tone, t))
    work["_price_cost"] = work["price_sensitivity"].apply(lambda p: _price_cost(fp_price, p))
    bias_diff = (work["rating_bias"].fillna(0.0) - fp_bias).abs()
    work["_bias_cost"] = (bias_diff / 2.0).clip(upper=1.0)
    work["_score"] = (
        0.4 * work["_tone_cost"] + 0.3 * work["_price_cost"] + 0.3 * work["_bias_cost"]
    )
    return work.nsmallest(n, "_score")


def build_prompt_context(similar_users: pd.DataFrame) -> str:
    """Render matched cohort rows as a compact text block for prompt injection."""
    if similar_users is None or similar_users.empty:
        return "No similar users found in the persona signals dataset."

    lines = ["Cohort of similar real users (grounding signal):"]
    for _, row in similar_users.iterrows():
        cats = row.get("top_categories") or []
        if not isinstance(cats, list):
            cats = []
        cats_str = ", ".join(str(c) for c in cats[:3]) or "—"
        lines.append(
            f"- src={row.get('source', '?')} avg_stars={row.get('avg_stars', 0):.2f} "
            f"pct_5star={row.get('pct_5star', 0):.2f} tone={row.get('tone_proxy', '?')} "
            f"price={row.get('price_sensitivity', '?')} cats=[{cats_str}]"
        )

    mean_5 = float(similar_users["pct_5star"].mean()) if "pct_5star" in similar_users else 0.0
    mean_bias = float(similar_users["rating_bias"].mean()) if "rating_bias" in similar_users else 0.0
    lines.append(
        f"Cohort aggregate: mean_pct_5star={mean_5:.2f} mean_rating_bias={mean_bias:.2f}"
    )
    return "\n".join(lines)
