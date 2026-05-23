"""Build the Task B recommendation catalog.

Deterministic and idempotent. Emits a JSON object keyed by canonical domain,
each value a list of catalog items. Items carry a `text_blob` so the Phase-2
vector-index builder can embed them without re-deriving text.

The seed below is real, hand-curated, Nigerian-aware data across all twelve
supported domains plus a `general lifestyle` fallback. In Phase 2 the
`merge_external()` hook folds in IMDb / MovieLens / Amazon / Jumia-Jiji dumps
when their cleaned parquet files are present under `data_prep/raw/`.

Run:
    python -m data_pipeline.build_catalog
Writes:
    data_prep/artifacts/catalog.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

OUT_PATH = Path(__file__).resolve().parents[1] / "data_prep" / "artifacts" / "catalog.json"

NG = "nigerian"  # tag shorthand


def _item(
    title: str,
    categories: list[str],
    niches: list[str],
    tags: list[str] | None = None,
    price_naira: int | None = None,
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tags = tags or []
    attributes = attributes or {}
    text_blob = " | ".join(
        [title, ", ".join(categories), ", ".join(niches), ", ".join(tags)]
    )
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return {
        "item_id": slug,
        "title": title,
        "categories": categories,
        "niches": niches,
        "tags": tags,
        "price_naira": price_naira,
        "attributes": attributes,
        "text_blob": text_blob,
    }


# --------------------------------------------------------------------------- #
# Seed catalog — real items, Nigerian-aware where relevant.
# --------------------------------------------------------------------------- #

SEED: dict[str, list[dict[str, Any]]] = {
    "movies": [
        _item("The Wedding Party", ["comedy", "romance"], ["nollywood", "comedy"], [NG]),
        _item("King of Boys", ["crime", "drama", "political"], ["nollywood", "thriller"], [NG]),
        _item("Gangs of Lagos", ["crime", "action", "drama"], ["nollywood", "action"], [NG]),
        _item("Citation", ["drama"], ["nollywood", "drama"], [NG]),
        _item("Anikulapo", ["fantasy", "drama"], ["nollywood", "epic", "yoruba"], [NG]),
        _item("Jagun Jagun", ["action", "epic"], ["nollywood", "yoruba", "action"], [NG]),
        _item("Lionheart", ["drama", "family"], ["nollywood", "drama"], [NG]),
        _item("The Black Book", ["thriller", "action"], ["nollywood", "thriller"], [NG]),
        _item("Brotherhood", ["crime", "action"], ["nollywood", "action"], [NG]),
        _item("Oppenheimer", ["drama", "history", "biopic"], ["hollywood", "drama"]),
        _item("Dune: Part Two", ["sci-fi", "adventure"], ["hollywood", "sci-fi"]),
        _item("Inception", ["sci-fi", "thriller"], ["hollywood", "thriller"]),
        _item("Spider-Man: Across the Spider-Verse", ["animation", "action"], ["hollywood", "animation"]),
        _item("Parasite", ["thriller", "drama"], ["world cinema", "thriller"]),
        _item("Black Panther", ["action", "sci-fi"], ["hollywood", "action"]),
    ],
    "food": [
        _item("Jollof Rice", ["rice", "main"], ["nigerian", "party food"], [NG], 2500),
        _item("Egusi Soup with Pounded Yam", ["soup", "swallow"], ["nigerian", "igbo", "yoruba"], [NG], 3000),
        _item("Amala and Ewedu", ["swallow", "soup"], ["nigerian", "yoruba"], [NG], 2200),
        _item("Suya", ["grill", "snack"], ["nigerian", "hausa", "street food"], [NG], 1500),
        _item("Pepper Soup", ["soup"], ["nigerian", "spicy"], [NG], 2800),
        _item("Ofada Rice and Ayamase", ["rice", "main"], ["nigerian", "yoruba"], [NG], 3200),
        _item("Nkwobi", ["delicacy"], ["nigerian", "igbo"], [NG], 3500),
        _item("Moi Moi", ["beans", "side"], ["nigerian", "side"], [NG], 800),
        _item("Akara and Pap", ["breakfast"], ["nigerian", "breakfast"], [NG], 700),
        _item("Tuwo Shinkafa and Miyan Kuka", ["swallow", "soup"], ["nigerian", "hausa"], [NG], 2400),
        _item("Shawarma", ["wrap", "fast food"], ["street food"], [], 2000),
        _item("Chicken and Chips", ["fast food"], ["fast food"], [], 3000),
        _item("Fried Rice and Chicken", ["rice", "main"], ["nigerian", "party food"], [NG], 2800),
        _item("Catfish Pepper Soup", ["soup", "fish"], ["nigerian", "spicy"], [NG], 4000),
    ],
    "books": [
        _item("Things Fall Apart", ["fiction", "classic"], ["nigerian", "literary"], [NG]),
        _item("Half of a Yellow Sun", ["fiction", "historical"], ["nigerian", "literary"], [NG]),
        _item("Purple Hibiscus", ["fiction", "coming-of-age"], ["nigerian", "literary"], [NG]),
        _item("Americanah", ["fiction", "contemporary"], ["nigerian", "literary"], [NG]),
        _item("Stay With Me", ["fiction", "contemporary"], ["nigerian", "literary"], [NG]),
        _item("The Fishermen", ["fiction", "literary"], ["nigerian", "literary"], [NG]),
        _item("Born a Crime", ["memoir", "non-fiction"], ["african", "memoir"]),
        _item("Atomic Habits", ["self-help", "non-fiction"], ["productivity", "self-help"]),
        _item("Rich Dad Poor Dad", ["finance", "non-fiction"], ["finance", "self-help"]),
        _item("The Alchemist", ["fiction", "philosophical"], ["inspirational", "fiction"]),
        _item("Sapiens", ["non-fiction", "history"], ["history", "science"]),
        _item("Dune", ["sci-fi", "fiction"], ["sci-fi", "fantasy"]),
    ],
    "music": [
        _item("Essence — Wizkid ft. Tems", ["afrobeats"], ["afrobeats", "rnb"], [NG]),
        _item("Last Last — Burna Boy", ["afrobeats"], ["afrobeats"], [NG]),
        _item("Calm Down — Rema", ["afrobeats"], ["afrobeats", "pop"], [NG]),
        _item("Unavailable — Davido", ["afrobeats"], ["afrobeats"], [NG]),
        _item("City Boys — Burna Boy", ["afrobeats"], ["afrobeats"], [NG]),
        _item("Soso — Omah Lay", ["afrobeats"], ["afrobeats", "rnb"], [NG]),
        _item("Rush — Ayra Starr", ["afrobeats"], ["afrobeats", "pop"], [NG]),
        _item("Ojapiano — Kcee", ["afrobeats", "highlife"], ["afrobeats", "igbo"], [NG]),
        _item("Amapiano Mix", ["amapiano"], ["amapiano", "dance"], []),
        _item("Gospel Praise Medley", ["gospel"], ["gospel", "worship"], [NG]),
        _item("Fuji Classics — KWAM 1", ["fuji"], ["fuji", "yoruba"], [NG]),
        _item("Highlife Legends", ["highlife"], ["highlife", "igbo"], [NG]),
    ],
    "skincare": [
        _item("Shea Butter Body Cream", ["moisturizer"], ["natural", "dry skin"], [NG], 3500),
        _item("African Black Soap", ["cleanser"], ["natural", "acne"], [NG], 2000),
        _item("Niacinamide Serum", ["serum"], ["brightening", "oily skin"], [], 6500),
        _item("Vitamin C Serum", ["serum"], ["brightening", "anti-aging"], [], 8000),
        _item("Sunscreen SPF 50", ["sunscreen"], ["protection", "daily"], [], 7000),
        _item("Hyaluronic Acid Moisturizer", ["moisturizer"], ["hydration"], [], 9000),
        _item("Salicylic Acid Cleanser", ["cleanser"], ["acne", "oily skin"], [], 5500),
        _item("Cocoa Butter Lotion", ["moisturizer"], ["natural", "dry skin"], [NG], 3000),
        _item("Retinol Night Cream", ["treatment"], ["anti-aging"], [], 12000),
        _item("Aloe Vera Gel", ["soothing"], ["natural", "sensitive skin"], [], 2500),
    ],
    "hotel": [
        _item("Eko Hotels & Suites, Lagos", ["luxury", "5-star"], ["lagos", "business"], [NG], 150000),
        _item("Transcorp Hilton, Abuja", ["luxury", "5-star"], ["abuja", "business"], [NG], 180000),
        _item("The Wheatbaker, Lagos", ["boutique", "luxury"], ["lagos", "boutique"], [NG], 130000),
        _item("Radisson Blu, Lagos", ["luxury", "4-star"], ["lagos", "business"], [NG], 120000),
        _item("BON Hotel, Ibadan", ["midrange", "3-star"], ["ibadan", "budget"], [NG], 45000),
        _item("Whispering Palms Resort, Badagry", ["resort"], ["leisure", "beach"], [NG], 60000),
        _item("Obudu Mountain Resort", ["resort", "mountain"], ["leisure", "nature"], [NG], 75000),
        _item("Ibom Icon Hotel, Uyo", ["luxury", "resort"], ["leisure", "golf"], [NG], 90000),
        _item("Budget Guesthouse, Enugu", ["budget", "guesthouse"], ["enugu", "budget"], [NG], 20000),
    ],
    "travel": [
        _item("Lekki Conservation Centre", ["nature", "attraction"], ["lagos", "nature"], [NG]),
        _item("Idanre Hills", ["nature", "hiking"], ["ondo", "adventure"], [NG]),
        _item("Olumo Rock, Abeokuta", ["landmark", "hiking"], ["ogun", "history"], [NG]),
        _item("Yankari National Park", ["safari", "nature"], ["bauchi", "wildlife"], [NG]),
        _item("Zuma Rock, Abuja", ["landmark"], ["abuja", "sightseeing"], [NG]),
        _item("Tarkwa Bay Beach", ["beach"], ["lagos", "beach"], [NG]),
        _item("Erin Ijesha Waterfalls", ["nature", "waterfall"], ["osun", "nature"], [NG]),
        _item("Nike Art Gallery, Lagos", ["culture", "art"], ["lagos", "culture"], [NG]),
        _item("Calabar Carnival", ["festival", "culture"], ["cross river", "festival"], [NG]),
        _item("Obudu Cattle Ranch", ["resort", "mountain"], ["cross river", "nature"], [NG]),
    ],
    "fitness": [
        _item("Home HIIT Program", ["workout", "cardio"], ["weight loss", "home"], []),
        _item("Strength Training Plan", ["workout", "strength"], ["muscle gain", "gym"], []),
        _item("Yoga for Beginners", ["yoga", "flexibility"], ["wellness", "home"], []),
        _item("Lagos Running Club", ["running", "community"], ["lagos", "cardio"], [NG]),
        _item("Resistance Bands Set", ["equipment"], ["home", "strength"], [], 8000),
        _item("Adjustable Dumbbells", ["equipment"], ["gym", "strength"], [], 45000),
        _item("Pilates Mat Routine", ["pilates", "core"], ["wellness", "home"], []),
        _item("Marathon Training Plan", ["running", "endurance"], ["cardio", "advanced"], []),
        _item("Calisthenics Park Workout", ["bodyweight"], ["outdoor", "strength"], []),
    ],
    "tech": [
        _item("Tecno Camon 30", ["smartphone"], ["budget", "android"], [NG], 320000),
        _item("Infinix Hot 40 Pro", ["smartphone"], ["budget", "android"], [NG], 230000),
        _item("Samsung Galaxy A55", ["smartphone"], ["midrange", "android"], [], 480000),
        _item("iPhone 15", ["smartphone"], ["premium", "ios"], [], 1300000),
        _item("Oraimo FreePods", ["audio", "earbuds"], ["budget", "accessories"], [NG], 25000),
        _item("Anker Power Bank 20000mAh", ["accessories", "power"], ["essential", "portable"], [], 35000),
        _item("HP Pavilion Laptop", ["laptop"], ["productivity", "midrange"], [], 750000),
        _item("MacBook Air M3", ["laptop"], ["premium", "productivity"], [], 1900000),
        _item("Solar Inverter 1.5kVA", ["power", "home"], ["essential", "power backup"], [NG], 450000),
        _item("Smart TV 43-inch", ["tv", "home entertainment"], ["midrange", "home"], [], 320000),
    ],
    "fashion": [
        _item("Ankara Print Dress", ["dress", "traditional"], ["nigerian", "ankara"], [NG], 18000),
        _item("Agbada Set", ["traditional", "formal"], ["nigerian", "yoruba", "owambe"], [NG], 55000),
        _item("Aso Oke Cap", ["accessory", "traditional"], ["nigerian", "yoruba"], [NG], 12000),
        _item("Senator Wear", ["traditional", "formal"], ["nigerian", "menswear"], [NG], 35000),
        _item("Adire Shirt", ["casual", "traditional"], ["nigerian", "yoruba"], [NG], 15000),
        _item("Sneakers (Unisex)", ["footwear", "casual"], ["streetwear"], [], 25000),
        _item("Leather Slippers (Palm)", ["footwear", "traditional"], ["nigerian", "menswear"], [NG], 14000),
        _item("Denim Jacket", ["outerwear", "casual"], ["streetwear"], [], 22000),
        _item("Gele Headtie", ["accessory", "traditional"], ["nigerian", "owambe"], [NG], 9000),
        _item("Corporate Suit", ["formal", "menswear"], ["business", "formal"], [], 70000),
    ],
    "sport": [
        _item("Premier League Match Pass", ["football", "subscription"], ["football", "epl"], []),
        _item("Super Eagles Jersey", ["football", "merchandise"], ["nigerian", "football"], [NG], 25000),
        _item("Local Football Viewing Centre", ["football", "experience"], ["nigerian", "community"], [NG], 500),
        _item("Lawn Tennis Lessons", ["tennis", "coaching"], ["coaching", "racquet"], []),
        _item("Basketball Court Booking", ["basketball", "facility"], ["facility", "team"], []),
        _item("Table Tennis Set", ["equipment"], ["indoor", "racquet"], [], 35000),
        _item("Gym Membership (Monthly)", ["fitness", "subscription"], ["gym", "membership"], [], 20000),
        _item("Cycling Group Ride", ["cycling", "community"], ["outdoor", "cardio"], []),
    ],
    "drink": [
        _item("Zobo (Hibiscus Drink)", ["non-alcoholic", "traditional"], ["nigerian", "healthy"], [NG], 500),
        _item("Chapman Cocktail", ["cocktail", "non-alcoholic"], ["nigerian", "party"], [NG], 1500),
        _item("Palm Wine", ["alcoholic", "traditional"], ["nigerian", "traditional"], [NG], 1000),
        _item("Kunu", ["non-alcoholic", "traditional"], ["nigerian", "hausa", "healthy"], [NG], 400),
        _item("Star Lager Beer", ["beer", "alcoholic"], ["nigerian", "beer"], [NG], 800),
        _item("Smoov Chapman (Bottled)", ["non-alcoholic", "bottled"], ["nigerian", "party"], [NG], 700),
        _item("Fresh Coconut Water", ["non-alcoholic", "natural"], ["healthy", "natural"], [], 600),
        _item("Tigernut Drink (Kunu Aya)", ["non-alcoholic", "traditional"], ["nigerian", "healthy"], [NG], 800),
        _item("Espresso / Coffee", ["coffee", "hot drink"], ["cafe", "caffeine"], [], 1500),
        _item("Smoothie Bowl Drink", ["non-alcoholic", "healthy"], ["healthy", "fruit"], [], 2500),
    ],
    "general lifestyle": [
        _item("Spa Day Package", ["wellness", "self-care"], ["relaxation", "self-care"], []),
        _item("Weekend Getaway Plan", ["travel", "leisure"], ["leisure", "relaxation"], []),
        _item("Owambe Party Planning", ["event", "social"], ["nigerian", "owambe"], [NG]),
        _item("Home Organization Service", ["home", "service"], ["home", "lifestyle"], []),
        _item("Personal Styling Session", ["fashion", "service"], ["fashion", "self-care"], []),
        _item("Meal Prep Subscription", ["food", "service"], ["healthy", "convenience"], []),
        _item("Book Club Membership", ["books", "community"], ["reading", "community"], []),
        _item("Photography Session", ["creative", "service"], ["lifestyle", "creative"], []),
    ],
}


def _merge_grouped(
    catalog: dict[str, list[dict[str, Any]]],
    extra: dict[str, list[dict[str, Any]]],
) -> None:
    """In-place merge: append items by domain, skipping duplicate item_ids."""
    for domain, items in extra.items():
        existing = {i["item_id"] for i in catalog.get(domain, [])}
        catalog.setdefault(domain, []).extend(
            i for i in items if isinstance(i, dict) and i.get("item_id") not in existing
        )


def merge_external(catalog: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Fold real external datasets into the seed catalog.

    - **Jumia** (committed under ``datasets/``) is always merged when present.
    - ``data_prep/raw/<domain>.json`` (per-domain canonical-item lists from
      future IMDb / MovieLens / Amazon adapters) is merged when present.
    """
    # 1. Jumia -- real Nigerian product catalog, ~150 unique items.
    try:
        from data_pipeline.sources.jumia import canonical_items as jumia_items
        _merge_grouped(catalog, jumia_items())
    except Exception as exc:  # noqa: BLE001 -- catalog must keep building
        print(f"[build_catalog] jumia merge skipped: {exc}")

    # 2. Pre-cleaned raw drops (forward-compat for IMDb / MovieLens / Amazon).
    raw_dir = OUT_PATH.parents[1] / "raw"
    if raw_dir.exists():
        for f in sorted(raw_dir.glob("*.json")):
            domain = f.stem
            try:
                extra = json.loads(f.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if isinstance(extra, list):
                _merge_grouped(catalog, {domain: extra})
    return catalog


def build() -> dict[str, list[dict[str, Any]]]:
    catalog = {k: list(v) for k, v in SEED.items()}
    catalog = merge_external(catalog)
    return catalog


def main() -> None:
    catalog = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    n_items = sum(len(v) for v in catalog.values())
    print(f"Wrote {OUT_PATH} — {len(catalog)} domains, {n_items} items.")


if __name__ == "__main__":
    main()
