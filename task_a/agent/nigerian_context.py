"""Nigerian-context node — layers cultural / linguistic / market context.

Mirrors `task_b/agent/nigerian_context.py`. Keep both in sync.
"""

from __future__ import annotations

import re
from typing import Any

from task_a.agent.state import AgentState


_PIDGIN_MARKERS = [
    "abeg", "sha", "wahala", "no wahala", "e be like", "na", "wey",
    "small small", "chai", "oya", "abi", "biko",
]

_CULTURAL_VOCAB = [
    "jollof", "suya", "danfo", "agbada", "naija", "owambe", "ankara",
    "buka", "amala", "egusi", "puff-puff", "garri", "akara", "moi-moi",
    "fufu", "pepper soup", "ofada", "asaba", "yoruba", "igbo", "hausa",
]

_FOOD_BRANDS = [
    "Chicken Republic", "Mama Put", "Iya Basira", "Sweet Sensation",
    "The Place", "Tantalizers", "Mr Bigg's",
]


_YORUBA_KEYS = [
    "lagos", "ibadan", "ogun", "oyo", "osun", "ekiti", "ondo", "kwara",
    "surulere", "lekki", "ikeja", "yaba", "ife", "abeokuta", "ilorin",
]
_IGBO_KEYS = [
    "enugu", "anambra", "imo", "abia", "ebonyi", "delta", "rivers",
    "cross river", "port harcourt", "onitsha", "owerri", "nsukka", "aba",
]
_HAUSA_KEYS = [
    "abuja", "kano", "kaduna", "katsina", "sokoto", "zamfara", "kebbi",
    "jigawa", "bauchi", "gombe", "yobe", "borno", "adamawa", "taraba",
    "plateau", "nasarawa", "niger state", "benue", "kogi", "wuse", "maitama",
]


_REGIONAL_PALETTE: dict[str, dict[str, Any]] = {
    "yoruba": {
        "food_nouns": ["amala", "ewedu", "gbegiri", "moin moin", "akara",
                       "ofada", "egusi", "efo riro", "pounded yam"],
        "interjections": ["omo", "shey", "ah ah"],
        "praise": "e sweet well well",
        "disappointment": "ah, this one no try o",
    },
    "igbo": {
        "food_nouns": ["ofe onugbu", "oha soup", "nkwobi", "ugba", "abacha",
                       "ofe akwu", "ji"],
        "interjections": ["nna", "nne", "chai"],
        "praise": "o di mma well well",
        "disappointment": "chai, tufiakwa",
    },
    "hausa": {
        "food_nouns": ["suya", "kilishi", "tuwo shinkafa", "miyan kuka",
                       "masa", "fura da nono", "dambu nama"],
        "interjections": ["wallahi", "dan Allah", "nagode"],
        "praise": "wallahi this place try",
        "disappointment": "abin takaici",
    },
    "pidgin_only": {
        "food_nouns": [],
        "interjections": ["abeg", "oga", "no wahala", "no dull yourself"],
        "praise": "e sweet well well",
        "disappointment": "e don do, this one no try",
    },
}


def _matches_any(text: str, keys: list[str]) -> bool:
    for k in keys:
        if " " in k:
            if k in text:
                return True
        else:
            if re.search(rf"\b{re.escape(k)}\b", text):
                return True
    return False


def _detect_region(persona_text: str, fingerprint: dict[str, Any]) -> str:
    markers = {str(m).lower() for m in (fingerprint.get("nigerian_markers") or [])}
    for r in ("yoruba", "igbo", "hausa"):
        if r in markers:
            return r

    text = (persona_text or "").lower()
    if not text:
        return "pidgin_only"

    if _matches_any(text, _YORUBA_KEYS):
        return "yoruba"
    if _matches_any(text, _IGBO_KEYS):
        return "igbo"
    if _matches_any(text, _HAUSA_KEYS):
        return "hausa"
    return "pidgin_only"


def derive_nigerian_context(
    fingerprint: dict[str, Any],
    persona_text: str = "",
) -> dict[str, Any]:
    """Build the Nigerian-context dict from a fingerprint."""
    tone = str(fingerprint.get("tone", "balanced") or "balanced").lower()
    markers = fingerprint.get("nigerian_markers") or []

    if tone == "pidgin":
        register = "pidgin"
    elif markers or tone == "casual":
        register = "nigerian_english"
    else:
        register = "neutral"

    apply_ctx = (tone in {"pidgin", "casual"}) or bool(markers) or True

    language_region = _detect_region(persona_text, fingerprint)
    regional_palette = _REGIONAL_PALETTE.get(language_region, _REGIONAL_PALETTE["pidgin_only"])

    return {
        "language_register": register,
        "language_region": language_region,
        "currency_symbol": "₦",
        "pidgin_markers": list(_PIDGIN_MARKERS),
        "cultural_vocab": list(_CULTURAL_VOCAB),
        "food_brands": list(_FOOD_BRANDS),
        "regional_palette": regional_palette,
        "apply": apply_ctx,
    }


def nigerian_context_node(state: AgentState) -> AgentState:
    """Read `fingerprint` + `persona`; write `nigerian_context`, `nigerian_context_applied`, `language_region`."""
    fingerprint = state.get("fingerprint", {}) or {}
    persona_text = state.get("persona", "") or ""
    ctx = derive_nigerian_context(fingerprint, persona_text)
    state["nigerian_context"] = ctx
    state["nigerian_context_applied"] = bool(ctx["apply"])
    state["language_region"] = ctx["language_region"]
    return state
