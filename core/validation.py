"""Pre-graph input validation: length / gibberish / keyboard-mash checks."""

from __future__ import annotations

import re


COMMON_WORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "in", "on", "at", "to", "of", "and",
    "or", "but", "if", "then", "with", "for", "by", "from", "as", "i", "you",
    "he", "she", "it", "we", "they", "my", "your", "his", "her", "our", "their",
    "me", "him", "us", "them", "who", "what", "where", "when", "why", "how",
    "loves", "likes", "hates", "eats", "watches", "reads", "prefers", "values",
    "writes", "rates", "lives", "works", "studies", "enjoys", "plays", "shops",
    "buys", "wants", "needs", "uses", "drinks", "cooks", "travels",
    "user", "person", "persona", "people", "food", "movie", "movies", "music",
    "book", "books", "restaurant", "blogger", "professional", "student",
    "lover", "fan", "reviewer", "consumer", "customer", "budget", "cheap",
    "expensive", "formal", "casual", "young", "old", "new", "male", "female",
    "reviews", "review", "rating", "ratings", "price", "quality", "year",
    "years", "thriller", "thrillers", "drama", "comedy", "action",
    "nigeria", "nigerian", "lagos", "abuja", "naija", "jollof", "suya",
    "nollywood", "afrobeats", "enugu", "ibadan", "kano", "kaduna", "ogun",
    "rivers", "port", "harcourt", "pidgin", "yoruba", "igbo", "hausa",
    "amala", "ofada", "fufu", "egusi", "akara", "onitsha",
}

_CONSONANT_RUN = re.compile(r"[bcdfghjklmnpqrstvwxyz]{4,}")
_STRIP_CHARS = ".,!?;:'\""


def _has_common_word(text: str) -> bool:
    return any(
        tok.lower().strip(_STRIP_CHARS) in COMMON_WORDS
        for tok in (text or "").split()
    )


def is_too_short(text: str, min_len: int) -> bool:
    return len((text or "").strip()) < min_len


def is_gibberish(text: str) -> bool:
    text = (text or "").strip()
    if not text or _has_common_word(text):
        return False
    tokens = text.split()
    if len(tokens) >= 3:
        short = sum(1 for t in tokens if len(t) < 3)
        if short / len(tokens) > 0.60:
            return True
    if _CONSONANT_RUN.search(text.lower()):
        return True
    return False


def is_keyboard_mash(text: str) -> bool:
    text = (text or "").strip()
    return len(text) > 8 and " " not in text and not _has_common_word(text)
