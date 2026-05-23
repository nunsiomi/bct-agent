"""Deprecated shim. Re-exports from ``core.validation``.

New code MUST import from ``core.validation`` directly.
"""

from core.validation import (  # noqa: F401
    COMMON_WORDS,
    is_gibberish,
    is_keyboard_mash,
    is_too_short,
)
