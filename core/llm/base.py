"""LLM provider protocol.

A provider takes (system, user, opts) and returns the model's text response.
JSON-structured output is implemented at the provider level when supported,
or fenced/free-text JSON the caller parses via `core.json_utils.parse_json_block`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Return the assistant's text response for a single-turn message."""
        ...
