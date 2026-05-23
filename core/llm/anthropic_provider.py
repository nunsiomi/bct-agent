"""Anthropic provider implementation."""

from __future__ import annotations

import anthropic

from core.config import LLM_MODEL


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or LLM_MODEL
        self._client = anthropic.Anthropic()

    @property
    def client(self) -> anthropic.Anthropic:
        return self._client

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        parts = [block.text for block in resp.content if getattr(block, "type", None) == "text"]
        return "".join(parts).strip()
