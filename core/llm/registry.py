"""Provider selection + backward-compatible legacy shims.

`get_llm()` returns the configured provider. The `call_claude()` /
`get_anthropic_client()` shims preserve the old call sites in nodes that
have not been migrated yet -- they route through `get_llm()` so swapping
providers via `LLM_PROVIDER=openai_compatible` works without code changes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import anthropic

from core.config import LLM_PROVIDER
from core.llm.base import LLMProvider


@lru_cache(maxsize=1)
def get_llm() -> LLMProvider:
    """Return a process-wide singleton provider per current env config."""
    if LLM_PROVIDER == "anthropic":
        from core.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider()
    if LLM_PROVIDER in ("openai_compatible", "openai", "groq", "together", "ollama", "vllm"):
        from core.llm.openai_compatible import OpenAICompatibleProvider
        return OpenAICompatibleProvider()
    raise ValueError(f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}")


# --------------------------------------------------------------------------- #
# Backward-compatible shims for existing call sites.
# --------------------------------------------------------------------------- #

def get_anthropic_client() -> anthropic.Anthropic:
    """Legacy: return the Anthropic SDK client (only if anthropic provider in use).

    Most call sites should migrate to `get_llm().complete(...)` directly.
    """
    provider = get_llm()
    client = getattr(provider, "client", None)
    if client is None:
        # Fall back to a fresh client so old code that bypasses the provider works.
        return anthropic.Anthropic()
    return client


def call_claude(
    client: Any,  # accepted for backward compatibility; unused by the provider path
    system: str,
    user: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """Legacy: route through the configured provider regardless of the passed client."""
    provider = get_llm()
    return provider.complete(system=system, user=user, max_tokens=max_tokens, temperature=temperature)
