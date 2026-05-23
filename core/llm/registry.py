"""Provider selection + backward-compatible legacy shims.

`get_llm()` returns the configured provider. The `call_claude()` /
`get_anthropic_client()` shims preserve the old call sites in nodes that
have not been migrated yet -- they route through `get_llm()` so swapping
providers via `LLM_PROVIDER=openai_compatible` works without code changes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

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

def get_anthropic_client() -> Any:
    """Legacy: return the Anthropic SDK client when the Anthropic provider is in use.

    When a non-Anthropic provider is active (e.g. Groq via openai_compatible),
    return None -- the ``call_claude`` shim ignores the ``client`` arg and
    routes through ``get_llm().complete()`` regardless.
    """
    provider = get_llm()
    client = getattr(provider, "client", None)
    if client is not None:
        return client
    try:
        import anthropic
        return anthropic.Anthropic()
    except Exception:  # noqa: BLE001 -- missing key / SDK not configured
        return None


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
