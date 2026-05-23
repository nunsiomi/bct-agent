"""LLM provider abstraction.

Use `get_llm()` from `core.llm.registry`. The legacy helpers
`get_anthropic_client()` and `call_claude()` are kept here as thin shims so
existing nodes keep working until they are migrated to the provider API.
"""

from core.llm.base import LLMProvider
from core.llm.registry import call_claude, get_anthropic_client, get_llm

__all__ = ["LLMProvider", "call_claude", "get_anthropic_client", "get_llm"]
