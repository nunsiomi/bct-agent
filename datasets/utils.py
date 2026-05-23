"""Deprecated shim. Re-exports from ``core`` for backward compatibility.

New code MUST import from ``core`` directly:
    from core.llm import call_claude, get_anthropic_client, get_llm
    from core.json_utils import parse_json_block, warn
    from core.persona_signals import (
        load_persona_signals, find_similar_users, build_prompt_context,
    )
"""

from core.json_utils import parse_json_block, warn  # noqa: F401
from core.llm import call_claude, get_anthropic_client, get_llm  # noqa: F401
from core.persona_signals import (  # noqa: F401
    build_prompt_context,
    find_similar_users,
    load_persona_signals,
)

# Legacy constant kept for any caller that still reads it.
from core.config import LLM_MODEL as CLAUDE_MODEL  # noqa: F401
