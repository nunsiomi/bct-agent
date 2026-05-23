"""OpenAI-compatible provider: works with vLLM, Ollama, Groq, Together, etc.

Configure with env vars:
    LLM_PROVIDER=openai_compatible
    LLM_MODEL=<model id at the endpoint>
    LLM_BASE_URL=<http(s)://host:port/v1>
    LLM_API_KEY_ENV=<env var holding the API key>   # default: OPENAI_API_KEY

Uses the official `openai` SDK if installed (it accepts a `base_url`). We
import lazily so this file is safe to ship without the `openai` dep.
"""

from __future__ import annotations

import os
from typing import Any

from core.config import LLM_API_KEY_ENV, LLM_BASE_URL, LLM_MODEL


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        self.model = model or LLM_MODEL
        self.base_url = base_url or LLM_BASE_URL
        env_key = api_key_env or LLM_API_KEY_ENV or "OPENAI_API_KEY"
        api_key = os.environ.get(env_key, "")
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "openai SDK not installed; pip install openai to use OpenAICompatibleProvider"
            ) from exc
        self._client: Any = OpenAI(api_key=api_key or "EMPTY", base_url=self.base_url)

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
