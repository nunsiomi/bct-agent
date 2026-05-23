"""Quick verification that the OpenAI-compatible provider talks to Groq.

Reads `.env` (gitignored), then routes a single LLM call through
``core.llm.get_llm().complete()``. Prints only model id + first 200 chars of
the response so the API key never appears in logs.

Run:
    python scripts/smoke_groq.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE any core.* import so module-level config picks it up.
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Now safe to import the provider stack.
from core.config import LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER  # noqa: E402
from core.llm import get_llm  # noqa: E402


def main() -> None:
    print(f"provider={LLM_PROVIDER!r} model={LLM_MODEL!r} base_url={LLM_BASE_URL!r}")
    if LLM_PROVIDER == "anthropic":
        print("LLM_PROVIDER is anthropic; expected openai_compatible. Aborting.")
        sys.exit(1)
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set in environment. Did .env load correctly?")
        sys.exit(1)

    provider = get_llm()
    print(f"provider class: {type(provider).__name__}")

    resp = provider.complete(
        system="You are a precise assistant. Reply in <=20 words.",
        user="In one short sentence, name two popular Nigerian afrobeats songs.",
        max_tokens=80,
        temperature=0.2,
    )
    snippet = resp.replace("\n", " ")[:200]
    print("\n--- response (200 chars max) ---")
    print(snippet)
    print("--- ok ---")


if __name__ == "__main__":
    main()
