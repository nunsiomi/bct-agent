"""Core package: framework-agnostic building blocks shared by Task A and Task B.

Layout
------
- `core.config`            -- typed settings (model id, provider, paths)
- `core.schemas`           -- Pydantic models for persona, recommendations, reviews
- `core.validation`        -- pre-graph input validation (gibberish / mash / length)
- `core.json_utils`        -- tolerant JSON parsing for LLM responses
- `core.persona_signals`   -- load / match / format the persona-signals dataset
- `core.persona_builder`   -- shared persona_builder node (deduped)
- `core.nigerian_context`  -- shared Nigerian-context node (deduped)
- `core.llm.*`             -- LLMProvider abstraction (Anthropic + OpenAI-compatible)
- `core.embeddings.*`      -- EmbeddingProvider abstraction (TF-IDF default, ST optional)

`datasets/` is intentionally data-only (CSVs). All shared code lives here.
"""
