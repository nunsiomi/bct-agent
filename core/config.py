"""Typed configuration. Single source of truth for paths and model selection.

Read from environment with sensible defaults. The runtime services and the
offline data pipeline both import paths from here -- no hardcoded literals.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Data paths.
DATASETS_DIR = ROOT / "datasets"
DATA_PREP_DIR = ROOT / "data_prep"
ARTIFACTS_DIR = DATA_PREP_DIR / "artifacts"
RAW_DIR = DATA_PREP_DIR / "raw"

CATALOG_PATH = ARTIFACTS_DIR / "catalog.json"
PERSONA_SIGNALS_PATH = DATASETS_DIR / "persona_signals.csv"
USER_HISTORIES_PATH = ARTIFACTS_DIR / "user_histories.parquet"
EVAL_HOLDOUT_PATH = ARTIFACTS_DIR / "eval_holdout.parquet"
VECTOR_INDEX_PATH = ARTIFACTS_DIR / "vector_index.npz"

# LLM selection.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL")  # for openai_compatible / Groq / Together / Ollama
LLM_API_KEY_ENV = os.environ.get("LLM_API_KEY_ENV", "ANTHROPIC_API_KEY")

# Embeddings.
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "tfidf").lower()  # tfidf | sentence_transformer
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
