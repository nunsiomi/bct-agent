# BCT Hackathon — Two-Task LLM Agent System

## Overview

This project implements a two-task LLM agent system for the BCT Hackathon.

- **Task A — Persona-Driven Review Generation**: Given a Nigerian persona and a product, generate a culturally grounded review with rating, review text, and a confidence score.
- **Task B — Persona-Driven Domain Recommendation**: Given a Nigerian persona, a domain, and an optional niche, return ranked recommendations with reasons and match scores.

Both tasks share a common persona builder and Nigerian-context layer, but each runs as an independent FastAPI service in its own Docker container, orchestrated by LangGraph and powered by `claude-sonnet-4-20250514` via the Anthropic Python SDK.

## Stack

- FastAPI (HTTP layer)
- LangGraph (agent graph orchestration)
- Anthropic Python SDK (direct `anthropic.Anthropic()` client — no LangChain LLM wrappers), behind a swappable provider layer (`core/llm/`)
- pandas, numpy, scikit-learn for the data layer; Task B retrieval scores a committed item catalog (`data_prep/artifacts/catalog.json`)

## Project Structure

```
bct-agent/
├── docker-compose.yml
├── .env.example
├── README.md
├── docs/                  # Documentation — design notes, write-ups, diagrams
├── data_prep/             # Data preparation notebook(s) — drop your .ipynb here
├── shared/                # Code shared between Task A and Task B
├── task_a/                # Review generation service
└── task_b/                # Recommendation service
```

### `docs/`

Drop any project documentation here — design notes, architecture diagrams, evaluation reports, submission write-ups, etc.

### `data_prep/`

This is where the data-preparation notebook lives. The notebook is run **once, offline** to build any artifacts (cleaned CSVs, embeddings, indices) that the runtime services consume. Place your `.ipynb` here, and write generated artifacts into `data_prep/artifacts/` (gitignored if large). The Docker services do not run the notebook — they only read its outputs.

## Setup

1. Clone the repo and `cd` into it.
2. Copy `.env.example` to `.env` and set your Anthropic key:
   ```
   cp .env.example .env
   # then edit .env and set ANTHROPIC_API_KEY=sk-ant-...
   ```
3. The Task B catalog (`data_prep/artifacts/catalog.json`) is committed, so the
   services work on a fresh clone with no extra steps. To regenerate or extend it:
   ```
   python -m data_pipeline.build_catalog
   ```
4. Smoke-test the data + retrieval layer (no API key needed):
   ```
   python -m pytest tests/test_smoke.py -q
   ```

## Run with Docker Compose

Build and start both services:

```bash
docker compose up --build
```

- Task A is exposed on **http://localhost:8001**
- Task B is exposed on **http://localhost:8002**

Stop with `Ctrl+C` and `docker compose down`.

## Endpoints

### Task A — `POST /generate-review`

Request:

```json
{
  "persona": "32-year-old Lagos tech worker, budget-conscious, uses public transport",
  "product": "Tecno Spark 20 smartphone"
}
```

Response:

```json
{
  "rating": 4.2,
  "review": "...",
  "confidence": 0.87,
  "nigerian_context_applied": true
}
```

Curl example:

```bash
curl -X POST http://localhost:8001/generate-review \
  -H "Content-Type: application/json" \
  -d '{
    "persona": "32-year-old Lagos tech worker, budget-conscious, uses public transport",
    "product": "Tecno Spark 20 smartphone"
  }'
```

### Task B — `POST /recommend`

Request:

```json
{
  "persona": "28-year-old Abuja medical student, vegetarian, loves Afrobeats",
  "domain": "music",
  "niche": "afrobeats"
}
```

Response:

```json
{
  "recommendations": [
    {"rank": 1, "title": "...", "reason": "...", "match_score": 0.93}
  ],
  "fallback_used": false
}
```

Curl example:

```bash
curl -X POST http://localhost:8002/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "persona": "28-year-old Abuja medical student, vegetarian, loves Afrobeats",
    "domain": "music",
    "niche": "afrobeats"
  }'
```

`niche` is optional — pass `null` or omit it to let the agent infer or ask for clarification.

## Local Development (without Docker)

Each service is a plain FastAPI app. From the repo root:

```bash
# Task A
pip install -r task_a/requirements.txt
uvicorn task_a.main:app --reload --port 8001

# Task B
pip install -r task_b/requirements.txt
uvicorn task_b.main:app --reload --port 8002
```

Make sure `ANTHROPIC_API_KEY` is set in your environment.
