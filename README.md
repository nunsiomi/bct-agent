# BCT Agent — A two-task LLM agent system for Nigerian users

A FastAPI + LangGraph system that simulates how real Nigerian users review and discover products. Built for the **DSN × BCT Hackathon 3.0** with a strong bias toward *real data, real ground truth, and an honest agentic architecture* — no synthetic catalogs, no self-referential metrics, no shoehorned Pidgin.

> **Live demo:** [https://frontend.lemonbay-7e904144.eastus.azurecontainerapps.io](https://frontend.lemonbay-7e904144.eastus.azurecontainerapps.io)
> **Headline numbers (n=100 held-out Jumia reviews):** Task A RMSE **0.653 (−54%)**, MAE **0.307 (−72%)**, ROUGE-L **0.071 (+37%)** vs offline baseline. See [docs/SOLUTION_PAPER.md](docs/SOLUTION_PAPER.md) for the full evaluation.

---

## What it does

| Task | Endpoint | Input | Output |
|---|---|---|---|
| **A. Persona-driven review generation** | `POST /generate-review` | A free-text Nigerian persona + a product/item | Rating (1–5), persona-voiced review text, confidence, and whether Nigerian context was applied |
| **B. Persona-driven domain recommendation** | `POST /recommend` | Persona + domain + optional niche | Ranked top-5 recommendations with personalised reasons and match scores |

Both tasks share a common `core/` package (persona builder, Nigerian-context layer, LLM/embedding abstraction, hybrid retriever) but run as two independent FastAPI services. The default model is **Claude Sonnet 4** (`claude-sonnet-4-20250514`) — but the LLM provider is swappable to **Groq Llama-3.3-70b** (or any OpenAI-compatible endpoint) via a single env-var flip.

### What's "agentic" about it

Both tasks are LangGraph state machines, not single LLM calls.

**Task A** (`task_a/agent/graph.py`):
```
persona_builder → nigerian_context → history_grounding → review_generator
    → critique --(passes | budget spent)--> quality_checker → END
              \--(fails + budget left)----> revise → critique  (back-edge)
```
- `history_grounding` retrieves 3 real exemplar reviews from a 15.2k-row Jumia user-history corpus.
- `critique` is heuristic (zero LLM calls) — checks length, product mention, Nigerian-marker landing, sentiment↔rating consistency, and rating-prior alignment.
- `revise` is an LLM re-prompt with the critique's issue list; capped at `max_revisions=1` so the loop always terminates.

**Task B** (`task_b/agent/graph.py`):
```
persona_builder → nigerian_context → domain_resolver → domain_validator
    |--(invalid)--> clarification --> retrieval
    |--(valid)----> retrieval → reasoning_ranker → verify → END
```
- `retrieval` is a `HybridRetriever` — TF-IDF + BM25 fused with Reciprocal Rank Fusion (k=30 pool).
- `verify` drops any title not in the catalog (hallucination guard), backfills from real candidates if needed.

---

## Quickstart (3 minutes)

```bash
# 0) clone and cd
git clone <repo>
cd bct-agent

# 1) install (one-time)
pip install -r task_a/requirements.txt

# 2) configure
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-... (or use the Groq block in the file)

# 3) build the data artifacts (~30 sec, idempotent — only catalog.json is committed)
python -m data_pipeline.build_catalog
python -m data_pipeline.build_user_histories
python -m data_pipeline.build_vector_index

# 4) confirm everything is wired (offline, no API key needed)
python -m pytest tests/ -q
# expected: 36 passed
```

Now you can either run the **demo UI** (3 terminals — Path A below) or just hit the **APIs directly** (Path B).

### Path A — full local demo (UI at `localhost:3000`)

```powershell
# Terminal 1
uvicorn task_a.main:app --port 8001

# Terminal 2
uvicorn task_b.main:app --port 8002

# Terminal 3
python -m http.server 3000 --directory frontend
# browse http://localhost:3000
```

### Path B — APIs only

```powershell
uvicorn task_a.main:app --port 8001
# (in another terminal)
uvicorn task_b.main:app --port 8002

# Task A
$bodyA = @{ persona = "Lagos office worker, mid-budget, casual"
            product = "Roushun Vitamin C Serum 30ml" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8001/generate-review -Method Post -ContentType "application/json" -Body $bodyA

# Task B
$bodyB = @{ persona = "Lagos professional, loves Afrobeats"
            domain  = "music"
            niche   = "afrobeats" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8002/recommend -Method Post -ContentType "application/json" -Body $bodyB
```

### Path C — no LLM at all, just the offline baseline + tests

If you don't have an API key but want to verify the data pipeline:

```bash
python -m pytest tests/ -q                         # 36 offline tests
python -m eval.run_eval --n 50                     # offline baseline metrics
```

---

## Data foundation

| Source | Use | Volume |
|---|---|---|
| **Jumia Nigeria** (scraped by team, `webscrapping/jumia_scrapper.py`) | Product catalog, user histories, held-out ground truth | **15,243 reviews, 8,092 reviewers, 155 products** across cookware/electronics/food/phones/skincare |
| **Yelp / Amazon / Goodreads** (via `data_prep/bct-dataprep.ipynb`) | Persona behavioural signals (`avg_stars`, tone proxy, price sensitivity, top categories) | aggregated into `datasets/persona_signals.csv` |
| **Curated seed catalog** | Items in domains Jumia doesn't cover (movies, books, music, hotel, travel, fitness, fashion, sport, drink, general lifestyle) | 137 items |

After running `python -m data_pipeline.build_catalog`, `catalog.json` contains **287 items across 13 domains** (seed + Jumia merged).

### Evaluation against real ground truth

`data_pipeline/build_eval_holdout.py` does a **chronological temporal split** of the Jumia corpus:

- **Train:** 12,193 rows (2019-11-25 → 2026-02-15)
- **Holdout:** 3,050 rows (2026-02-16 → 2026-05-22)

For Task B, only holdout rows with `rating ≥ 4` count as relevant positives (canonical recsys threshold — predicting an item the user *hated* shouldn't be punished as a miss).

```bash
# Offline baseline (no LLM, deterministic, ~5 sec)
python -m eval.run_eval --n 200 --min-relevance-rating 4

# Full agent in-process (uses whichever provider .env selects; needs API key)
python -m eval.run_eval --in-process --use-reviewer-id --min-relevance-rating 4 --n 100

# Or hit running FastAPI services (live mode)
python -m eval.run_eval --live --n 30
```

Results land in `data_prep/artifacts/evaluation_results.json` (or wherever `--out` points), and the live frontend's **Held-Out Benchmark** panel fetches the latest run from `GET /eval_metrics`.

---

## Provider — Groq

The LLM is abstracted behind `core.llm.LLMProvider`:

```bash
# Groq (or any OpenAI-compatible endpoint)
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
LLM_API_KEY_ENV=GROQ_API_KEY
GROQ_API_KEY=gsk_...
```

See [.env.example](.env.example) for both blocks plus a Together AI example.

Embeddings are likewise abstracted (`core.embeddings`) — TF-IDF is the default; opting into sentence-transformers requires `pip install sentence-transformers` and `EMBEDDING_BACKEND=sentence_transformer`.

---

## Run with Docker Compose

```bash
docker compose up --build
```

- Task A → http://localhost:8001
- Task B → http://localhost:8002
- Frontend → http://localhost:3000

The Dockerfiles bake the data artifacts (catalog + vector index + user histories) into the image, so containers start ready — no runtime data-prep step.

---

## API reference

### `POST /generate-review` (Task A, port 8001)

```jsonc
// Request
{
  "persona": "32-year-old Lagos tech worker, budget-conscious, uses public transport",
  "product": "Tecno Spark 20 smartphone"
}

// Response
{
  "rating": 4.2,
  "review": "This Spark 20 dey try sha, screen sharp, battery hold up to evening ...",
  "confidence": 0.87,
  "nigerian_context_applied": true
}
```

### `POST /recommend` (Task B, port 8002)

```jsonc
// Request
{
  "persona": "28-year-old Abuja medical student, vegetarian, loves Afrobeats",
  "domain": "music",
  "niche": "afrobeats"  // optional; null/missing lets the agent infer or ask for clarification
}

// Response
{
  "recommendations": [
    { "rank": 1, "title": "Essence — Wizkid ft. Tems", "reason": "...", "match_score": 0.93 },
    { "rank": 2, "title": "Calm Down — Rema",          "reason": "...", "match_score": 0.88 }
    // ... up to 5
  ],
  "fallback_used": false
}
```

### `GET /eval_metrics` (Task A, port 8001)

Returns the most-recent held-out evaluation results (read by the frontend's Held-Out Benchmark panel). No payload, no auth.

### `GET /health`

Liveness probe on both services.

---

## Project structure

```
bct-agent/
├── core/                      # framework-agnostic building blocks
│   ├── llm/                   #   provider abstraction (Anthropic + OpenAI-compatible)
│   ├── embeddings/            #   TF-IDF (default) + sentence-transformers (opt-in)
│   ├── retrieval/             #   HybridRetriever (TF-IDF + BM25 + RRF)
│   ├── persona_builder.py     #   free text → structured fingerprint
│   ├── nigerian_context.py    #   region detection + regional palettes
│   ├── history_grounding.py   #   Phase 5: real-exemplar retrieval over Jumia
│   ├── schemas.py, config.py, validation.py, json_utils.py
│   └── persona_signals.py     #   loads the Yelp/Amazon/Goodreads signals CSV
├── datasets/                  # DATA ONLY
│   ├── persona_signals.csv    #   built by the notebook from Yelp/Amazon/Goodreads
│   ├── jumia_reviews_*.csv    #   5 files, 15.2k rows total
│   ├── utils.py, validation.py #  deprecated re-export shims → core.*
├── data_pipeline/             # offline data build (run once, idempotent)
│   ├── build_catalog.py
│   ├── build_user_histories.py
│   ├── build_eval_holdout.py
│   ├── build_vector_index.py
│   └── sources/jumia.py       #   canonical-item + user-history adapter
├── data_prep/
│   ├── artifacts/             #   catalog.json (committed), vector_index.npz, etc.
│   └── bct-dataprep.ipynb     #   Yelp/Amazon/Goodreads → persona_signals.csv
├── task_a/                    # Review generation service
│   ├── main.py                #   FastAPI app, /generate-review + /eval_metrics
│   └── agent/                 #   persona_builder, nigerian_context, history_grounding,
│                              #     review_generator, critique, revise, quality_checker
├── task_b/                    # Recommendation service
│   ├── main.py                #   FastAPI app, /recommend
│   └── agent/                 #   persona_builder, nigerian_context, domain_resolver,
│                              #     domain_validator, retrieval, reasoning_ranker, verify
├── eval/                      # ground-truth metric harness
│   ├── run_eval.py            #   --mode offline | live | in_process
│   ├── metrics.py             #   ROUGE-L, RMSE, NDCG@k, HR@k, MRR (pure-Python)
│   ├── persona_reconstruction.py
│   └── retrieval_baseline.py
├── frontend/                  # static demo UI (vanilla HTML/JS + nginx)
├── tests/                     # 36 offline tests (no API key needed)
├── scripts/                   # ad-hoc tools (smoke_groq, diag_task_b, demo_languages)
├── webscrapping/              # the Playwright scraper that produced the Jumia CSVs
├── docs/
│   ├── SOLUTION_PAPER.md      # READ THIS — full writeup with numbers + diagnosis
│   └── OVERHAUL_PLAN.md       # design doc the codebase implements
├── docker-compose.yml         # local 3-service stack (task_a + task_b + nginx frontend)
├── task_a/Dockerfile, task_b/Dockerfile, frontend/Dockerfile
└── .env.example
```

---

## Tests

**36/36 passing offline.** No API key, no network, no LLM calls (all mocked).

```bash
python -m pytest tests/ -q
```

| File | Covers |
|---|---|
| `tests/test_smoke.py` | Phase 0 — catalog committed, retrieval non-empty per domain, ranker fallback |
| `tests/test_phase4.py` | Hybrid retriever across all domains, verify node, full Task B graph end-to-end with mocked LLM |
| `tests/test_phase5.py` | History grounding primitives, known-reviewer + cold-start paths, full Task A graph |
| `tests/test_phase6.py` | Critique heuristics (length / product / Nigerian / sentiment / prior), conditional-edge routing, full critique→revise loop with mocked LLM |

---

## Deployment

Already live on Azure under the `Nunsi.Shiaki@studentambassadors.com` subscription:

- Resource group: `bct-agent-rg` (eastus)
- ACR: `bctagented31c90e.azurecr.io` — images tagged `:phase6` + `:latest`
- 3 Container Apps: `task-a`, `task-b` (internal ingress), `frontend` (public, nginx reverse-proxies `/api/task-a` and `/api/task-b`)

Public URL: **[https://frontend.lemonbay-7e904144.eastus.azurecontainerapps.io](https://frontend.lemonbay-7e904144.eastus.azurecontainerapps.io)**

To redeploy after code changes:
```bash
az acr build --registry bctagented31c90e --image bct-agent-task_a:phase6 --image bct-agent-task_a:latest -f task_a/Dockerfile .
az acr build --registry bctagented31c90e --image bct-agent-task_b:phase6 --image bct-agent-task_b:latest -f task_b/Dockerfile .
az acr build --registry bctagented31c90e --image bct-agent-frontend:phase6 --image bct-agent-frontend:latest -f frontend/Dockerfile .

az containerapp update -n task-a -g bct-agent-rg --image bctagented31c90e.azurecr.io/bct-agent-task_a:phase6
az containerapp update -n task-b -g bct-agent-rg --image bctagented31c90e.azurecr.io/bct-agent-task_b:phase6
az containerapp update -n frontend -g bct-agent-rg --image bctagented31c90e.azurecr.io/bct-agent-frontend:phase6
```

Each update spins a new revision in ~30 s with zero downtime.

---

## Want to go deeper?

- **[docs/SOLUTION_PAPER.md](docs/SOLUTION_PAPER.md)** — the writeup. Architecture, methods, experiments, per-domain analysis, honest limitations.
- **[docs/OVERHAUL_PLAN.md](docs/OVERHAUL_PLAN.md)** — the design doc this codebase implements (Phases 0–9; we landed 0–6).
- **[scripts/diag_task_b.py](scripts/diag_task_b.py)** — runs one held-out tech row through Task B end-to-end and prints retrieval / ranker / verify state. Useful for understanding *why* Task B is harder on tech than on skincare.
- **[scripts/demo_languages.py](scripts/demo_languages.py)** — same product through four personas (Pidgin / Yoruba / Igbo / Hausa) to show the regional code-switching live.
