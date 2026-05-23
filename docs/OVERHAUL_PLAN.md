# BCT Agent — Complete Overhaul Plan

> Goal: turn the current two-service prototype into a system that **actually works** — grounded in real data, with real retrieval, real agentic loops, real multi-turn, real evaluation against ground truth, a swappable open-source model layer, and a deepened Nigerian-context capability. Timeline is intentionally ignored; this is the "what it should be" target. Phasing at the end lets it be executed incrementally.

This plan is written against the DSN × BCT rubric:

- **Task A** — Review Text Quality (ROUGE/BERTScore), Rating Accuracy (RMSE), Behavioural Fidelity (human eval), Solution Paper, Code Reproducibility.
- **Task B** — Ranking Quality NDCG@10/Hit Rate (30), Cold-Start & Cross-Domain (25), Contextual Relevance human eval (20), Solution Paper (15), Code Reproducibility (10).
- **Bonus** — sounds/behaves Nigerian.
- **Cross-cutting** — the *paper is read first*; experiments + ablations are the talent signal.

---

## 0. Design principles

1. **Ground everything in real data.** No hand-authored catalogs, no synthetic confidence. Every output traces back to real Yelp/Amazon/Goodreads/IMDb/MovieLens/Nigerian data.
2. **Measure against ground truth, not against ourselves.** The current eval is self-referential (NDCG from the model's own scores). Replace with held-out real interactions.
3. **Model-agnostic core.** Claude is one provider behind an interface; open-source models (Qwen2.5, Llama-3.1, embeddings) are first-class and benchmarked head-to-head.
4. **Truly agentic.** Reason → act → verify → (re)act loops, not linear pipelines. Multi-turn with real session state.
5. **Reproducible by construction.** A clean clone + one command must produce identical artifacts and a working demo. No gitignored runtime dependencies.
6. **Nigerian context is a feature, not a sprinkle.** Region-aware generation, Naija data grounding, and an optional fine-tune.

---

## 1. Target architecture

### 1.1 Repository restructure (kill the duplication)

Current state duplicates `persona_builder.py` and `nigerian_context.py` across `task_a/` and `task_b/`. Consolidate into a real shared core:

```
bct-agent/
├── core/                       # the brain, framework-agnostic
│   ├── llm/                    # provider abstraction
│   │   ├── base.py             # LLMProvider protocol
│   │   ├── anthropic_provider.py
│   │   ├── openai_compatible.py  # vLLM / Ollama / Together / Groq
│   │   └── registry.py         # provider selection from config/env
│   ├── embeddings/             # embedding provider abstraction + cache
│   ├── persona/                # persona_builder + fingerprint schema (shared)
│   ├── nigerian/               # nigerian_context, regional palettes, detection
│   ├── retrieval/              # vector index, hybrid retriever, rerankers
│   ├── schemas.py              # Pydantic models for ALL I/O + state
│   └── config.py               # typed settings (pydantic-settings)
├── data_pipeline/              # offline: ingest → clean → embed → build artifacts
│   ├── sources/                # one module per dataset (yelp, amazon, goodreads, imdb, movielens, naija)
│   ├── build_catalog.py
│   ├── build_user_histories.py
│   ├── build_eval_holdout.py
│   └── build_vector_index.py
├── task_a/                     # thin FastAPI service → core
├── task_b/                     # thin FastAPI service → core
├── eval/                       # real metric harness + human-eval tooling
├── artifacts/                  # built data (committed or DVC-tracked, NOT gitignored away)
├── paper/                      # solution paper, figures, ablation tables
└── docker-compose.yml          # services + model server + vector db
```

**Acceptance:** `persona_builder` / `nigerian_context` exist in exactly one place; both services import from `core/`.

### 1.2 Typed state and contracts

Replace loose `dict`-based LangGraph state and ad-hoc parsing with **Pydantic schemas** for: `PersonaFingerprint`, `NigerianContext`, `Candidate`, `Recommendation`, `ReviewDraft`, plus per-task request/response. Use Anthropic / OpenAI **structured-output / tool-calling** modes so JSON parsing failures (currently handled by a hand-rolled brace-matcher in `shared/utils.py`) become impossible.

**Acceptance:** zero `parse_json_block` fallbacks needed; malformed model output is impossible by construction or retried with a schema-repair step.

---

## 2. Data foundation (the part that makes it "actually work")

The single biggest gap today: Task B's `catalog.json` is hand-built and **not committed**, and there is no real ground truth for evaluation. Fix the foundation first.

### 2.1 Datasets to ingest

| Source | Use | Task | License note |
|---|---|---|---|
| **Yelp Open Dataset** | businesses + reviews + user histories | A & B (food/restaurants) | research use, disclose |
| **Amazon Reviews 2023 (McAuley)** | products + reviews + ratings | A & B (tech/fashion/etc.) | research use |
| **Goodreads (UCSD)** | books + reviews + shelves | A & B (books) | research use |
| **IMDb official datasets** | movie catalog + aggregate ratings | B (movies) | non-commercial |
| **MovieLens (GroupLens)** | user→item→rating ground truth | B eval (movies) | research use |
| **NaijaSenti / Masakhane** | Naija/Pidgin/Yoruba/Igbo/Hausa text | A voice + fine-tune | research use |
| **Jumia/Jiji community dumps** *(optional)* | Nigerian products + ₦ prices | B catalog + A grounding | use existing published dumps, not self-scraped; disclose |

> Jumia/Jiji: prefer existing Kaggle/HF dumps over self-scraping (ToS + reliability). Inspect quality before trusting; treat as enrichment, not a dependency.

### 2.2 Ingestion pipeline (`data_pipeline/`)

Deterministic, reproducible, idempotent. Each source module exposes `download()`, `clean()`, `to_canonical()`:

1. **Canonical item schema** across domains: `{item_id, domain, title, categories[], niches[], tags[], price_naira?, attributes{}, popularity, avg_rating, text_blob}`.
2. **Canonical user-history schema**: `{user_id, source, reviews:[{item_id, rating, text, ts}], derived_signals}` — this replaces the aggregate-only `persona_signals.csv` so Task A can ground on a real user's **actual past reviews**, not just cohort stats.
3. **`build_catalog.py`** → `artifacts/catalog/<domain>.parquet` + a unified item table.
4. **`build_user_histories.py`** → `artifacts/user_histories/` (sampled, deduped).
5. **`build_eval_holdout.py`** → held-out (user, item, true_rating, true_review_text) for Task A RMSE/ROUGE/BERTScore, and held-out user→item interactions for Task B NDCG/Hit-Rate. **Strict temporal or user-level split** to avoid leakage.
6. **`build_vector_index.py`** → embeds every catalog item, persists a FAISS/Chroma index + the embedding model id/version.

**Reproducibility:** artifacts are either committed (if small) or **DVC/Git-LFS tracked** with a manifest of source URLs + checksums + the exact build command. A `make data` target rebuilds everything from scratch. The "no catalog on clean clone" failure must be structurally impossible.

**Acceptance:** `git clone && make data && docker compose up` yields a working Task B with real recommendations.

---

## 3. Task A overhaul — User Modeling that's actually grounded

Today: a free-text persona → heuristic cohort of similar users (aggregate stats only) → single-shot generation → canned-string fallback. No real user history, no reflection, magic-number confidence.

### 3.1 Real user-history grounding (Retrieval-Augmented Generation)
- Given a persona, **match to real users** in the history store (semantic + signal match), and retrieve their **actual review texts** for similar items.
- Inject 2–4 real exemplar reviews (style, length, rating pattern) as few-shot grounding — this is what moves *behavioural fidelity* and ROUGE/BERTScore against real references.
- For the "unseen item" simulation in the brief: retrieve the user's history + the **item's metadata + other users' reviews of that item**, then simulate how *this* user would rate/review it.

### 3.2 Agentic generate→critique→revise loop
Replace the linear pipeline + canned fallback with a reflection loop:
```
persona_builder → context → retrieve_grounding → draft → self_critique → (revise ↺ up to N) → finalize
```
- **self_critique** node scores the draft on: persona-voice match, factuality vs item metadata, rating↔text consistency, Nigerian-register appropriateness. If below threshold, it emits targeted feedback and the draft node revises. This is the "agentic workflow logic" the rubric rewards.

### 3.3 Calibrated rating + confidence
- Predict rating via the **user's empirical rating distribution + item priors** (the cohort already has `rating_bias`, `pct_5star`), blended with the LLM's estimate — not a raw LLM float.
- Replace magic-number confidence (`base = 0.85 …`) with a **calibrated** score: fit a small calibrator (isotonic/Platt) on the holdout so confidence correlates with actual error. Report calibration curve in the paper.

**Acceptance:** RMSE measured against **real held-out ratings**; ROUGE-L/BERTScore against **real held-out review text**; ablation showing RAG-grounding and the critique loop each improve fidelity.

---

## 4. Task B overhaul — Recommendation that reasons

Today: substring-overlap scoring over a hand-built catalog, a conditional edge that routes to clarification but proceeds anyway, single-shot, no real multi-turn, NDCG measured from the model's own scores.

### 4.1 Hybrid semantic retrieval (replaces `_overlap`)
- **Dense:** embed persona+query, ANN search over the catalog vector index (FAISS/Chroma).
- **Sparse:** BM25 over item text for lexical/niche precision.
- **Fusion:** reciprocal-rank fusion of dense+sparse → top-k candidates.
- **Cross-encoder rerank:** a reranker (e.g., `bge-reranker`) or LLM reranker for the final ordering that NDCG@10 sees.

### 4.2 Agentic reason-before-recommend
```
persona_builder → context → query_planner → (retrieve ⇄ reflect: "enough signal?") 
   → rerank → reason_and_explain → verify (titles real? diverse? on-domain?) → finalize
```
- **query_planner** decomposes vague personas into structured retrieval intents (e.g., "budget-conscious Lagos foodie" → price tier + cuisine + locale filters).
- **reflect** node decides whether retrieved candidates are sufficient or a re-query/broadening is needed (handles thin niches).
- **verify** node guarantees titles exist in the catalog (kills hallucinated recommendations), enforces diversity, and checks domain fit.

### 4.3 Real multi-turn / conversational
- Introduce **session state** (Redis or in-process store keyed by `session_id`).
- The clarification node **actually pauses**: if the domain/niche is ambiguous, return a clarifying question and *suspend* the graph (LangGraph checkpointer / interrupt), resuming on the user's next turn. Today the graph asks then proceeds anyway — fix that.
- Support follow-ups ("more like #2", "cheaper", "something Nigerian") by carrying prior recommendations + feedback into the next turn.

### 4.4 Cold-start & cross-domain (25 pts — currently weak)
- **Cold-start:** when persona has no history match, fall back to **population priors + content-based** retrieval (already partially present) and lean on the LLM's world knowledge for the niche — measure explicitly on cold personas.
- **Cross-domain:** build a **shared embedding space** so a books-lover persona can get music/movie crossovers; add an explicit cross-domain transfer step ("you liked X in books → Y in film"). Evaluate cross-domain transfer as its own metric.

**Acceptance:** NDCG@10 / Hit-Rate measured against **real held-out interactions** (MovieLens etc.); cold-start and cross-domain each have dedicated eval slices; zero hallucinated titles.

---

## 5. Model layer — Claude + open-source, swappable and benchmarked

### 5.1 Provider abstraction
`core/llm/base.py` defines an `LLMProvider` protocol: `complete(messages, schema?, **opts) -> structured`. Implementations:
- **AnthropicProvider** (Claude, default for demo) — with prompt caching, retries/backoff, tool-use structured output.
- **OpenAICompatibleProvider** — points at **vLLM / Ollama / Together / Groq**, so any OSS model is a config change, not a code change.

Selection via `core/config.py` / env (`LLM_PROVIDER`, `LLM_MODEL`). Same for embeddings (`EmbeddingProvider`).

### 5.2 Open-source model choices
- **Generation/reasoning:** `Qwen2.5-7B/14B-Instruct` (top pick — strong JSON + reasoning) or `Llama-3.1-8B-Instruct`. Served via vLLM (OpenAI-compatible) or Ollama for the demo.
- **Embeddings (retrieval):** `BAAI/bge-small-en-v1.5` / `intfloat/multilingual-e5-base` (Naija-friendly). Reranker: `BAAI/bge-reranker-base`.
- **Nigerian voice:** base instruct model + few-shot Naija exemplars; optional `Aya-23-8B` for broader African-language coverage.

### 5.3 Nigerian-voice fine-tune (the bonus, done properly)
- Build an instruction dataset from **NaijaSenti + real Nigerian reviews** (persona+item → Naija review) and **LoRA fine-tune** Qwen2.5-7B.
- Serve the adapter via vLLM; A/B it against base + Claude on a **human-eval Naija-authenticity rubric**.
- This becomes a headline ablation in the paper.

### 5.4 Head-to-head ablation (paper gold)
Run the full eval across {Claude, Qwen2.5-7B base, Qwen2.5-7B+LoRA, Llama-3.1-8B} × {dense-only, hybrid, hybrid+rerank}. Produce a table: cost, latency, RMSE, ROUGE/BERTScore, NDCG, Naija-authenticity. **Disclose all models/datasets** per the rubric.

---

## 6. Evaluation harness — replace the self-referential metrics

Today `eval` measures consistency between two runs and NDCG from the model's own match_scores → always ~1.0. Rebuild:

- **Task A:** ROUGE-L + BERTScore of generated vs **real held-out review text**; RMSE of predicted vs **real held-out star rating**; per-persona fidelity.
- **Task B:** **NDCG@10 + Hit-Rate + MRR** against held-out interactions; separate cold-start and cross-domain slices.
- **Human-eval protocol:** a small annotation app/spreadsheet + rubric (persona fidelity, contextual relevance, Naija authenticity, 1–5 each), ≥2 raters, report inter-rater agreement — mirrors the judges' human-eval columns.
- **Regression CI:** eval runs in CI on a fixed sample; metrics tracked over commits so the overhaul's gains are provable.

**Acceptance:** every rubric metric has a real number from real ground truth, reproducible via `make eval`.

---

## 7. Deepened Nigerian contextualization

Build on the already-strong `nigerian_context.py`:
- Promote region detection from keyword lists to an **LLM/embedding classifier** with confidence, but keep the deterministic keyword layer as a fast/cheap prior.
- Ground ₦ pricing and product realism in **actual Nigerian catalog data** (Jumia/Jiji dumps) rather than a static brand list.
- Register-aware generation: code-switching depth scales with persona formality (already gated by `tone == formal`); validate with the Naija human-eval rubric.

---

## 8. Infra, DevEx, reproducibility

- **Config:** `pydantic-settings`, one `.env`, typed; no hardcoded model strings (today `CLAUDE_MODEL` is hardcoded in `shared/utils.py`).
- **Docker:** compose adds a **model server** (vLLM/Ollama) and a **vector DB** (or file-based FAISS) alongside Task A/B; healthchecks; pinned deps.
- **Observability:** structured logging + LangSmith/Langfuse tracing of every node (great for the paper's workflow diagrams).
- **Testing:** unit tests for each node/schema; contract tests for the API; a smoke test that asserts non-empty, valid, on-catalog outputs (would have caught the empty-catalog bug).
- **Reproducibility:** `make data && make index && make serve && make eval`; artifacts versioned (DVC/LFS) with a source manifest + checksums; pinned model versions.
- **README:** rewritten to match reality (current README claims sentence-transformers retrieval that isn't wired in).

---

## 9. Solution paper (read first — the primary signal)

Plan the 4–8 pages in parallel with the build:
1. **Problem framing** — users as dynamic, context-sensitive agents.
2. **Architecture** — the agentic graphs (with traced diagrams), provider abstraction, data pipeline.
3. **Data** — sources, cleaning, holdout design, **disclosure table**.
4. **Methods** — RAG grounding, hybrid retrieval+rerank, agentic loops, multi-turn, calibration.
5. **Experiments & ablations** — the model × retrieval grid; cold-start/cross-domain slices; calibration curves.
6. **Nigerian contextualization** — region-aware generation + LoRA fine-tune results + human-eval authenticity.
7. **Limitations & future work.**

---

## 10. Phased roadmap (incremental, dependency-ordered)

> Phases are dependency-ordered, not date-bound. Each ends with a green acceptance check.

- **Phase 0 — Reproducibility floor.** Commit/track a real `catalog.json`; smoke test asserts non-empty Task B on clean clone; fix README. *(De-risks the whole thing.)*
- **Phase 1 — Core refactor.** `core/` extraction, Pydantic schemas, structured outputs, provider abstraction (Claude only, behind the interface).
- **Phase 2 — Data foundation.** Ingestion pipeline; canonical catalog + user histories; held-out eval sets; vector index.
- **Phase 3 — Real evaluation.** Replace `evaluate_metrics.py` with ground-truth ROUGE/BERTScore/RMSE + NDCG/Hit-Rate; baseline numbers recorded.
- **Phase 4 — Task B retrieval.** Hybrid dense+sparse+rerank; verify node; cold-start + cross-domain slices. *(Targets the 30+25 pts.)*
- **Phase 5 — Task A grounding + reflection.** RAG on real histories; generate→critique→revise; calibrated rating/confidence.
- **Phase 6 — Multi-turn + agentic.** Session state, real clarification pause/resume, follow-up handling.
- **Phase 7 — OSS model layer.** Wire vLLM/Ollama + Qwen2.5/Llama; embeddings swap; full ablation grid.
- **Phase 8 — Nigerian fine-tune.** LoRA on Naija data; human-eval authenticity; A/B vs base + Claude.
- **Phase 9 — Infra hardening + paper.** Tracing, CI eval regression, Docker/model-server compose, finalize solution paper with figures and tables.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Dataset licenses / Jumia-Jiji ToS | Use official/community-published dumps; disclose; treat Nigerian product data as enrichment, not a hard dependency |
| OSS model underperforms Claude on JSON/quality | Keep Claude as demo default; OSS proven via ablation; structured-output mode + schema-repair retry |
| Eval leakage inflating metrics | Strict temporal/user-level holdout splits; document split logic |
| Local GPU unavailable | Use hosted OSS (Groq/Together) behind the same OpenAI-compatible provider |
| Scope creep | Phases are independently shippable; Phase 0–3 already make it "actually work" |
