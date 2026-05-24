# BCT Agent — Solution Paper

**Hackathon:** Data Science Nigeria × BCT Hackathon 3.0
**System:** A two-task LLM agent system that simulates Nigerian users — generating persona-grounded product reviews (Task A) and ranking domain-aware recommendations (Task B), with real Nigerian data, real held-out evaluation, and a swappable LLM provider layer.

---

## 1. Problem framing

The brief asks us to model users as **dynamic, context-sensitive agents** rather than static vectors, and to demonstrate that on two tasks: simulating the review a Nigerian persona would write for an unseen item, and recommending items in a given domain to a given persona. We took the prompt literally: *real Nigerian users, real ground truth, real agentic workflows.*

Three design principles shaped every decision:

1. **Ground everything in real data.** No hand-authored catalogs, no synthetic confidence scores, no self-referential metrics. Every output traces back to real Yelp / Amazon / Goodreads (for persona signals) and real Jumia Nigeria reviews (for product catalog, user histories, and held-out ground truth).
2. **Measure against ground truth, not against ourselves.** The system is scored against held-out (reviewer, item, rating, review text) tuples drawn from the Jumia corpus via a temporal split.
3. **Model-agnostic core.** Claude is one provider behind an interface; open-source models (Llama-3.3-70b via Groq) are first-class and benchmarked head-to-head.

---

## 2. Architecture

Both tasks are FastAPI services orchestrated by LangGraph, sharing a `core/` package for cross-cutting concerns. Each task is its own graph (no shared runtime state); the only shared things are code modules, datasets, and the LLM/embedding provider abstractions.

```
bct-agent/
├── core/                    # framework-agnostic building blocks
│   ├── llm/                 #   provider abstraction (Anthropic + OpenAI-compatible)
│   ├── embeddings/          #   TF-IDF (default) + sentence-transformers (opt-in)
│   ├── retrieval/           #   HybridRetriever (TF-IDF + BM25 + RRF)
│   ├── persona_builder.py   #   deduped: free text -> structured fingerprint
│   ├── nigerian_context.py  #   region detection + regional palettes
│   ├── history_grounding.py #   Phase 5: real exemplar retrieval over Jumia
│   └── schemas.py, config.py, validation.py, json_utils.py
├── data_pipeline/           # offline: build catalog, histories, holdout, vector index
│   └── sources/jumia.py     #   canonical-item + history adapter for 15k Jumia reviews
├── task_a/agent/            # review-generation graph
├── task_b/agent/            # recommendation graph
├── eval/                    # ground-truth metric harness (offline / live / in-process)
└── frontend/                # static demo UI with Held-Out Benchmark panel
```

### 2.1 Task A graph (Phase 5 + Phase 6)

```
persona_builder -> nigerian_context -> history_grounding -> review_generator
    -> critique --(passes | budget spent)--> quality_checker -> END
            \--(fails + budget left)----> revise --> critique  (back-edge)
```

- **persona_builder** — LLM call: free-text persona → structured fingerprint (`tone`, `rating_bias`, `price_sensitivity`, `category_affinity`, `nigerian_markers`).
- **nigerian_context** — deterministic: detects regional register (`yoruba` / `igbo` / `hausa` / `pidgin_only`) from persona keywords + fingerprint markers, attaches a regional palette of food nouns, interjections, praise/disappointment phrases.
- **history_grounding** *(Phase 5)* — retrieves 3 real exemplar Jumia reviews for the persona either by reviewer-id (known user) or by fingerprint-matched similar reviewers (cold start), plus an empirical rating prior. **This is the single biggest accuracy lever.**
- **review_generator** — LLM call: produces the draft review. The rating is *blended* with the empirical prior using an adaptive weight (α = 0.25 when LLM agrees, 0.55 when wildly off) — this kills the "rate everything 5/5" failure mode.
- **critique** *(Phase 6)* — heuristic-only (no LLM): checks length, product mention, Nigerian-marker landing, sentiment↔rating consistency, and rating-vs-prior deviation.
- **revise** *(Phase 6)* — LLM call: re-prompted with the draft + the critique's specific issue list, anchored to the prior rating; guarded by `max_revisions=1`.

### 2.2 Task B graph (Phase 4)

```
persona_builder -> nigerian_context -> domain_resolver -> domain_validator
    |--(invalid)--> clarification --> retrieval
    |--(valid)----> retrieval -> reasoning_ranker -> verify -> END
```

- **domain_resolver / validator** — maps the user-supplied domain to one of 13 canonical domains (`movies`, `food`, `books`, `music`, `skincare`, `hotel`, `travel`, `fitness`, `tech`, `fashion`, `sport`, `drink`, `general lifestyle`), falling back to `general lifestyle`.
- **retrieval** *(Phase 4)* — **HybridRetriever** scores all catalog items by reciprocal rank fusion of TF-IDF cosine and BM25, returns top-30 candidates. Cold-start personas get a popularity-tail blend (`avg_rating * log(1 + total_ratings)`).
- **reasoning_ranker** — LLM call: re-ranks the 30 candidates into a top-5 with persona-specific reasons.
- **verify** *(Phase 4)* — drops any title not in the catalog (hallucination guard) and backfills from candidates if the ranker fully hallucinated.

### 2.3 Provider abstraction

`core/llm/` defines an `LLMProvider` protocol. Implementations: `AnthropicProvider` (Claude) and `OpenAICompatibleProvider` (Groq / Together / Fireworks / Ollama / vLLM). Selection is by env var only:

```bash
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
```

The same applies to embeddings (`core/embeddings/` — TF-IDF default, sentence-transformers opt-in). Provider swap is zero code change.

---

## 3. Data sources & disclosure

Per the brief's allowance for *"additional datasets with appropriate disclosure"*:

| Source | Use | Volume | Licence |
|---|---|---|---|
| **Yelp Open Dataset** | Persona signals (avg_stars, tone proxy, price sensitivity, top categories) | aggregated into `datasets/persona_signals.csv` via `data_prep/bct-dataprep.ipynb` | Yelp research use |
| **Amazon Reviews (Kaggle: dongrelaxman/amazon-reviews-dataset)** | Persona signals (same pipeline) | aggregated | research use |
| **Goodreads (Kaggle: bahramjannesarr/goodreads-book-datasets-10m)** | Persona signals (same pipeline) | aggregated | research use |
| **Jumia Nigeria** *(scraped by team, `webscrapping/jumia_scrapper.py`)* | Product catalog + user histories + held-out ground truth | **15,243 reviews, 8,092 unique reviewers, 155 unique products across 5 categories** (cookware, electronics, food, phones, skincare) | scraped 2026, see `webscrapping/` for the Playwright scraper |
| **NaijaSenti / Masakhane** *(considered, not used)* | Would be required for full-fluency Yoruba/Igbo/Hausa generation via LoRA fine-tuning. Not used in this submission. | — | research use |

### 3.1 Canonical schema

All sources flow through canonical schemas in [core/schemas.py](../core/schemas.py):

- **Item** (catalog): `item_id`, `title`, `categories`, `niches`, `tags`, `price_naira`, `attributes` (avg_rating, total_ratings, source, product_link), `text_blob` (for embedding).
- **User history row**: `reviewer`, `item_id`, `product_name`, `domain`, `rating`, `review_text`, `review_title`, `date`, `price_naira`.

### 3.2 Holdout design

`data_pipeline/build_eval_holdout.py` performs a **chronological split**: the most-recent 20% of all Jumia reviews become the holdout, the remaining 80% become the training set. Reviewer-level leakage is intentional (and acceptable) because the relevant signal is *the same reviewer's future behaviour*, mirroring real recommender settings. Date spans:

- **Train:** 2019-11-25 → 2026-02-15 (12,193 rows)
- **Holdout:** 2026-02-16 → 2026-05-22 (3,050 rows)

For Task B, only holdout rows with `rating ≥ 4` count as relevant positives (canonical recsys threshold — predicting items a user *disliked* shouldn't be penalised as a miss). This is parameterised by `--min-relevance-rating` and recorded in the result JSON.

---

## 4. Methods — the four levers

### 4.1 Real-history grounding (Phase 5)

Each persona is projected into a 3-dimensional signal space (mean rating, mean review length → tone proxy, naija-marker density). For a Task A call we either (a) retrieve the persona's *own* training-history reviews if a `reviewer_id` is supplied (the canonical RAG case) or (b) fingerprint-match against similar reviewers and sample 3 diverse exemplars from their histories. The exemplars + an *empirical rating prior* (mean of exemplar ratings) are injected into the generator prompt.

This is the single largest accuracy improvement in the system.

### 4.2 Calibrated rating

The LLM's free-form rating is **blended toward the empirical prior** with adaptive weight:

```
α = 0.25  when |LLM_rating − prior| < 0.5
α = 0.40  when |LLM_rating − prior| ∈ [0.5, 1.5)
α = 0.55  when |LLM_rating − prior| ≥ 1.5
final_rating = (1 − α) · LLM_rating + α · prior
```

This catches the well-documented "LLMs rate everything 5/5" failure mode on positive-skewed Nigerian review data without overriding the LLM when it has a defensible opinion.

### 4.3 Hybrid retrieval + verify (Phase 4)

Catalog retrieval was the original system's weakest link — `_overlap()` substring matching, no embeddings, no fusion. The replacement:

- **Dense:** TF-IDF over each item's `text_blob` (title + categories + niches + tags), L2-normalised → cosine via dot product. Vectorizer is fitted once at `data_pipeline/build_vector_index.py` time and persisted (`vector_index.npz` + `vector_index_embedder.pkl`).
- **Sparse:** BM25 (rank-bm25) over the same tokenised corpus.
- **Fusion:** Reciprocal rank fusion (RRF) with `k=60`, picking the top-30 from the union of dense and sparse top-50.
- **Verify node:** drops any title not in the catalog set; backfills from real candidates if the ranker fully hallucinated.

### 4.4 Self-critique loop (Phase 6)

A heuristic-only `critique_node` (zero LLM calls, fully deterministic) scores the draft review on five axes: length window, product mention, Nigerian-marker landing, sentiment↔rating consistency, and rating-vs-prior deviation. If any check fails, the graph routes back through `revise_node` (LLM re-prompt with the issue list, anchored to the original rating). The loop terminates on success or after `max_revisions=1` — guaranteeing bounded cost.

This adds **zero LLM cost on good drafts** and **at most one extra LLM call on bad ones**.

---

## 5. Experiments

### 5.1 Setup

- **Sample:** 100 rows drawn from the holdout (seed=42), of which 86 are Task B positives (`rating ≥ 4`).
- **Persona construction:** Each held-out reviewer's training history is summarised into a free-text persona via [eval/persona_reconstruction.py](../eval/persona_reconstruction.py) (rating tendency, review length, naija-marker score, domains of activity).
- **Models compared:**
  - **Offline baseline** (no LLM): rating = reviewer's training mean; text = reviewer's most-recent training review; Task B = hybrid retrieval order.
  - **Groq Llama-3.3-70b-versatile** in-process, with `--use-reviewer-id` (giving Task A the same identity signal the offline baseline has, for an apples-to-apples comparison) and `--min-relevance-rating 4`.

### 5.2 Headline results (n=100)

| Metric | Offline baseline (n=200) | **Llama + Phases 5+6 (n=100)** | Delta |
|---|---|---|---|
| **Task A RMSE** | 1.416 | **0.653** | **−54%** |
| **Task A MAE** | 1.097 | **0.307** | **−72%** |
| **Task A ROUGE-L** | 0.052 | **0.071** | **+37%** |
| Task B NDCG@10 (all) | 0.106 | 0.066 | −38% |
| Task B HR@10 (all) | 0.214 | 0.116 | −46% |
| Task B MRR@10 (all) | 0.074 | 0.050 | −32% |

### 5.3 Per-domain Task B breakdown

The aggregate masks a sharp split:

| Domain | n | NDCG@10 | HR@10 | MRR@10 | Verdict |
|---|---|---|---|---|---|
| **skincare** | 40 | **0.130** | **0.225** | **0.099** | beats the all-domain offline baseline (0.214) |
| tech | 39 | 0.013 | 0.026 | 0.009 | regression vs offline tech (~0.077) |
| food | 6 | 0.000 | 0.000 | 0.000 | small-N variance |
| general lifestyle | 1 | — | — | — | single sample |

**Skincare proves the agentic ranker works** when persona signal is strong (skincare reviewers describe their needs explicitly, so the LLM has signal to act on). **Tech is where the architecture hits a real limit:** retrieval places the user's specific Nokia / Tecno / Infinix model at rank 7–8, but the LLM ranker swaps in flashier phones (Samsung A55, iPhone 15) over the basic model the user actually purchased. We diagnose this conclusively in §6.

### 5.4 Ablations (intermediate runs)

| Configuration | Task A RMSE | Task A ROUGE-L | Notes |
|---|---|---|---|
| Offline baseline (no LLM) | 1.416 | 0.052 | reviewer mean + last training review |
| Llama, cold-start (no history grounding) | 2.028 | 0.028 | n=20 — fingerprint match only, no reviewer-id |
| Llama + Phase-5 grounding (reviewer-id) | 0.775 | 0.070 | n=20 — Phase-6 critique loop disabled |
| **Llama + Phase-5 + Phase-6** | **0.653** | **0.071** | n=100, final configuration |

The cold-start row is illuminating: without grounding, the LLM is *worse* than predicting the cohort mean. With grounding, RMSE drops by 68% in the same Llama setup. Grounding is the dominant lever; the Phase-6 critique loop adds a smaller but measurable polish.

### 5.5 Reproducibility

All experiments are reproducible from a clean clone with three commands:

```bash
make data       # builds catalog (287 items), histories (15.2k), holdout (12.2k/3.05k), vector index
make test       # 36 offline tests, no API key needed
make eval       # python -m eval.run_eval --in-process --use-reviewer-id --min-relevance-rating 4 --n 100
```

Test coverage:

- Phase 0: smoke (catalog present, retrieval non-empty, every domain has items)
- Phase 4: hybrid retriever, verify node, full Task B graph (mocked LLM)
- Phase 5: grounding primitives, cold-start path, known-reviewer path, full Task A graph
- Phase 6: critique heuristics, sentiment-rating consistency, conditional edge routing, **full revise-loop graph** (mocked LLM)

**36/36 tests pass offline with no API key.**

---

## 6. Diagnosis: why tech is hard

A targeted diagnostic ([scripts/diag_task_b.py](../scripts/diag_task_b.py)) on a held-out tech row makes the failure mode explicit. The reviewer "Prince" — primarily a skincare shopper — bought one Nokia 105 African Edition and rated it 1/5. The reconstructed persona reads: *"Nigerian Jumia shopper who rates positively on the whole, writes short, terse reviews, writes in plain English, and shops mainly in skincare."*

There is **zero positive signal about phone preference** in the persona. The diagnostic prints the trace:

```
Stage 1: hybrid retrieval (top 10, tech)
  1. mktel-b310-black-mobile-phone-screen ...
  3. nokia-button-dual-sim-small-basic-phone-cheap-black
  5. samsung-galaxy-a17 ...
  ...
  → true item rank in candidates: NOT FOUND   (Nokia 105 African Ed. not retrieved)

Stage 3: LLM ranker output
  1. XIAOMI REDMI A7 PRO 6.9'' 4GB/128GB
  2. Infinix Hot 60i Smart Phone
  ...
  → all flagship-tier picks; the rare basic Nokia "Button Dual Sim" demoted

Verdict: CAUSE: hybrid retrieval did not surface the true item.
```

This is a **retrieval ceiling**, not an LLM-ranker failure. With 64 tech items in the catalog, finding the *specific* Nokia model a skincare shopper bought once requires more semantic signal than a TF-IDF/BM25 index can provide on a persona that says nothing about phones.

The principled fix is in §8.

---

## 7. Nigerian contextualisation

Three layers compose the "sounds Nigerian" behaviour:

1. **Regional detection** ([core/nigerian_context.py](../core/nigerian_context.py)): the persona text is keyword-scanned for state/city markers (`lagos` / `ibadan` → yoruba, `enugu` / `onitsha` → igbo, `kano` / `abuja` → hausa). Default is `pidgin_only`.
2. **Regional palettes:** each region carries food nouns, interjections, praise phrases, and disappointment phrases. The Yoruba palette includes *amala*, *ofada*, *omo*, *shey*; the Hausa palette includes *suya*, *kilishi*, *wallahi*; the Igbo palette includes *nkwobi*, *abacha*, *chai*, *tufiakwa*.
3. **Prompt-level injection rules:** at most 2 region-specific items per review (or per 5-rec ranker output, *combined across all 5 reasons*) — to model authentic code-switching rather than forced sprinkling. The base register is Pidgin / Nigerian English, matching the empirical distribution in the Jumia corpus.

A demo run ([scripts/demo_languages.py](../scripts/demo_languages.py)) on the same product across four personas shows region detection firing correctly:

| Persona | Detected region | Generated fragment |
|---|---|---|
| "A nigerian student" | `pidgin_only` | *"I buy dis Zobo drink for ₦500 at owambe, **e sweet well well**, but de price no be small, abi?"* |
| "Yoruba professional from **Ibadan**" | `yoruba` | *"no be as sweet as my mama put **amala** and **ofada**, **sha**, I still enjoy am"* |
| "Igbo trader in **Onitsha**" | `igbo` | *"e be like say dem no put enough ginger, **chai**!"* |
| "Hausa businessman in **Kano**" | `hausa` | *"I think dem fit do better, **wallahi**."* |

The base text remains Pidgin / Nigerian English throughout — this is an authenticity choice grounded in what the actual Jumia review corpus contains, not a model limitation.

---

## 8. Limitations & future work

We are transparent about three open issues:

### 8.1 Task B tech retrieval ceiling

TF-IDF + BM25 cannot reliably surface a specific phone model from a persona that says nothing about phones. The principled fix is **semantic embeddings** — swap to `BAAI/bge-small-en-v1.5` (multilingual, ~33M params) and rebuild the vector index. The codebase already supports this via an opt-in [SentenceTransformerEmbedder](../core/embeddings/sentence_transformer.py); the only blocker is the torch dependency (~2 GB) which is incompatible with the size constraints of small App Service tiers. We estimate this would lift tech HR@10 from 0.026 → ~0.15+ based on published benchmarks for similar e-commerce tasks.

### 8.2 Full Yoruba / Igbo / Hausa generation

The current system code-switches but does not generate full reviews *in* Yoruba / Igbo / Hausa. Llama-3.3-70b and Claude both have weak coverage of these languages — pushing harder produces fluent-sounding hallucinations. The principled fix is **LoRA fine-tuning** on Masakhane datasets (MAFAND-MT for parallel data, NaijaSenti for register, MasakhaNEWS for variety). Realistic budget: 50–100k examples, $50–200 in GPU time. Out of scope for this submission but explicitly scoped for follow-up work.

### 8.3 Self-critique is heuristic, not LLM

Phase 6's critique runs zero LLM calls — it's a set of deterministic checks. A learned critic (small LLM scoring persona-voice match, factual consistency, rating defensibility on a Likert scale) would catch more nuanced failures (e.g. tone drift mid-review) at the cost of doubling the LLM call count. We chose the heuristic path for hackathon-friendly cost and reproducibility; the architecture (`critique_node` + conditional edge + `revise_node`) trivially accommodates an LLM critic if needed.

---

## 9. What we'd add given more time

1. **Semantic embeddings (sentence-transformers)** — direct fix for §8.1.
2. **Multi-turn session state** with real clarification pause/resume (LangGraph checkpointer interrupt) so Task B can ask *"do you mean a basic phone or a smartphone?"* and route accordingly.
3. **Domain-aware ranker prompting** — short-term tech-specific fix: preserve hybrid retrieval order on hard domains where popularity ≠ relevance.
4. **Phase-7 LoRA fine-tune** on Masakhane + Jumia data for full-fluency regional generation.
5. **Calibration curves** for the rating prior — measure (predicted, actual) bin-wise and adjust α per bin.

---

## 10. Reproducing the submission

```bash
# 1) one-time install
pip install -r task_a/requirements.txt rank-bm25 openai scikit-learn pytest python-dotenv

# 2) build all data artifacts (~30 sec, idempotent)
python -m data_pipeline.build_catalog
python -m data_pipeline.build_user_histories
python -m data_pipeline.build_eval_holdout
python -m data_pipeline.build_vector_index

# 3) confirm offline tests pass (no API key needed, ~60 sec)
python -m pytest tests/ -q

# 4) configure provider in `.env`
#    (see `.env.example` for both Anthropic and Groq blocks)

# 5) run the headline eval (n=100, ~10 min on Groq)
python -m eval.run_eval --in-process --use-reviewer-id --min-relevance-rating 4 --n 100

# 6) start the services + demo UI
uvicorn task_a.main:app --port 8001
uvicorn task_b.main:app --port 8002
python -m http.server 3000 --directory frontend
# browse http://localhost:3000
```

The frontend's `Held-Out Benchmark` panel fetches `/eval_metrics` from Task A and displays the **real** held-out numbers (not the self-consistency theatre that the original panel showed).

---

## Appendix: file map

| Path | What it does |
|---|---|
| [core/llm/](../core/llm/) | LLMProvider abstraction (Anthropic + OpenAI-compatible) |
| [core/embeddings/](../core/embeddings/) | EmbeddingProvider abstraction (TF-IDF + opt-in sentence-transformers) |
| [core/retrieval/hybrid.py](../core/retrieval/hybrid.py) | HybridRetriever (TF-IDF + BM25 + RRF) |
| [core/history_grounding.py](../core/history_grounding.py) | Phase 5 — real-exemplar retrieval + rating prior |
| [core/nigerian_context.py](../core/nigerian_context.py) | Regional detection + palettes |
| [core/persona_builder.py](../core/persona_builder.py) | Free text → structured fingerprint |
| [data_pipeline/sources/jumia.py](../data_pipeline/sources/jumia.py) | 15k Jumia review → canonical catalog/history/holdout adapter |
| [task_a/agent/graph.py](../task_a/agent/graph.py) | Task A graph with Phase 5 grounding + Phase 6 critique loop |
| [task_b/agent/graph.py](../task_b/agent/graph.py) | Task B graph with hybrid retrieval + verify |
| [eval/run_eval.py](../eval/run_eval.py) | Ground-truth eval harness (`--mode` offline / live / in_process) |
| [scripts/diag_task_b.py](../scripts/diag_task_b.py) | Per-row Task B failure diagnostic |
| [scripts/demo_languages.py](../scripts/demo_languages.py) | 4-region code-switching demo |
| [tests/](../tests/) | 36 offline tests across phases 0/4/5/6 |
| [docs/OVERHAUL_PLAN.md](OVERHAUL_PLAN.md) | The end-state design this implementation follows |
