"""Ground-truth evaluation harness.

Replaces the legacy self-referential ``evaluate_metrics.py``. For each held-out
Jumia (reviewer, item, rating, review_text) tuple, reconstructs the reviewer's
persona from their *training* history, then runs Task A and Task B against
real ground truth.

Modes
-----
- ``--offline`` (default): no API key needed. Task A predicts rating = the
  reviewer's training-history mean (and copies their most-recent training
  review as the text prior). Task B uses TF-IDF retrieval over the catalog.
- ``--live``  : POSTs to the running FastAPI services (Task A on :8001,
  Task B on :8002). Requires ANTHROPIC_API_KEY to be set in those services.

Run:
    python -m eval.run_eval --offline --n 100
    python -m eval.run_eval --live --n 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

# Load .env BEFORE any core.* import so module-level config picks up provider choice.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

from core.config import ARTIFACTS_DIR, LLM_MODEL, LLM_PROVIDER  # noqa: E402
from core.retrieval import get_retriever  # noqa: E402
from eval.metrics import (  # noqa: E402
    hit_rate_at_k,
    mae,
    mean,
    ndcg_at_k,
    reciprocal_rank,
    rmse,
    rouge_l,
)
from eval.persona_reconstruction import reconstruct_persona  # noqa: E402

HOLDOUT_PATH = ARTIFACTS_DIR / "eval_holdout.csv"
TRAIN_PATH = ARTIFACTS_DIR / "eval_train.csv"
RESULTS_PATH = ARTIFACTS_DIR / "evaluation_results.json"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not HOLDOUT_PATH.exists() or not TRAIN_PATH.exists():
        raise SystemExit(
            "missing eval splits; run `python -m data_pipeline.build_eval_holdout`"
        )
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    holdout = pd.read_csv(HOLDOUT_PATH, parse_dates=["date"])
    return train, holdout


def _sample(holdout: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample n holdout rows, preferring reviewers WITH training history."""
    return holdout.sample(n=min(n, len(holdout)), random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Task A predictors
# --------------------------------------------------------------------------- #

def _task_a_offline(persona_signals: dict[str, Any], train_for_reviewer: pd.DataFrame, product: str) -> dict[str, Any]:
    """Baseline Task A: predict rating = reviewer's mean; text = empty."""
    rating = float(persona_signals.get("avg_rating") or 3.0)
    # Use the reviewer's most-recent training review as a stylistic prior text
    # ONLY for ROUGE-L sanity (it should not score perfectly -- the holdout
    # text is different by construction since dates are after the split).
    text_prior = ""
    if not train_for_reviewer.empty:
        last = train_for_reviewer.sort_values("date").iloc[-1]
        text_prior = str(last.get("review_text") or "")
    return {"rating": rating, "review": text_prior}


def _task_a_live(persona_text: str, product: str, base_url: str) -> dict[str, Any]:
    r = requests.post(
        f"{base_url}/generate-review",
        json={"persona": persona_text, "product": product},
        timeout=60,
    )
    r.raise_for_status()
    j = r.json()
    return {"rating": float(j.get("rating", 3.0)), "review": str(j.get("review", ""))}


_TASK_A_GRAPH = None
_TASK_B_GRAPH = None


def _task_a_in_process(persona_text: str, product: str, reviewer_id: str | None = None) -> dict[str, Any]:
    """Run the real Task A graph in-process (uses whichever LLM provider .env selects)."""
    global _TASK_A_GRAPH
    if _TASK_A_GRAPH is None:
        from task_a.agent.graph import build_graph as build_a
        _TASK_A_GRAPH = build_a()
    state: dict[str, Any] = {"persona": persona_text, "product": product}
    if reviewer_id is not None:
        state["reviewer_id"] = reviewer_id
    result = _TASK_A_GRAPH.invoke(state)
    return {"rating": float(result.get("rating", 3.0)), "review": str(result.get("review", ""))}


def _task_b_in_process(persona_text: str, domain: str, k: int = 10) -> list[str]:
    """Run the real Task B graph in-process, then map ranked titles back to item_ids."""
    global _TASK_B_GRAPH
    if _TASK_B_GRAPH is None:
        from task_b.agent.graph import build_graph as build_b
        _TASK_B_GRAPH = build_b()
    result = _TASK_B_GRAPH.invoke({"persona": persona_text, "domain": domain, "niche": None})
    recs = result.get("recommendations", []) or []

    from core.config import CATALOG_PATH
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    title_to_id: dict[str, str] = {it["title"]: it["item_id"] for items in catalog.values() for it in items}
    return [title_to_id.get(r.get("title", ""), r.get("title", "")) for r in recs[:k]]


# --------------------------------------------------------------------------- #
# Task B predictors
# --------------------------------------------------------------------------- #

def _task_b_offline(persona_text: str, domain: str, k: int = 10) -> list[str]:
    """Offline Task B baseline: hybrid TF-IDF + BM25 retrieval (same code as Task B's runtime)."""
    hits = get_retriever().retrieve(persona_text, domain=domain, k=k)
    return [h["item_id"] for h in hits]


def _task_b_live(persona_text: str, domain: str, base_url: str, k: int = 10) -> list[str]:
    r = requests.post(
        f"{base_url}/recommend",
        json={"persona": persona_text, "domain": domain, "niche": None},
        timeout=60,
    )
    r.raise_for_status()
    recs = r.json().get("recommendations", []) or []
    # Live service returns titles; map back to item_ids via catalog.
    from core.config import CATALOG_PATH
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    title_to_id: dict[str, str] = {}
    for items in catalog.values():
        for it in items:
            title_to_id[it["title"]] = it["item_id"]
    return [title_to_id.get(r.get("title", ""), r.get("title", "")) for r in recs[:k]]


# --------------------------------------------------------------------------- #
# Eval loop
# --------------------------------------------------------------------------- #

def evaluate(
    n: int = 100,
    seed: int = 42,
    mode: str = "offline",  # offline | live | in_process
    task_a_url: str = "http://localhost:8001",
    task_b_url: str = "http://localhost:8002",
    use_reviewer_id: bool = False,
) -> dict[str, Any]:
    train, holdout = _load()
    sample = _sample(holdout, n=n, seed=seed)

    a_pred_ratings: list[float] = []
    a_true_ratings: list[float] = []
    a_pred_texts: list[str] = []
    a_true_texts: list[str] = []

    b_ndcg: list[float] = []
    b_hit: list[float] = []
    b_mrr: list[float] = []

    per_domain: dict[str, dict[str, list[float]]] = {}

    for i, row in sample.iterrows():
        reviewer = row["reviewer"]
        item_id = row["item_id"]
        domain = row["domain"]
        true_rating = float(row["rating"])
        true_text = str(row.get("review_text") or "")
        product = str(row.get("product_name") or row["item_id"])

        train_for_reviewer = train[train["reviewer"] == reviewer]
        signals = reconstruct_persona(train_for_reviewer)
        persona_text = signals["persona"]

        # Task A
        try:
            if mode == "live":
                pred_a = _task_a_live(persona_text, product, task_a_url)
            elif mode == "in_process":
                rid = reviewer if use_reviewer_id else None
                pred_a = _task_a_in_process(persona_text, product, reviewer_id=rid)
            else:
                pred_a = _task_a_offline(signals, train_for_reviewer, product)
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] task_a row {i} failed: {exc}", file=sys.stderr)
            continue

        a_pred_ratings.append(pred_a["rating"])
        a_true_ratings.append(true_rating)
        a_pred_texts.append(pred_a["review"])
        a_true_texts.append(true_text)

        # Task B
        try:
            if mode == "live":
                ranked = _task_b_live(persona_text, domain, task_b_url, k=10)
            elif mode == "in_process":
                ranked = _task_b_in_process(persona_text, domain, k=10)
            else:
                ranked = _task_b_offline(persona_text, domain, k=10)
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] task_b row {i} failed: {exc}", file=sys.stderr)
            continue

        ndcg = ndcg_at_k(ranked, item_id, k=10)
        hit = hit_rate_at_k(ranked, item_id, k=10)
        mrr = reciprocal_rank(ranked, item_id)
        b_ndcg.append(ndcg)
        b_hit.append(hit)
        b_mrr.append(mrr)

        slot = per_domain.setdefault(domain, {"ndcg": [], "hit": [], "mrr": []})
        slot["ndcg"].append(ndcg)
        slot["hit"].append(hit)
        slot["mrr"].append(mrr)

    results = {
        "mode": mode,
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "n_evaluated": len(a_pred_ratings),
        "task_a": {
            "rmse": rmse(a_pred_ratings, a_true_ratings),
            "mae": mae(a_pred_ratings, a_true_ratings),
            "rouge_l_mean": mean(rouge_l(h, r) for h, r in zip(a_pred_texts, a_true_texts)),
        },
        "task_b": {
            "ndcg_at_10": mean(b_ndcg),
            "hit_rate_at_10": mean(b_hit),
            "mrr_at_10": mean(b_mrr),
        },
        "task_b_per_domain": {
            d: {
                "n": len(s["ndcg"]),
                "ndcg_at_10": mean(s["ndcg"]),
                "hit_rate_at_10": mean(s["hit"]),
                "mrr_at_10": mean(s["mrr"]),
            }
            for d, s in sorted(per_domain.items())
        },
    }
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100, help="number of holdout rows to score")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mode",
        choices=["offline", "live", "in_process"],
        default="offline",
        help=(
            "offline = deterministic baseline (no LLM); "
            "live = POST to running FastAPI services; "
            "in_process = build graphs in-process and call current LLM provider directly."
        ),
    )
    p.add_argument("--live", action="store_true", help="shortcut for --mode live")
    p.add_argument("--in-process", action="store_true", help="shortcut for --mode in_process")
    p.add_argument("--use-reviewer-id", action="store_true",
                   help="(in_process Task A only) pass the held-out reviewer's id into grounding")
    p.add_argument("--task-a-url", default="http://localhost:8001")
    p.add_argument("--task-b-url", default="http://localhost:8002")
    p.add_argument("--out", default=str(RESULTS_PATH))
    args = p.parse_args()

    mode = args.mode
    if args.live:
        mode = "live"
    elif args.in_process:
        mode = "in_process"

    results = evaluate(
        n=args.n,
        seed=args.seed,
        mode=mode,
        task_a_url=args.task_a_url,
        task_b_url=args.task_b_url,
        use_reviewer_id=args.use_reviewer_id,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
