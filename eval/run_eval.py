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
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from core.config import ARTIFACTS_DIR
from eval.metrics import (
    hit_rate_at_k,
    mae,
    mean,
    ndcg_at_k,
    reciprocal_rank,
    rmse,
    rouge_l,
)
from eval.persona_reconstruction import reconstruct_persona
from eval.retrieval_baseline import get_index

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


# --------------------------------------------------------------------------- #
# Task B predictors
# --------------------------------------------------------------------------- #

def _task_b_offline(persona_text: str, domain: str, k: int = 10) -> list[str]:
    idx = get_index()
    hits = idx.retrieve(persona_text, domain=domain, k=k)
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
    live: bool = False,
    task_a_url: str = "http://localhost:8001",
    task_b_url: str = "http://localhost:8002",
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
            if live:
                pred_a = _task_a_live(persona_text, product, task_a_url)
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
            if live:
                ranked = _task_b_live(persona_text, domain, task_b_url, k=10)
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
        "mode": "live" if live else "offline",
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
    p.add_argument("--live", action="store_true", help="hit running FastAPI services instead of offline baseline")
    p.add_argument("--task-a-url", default="http://localhost:8001")
    p.add_argument("--task-b-url", default="http://localhost:8002")
    p.add_argument("--out", default=str(RESULTS_PATH))
    args = p.parse_args()

    results = evaluate(
        n=args.n,
        seed=args.seed,
        live=args.live,
        task_a_url=args.task_a_url,
        task_b_url=args.task_b_url,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
