"""Diagnose Task B's underperformance.

Picks one held-out tech-domain row (or `--domain X` to override), reconstructs
the reviewer's persona from training history, then runs the full Task B graph
with instrumentation that captures:

  - the top-10 hybrid-retrieval candidates and whether the true item is among them
  - the raw LLM-ranker JSON (before parsing)
  - what the ranker emitted (parsed, but before the verify-node)
  - what verify_node dropped, and the final 5 recommendations
  - the rank of the true item at each stage

Reads .env first so the current LLM provider (Groq / Anthropic) is used.

Run:
    python scripts/diag_task_b.py                # default: one tech row
    python scripts/diag_task_b.py --domain skincare --seed 7
    python scripts/diag_task_b.py --n 3          # multiple rows
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from core.config import ARTIFACTS_DIR, LLM_MODEL, LLM_PROVIDER  # noqa: E402
from core.retrieval import get_retriever  # noqa: E402
from eval.persona_reconstruction import reconstruct_persona  # noqa: E402


def _instrument_ranker():
    """Wrap reasoning_ranker.call_claude so we see the raw LLM output."""
    import task_b.agent.reasoning_ranker as rr
    box: dict = {"raw": None}
    orig = rr.call_claude

    def wrapped(*args, **kwargs):
        out = orig(*args, **kwargs)
        box["raw"] = out
        return out

    rr.call_claude = wrapped  # type: ignore
    return box, lambda: setattr(rr, "call_claude", orig)


def _rank_of(item_id: str, items: list[dict]) -> str:
    for i, it in enumerate(items, start=1):
        if it.get("item_id") == item_id:
            return f"#{i}"
    return "NOT FOUND"


def diagnose_row(row: pd.Series, train: pd.DataFrame) -> None:
    reviewer = row["reviewer"]
    true_item_id = row["item_id"]
    true_product = row["product_name"]
    domain = row["domain"]
    true_rating = int(row["rating"])

    train_for_reviewer = train[train["reviewer"] == reviewer]
    signals = reconstruct_persona(train_for_reviewer)
    persona_text = signals["persona"]

    print("=" * 78)
    print(f"reviewer:     {reviewer}")
    print(f"true product: {true_product[:70]}")
    print(f"  item_id:    {true_item_id}")
    print(f"  domain:     {domain}")
    print(f"  true rating: {true_rating}/5")
    print(f"persona:      {persona_text}")
    print()

    # Stage 1: hybrid retrieval directly.
    retriever = get_retriever()
    candidates = retriever.retrieve(query=persona_text, domain=domain, k=10)
    print(f"--- Stage 1: hybrid retrieval (top 10, {domain}) ---")
    for i, c in enumerate(candidates, start=1):
        marker = "  <-- TRUE" if c["item_id"] == true_item_id else ""
        print(f"  {i:>2}. {c['item_id']:60s} score={c['score']:.4f}{marker}")
    retrieval_rank = _rank_of(true_item_id, candidates)
    print(f"\n  true item rank in candidates: {retrieval_rank}")
    print()

    # Stage 2: full Task B graph (uses hybrid + LLM ranker + verify).
    box, restore = _instrument_ranker()
    try:
        from task_b.agent.graph import build_graph
        graph = build_graph()
        result = graph.invoke({"persona": persona_text, "domain": domain, "niche": None})
    finally:
        restore()

    full_candidates = result.get("candidates") or []
    print(f"--- Stage 2: graph candidates (after retrieval_node) ({len(full_candidates)} items) ---")
    for i, c in enumerate(full_candidates, start=1):
        marker = "  <-- TRUE" if c.get("item_id") == true_item_id else ""
        print(f"  {i:>2}. {c.get('item_id'):60s} score={c.get('raw_score', 0):.4f}{marker}")
    graph_retrieval_rank = _rank_of(true_item_id, full_candidates)
    print(f"\n  true item rank in graph candidates: {graph_retrieval_rank}")
    print()

    print("--- Stage 3: raw LLM ranker output (Groq/Anthropic JSON) ---")
    raw = box["raw"] or ""
    print(raw[:1200] + (" ...[truncated]" if len(raw) > 1200 else ""))
    print()

    final = result.get("recommendations") or []
    dropped = result.get("verification_dropped") or []
    print(f"--- Stage 4: verify_node drops ---")
    if dropped:
        for d in dropped:
            print(f"  dropped: {d}")
    else:
        print("  (none)")
    print()

    print(f"--- Stage 5: final 5 recommendations ---")
    for r in final:
        marker = "  <-- TRUE" if r.get("title") and any(
            c.get("title") == r["title"] and c.get("item_id") == true_item_id
            for c in full_candidates
        ) else ""
        print(f"  #{r.get('rank')} {r.get('title'):50s} score={r.get('match_score', 0)}{marker}")
    print()

    # Verdict.
    final_titles = [r.get("title") for r in final]
    true_title = next((c["title"] for c in full_candidates if c.get("item_id") == true_item_id), None)
    final_rank = "NOT FOUND"
    if true_title:
        for i, t in enumerate(final_titles, start=1):
            if t == true_title:
                final_rank = f"#{i}"
                break
    print(f"--- Verdict ---")
    print(f"  retrieval surfaced true item: {retrieval_rank}")
    print(f"  graph passed it to ranker:    {graph_retrieval_rank}")
    print(f"  ranker kept it in final 5:    {final_rank}")
    if retrieval_rank == "NOT FOUND":
        print("  CAUSE: hybrid retrieval did not surface the true item (retrieval ceiling).")
    elif graph_retrieval_rank == "NOT FOUND":
        print("  CAUSE: the retrieval_node post-processing dropped the true item.")
    elif final_rank == "NOT FOUND":
        print("  CAUSE: the LLM ranker (or verify_node) discarded a retrieved true item.")
    else:
        print("  RESULT: true item is in the final list -- this row was actually a hit.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="tech", choices=["tech", "skincare", "food", "general lifestyle"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=1, help="how many rows to diagnose")
    args = p.parse_args()

    train = pd.read_csv(ARTIFACTS_DIR / "eval_train.csv", parse_dates=["date"])
    holdout = pd.read_csv(ARTIFACTS_DIR / "eval_holdout.csv", parse_dates=["date"])
    in_domain = holdout[holdout["domain"] == args.domain]
    if in_domain.empty:
        raise SystemExit(f"no holdout rows for domain={args.domain}")
    sample = in_domain.sample(n=min(args.n, len(in_domain)), random_state=args.seed).reset_index(drop=True)

    print(f"# provider={LLM_PROVIDER} model={LLM_MODEL}")
    print(f"# diagnosing {len(sample)} {args.domain} holdout row(s)\n")
    for _, row in sample.iterrows():
        diagnose_row(row, train)


if __name__ == "__main__":
    main()
