"""Build the held-out eval slice (Phase 3 ground truth).

A chronological split of the Jumia corpus: the most-recent 20% of reviews
become the holdout set. Output: ``data_prep/artifacts/eval_holdout.csv``
with columns: reviewer, item_id, domain, rating, review_text, review_title,
date, price_naira.

Phase 3 uses this for:
- **Task A**: RMSE of predicted rating vs ``rating``; ROUGE-L of generated
  review vs ``review_text``.
- **Task B**: NDCG@10 / Hit-Rate@10 by treating each (reviewer, item_id) in
  the holdout as a positive interaction the system should rank high.

Run:
    python -m data_pipeline.build_eval_holdout
"""

from __future__ import annotations

import argparse

from core.config import ARTIFACTS_DIR
from data_pipeline.sources.jumia import load_raw_reviews, temporal_split

OUT_PATH = ARTIFACTS_DIR / "eval_holdout.csv"
TRAIN_PATH = ARTIFACTS_DIR / "eval_train.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--holdout-frac", type=float, default=0.2)
    args = p.parse_args()

    reviews = load_raw_reviews()
    train, holdout = temporal_split(reviews, holdout_frac=args.holdout_frac)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    keep = ["reviewer", "item_id", "product_name", "domain", "rating",
            "review_text", "review_title", "date", "price"]
    train[keep].rename(columns={"price": "price_naira"}).to_csv(TRAIN_PATH, index=False)
    holdout[keep].rename(columns={"price": "price_naira"}).to_csv(OUT_PATH, index=False)

    train_min, train_max = train["date"].min(), train["date"].max()
    holdout_min, holdout_max = holdout["date"].min(), holdout["date"].max()
    print(f"Wrote {TRAIN_PATH} -- {len(train):>6} rows ({train_min} -> {train_max})")
    print(f"Wrote {OUT_PATH}    -- {len(holdout):>6} rows ({holdout_min} -> {holdout_max})")


if __name__ == "__main__":
    main()
