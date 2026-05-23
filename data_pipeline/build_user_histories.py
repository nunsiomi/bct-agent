"""Build per-reviewer review histories from the Jumia corpus.

Output: ``data_prep/artifacts/user_histories.csv`` (one row per review).
Phase 3 retrieves a reviewer's prior reviews to ground Task A generation, and
Phase 4 uses them as cold-start backstop signal.

Run:
    python -m data_pipeline.build_user_histories
"""

from __future__ import annotations

from core.config import ARTIFACTS_DIR, USER_HISTORIES_PATH
from data_pipeline.sources.jumia import load_raw_reviews, to_user_histories

# Override default parquet path -- we ship CSV so no parquet dep is required.
OUT_PATH = ARTIFACTS_DIR / "user_histories.csv"


def main() -> None:
    reviews = load_raw_reviews()
    hist = to_user_histories(reviews)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(OUT_PATH, index=False)
    n_users = hist["reviewer"].nunique()
    repeaters = (hist.groupby("reviewer").size() > 1).sum()
    print(
        f"Wrote {OUT_PATH} -- {len(hist)} review-rows, "
        f"{n_users} unique reviewers ({repeaters} with >1 review)."
    )
    # Mirror to the canonical config path for any caller that uses parquet.
    _ = USER_HISTORIES_PATH


if __name__ == "__main__":
    main()
