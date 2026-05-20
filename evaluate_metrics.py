# =============================================================================
# BCT Agent — Metrics Evaluation Script
# Run this after your API is up to score your own outputs before submission.
# Covers all metrics from the hackathon brief rubric.
#
# Install: pip install rouge-score bert-score scikit-learn requests numpy
# Usage:   python evaluate_metrics.py
# =============================================================================

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import requests
import json
import numpy as np
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
TASK_A_URL = "http://localhost:8001/generate-review"
TASK_B_URL = "http://localhost:8002/recommend"
RESULTS_FILE = "evaluation_results.json"

# =============================================================================
# TASK A METRICS
# =============================================================================

def evaluate_task_a():
    """
    Evaluates Task A outputs against three metrics from the brief:
      1. ROUGE-L  — text quality / fluency
      2. BERTScore — semantic similarity
      3. RMSE      — rating accuracy
    
    Since we have no ground-truth reviews (we're generating them), we use
    self-consistency evaluation: run the same persona twice and measure
    how consistent the outputs are. Also measure against reference reviews
    from the persona_signals.csv where available.
    """
    print("\n" + "="*60)
    print("TASK A EVALUATION — User Modeling")
    print("="*60)

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    except ImportError:
        print("  ⚠ Install rouge-score: pip install rouge-score")
        scorer = None

    try:
        from bert_score import score as bert_score
        bert_available = True
    except ImportError:
        print("  ⚠ Install bert-score: pip install bert-score")
        bert_available = False

    # Load test cases
    with open("tests/test_samples.json", encoding="utf-8") as f:
        samples = json.load(f)["task_a"]["tests"]

    results = []

    for test in samples:
        print(f"\n  Running {test['id']} — {test['label']}")
        inp = test["input"]

        # Run twice to measure consistency
        try:
            r1 = requests.post(TASK_A_URL, json=inp, timeout=60).json()
            r2 = requests.post(TASK_A_URL, json=inp, timeout=60).json()
        except Exception as e:
            print(f"    ✗ API call failed: {e}")
            continue

        review1 = r1.get("review", "")
        review2 = r2.get("review", "")
        rating1 = float(r1.get("rating", 3.0))
        rating2 = float(r2.get("rating", 3.0))
        confidence = float(r1.get("confidence", 1.0))
        ng_applied = r1.get("nigerian_context_applied", False)

        result = {
            "test_id": test["id"],
            "label": test["label"],
            "rating_run1": rating1,
            "rating_run2": rating2,
            "confidence": confidence,
            "nigerian_context_applied": ng_applied,
            "review_length_chars": len(review1),
            "review_preview": review1[:120] + "..." if len(review1) > 120 else review1,
        }

        # ── ROUGE-L (self-consistency between two runs) ───────────────────────
        if scorer and review1 and review2:
            rouge_result = scorer.score(review1, review2)
            result["rougeL_consistency"] = round(rouge_result['rougeL'].fmeasure, 4)
            print(f"    ROUGE-L consistency: {result['rougeL_consistency']:.4f} (target > 0.35)")
        else:
            result["rougeL_consistency"] = None

        # ── RMSE (rating consistency across two runs) ─────────────────────────
        rmse = float(np.sqrt((rating1 - rating2) ** 2))
        result["rmse_consistency"] = round(rmse, 4)
        print(f"    RMSE consistency: {rmse:.4f} (target < 0.5)")

        # ── Rating validity check ─────────────────────────────────────────────
        result["rating_valid"] = 1.0 <= rating1 <= 5.0
        print(f"    Rating: {rating1} ({'✓ valid' if result['rating_valid'] else '✗ out of range'})")
        print(f"    Nigerian context: {'✓ applied' if ng_applied else '✗ not applied'}")
        print(f"    Confidence: {confidence}")
        print(f"    Review length: {len(review1)} chars")

        # ── BERTScore (semantic consistency) ─────────────────────────────────
        if bert_available and review1 and review2:
            try:
                P, R, F1 = bert_score([review1], [review2], lang="en", verbose=False)
                result["bertscore_consistency"] = round(F1.mean().item(), 4)
                print(f"    BERTScore F1: {result['bertscore_consistency']:.4f} (target > 0.75)")
            except Exception as e:
                print(f"    BERTScore error: {e}")
                result["bertscore_consistency"] = None
        else:
            result["bertscore_consistency"] = None

        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n  ── Task A Summary ──")
    valid_results = [r for r in results if r.get("rougeL_consistency") is not None]

    if valid_results:
        avg_rouge = np.mean([r["rougeL_consistency"] for r in valid_results])
        avg_rmse  = np.mean([r["rmse_consistency"] for r in results])
        ng_rate   = np.mean([r["nigerian_context_applied"] for r in results])
        avg_conf  = np.mean([r["confidence"] for r in results])

        print(f"  Avg ROUGE-L consistency: {avg_rouge:.4f}")
        print(f"  Avg RMSE consistency:    {avg_rmse:.4f}")
        print(f"  Nigerian context rate:   {ng_rate*100:.0f}% of outputs")
        print(f"  Avg confidence:          {avg_conf:.4f}")

        if valid_results[0].get("bertscore_consistency"):
            avg_bert = np.mean([r["bertscore_consistency"] for r in valid_results if r.get("bertscore_consistency")])
            print(f"  Avg BERTScore:           {avg_bert:.4f}")

    return results


# =============================================================================
# TASK B METRICS
# =============================================================================

def evaluate_task_b():
    """
    Evaluates Task B outputs against metrics from the brief:
      1. NDCG@10   — ranking quality
      2. Hit Rate  — did relevant items appear in top 10
      3. Cold-start handling — does it work with minimal persona
      4. Fallback  — does unrecognised domain trigger correctly
    
    Since we have no ground-truth ranking, we use proxy evaluation:
    - Relevance is judged by whether recommendations match domain + persona signals
    - NDCG is computed from match_score values returned by the agent
    - Hit rate = fraction of tests that returned at least 3 recommendations
    """
    print("\n" + "="*60)
    print("TASK B EVALUATION — Recommendation")
    print("="*60)

    with open("tests/test_samples.json", encoding="utf-8") as f:
        samples = json.load(f)["task_b"]["tests"]

    results = []

    for test in samples:
        print(f"\n  Running {test['id']} — {test['label']}")
        inp = test["input"]

        try:
            response = requests.post(TASK_B_URL, json=inp, timeout=60).json()
        except Exception as e:
            print(f"    ✗ API call failed: {e}")
            continue

        recs = response.get("recommendations", [])
        fallback_used = response.get("fallback_used", False)

        result = {
            "test_id": test["id"],
            "label": test["label"],
            "num_recommendations": len(recs),
            "fallback_used": fallback_used,
            "expected_fallback": test["id"] == "TB-06",
        }

        # ── Hit Rate proxy — did we get at least 3 recommendations? ──────────
        hit = len(recs) >= 3
        result["hit"] = hit
        print(f"    Recommendations returned: {len(recs)} ({'✓' if hit else '✗ too few'})")

        # ── NDCG@10 proxy — compute from match_scores ─────────────────────────
        # Real NDCG needs ground truth relevance labels.
        # Proxy: treat match_score as relevance, compute DCG/IDCG
        if recs:
            scores = [float(r.get("match_score", 0)) / 100.0 for r in recs]
            dcg  = sum(s / np.log2(i + 2) for i, s in enumerate(scores))
            ideal = sorted(scores, reverse=True)
            idcg = sum(s / np.log2(i + 2) for i, s in enumerate(ideal)) or 1
            ndcg = dcg / idcg
            result["ndcg_proxy"] = round(ndcg, 4)
            print(f"    NDCG@{len(recs)} proxy: {ndcg:.4f} (target > 0.7)")

            # Preview top rec
            top = recs[0]
            print(f"    Top rec: {top.get('title', 'N/A')} (score: {top.get('match_score', 'N/A')})")
            print(f"    Reason preview: {str(top.get('reason', ''))[:80]}...")
        else:
            result["ndcg_proxy"] = 0.0
            print("    ✗ No recommendations returned")

        # ── Fallback check ────────────────────────────────────────────────────
        if result["expected_fallback"]:
            fallback_correct = fallback_used == True
            result["fallback_correct"] = fallback_correct
            print(f"    Fallback triggered: {'✓ correct' if fallback_correct else '✗ should have triggered'}")
        else:
            result["fallback_correct"] = not fallback_used  # should NOT trigger for valid domains
            if fallback_used:
                print(f"    ⚠ Fallback triggered unexpectedly for valid domain")

        # ── Cold-start check ──────────────────────────────────────────────────
        if test["id"] in ["TB-02", "TB-08"]:
            result["cold_start_handled"] = len(recs) >= 3
            print(f"    Cold-start handled: {'✓' if result['cold_start_handled'] else '✗'}")

        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n  ── Task B Summary ──")
    hit_rate    = np.mean([r["hit"] for r in results])
    avg_ndcg    = np.mean([r.get("ndcg_proxy", 0) for r in results])
    fallback_ok = all(r["fallback_correct"] for r in results if "fallback_correct" in r)
    cold_ok     = all(r.get("cold_start_handled", True) for r in results)

    print(f"  Hit Rate:        {hit_rate*100:.0f}% of tests returned ≥3 recommendations")
    print(f"  Avg NDCG proxy:  {avg_ndcg:.4f}")
    print(f"  Fallback logic:  {'✓ working' if fallback_ok else '✗ check domain_validator'}")
    print(f"  Cold-start:      {'✓ handled' if cold_ok else '✗ check retrieval_node'}")

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("BCT Agent — Evaluation Suite")
    print("Ensure both APIs are running before proceeding.")
    print(f"  Task A: {TASK_A_URL}")
    print(f"  Task B: {TASK_B_URL}")

    task_a_results = evaluate_task_a()
    task_b_results = evaluate_task_b()

    # Save full results to JSON
    all_results = {
        "task_a": task_a_results,
        "task_b": task_b_results
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Full results saved to {RESULTS_FILE}")
    print("\n── FINAL SCORECARD ──")
    if task_a_results:
        valid = [r for r in task_a_results if r.get("rougeL_consistency")]
        if valid:
            print(f"  Task A ROUGE-L:  {np.mean([r['rougeL_consistency'] for r in valid]):.4f}")
            print(f"  Task A RMSE:     {np.mean([r['rmse_consistency'] for r in task_a_results]):.4f}")
    if task_b_results:
        print(f"  Task B NDCG:     {np.mean([r.get('ndcg_proxy',0) for r in task_b_results]):.4f}")
        print(f"  Task B Hit Rate: {np.mean([r['hit'] for r in task_b_results])*100:.0f}%")
