"""Independent metric calculation for saved rankings; no training imports."""
import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def independent_metrics(uids, items, scores, truth, k):
    """Explicit ranking calculation from scores and full external truth."""
    require(len(uids) == len(items) == len(scores), "Prediction lengths differ")
    require(np.isfinite(scores).all(), "Nonfinite predictions")
    # NumPy lexsort's final key is primary: user, descending score, item ID.
    order = np.lexsort((items, -scores, uids))
    sorted_users, sorted_items = uids[order], items[order]
    unique, starts, counts = np.unique(sorted_users, return_index=True, return_counts=True)
    groups = {int(u): sorted_items[s:s+n] for u, s, n in zip(unique, starts, counts)}
    discounts = np.reciprocal(np.log2(np.arange(2, k + 2, dtype=float)))
    records = []
    for uid, relevant in sorted(truth.items()):
        ranked = groups.get(uid, np.empty(0, dtype=np.int64))
        relevance = np.fromiter((int(i) in relevant for i in ranked), dtype=bool, count=len(ranked))
        top = relevance[:k].astype(float)
        n_truth, n_hit, n_pool_hit = len(relevant), float(top.sum()), int(relevance.sum())
        ideal = float(discounts[:min(k, n_truth)].sum())
        dcg = float(np.dot(top, discounts[:len(top)]))
        precisions = np.cumsum(top) / np.arange(1, len(top) + 1)
        ap = float((top * precisions).sum()) / min(k, n_truth) if n_truth else 0.0
        records.append({
            "user_id": uid, "ndcg": dcg / ideal if ideal else 0.0,
            "recall": n_hit / n_truth if n_truth else 0.0, "map": ap,
            "hitrate": float(n_hit > 0),
            "candidate_recall": n_pool_hit / n_truth if n_truth else 0.0,
            "n_candidates": len(ranked), "n_truth": n_truth,
            "oracle_ndcg": float(discounts[:min(k, n_pool_hit)].sum()) / ideal if ideal else 0.0,
            "oracle_recall": min(k, n_pool_hit) / n_truth if n_truth else 0.0,
        })
    per = pd.DataFrame.from_records(records).set_index("user_id")
    summary = {f"{name}@{k}": float(per[name].mean()) for name in ("ndcg", "recall", "map", "hitrate")}
    summary.update(candidate_recall=float(per.candidate_recall.mean()), n_users=len(per),
                   mean_candidates=float(per.n_candidates.mean()))
    oracle = {"oracle_ndcg": float(per.oracle_ndcg.mean()),
              "oracle_recall": float(per.oracle_recall.mean()),
              "no_hit_fraction": float((per.candidate_recall == 0).mean()),
              "n_users_no_hit": int((per.candidate_recall == 0).sum())}
    return summary, per, oracle
