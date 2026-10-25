"""Candidate-aware ranking evaluation and paired user bootstrap.

Ranking scores need only preserve order. They are never silently interpreted
as probabilities for log loss or other calibration-dependent metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from typing import Any

import numpy as np
import pandas as pd


def evaluate_ranking(
    candidates: pd.DataFrame,
    scores: Any,
    truth: Mapping[int, Set[int]],
    k: int = 10,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Evaluate binary relevance over every user in ``truth``.

    All truth items count in recall and ideal-DCG denominators, including items
    that candidate generation missed. A user with no candidates contributes
    zeros. Ties are resolved by ascending item_id. ``label`` is required as a
    schema check; relevance is derived from the complete external ``truth``.
    """
    required = {"user_id", "item_id", "label"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"Missing candidate columns: {sorted(required - set(candidates.columns))}")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if not truth:
        raise ValueError("truth must contain at least one evaluation user")
    prediction = np.asarray(scores, dtype=float)
    if prediction.ndim != 1 or len(prediction) != len(candidates):
        raise ValueError("scores must be one-dimensional with one value per candidate")
    if not np.isfinite(prediction).all():
        raise ValueError("Ranking scores must be finite")
    if candidates[["user_id", "item_id"]].isna().any().any():
        raise ValueError("Candidate identifiers must not be missing")
    if candidates.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Candidates must contain unique user/item pairs")

    frame = candidates.loc[:, ["user_id", "item_id"]].copy()
    frame["score"] = prediction
    frame = frame.sort_values(
        ["user_id", "score", "item_id"], ascending=[True, False, True], kind="stable"
    )
    by_user = {uid: group["item_id"].tolist() for uid, group in frame.groupby("user_id", sort=False)}
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    rows = []
    for uid, relevant in truth.items():
        relevant = set(relevant)
        all_items = by_user.get(uid, [])
        top_items = all_items[:k]
        hits = np.asarray([item in relevant for item in top_items], dtype=float)
        n_truth, n_hits = len(relevant), float(hits.sum())
        ideal_length = min(k, n_truth)
        idcg = float(discounts[:ideal_length].sum())
        dcg = float(np.dot(hits, discounts[:len(hits)]))
        ap_numerator = float(np.sum(hits * np.cumsum(hits) / np.arange(1, len(hits) + 1)))
        rows.append({
            "user_id": uid,
            "ndcg": dcg / idcg if idcg else 0.0,
            "recall": n_hits / n_truth if n_truth else 0.0,
            "map": ap_numerator / ideal_length if ideal_length else 0.0,
            "hitrate": float(n_hits > 0),
            "candidate_recall": len(set(all_items) & relevant) / n_truth if n_truth else 0.0,
            "n_candidates": len(all_items),
            "n_truth": n_truth,
        })
    per_user = pd.DataFrame(rows)
    summary: dict[str, float | int] = {
        f"{metric}@{k}": float(per_user[metric].mean())
        for metric in ("ndcg", "recall", "map", "hitrate")
    }
    summary.update({
        "candidate_recall": float(per_user["candidate_recall"].mean()),
        "n_users": len(per_user),
        "mean_candidates": float(per_user["n_candidates"].mean()),
    })
    return summary, per_user


def paired_bootstrap(
    per_user_a: pd.DataFrame,
    per_user_b: pd.DataFrame,
    metric: str = "ndcg",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired percentile CI for mean(B - A), resampling users with replacement.

    Both tables must cover exactly the same unique user cohort. The interval
    captures uncertainty across these users, conditional on the fixed time
    split; it is not an interval over future dates or hyperparameter searches.
    """
    if not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    for frame in (per_user_a, per_user_b):
        if not {"user_id", metric}.issubset(frame.columns):
            raise ValueError(f"Each table must contain user_id and {metric}")
        if frame["user_id"].isna().any() or frame["user_id"].duplicated().any():
            raise ValueError("Each user must occur exactly once without missing IDs")
    if len(per_user_a) == 0 or set(per_user_a["user_id"]) != set(per_user_b["user_id"]):
        raise ValueError("Paired bootstrap requires the same nonempty user cohort")
    joined = per_user_a[["user_id", metric]].merge(
        per_user_b[["user_id", metric]], on="user_id", suffixes=("_a", "_b"), validate="one_to_one"
    )
    differences = joined[f"{metric}_b"].to_numpy(dtype=float) - joined[f"{metric}_a"].to_numpy(dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError("Per-user metric values must be finite")
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_boot)
    # Bound the index matrix to roughly 8 MB even for large evaluation cohorts.
    batch_size = max(1, min(n_boot, 1_000_000 // len(differences)))
    for start in range(0, n_boot, batch_size):
        end = min(start + batch_size, n_boot)
        indices = rng.integers(0, len(differences), size=(end - start, len(differences)))
        bootstrap_means[start:end] = differences[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "metric": metric, "delta": float(differences.mean()),
        "ci_low": float(ci_low), "ci_high": float(ci_high),
        "n_users": len(differences), "n_boot": n_boot, "seed": seed,
        "direction": "b_minus_a",
    }
