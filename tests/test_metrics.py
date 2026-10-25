import numpy as np
import pandas as pd
import pytest

from kion_recsys.metrics import evaluate_ranking, paired_bootstrap


def test_unretrieved_truth_items_stay_in_metric_denominators():
    candidates = pd.DataFrame({"user_id": [1, 1], "item_id": [10, 20], "label": [1, 0]})
    summary, users = evaluate_ranking(candidates, [2, 1], {1: {10, 30, 40}}, k=3)
    assert summary["recall@3"] == pytest.approx(1 / 3)
    assert summary["map@3"] == pytest.approx(1 / 3)
    assert summary["ndcg@3"] == pytest.approx(1 / (1 + 1 / np.log2(3) + 0.5))
    assert summary["candidate_recall"] == pytest.approx(1 / 3)
    assert users.iloc[0]["n_truth"] == 3


def test_users_without_candidates_contribute_zero_to_macro_means():
    candidates = pd.DataFrame({"user_id": [1], "item_id": [10], "label": [1]})
    summary, users = evaluate_ranking(candidates, [1], {1: {10}, 2: {20}}, k=10)
    assert summary["n_users"] == 2
    assert summary["ndcg@10"] == pytest.approx(0.5)
    assert summary["candidate_recall"] == pytest.approx(0.5)
    assert summary["mean_candidates"] == pytest.approx(0.5)
    assert users.set_index("user_id").loc[2, "n_candidates"] == 0


def test_ties_use_item_id_ascending_and_accept_uncalibrated_rank_scores():
    candidates = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [30, 10, 20], "label": [0, 1, 0]})
    summary, _ = evaluate_ranking(candidates, [-99, -99, -99], {1: {10}}, k=1)
    assert summary["ndcg@1"] == 1
    assert "logloss" not in summary


def test_no_candidates_still_evaluates_all_truth_users():
    frame = pd.DataFrame(columns=["user_id", "item_id", "label"])
    summary, users = evaluate_ranking(frame, [], {1: {3}, 2: {4}})
    assert summary["n_users"] == 2
    assert summary["ndcg@10"] == 0
    assert users["recall"].eq(0).all()


def test_duplicate_candidates_are_rejected():
    frame = pd.DataFrame({"user_id": [1, 1], "item_id": [3, 3], "label": [1, 1]})
    with pytest.raises(ValueError, match="unique"):
        evaluate_ranking(frame, [1, 2], {1: {3}})


def test_identical_predictions_have_zero_paired_interval_even_after_shuffle():
    users = pd.DataFrame({"user_id": [1, 2, 3], "ndcg": [0.1, 0.7, 1.0]})
    result = paired_bootstrap(users, users.iloc[::-1], n_boot=100)
    assert result["delta"] == result["ci_low"] == result["ci_high"] == 0


def test_paired_bootstrap_direction_is_b_minus_a_and_cohort_must_match():
    a = pd.DataFrame({"user_id": [1, 2, 3], "ndcg": [0.1, 0.2, 0.4]})
    b = pd.DataFrame({"user_id": [3, 1, 2], "ndcg": [0.6, 0.3, 0.4]})
    result = paired_bootstrap(a, b, n_boot=100)
    assert result["delta"] == pytest.approx(0.2)
    assert result["ci_low"] == pytest.approx(0.2)
    assert result["ci_high"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="cohort"):
        paired_bootstrap(a, b.iloc[:2])
