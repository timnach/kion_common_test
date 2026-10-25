"""Behavioral checks for causal feature generation, using tiny known histories."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kion_recsys.features import DecayConfig, SnapshotFeatures, decay_affinity


CUTOFF = pd.Timestamp("2021-07-10")
DECAYS = {
    "fixed": DecayConfig(half_life=7),
    "adaptive": DecayConfig(half_life=3, alpha=2, beta=1, genre_weight=.75, adaptive_power=.5),
}


def feature_tables():
    events = pd.DataFrame({
        "user_id": [1, 1, 1, 1, 2, 2, 2, 1, 1, 3],
        "item_id": [1, 2, 3, 4, 1, 4, 5, 6, 7, 2],
        "timestamp": pd.to_datetime([
            "2021-06-10", "2021-07-03", "2021-07-07", "2021-07-09",
            "2021-07-01", "2021-07-08", "2021-07-09",
            "2021-07-10", "2021-07-10 12:00:00", "2021-07-11",
        ], format="mixed"),
        "label": [1] * 10,
        "total_dur": [100] * 10,
        "watched_pct": [90] * 10,
    })
    items = pd.DataFrame({
        "item_id": list(range(1, 9)),
        "genres": ["drama", "comedy", "drama, comedy", "drama", None, "action", "drama", None],
        "content_type": ["film", "series", "film", "film", None, "film", "film", None],
        "release_year": [2010, 2020, 2011, 2019, 2018, 2021, 2021, 2020],
    })
    users = pd.DataFrame({"user_id": [1, 2], "age": ["25_34", "35_44"]})
    candidates = pd.DataFrame({
        "query_id": ["a", "b", "c", "d", "e"],
        "user_id": [1, 2, 99, 1, 1],
        "item_id": [6, 3, 1, 8, 2],
        "label": [0, 1, 0, 1, 0],
        "retrieval_score": [1., 2., 1., 0., .5],
    }, index=[90, 4, 22, 17, 10])
    return events, items, users, candidates


def transform(events, items, users, candidates, *, cutoff=CUTOFF, configs=None):
    snapshot = SnapshotFeatures(events, items, users, cutoff)
    result = snapshot.transform(candidates, DECAYS if configs is None else configs)
    return snapshot, result


def test_future_and_same_day_perturbations_leave_every_feature_unchanged():
    events, items, users, candidates = feature_tables()
    history = events.loc[events.timestamp < CUTOFF].copy()
    original, expected = transform(history, items, users, candidates)
    future = events.loc[events.timestamp >= CUTOFF].copy()
    future["user_id"] = 1
    future["item_id"] = 2
    future["label"] = 0
    future["total_dur"] = 10**12
    augmented = pd.concat([history, *[future] * 20], ignore_index=True)
    perturbed, actual = transform(augmented, items, users, candidates)
    assert_frame_equal(actual, expected)
    assert perturbed.history.timestamp.max() < CUTOFF
    assert original.manual_columns == perturbed.manual_columns
    assert original.bank_columns == perturbed.bank_columns
    assert original.trend_columns == perturbed.trend_columns


def test_labels_and_aggregate_watch_signals_are_not_consumed_as_features():
    events, items, users, candidates = feature_tables()
    snapshot, before = transform(events, items, users, candidates)
    changed = events.copy()
    changed["label"] = 1 - changed.label
    changed["total_dur"] = np.arange(len(events)) * 10**9
    changed["watched_pct"] = np.nan
    changed_candidates = candidates.copy()
    changed_candidates["label"] = 1 - changed_candidates.label
    _, after = transform(changed, items, users, changed_candidates)
    assert_frame_equal(before.drop(columns="label"), after.drop(columns="label"))
    exposed = snapshot.manual_columns + snapshot.bank_columns + snapshot.trend_columns
    assert not {"label", "total_dur", "watched_pct", "query_id"}.intersection(exposed)
    assert after.label.tolist() == changed_candidates.label.tolist()


def test_history_order_and_dataframe_indices_do_not_change_features():
    events, items, users, candidates = feature_tables()
    _, expected = transform(events, items, users, candidates)
    _, actual = transform(events.sample(frac=1, random_state=3), items.sample(frac=1, random_state=4),
                          users, candidates)
    assert_frame_equal(actual, expected)
    assert actual.query_id.tolist() == candidates.query_id.tolist()
    assert actual.item_id.tolist() == candidates.item_id.astype(str).tolist()
    assert actual.index.tolist() == list(range(len(candidates)))


def test_window_boundaries_and_disjoint_lags_cover_fractional_days():
    # Two copies with distinct user-item pairs make user and item totals agree.
    ages = np.array([.25, 1, 1.01, 2, 7, 7.001, 14, 14.01, 21, 21.1, 30, 31, 90, 90.1])
    dates = CUTOFF - pd.to_timedelta(ages, unit="D")
    n = len(ages)
    events = pd.DataFrame({
        "user_id": [1] * n + list(range(100, 100 + n)) + [1],
        "item_id": list(range(1, n + 1)) + [99] * n + [100],
        "timestamp": list(dates) + list(dates) + [CUTOFF],
    })
    items = pd.DataFrame({"item_id": list(range(1, n + 1)) + [99, 100], "genres": "drama"})
    users = pd.DataFrame({"user_id": [1]})
    candidates = pd.DataFrame({"user_id": [1], "item_id": [99]})
    _, result = transform(events, items, users, candidates, configs={})
    row = result.iloc[0]
    for suffix, count in {"1d": 2, "7d": 5, "30d": 11}.items():
        assert row[f"user_count_{suffix}"] == count
        assert row[f"item_count_{suffix}"] == count
    expected_lags = {"1_1": 2, "2_2": 2, "7_7": 1, "8_8": 1, "14_14": 1,
                     "15_21": 2, "22_30": 2, "31_45": 1, "61_90": 1}
    for suffix, count in expected_lags.items():
        assert row[f"auto_item_lag_{suffix}d"] == count
        assert row[f"auto_user_lag_{suffix}d"] == count
        assert row[f"auto_genre_lag_{suffix}d"] == pytest.approx(count)
    # The most recent 90 days are partitioned exactly once; the 90.1d event is outside.
    assert row.filter(like="auto_user_lag_").sum() == 13
    assert row.filter(like="auto_item_lag_").sum() == 13


def test_positive_only_history_still_has_candidate_dependent_affinity():
    ages = np.array([1., 2., 3.])
    similarity = np.array([[1., 1., 1.], [0., 0., 0.], [.5, 0., 1.]])
    scores = decay_affinity(ages, similarity, np.zeros(3), DecayConfig(half_life=2))
    assert scores[0, 0] == pytest.approx(1)
    assert scores[1, 0] == 0
    assert 0 < scores[2, 0] < 1
    assert scores[0, 1] > scores[1, 1]
    assert np.all((scores[:, 2] >= 1) & (scores[:, 2] <= 3))


def test_uniform_age_shift_preserves_normalized_affinity_but_decreases_mass():
    ages = np.array([1., 3., 8.])
    similarity = np.array([[1., .5, 0.], [0., 0., 1.]])
    regularity = np.array([.2, .4, .6])
    config = DecayConfig(half_life=7, alpha=.5, beta=1, adaptive_power=0)
    before = decay_affinity(ages, similarity, regularity, config)
    after = decay_affinity(ages + 7, similarity, regularity, config)
    np.testing.assert_allclose(before[:, 0], after[:, 0], rtol=1e-12)
    np.testing.assert_allclose(before[:, 2], after[:, 2], rtol=1e-12)
    np.testing.assert_allclose(np.expm1(after[:, 1]), .5 * np.expm1(before[:, 1]), rtol=1e-12)
    assert np.all(after[:, 1] < before[:, 1])


def test_category_regularity_varies_by_event_and_changes_affinity():
    events = pd.DataFrame({"user_id": [1] * 4, "item_id": [1, 2, 3, 4],
                           "timestamp": CUTOFF - pd.to_timedelta([1, 3, 5, 2], unit="D")})
    items = pd.DataFrame({"item_id": [1, 2, 3, 4, 10, 11],
                          "genres": ["drama", "drama", "drama", "comedy", "drama", "comedy"]})
    users = pd.DataFrame({"user_id": [1]})
    candidates = pd.DataFrame({"user_id": [1, 1], "item_id": [10, 11]})
    configs = {"plain": DecayConfig(half_life=7, alpha=0),
               "regular": DecayConfig(half_life=7, alpha=4)}
    _, result = transform(events, items, users, candidates, configs=configs)
    # Three equally spaced drama dates provide support; a single comedy date does not.
    assert result.loc[0, "regular_affinity"] > result.loc[0, "plain_affinity"]
    assert result.loc[1, "regular_affinity"] < result.loc[1, "plain_affinity"]


@pytest.mark.parametrize("drop_genres", [False, True])
def test_cold_start_and_missing_metadata_are_finite(drop_genres):
    events, items, users, _ = feature_tables()
    if drop_genres:
        items = items.drop(columns="genres")
    candidates = pd.DataFrame({"user_id": [99, 1], "item_id": [8, 8]})
    _, result = transform(events, items, users, candidates)
    assert not result.isna().any().any()
    assert np.isfinite(result.select_dtypes(include=np.number).to_numpy()).all()
    assert result.loc[0, "user_count_all"] == 0
    assert result.loc[0, "user_recency"] == 365
    # Sentinel is measured in days, not seconds divided by 86400.
    assert (result.item_recency == 365).all()
    assert result.loc[1, "user_count_all"] == 4
    assert (result.fixed_affinity == 0).all()
    assert (result.adaptive_affinity == 0).all()
    assert result.loc[1, "fixed_ess"] > 0  # Unknown genres must not erase the history.


def test_empty_history_is_supported():
    events, items, users, candidates = feature_tables()
    _, cold = transform(events.iloc[:0], items, users, candidates)
    assert not cold.isna().any().any()
    assert (cold.user_count_all == 0).all()
    assert (cold.fixed_affinity == 0).all()


def test_empty_candidate_table_is_supported():
    events, items, users, candidates = feature_tables()
    _, empty = transform(events, items, users, candidates.iloc[:0])
    assert empty.empty
    assert "fixed_affinity" in empty


@pytest.mark.parametrize("age", [0, -1])
def test_decay_rejects_nonpast_history(age):
    with pytest.raises(ValueError, match="strictly precede"):
        decay_affinity([age], [[1.]], [0.], DecayConfig())
