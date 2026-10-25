"""Regression checks for causality and the actual ItemKNN score definition."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from kion_recsys.collaborative import CollaborativeRetriever


def make_events(pairs, date="2021-01-01"):
    frame = pd.DataFrame(pairs, columns=["user_id", "item_id"])
    frame = frame.astype({"user_id": "int64", "item_id": "int64"})
    frame["timestamp"] = pd.Timestamp(date)
    return frame


@pytest.fixture
def fixture():
    events = make_events([(1, 10), (2, 10), (2, 20), (3, 10), (3, 20), (4, 20), (4, 30), (5, 30)])
    items = pd.DataFrame({"item_id": [30, 20, 10, 40, 50]})
    return events, items


def test_binary_cosine_matches_hand_calculation(fixture):
    events, items = fixture
    model = CollaborativeRetriever(events, items, "2021-01-05", neighbors=10, shrinkage=0, block_size=2)
    # Frequencies: A=3, B=3, C=2.  Common users: AB=2, BC=1, AC=0.
    pairs = pd.DataFrame({"user_id": [1, 1, 2, 5], "item_id": [20, 30, 30, 20]})
    np.testing.assert_allclose(model.scores_for(pairs), [2 / 3, 0, 1 / np.sqrt(6), 1 / np.sqrt(6)], rtol=1e-6)
    assert model.n_interactions == 8
    assert sparse.isspmatrix_csr(model.similarity)
    assert np.count_nonzero(model.similarity.diagonal()) == 0
    assert model.eligible_items == {10, 20, 30}
    assert model.max_block_nnz <= 2 * model.n_items
    assert model.fit_seconds >= 0


def test_shrinkage_discounts_low_support(fixture):
    events, items = fixture
    model = CollaborativeRetriever(events, items, "2021-01-05", shrinkage=2)
    actual = model.scores_for(pd.DataFrame({"user_id": [1, 5], "item_id": [20, 20]}))
    np.testing.assert_allclose(actual, [(2 / 3) * (2 / 4), (1 / np.sqrt(6)) * (1 / 3)], rtol=1e-6)


def test_strict_cutoff_and_future_invariance(fixture):
    events, items = fixture
    cutoff = "2021-01-05 12:00:00"
    exact = make_events([(1, 30), (8, 40)], cutoff)
    future = make_events([(1, 50), (2, 40), (90, 30)], "2021-01-06")
    past_model = CollaborativeRetriever(events, items, cutoff, shrinkage=0)
    augmented_model = CollaborativeRetriever(pd.concat([events, exact, future]), items, cutoff, shrinkage=0)
    pd.testing.assert_frame_equal(past_model.candidates_for([1, 2, 8, 90]), augmented_model.candidates_for([1, 2, 8, 90]))
    assert augmented_model.n_history_rows == len(events)
    assert augmented_model.eligible_items == past_model.eligible_items
    assert (augmented_model.similarity != past_model.similarity).nnz == 0
    # A time one second before the exact cutoff is retained, without truncating
    # an intraday prediction back to midnight.
    before = make_events([(8, 10)], "2021-01-05 11:59:59")
    extended = CollaborativeRetriever(pd.concat([events, before]), items, cutoff)
    assert 8 in extended.user_ids


def test_retrieval_excludes_seen_and_unavailable_and_cold_users(fixture):
    events, items = fixture
    # A past event without item metadata cannot establish catalogue eligibility.
    with_unknown = pd.concat([events, make_events([(1, 99), (9, 99)])])
    model = CollaborativeRetriever(with_unknown, items, "2021-01-05", shrinkage=0)
    result = model.candidates_for([5, 2, 1, 9, 999])
    seen = set(zip(events.user_id, events.item_id))
    assert all(pair not in seen for pair in zip(result.user_id, result.item_id))
    assert set(result.item_id) <= {10, 20, 30}
    assert not {9, 999}.intersection(result.user_id)
    assert result.loc[result.user_id == 2, "item_id"].tolist() == [30]
    assert result.loc[result.user_id == 1, "knn_rank"].tolist() == [1]


def test_binary_duplicates_do_not_change_weight(fixture):
    events, items = fixture
    original = CollaborativeRetriever(events, items, "2021-01-05", shrinkage=0)
    duplicated = CollaborativeRetriever(pd.concat([events, events.iloc[[0, 1, 1]]]), items, "2021-01-05", shrinkage=0)
    assert duplicated.n_interactions == original.n_interactions
    assert duplicated.n_history_rows == original.n_history_rows + 3
    pd.testing.assert_frame_equal(original.candidates_for([1, 2, 5]), duplicated.candidates_for([1, 2, 5]))


def test_ties_are_deterministic_at_neighbor_and_retrieval_boundaries():
    events = make_events([(1, 10), (2, 10), (2, 20), (3, 10), (3, 30), (4, 10), (4, 40)])
    items = pd.DataFrame({"item_id": [40, 30, 20, 10]})
    model = CollaborativeRetriever(events, items, "2021-01-05", neighbors=2, shrinkage=0, block_size=1)
    shuffled = CollaborativeRetriever(events.sample(frac=1, random_state=4), items.iloc[::-1], "2021-01-05", neighbors=2, shrinkage=0, block_size=3)
    assert model.candidates_for([1], k=1).item_id.tolist() == [20]
    assert model.candidates_for([1], k=10).item_id.tolist() == [20, 30]
    pd.testing.assert_frame_equal(model.candidates_for([4, 1, 2, 1]), shuffled.candidates_for([2, 4, 1]))
    assert np.max(np.diff(model.similarity.indptr)) <= 2


def test_scores_for_preserves_order_duplicates_and_nonretrieved_pairs(fixture):
    events, items = fixture
    model = CollaborativeRetriever(events, items, "2021-01-05", shrinkage=0)
    pairs = pd.DataFrame({"user_id": [5, 999, 2, 1, 5, 1], "item_id": [20, 10, 30, 40, 20, 20]}, index=[9, 2, 2, 8, 7, 1])
    result = model.scores_for(pairs)
    np.testing.assert_allclose(result, [1 / np.sqrt(6), 0, 1 / np.sqrt(6), 0, 1 / np.sqrt(6), 2 / 3], rtol=1e-6)
    assert result.dtype == np.float64
    # Scoring is independent of whether retrieval was called and its cap.
    model.candidates_for([1, 2, 5], k=0)
    np.testing.assert_array_equal(result, model.scores_for(pairs))


def test_sparse_operations_do_not_need_dense_matrix(monkeypatch, fixture):
    def forbidden(*args, **kwargs):
        raise AssertionError("dense conversion is forbidden")
    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbidden)
    monkeypatch.setattr(sparse.csr_matrix, "todense", forbidden)
    events, items = fixture
    model = CollaborativeRetriever(events, items, "2021-01-05", block_size=1)
    result = model.candidates_for([1, 2, 5])
    assert len(result) > 0
    np.testing.assert_allclose(model.scores_for(result), result.knn_score)
    assert model.max_block_nnz <= model.n_items


def test_empty_snapshot_and_single_item_are_well_defined():
    items = pd.DataFrame({"item_id": [10]})
    empty = CollaborativeRetriever(make_events([(1, 10)]), items, "2020-12-01")
    assert empty.candidates_for([1]).empty
    np.testing.assert_array_equal(empty.scores_for(pd.DataFrame({"user_id": [1], "item_id": [10]})), [0])
    singleton = CollaborativeRetriever(make_events([(1, 10), (2, 10)]), items, "2021-01-05")
    assert singleton.similarity.nnz == 0
    assert singleton.candidates_for([1]).empty
    no_rows = CollaborativeRetriever(pd.DataFrame(columns=["user_id", "item_id", "timestamp"]), items, "2021-01-05")
    assert no_rows.n_items == 0


@pytest.mark.parametrize("kwargs", [{"neighbors": 0}, {"neighbors": 1.5}, {"block_size": 0}, {"shrinkage": -1}, {"shrinkage": np.inf}])
def test_invalid_fit_parameters(kwargs, fixture):
    events, items = fixture
    with pytest.raises(ValueError):
        CollaborativeRetriever(events, items, "2021-01-05", **kwargs)
