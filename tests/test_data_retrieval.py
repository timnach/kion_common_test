import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from kion_recsys.data import data_summary, load_data, parse_genres
from kion_recsys.retrieval import SnapshotRetriever


def tables():
    events = pd.DataFrame({
        "user_id": [1, 2, 3, 3, 4, 5, 6],
        "item_id": [1, 2, 2, 3, 3, 4, 5],
        "timestamp": pd.to_datetime(["2021-07-01", "2021-07-05", "2021-07-06", "2021-07-06",
                                     "2021-07-07", "2021-07-10", "2021-07-11"]),
    })
    items = pd.DataFrame({"item_id": [1, 2, 3, 4, 5, 6],
                          "genres": ["Drama, comedy", "drama", "comedy", None, "future", "unseen"]})
    return events, items


def test_future_rows_and_same_timestamp_cannot_change_candidates():
    events, items = tables()
    original = SnapshotRetriever(events, items, "2021-07-10", global_k=10, recent_k=10)
    past = events[events.timestamp < pd.Timestamp("2021-07-10")]
    truncated = SnapshotRetriever(past, items, "2021-07-10", global_k=10, recent_k=10)
    assert_frame_equal(original.candidates_for([1, 99]), truncated.candidates_for([1, 99]))
    assert_series_equal(original.item_counts, truncated.item_counts)
    assert original.eligible_items == {1, 2, 3}
    assert original.history.timestamp.max() < original.cutoff
    assert not {4, 5, 6}.intersection(original.candidates_for([99]).item_id)


def test_seen_exclusion_cold_fallback_and_source_dtypes():
    events, items = tables()
    retriever = SnapshotRetriever(events, items, "2021-07-10", global_k=10, recent_k=10)
    candidates = retriever.candidates_for([1, 99])
    assert set(candidates[candidates.user_id == 1].item_id) == {2, 3}
    assert set(candidates[candidates.user_id == 99].item_id) == {1, 2, 3}
    assert candidates.loc[candidates.user_id == 99, "source_genre"].sum() == 0
    assert (candidates[["source_global", "source_recent", "source_genre"]].dtypes == np.dtype("uint8")).all()
    assert (candidates.retrieval_score > 0).all()


def test_deterministic_ties_cap_and_input_order():
    events, items = tables()
    a = SnapshotRetriever(events, items, "2021-07-10", global_k=2, recent_k=0, genre_k=0, max_candidates=1)
    b = SnapshotRetriever(events.sample(frac=1, random_state=5), items.sample(frac=1, random_state=4),
                          "2021-07-10", global_k=2, recent_k=0, genre_k=0, max_candidates=1)
    assert_frame_equal(a.candidates_for([99, 1, 99]), b.candidates_for([1, 99]))
    # Items 2 and 3 have the same popularity; the smaller ID wins for a cold user.
    assert a.candidates_for([99]).item_id.tolist() == [2]
    assert a.candidates_for([]).empty


@pytest.mark.parametrize("value, expected", [
    (None, ()), (np.nan, ()), (pd.NA, ()), ("", ()), ("null", ()),
    ("comedy, Drama, comedy", ("comedy", "drama")),
    (["Drama", None, "comedy"], ("comedy", "drama")),
    ('["drama", "comedy"]', ("comedy", "drama")),
])
def test_parse_genres(value, expected):
    assert parse_genres(value) == expected


def write_data(path, *, duplicate=False, invalid_id=False):
    path.mkdir(parents=True, exist_ok=True)
    interactions = pd.DataFrame({"user_id": [1, 1 if duplicate else 2], "item_id": [1, 1 if duplicate else 2],
                                 "last_watch_dt": ["2021-07-01", "2021-07-02"],
                                 "total_dur": [0, 60], "watched_pct": [0, 50]})
    if invalid_id:
        interactions["item_id"] = [1.5, 2]
    interactions.to_csv(path / "interactions.csv", index=False)
    pd.DataFrame({"item_id": [1, 2], "genres": ["drama", None]}).to_csv(path / "items.csv", index=False)
    # Missing metadata for user 2 must not drop their interaction.
    pd.DataFrame({"user_id": [1], "age": ["age_25_34"]}).to_csv(path / "users.csv", index=False)


@pytest.mark.parametrize("nested", [False, True])
def test_load_preserves_low_quality_and_missing_user_metadata(tmp_path, nested):
    source = tmp_path / "data_original" if nested else tmp_path
    write_data(source)
    events, items, users = load_data(tmp_path)
    assert len(events) == 2
    assert events["total_dur"].tolist() == [0, 60]
    assert events["watched_pct"].tolist() == [0, 50]
    assert events.timestamp.dt.hour.eq(0).all()
    summary = data_summary(events, items, users)
    assert summary["event_users_without_metadata"] == 1
    assert summary["duplicate_user_item_pairs"] == 0
    assert summary["n_calendar_days"] == 2
    json.dumps(summary, allow_nan=False)


def test_duplicate_pair_rejected(tmp_path):
    write_data(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="duplicate user-item"):
        load_data(tmp_path)


def test_fractional_id_rejected(tmp_path):
    write_data(tmp_path, invalid_id=True)
    with pytest.raises(ValueError, match="nonnegative integer"):
        load_data(tmp_path)


def test_empty_snapshot_and_metadata_without_genres():
    events, items = tables()
    retriever = SnapshotRetriever(events, items.drop(columns="genres"), "2021-06-01")
    assert retriever.candidates_for([1, 99]).empty
    assert all(not value for value in retriever.genre_map.values())
