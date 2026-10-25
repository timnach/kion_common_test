"""Causality and relational-aggregation checks with analytically known answers."""

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kion_recsys.relational import (
    FeatureSpec,
    SnapshotRelationalFeatures,
    candidate_specs,
    manual_specs,
)


CUTOFF = pd.Timestamp("2021-07-10")


def tables():
    events = pd.DataFrame({
        "user_id": [1, 1, 1, 1, 2, 2, 1, 1],
        "item_id": [1, 2, 3, 4, 1, 5, 5, 5],
        "timestamp": pd.to_datetime([
            "2021-07-09", "2021-07-07", "2021-07-03", "2021-07-03",
            "2021-07-08", "2021-07-01", "2021-07-10", "2021-07-11",
        ]),
        "label": [1] * 8,
        "total_dur": [1000] * 8,
    })
    items = pd.DataFrame({
        "item_id": [1, 2, 3, 4, 5, 6, 7],
        "genres": ["A, B", "B, C", "A", None, "A, B", None, "unknown"],
        "directors": ["D1, D2", "D2", "D3", None, "D1, D2", None, "unknown"],
        "countries": ["RU, US", "US", "FR", None, "RU, US", None, "unknown"],
        "content_type": ["film", "film", "series", None, "film", None, "unknown"],
    })
    candidates = pd.DataFrame({
        "user_id": [1, 2, 99, 1, 1, 1],
        "item_id": [5, 1, 5, 6, 7, 999],
        "label": [1, 0, 1, 1, 0, 1],
        "query_id": list("abcdef"),
    }, index=[5, 2, 5, 10, 11, 1])
    return events, items, candidates


def run(events, items, candidates, specs, cutoff=CUTOFF):
    return SnapshotRelationalFeatures(events, items, cutoff).transform(candidates, specs)


def test_future_target_and_event_value_invariance():
    events, items, candidates = tables()
    specs = candidate_specs()
    past = events.loc[events.timestamp < CUTOFF]
    expected = run(past, items, candidates, specs)
    changed = events.copy()
    changed["label"] = 0
    changed["total_dur"] = 10**12
    future = changed.loc[changed.timestamp >= CUTOFF].copy()
    future["item_id"] = 1
    future["user_id"] = 2
    changed = pd.concat([changed, *[future] * 20], ignore_index=True)
    other_candidates = candidates.copy()
    other_candidates["label"] = 1 - candidates.label
    other_candidates["query_id"] = "unrelated"
    actual = run(changed, items, other_candidates, specs)
    assert_frame_equal(actual, expected)
    assert set(actual.columns) == {spec.name for spec in specs}


def test_multi_value_any_overlap_counts_each_history_event_once():
    events, items, candidates = tables()
    specs = [FeatureSpec(relation, agg) for relation in ("genre", "director", "country", "content_type")
             for agg in ("count", "share", "recency", "active_days")]
    first = run(events, items, candidates.iloc[:1], specs).iloc[0]
    # Candidate A,B intersects the first history item twice but counts it once.
    # Three matching events occur on three dates; one metadata-less event is in
    # the share denominator, as absence of metadata does not erase consumption.
    expected = {"genre": (3, .75, 1, 3), "director": (2, .5, 1, 2),
                "country": (2, .5, 1, 2), "content_type": (2, .5, 1, 2)}
    for relation, answers in expected.items():
        for agg, answer in zip(("count", "share", "recency", "active_days"), answers):
            assert first[FeatureSpec(relation, agg).name] == pytest.approx(answer)


def test_user_and_item_relations_have_different_history_scopes():
    events, items, candidates = tables()
    specs = [FeatureSpec(relation, agg) for relation in ("user", "item")
             for agg in ("count", "share", "recency", "active_days")]
    result = run(events, items, candidates, specs)
    first = result.iloc[0]
    assert first[FeatureSpec("user", "count").name] == 4
    assert first[FeatureSpec("user", "active_days").name] == 3
    assert first[FeatureSpec("user", "share").name] == 1
    # Candidate item 5 has one past interaction, by user 2 nine days ago.
    assert first[FeatureSpec("item", "count").name] == 1
    assert first[FeatureSpec("item", "share").name] == pytest.approx(1 / 6)
    assert first[FeatureSpec("item", "recency").name] == 9
    assert first[FeatureSpec("item", "active_days").name] == 1
    # Cold user has no personal history but global item information is retained.
    assert result.iloc[2][FeatureSpec("user", "count").name] == 0
    assert result.iloc[2][FeatureSpec("user", "recency").name] == 365
    assert result.iloc[2][FeatureSpec("item", "count").name] == 1


def test_exact_interval_boundaries_partition_fractional_days():
    ages = [0, .25, 1, 1.001, 3, 3.001, 7, 7.001, 30]
    events = pd.DataFrame({"user_id": [1] * len(ages), "item_id": [1] * len(ages),
                           "timestamp": CUTOFF - pd.to_timedelta(ages, unit="D")})
    items = pd.DataFrame({"item_id": [1, 2], "genres": ["a,b", "a,b"]})
    candidates = pd.DataFrame({"user_id": [1], "item_id": [1]})
    specs = [FeatureSpec(relation, "count", lo, hi)
             for relation in ("user", "item", "genre")
             for lo, hi in ((0, 1), (1, 3), (3, 7), (7, None))]
    result = run(events, items, candidates, specs).iloc[0]
    assert np.array_equal(result.to_numpy(), [2] * 12)
    assert run(events, items, candidates, [FeatureSpec("user", "count")]).iloc[0, 0] == 8


def test_decay_numerators_and_denominators_use_the_same_kernel():
    events = pd.DataFrame({"user_id": [1, 1, 1, 2], "item_id": [1, 2, 3, 1],
                           "timestamp": CUTOFF - pd.to_timedelta([1, 2, 3, 2], unit="D")})
    items = pd.DataFrame({"item_id": [1, 2, 3, 4], "genres": ["a", "b", "a", "a"]})
    candidates = pd.DataFrame({"user_id": [1, 1], "item_id": [4, 1]})
    specs = [FeatureSpec("genre", agg, half_life=1) for agg in ("count", "share")]
    specs += [FeatureSpec("genre", agg, 1, 3, half_life=1) for agg in ("count", "share")]
    specs += [FeatureSpec("item", agg, half_life=1) for agg in ("count", "share")]
    result = run(events, items, candidates, specs)
    assert result.iloc[0, 0] == pytest.approx(.5 + .125)
    assert result.iloc[0, 1] == pytest.approx(.625 / .875)
    assert result.iloc[0, 2] == pytest.approx(.125)
    assert result.iloc[0, 3] == pytest.approx(.125 / .375)
    assert result.iloc[1, 4] == pytest.approx(.75)
    assert result.iloc[1, 5] == pytest.approx(.75 / 1.125)


def test_active_days_deduplicates_dates_and_recency_is_capped():
    events = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [1, 1, 2],
                           "timestamp": pd.to_datetime(["2021-07-09 01:00", "2021-07-09 23:00", "2020-01-01 00:00"])})
    items = pd.DataFrame({"item_id": [1, 2, 3], "genres": ["a", "b", "b"]})
    candidates = pd.DataFrame({"user_id": [1, 1], "item_id": [1, 3]})
    specs = [FeatureSpec(relation, agg) for relation in ("user", "item", "genre")
             for agg in ("active_days", "recency")]
    result = run(events, items, candidates, specs)
    assert result.iloc[0][FeatureSpec("user", "active_days").name] == 2
    assert result.iloc[0][FeatureSpec("item", "active_days").name] == 1
    assert result.iloc[0][FeatureSpec("genre", "active_days").name] == 1
    assert result.iloc[0][FeatureSpec("genre", "recency").name] == pytest.approx(1 / 24)
    assert result.iloc[1][FeatureSpec("genre", "recency").name] == 365


def test_order_duplicate_indices_and_repeated_calls_are_preserved():
    events, items, candidates = tables()
    specs = candidate_specs()
    compiler = SnapshotRelationalFeatures(events, items, CUTOFF)
    expected = compiler.transform(candidates, specs)
    actual = run(events.sample(frac=1, random_state=2), items.sample(frac=1, random_state=3), candidates, specs)
    assert_frame_equal(actual, expected)
    assert actual.index.tolist() == candidates.index.tolist()
    order = [5, 1, 2, 0, 4, 3]
    assert_frame_equal(compiler.transform(candidates.iloc[order], specs), expected.iloc[order])
    assert_frame_equal(compiler.transform(candidates, specs), expected)


@pytest.mark.parametrize("remove_columns", [False, True])
def test_missing_metadata_never_matches_and_all_values_are_finite(remove_columns):
    events, items, candidates = tables()
    if remove_columns:
        items = items[["item_id"]]
    specs = candidate_specs()
    result = run(events, items, candidates, specs)
    assert np.isfinite(result.to_numpy()).all()
    relevant_rows = range(len(candidates)) if remove_columns else [2, 3, 4, 5]
    for spec in specs:
        if spec.relation not in {"user", "item"}:
            expected = 365 if spec.aggregation == "recency" else 0
            assert (result.iloc[list(relevant_rows)][spec.name] == expected).all()


def test_empty_inputs_and_all_future_history_have_explicit_defaults():
    events, items, candidates = tables()
    specs = candidate_specs()
    expected = run(events.iloc[:0], items, candidates, specs)
    actual = run(events.assign(timestamp=CUTOFF), items, candidates, specs)
    assert_frame_equal(actual, expected)
    for spec in specs:
        assert (actual[spec.name] == (365 if spec.aggregation == "recency" else 0)).all()
    empty = run(events, items, candidates.iloc[:0], specs)
    assert empty.shape == (0, len(specs))
    assert run(events, items, candidates, []).shape == (len(candidates), 0)


def test_specs_are_validated_immutable_serializable_and_uniquely_named():
    spec = FeatureSpec("genre", "share", 1, 3, 7)
    assert FeatureSpec(**json.loads(json.dumps(spec.to_dict()))) == spec
    assert spec.name == FeatureSpec("genre", "share", 1.0, 3.0, 7.0).name
    with pytest.raises(FrozenInstanceError):
        spec.relation = "country"
    manual = manual_specs()
    candidates = candidate_specs()
    assert len(manual) == 21
    assert len(candidates) == 157
    assert len(set(spec.name for spec in candidates)) == len(candidates)
    assert set(manual).issubset(candidates)
    assert candidates == candidate_specs()


@pytest.mark.parametrize("kwargs", [
    {"relation": "label"}, {"aggregation": "label"}, {"time_kind": "future"},
    {"lower_age_days": -1}, {"lower_age_days": None}, {"lower_age_days": True},
    {"upper_age_days": 0}, {"upper_age_days": np.inf}, {"half_life": np.nan},
    {"half_life": 0}, {"aggregation": "recency", "half_life": 7},
])
def test_invalid_specs_fail_before_computation(kwargs):
    settings = {"relation": "genre", "aggregation": "count", **kwargs}
    with pytest.raises(ValueError):
        FeatureSpec(**settings)


def test_duplicate_specs_and_ambiguous_metadata_are_rejected():
    events, items, candidates = tables()
    spec = FeatureSpec("genre", "count")
    with pytest.raises(ValueError, match="Duplicate feature"):
        run(events, items, candidates, [spec, spec])
    with pytest.raises(ValueError, match="Duplicate item_id"):
        run(events, pd.concat([items, items.iloc[:1]]), candidates, [spec])
