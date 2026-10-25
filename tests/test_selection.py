import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from kion_recsys.selection import CMIMSelector, discrete_cmi
import kion_recsys.selection as selection_module


def _reference_cmi(x, y, z):
    """Independent literal frequency formula for optimized-path verification."""
    context = [tuple(row) for row in np.asarray(z).reshape(len(x), -1)]
    nxyz = Counter(zip(x, y, context))
    nxz = Counter(zip(x, context))
    nyz = Counter(zip(y, context))
    nz = Counter(context)
    return sum(
        count / len(x) * np.log(count * nz[c] / (nxz[a, c] * nyz[b, c]))
        for (a, b, c), count in nxyz.items()
    )


def test_xor_has_conditional_but_not_marginal_information():
    x = np.tile([0, 0, 1, 1], 100)
    z = np.tile([0, 1, 0, 1], 100)
    y = x ^ z
    assert discrete_cmi(x, y) == pytest.approx(0.0, abs=1e-12)
    assert discrete_cmi(x, y, z) == pytest.approx(np.log(2))


def test_redundancy_is_suppressed_and_constant_is_ignored():
    rng = np.random.default_rng(123)
    a = rng.integers(0, 2, 20_000)
    b = rng.integers(0, 2, len(a))
    y = a | b
    frame = pd.DataFrame({"signal": a, "duplicate": a, "complement": b, "constant": 1})
    selector = CMIMSelector(n_features=4).fit(frame, y)
    assert "complement" in selector.selected_features_
    assert ("signal" in selector.selected_features_) != ("duplicate" in selector.selected_features_)
    assert "constant" not in selector.selected_features_
    assert len(selector.selected_features_) == 2
    assert all(row["score"] > 0 for row in selector.scores_)


def test_train_bins_preserve_zero_and_are_not_refit_at_transform():
    x = np.repeat([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 20)
    frame = pd.DataFrame({"count": x})
    selector = CMIMSelector(n_bins=4).fit(frame, x > 2)
    before = json.dumps(selector.to_dict(), sort_keys=True)
    bins = selector._apply_bins(x, selector.bin_edges_["count"])
    assert set(bins[x == 0]).isdisjoint(bins[x != 0])
    transformed = selector.transform(pd.DataFrame({"count": [-1000, 1e9]}))
    assert transformed["count"].tolist() == [-1000, 1e9]
    assert json.dumps(selector.to_dict(), sort_keys=True) == before


def test_constants_and_missing_values_are_well_defined():
    frame = pd.DataFrame({"zero": [0] * 8, "one": [1] * 8, "missing": [np.nan] * 8})
    selector = CMIMSelector().fit(frame, np.tile([0, 1], 4))
    assert selector.selected_features_ == []
    assert set(selector.constant_features_) == set(frame.columns)
    assert selector.transform(frame).shape == (8, 0)
    assert discrete_cmi(np.ones(8), np.tile([0, 1], 4)) == 0


def test_quantile_bins_keep_rare_nonzero_discrete_values():
    x = np.r_[np.ones(990), np.full(10, 2)]
    selector = CMIMSelector().fit(pd.DataFrame({"rare": x}), x == 2)
    assert selector.selected_features_ == ["rare"]


def test_context_removes_a_coarse_confounder_and_rejects_identifiers():
    strata = np.repeat([0, 1], 100)
    frame = pd.DataFrame({"activity_proxy": strata})
    assert CMIMSelector().fit(frame, strata).selected_features_ == ["activity_proxy"]
    assert CMIMSelector().fit(frame, strata, pd.DataFrame({"activity": strata})).selected_features_ == []
    with pytest.raises(ValueError, match="IDs"):
        CMIMSelector().fit(frame, strata, pd.DataFrame({"user_id": strata}))
    with pytest.raises(ValueError, match="16"):
        CMIMSelector().fit(frame, strata, pd.DataFrame({"activity": np.arange(len(strata))}))


def test_marginal_relevance_uses_same_train_bins_without_context():
    strata = np.repeat([0, 1], 100)
    frame = pd.DataFrame({"activity_proxy": strata, "constant": 0})
    selector = CMIMSelector().fit(frame, strata, pd.DataFrame({"activity": strata}))
    assert selector.relevance_["activity_proxy"] == 0
    assert selector.marginal_relevance_["activity_proxy"] == pytest.approx(np.log(2))
    assert "constant" not in selector.marginal_relevance_
    record = json.loads(json.dumps(selector.to_dict(), allow_nan=False))
    assert record["marginal_relevance"] == selector.marginal_relevance_


@pytest.mark.parametrize("path", ["dense", "sparse", "overflow_fallback"])
def test_counting_paths_match_independent_frequency_formula(path, monkeypatch):
    rng = np.random.default_rng(1024)
    x = rng.integers(0, 9, 2000)
    y = ((x > 4) ^ rng.integers(0, 2, len(x))).astype(int)
    z = rng.integers(0, 3, size=(len(x), 3))
    expected = _reference_cmi(x, y, z)
    if path != "dense":
        monkeypatch.setattr(selection_module, "_MAX_DENSE_CELLS", 1)
    if path == "overflow_fallback":
        monkeypatch.setattr(selection_module, "_MAX_ENCODED_STATE", 1)
    assert discrete_cmi(x, y, z) == pytest.approx(expected, abs=1e-13)


def test_high_cardinality_and_large_raw_ids_use_sparse_observed_cells():
    x = np.arange(2000, dtype=np.int64) + 2**62
    # 4 million possible joint cells, but only 2000 observed cells.
    assert discrete_cmi(x, x) == pytest.approx(np.log(len(x)))


def test_missing_states_and_multicolumn_context_are_supported():
    x = np.array([None, "a", None, "a", "b", "b"], dtype=object)
    y = np.array([0, 1, 0, 1, 0, 1])
    z = np.array([[None, 0], [None, 0], ["s", 1], ["s", 1], ["s", 1], ["s", 1]], dtype=object)
    assert discrete_cmi(x, y, z) == pytest.approx(_reference_cmi(x, y, z))


def test_bad_shapes_are_rejected():
    with pytest.raises(ValueError):
        discrete_cmi([0, 1], [0])
    with pytest.raises(ValueError):
        discrete_cmi([], [])
    with pytest.raises(ValueError):
        discrete_cmi([0, 1], [0, 1], [[0]])
