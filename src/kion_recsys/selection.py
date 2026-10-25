"""Train-only, discretized CMIM feature selection.

This is a pairwise conditional-mutual-information heuristic, not an information
bottleneck objective and not joint I(X; Y | all selected features). Histogram
estimates are biased in small samples; held-out ranking quality remains the
criterion for deciding whether the selected feature set is useful.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_MAX_DENSE_CELLS = 1_000_000
_MAX_ENCODED_STATE = np.iinfo(np.int64).max


def _vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _codes(values: Any) -> np.ndarray:
    """Encode a discrete variable, treating missing values as one state."""
    return pd.factorize(values, sort=False, use_na_sentinel=False)[0]


def _context_codes(context: Any, n_rows: int) -> np.ndarray:
    if context is None:
        return np.zeros(n_rows, dtype=np.int64)
    array = np.asarray(context)
    if array.ndim == 1:
        if len(array) != n_rows:
            raise ValueError("context must have one row per observation")
        return _codes(array)
    if array.ndim != 2 or array.shape[0] != n_rows:
        raise ValueError("context must have one row per observation")
    if array.shape[1] == 0:
        return np.zeros(n_rows, dtype=np.int64)
    encoded = _codes(array[:, 0])
    # Compact after each column, so even many context columns do not create a
    # combinatorial allocation. The ordinary low-cardinality path hashes 1-D
    # integer keys instead of repeatedly sorting a matrix of training rows.
    for column in range(1, array.shape[1]):
        next_codes = _codes(array[:, column])
        next_states = int(next_codes.max()) + 1
        states = int(encoded.max()) + 1
        if states * next_states <= _MAX_ENCODED_STATE:
            encoded = _codes(encoded * next_states + next_codes)
        else:
            encoded = np.unique(
                np.column_stack([encoded, next_codes]), axis=0, return_inverse=True
            )[1]
    return encoded


def _cmi_encoded(xc: np.ndarray, yc: np.ndarray, zc: np.ndarray) -> float:
    """CMI for compact nonnegative integer codes; never allocate by raw IDs."""
    n_rows = len(xc)
    nx, ny, nz_states = (int(c.max()) + 1 for c in (xc, yc, zc))
    n_cells = nx * ny * nz_states  # Python integers keep overflow checks safe.
    if n_cells <= _MAX_DENSE_CELLS:
        joint = (zc * nx + xc) * ny + yc
        tensor = np.bincount(joint, minlength=n_cells).reshape(nz_states, nx, ny)
        zi, xi, yi = np.nonzero(tensor)
        counts = tensor[zi, xi, yi].astype(float)
        nz = tensor.sum(axis=(1, 2))[zi].astype(float)
        nzx = tensor.sum(axis=2)[zi, xi].astype(float)
        nzy = tensor.sum(axis=1)[zi, yi].astype(float)
    else:
        # High-cardinality inputs use only observed cells, not a potentially
        # enormous dense tensor. Retain a matrix fallback if integer packing
        # cannot represent the full Cartesian state space safely.
        if n_cells <= _MAX_ENCODED_STATE:
            joint = (zc * nx + xc) * ny + yc
            observed, counts = np.unique(joint, return_counts=True)
            zi, xi, yi = observed // (nx * ny), (observed // ny) % nx, observed % ny
            _, zx_inverse = np.unique(observed // ny, return_inverse=True)
            _, zy_inverse = np.unique(zi * ny + yi, return_inverse=True)
        else:
            cells, counts = np.unique(
                np.column_stack([zc, xc, yc]), axis=0, return_counts=True
            )
            zi = cells[:, 0]
            _, zx_inverse = np.unique(cells[:, :2], axis=0, return_inverse=True)
            _, zy_inverse = np.unique(cells[:, [0, 2]], axis=0, return_inverse=True)
        counts = counts.astype(float)
        nz = np.bincount(zc)[zi].astype(float)
        nzx = np.bincount(zx_inverse, weights=counts)[zx_inverse]
        nzy = np.bincount(zy_inverse, weights=counts)[zy_inverse]
    ratio = counts * nz / (nzx * nzy)
    result = np.sum((counts / n_rows) * np.log(ratio))
    return max(0.0, float(result))  # Roundoff can produce a tiny negative MI.


def discrete_cmi(x: Any, y: Any, z: Any = None) -> float:
    """Empirical I(X; Y | Z) in nats for discrete observations.

    ``z`` may be one variable or a matrix of jointly conditioned variables.
    Inputs are discrete states, not continuous measurements: callers must bin
    continuous features first. No probability smoothing is applied.
    """
    x_array, y_array = _vector(x, "x"), _vector(y, "y")
    n_rows = len(x_array)
    if n_rows == 0 or len(y_array) != n_rows:
        raise ValueError("x and y must have the same nonzero length")
    return _cmi_encoded(_codes(x_array), _codes(y_array), _context_codes(z, n_rows))


class CMIMSelector:
    """Greedy CMIM with optional low-cardinality historical context.

    First choose the largest I(X; Y | C). At subsequent steps score X as
    min(I(X; Y | C), min_{S selected} I(X; Y | C, S)). This penalizes pairwise
    redundancy, but can miss pure higher-order interactions such as XOR.

    All bin boundaries and scores are fitted using the provided training rows.
    ``transform`` returns the selected original numeric columns, suitable for
    CatBoost; discretized values are used only by the selector. ``relevance_``
    contains I(X; Y | C); ``marginal_relevance_`` contains I(X; Y), using exactly
    the same training bins for a genuine marginal-MI control.
    """

    def __init__(self, n_features: int = 12, n_bins: int = 5, min_score: float = 0.0,
                 max_context_columns: int = 3, max_context_states: int = 32):
        if not isinstance(n_features, int) or n_features < 1:
            raise ValueError("n_features must be a positive integer")
        if not isinstance(n_bins, int) or n_bins < 2:
            raise ValueError("n_bins must be an integer >= 2")
        if not np.isfinite(min_score) or min_score < 0:
            raise ValueError("min_score must be finite and nonnegative")
        if not isinstance(max_context_columns, int) or not 1 <= max_context_columns <= 4:
            raise ValueError("max_context_columns must be between 1 and 4")
        if not isinstance(max_context_states, int) or not 1 <= max_context_states <= 128:
            raise ValueError("max_context_states must be between 1 and 128")
        self.max_context_columns = max_context_columns
        self.max_context_states = max_context_states
        self.n_features = n_features
        self.n_bins = n_bins
        self.min_score = float(min_score)

    def _fit_bins(self, values: np.ndarray) -> dict[str, Any]:
        finite = values[np.isfinite(values)]
        separate_zero = bool(np.any(finite == 0))
        population = finite[finite != 0] if separate_zero else finite
        n_quantile_bins = self.n_bins - int(separate_zero)
        if len(population) and n_quantile_bins > 1:
            quantiles = np.arange(1, n_quantile_bins) / n_quantile_bins
            edges = np.unique(np.quantile(population, quantiles))
            # A cut at the minimum is useful for sparse/low-cardinality
            # variables when equality goes to the lower bin. A maximum cut
            # cannot separate observations and is removed.
            edges = edges[(edges >= population.min()) & (edges < population.max())]
        else:
            edges = np.empty(0)
        return {"zero_separate": separate_zero, "edges": edges.tolist()}

    @staticmethod
    def _apply_bins(values: np.ndarray, record: dict[str, Any]) -> np.ndarray:
        result = np.full(len(values), -1, dtype=np.int64)
        finite = np.isfinite(values)
        separate_zero = record["zero_separate"]
        ordinary = finite & (values != 0) if separate_zero else finite
        result[ordinary] = np.searchsorted(
            np.asarray(record["edges"]), values[ordinary], side="left"
        ) + int(separate_zero)
        if separate_zero:
            result[finite & (values == 0)] = 0
        return result

    @staticmethod
    def _validate_frame(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("X must be a DataFrame with unique columns")
        if not all(isinstance(column, str) for column in X.columns):
            raise ValueError("X column names must be strings")
        if any(not pd.api.types.is_numeric_dtype(X[c]) for c in X):
            raise ValueError("CMIM candidates must be numeric")

    @staticmethod
    def _validate_context(context: pd.DataFrame | None, n_rows: int, max_columns: int = 3, max_states: int = 32) -> np.ndarray:
        if context is None:
            return np.zeros(n_rows, dtype=np.int64)
        if not isinstance(context, pd.DataFrame) or len(context) != n_rows or not context.columns.is_unique:
            raise ValueError("context must have unique columns and one row per training row")
        if context.shape[1] > max_columns:
            raise ValueError(f"Use at most {max_columns} coarse historical context columns")
        if any(str(c).lower() in {"user_id", "item_id", "uid", "iid"} for c in context):
            raise ValueError("Raw user/item IDs are not suitable CMIM context")
        if any(context[c].nunique(dropna=False) > 16 for c in context):
            raise ValueError("Bin each context column into at most 16 historical strata")
        encoded = _context_codes(context, n_rows)
        if len(np.unique(encoded)) > max_states:
            raise ValueError(f"Context is too sparse: use at most {max_states} joint context states")
        return encoded

    def fit(self, X: pd.DataFrame, y: Any, context: pd.DataFrame | None = None) -> "CMIMSelector":
        self._validate_frame(X)
        labels = _vector(y, "y")
        if len(X) == 0 or len(labels) != len(X):
            raise ValueError("X and y must have the same nonzero number of rows")
        if pd.isna(labels).any():
            raise ValueError("Training labels must not be missing")
        context_codes = self._validate_context(context, len(X), self.max_context_columns, self.max_context_states)
        self.feature_names_in_ = list(X.columns)
        self.bin_edges_: dict[str, dict[str, Any]] = {}
        binned: dict[str, np.ndarray] = {}
        self.constant_features_: list[str] = []
        for column in self.feature_names_in_:
            values = X[column].to_numpy(dtype=float, na_value=np.nan)
            self.bin_edges_[column] = self._fit_bins(values)
            codes = self._apply_bins(values, self.bin_edges_[column])
            if codes.min() == codes.max():
                self.constant_features_.append(column)
            else:
                binned[column] = _codes(codes)

        label_codes = _codes(labels)
        relevance = {c: _cmi_encoded(v, label_codes, context_codes) for c, v in binned.items()}
        n_context_states = int(context_codes.max()) + 1
        if n_context_states == 1:
            self.marginal_relevance_ = relevance.copy()
        else:
            no_context = np.zeros(len(X), dtype=np.int64)
            self.marginal_relevance_ = {
                c: _cmi_encoded(v, label_codes, no_context) for c, v in binned.items()
            }
        candidate_scores = relevance.copy()
        remaining = list(binned)
        self.selected_features_: list[str] = []
        self.scores_: list[dict[str, Any]] = []
        while remaining and len(self.selected_features_) < self.n_features:
            # Python's stable max resolves ties in original column order.
            best = max(remaining, key=candidate_scores.__getitem__)
            score = candidate_scores[best]
            if score <= max(self.min_score, 1e-12):
                break
            self.selected_features_.append(best)
            self.scores_.append({
                "step": len(self.selected_features_), "feature": best,
                "score": float(score), "relevance": float(relevance[best]),
            })
            remaining.remove(best)
            condition = _context_codes(np.column_stack([context_codes, binned[best]]), len(X))
            for candidate in remaining:
                candidate_scores[candidate] = min(
                    candidate_scores[candidate],
                    _cmi_encoded(binned[candidate], label_codes, condition),
                )
        self.relevance_ = relevance
        self.n_training_rows_ = len(X)
        self.n_context_states_ = n_context_states
        self.context_columns_ = [] if context is None else [str(c) for c in context.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("Fit the selector before transform")
        missing = set(self.selected_features_) - set(X.columns)
        if missing:
            raise ValueError(f"Missing selected features: {sorted(missing)}")
        return X.loc[:, self.selected_features_].copy()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable audit record, including train-only bins."""
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("Fit the selector before serializing")
        return {
            "method": "pairwise_cmim", "information_unit": "nat",
            "n_features": self.n_features, "n_bins": self.n_bins,
            "min_score": self.min_score, "selected_features": self.selected_features_,
            "scores": self.scores_, "relevance": self.relevance_,
            "marginal_relevance": self.marginal_relevance_,
            "bin_edges": self.bin_edges_, "constant_features": self.constant_features_,
            "n_training_rows": self.n_training_rows_,
            "context_columns": self.context_columns_,
            "n_context_states": self.n_context_states_,
            "max_context_columns": self.max_context_columns, "max_context_states": self.max_context_states,
        }
