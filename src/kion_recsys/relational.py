"""Small, causal language for relational history features.

An interval means ``lower_age_days < age <= upper_age_days``. All ages are
measured at a fixed midnight cutoff; the target day and future are excluded.
``count`` counts input rows, not reconstructed views: KION has one aggregate
per user--item pair. Category matches use ANY overlap and count an event once,
even when several category values match. Missing values never match.

User and category relations use the user's complete available history. Item
relations use all users' history. Shares use the same temporal kernel in both
numerator and denominator: user history for category shares, global history
for item shares. These are observed-interaction shares, not click-through rates.
Static item metadata is assumed available at the cutoff, as in the baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np
import pandas as pd

from .data import parse_genres


_RELATION_COLUMNS = {
    "genre": "genres",
    "director": "directors",
    "country": "countries",
    "content_type": "content_type",
}
_RELATIONS = frozenset({"user", "item", *_RELATION_COLUMNS})
_AGGREGATIONS = frozenset({"count", "share", "recency", "active_days"})


def _number_name(value: float) -> str:
    """Stable, non-lossy identifier component for finite nonnegative floats."""
    return np.format_float_positional(value, trim="-").replace(".", "p")


@dataclass(frozen=True)
class FeatureSpec:
    relation: str
    aggregation: str
    lower_age_days: float = 0.0
    upper_age_days: float | None = None
    half_life: float | None = None
    time_kind: str = "interval"

    def __post_init__(self):
        if self.relation not in _RELATIONS:
            raise ValueError(f"Unknown relation: {self.relation}")
        if self.aggregation not in _AGGREGATIONS:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        if self.time_kind != "interval":
            raise ValueError("Only interval time kernels are supported")
        for field in ("lower_age_days", "upper_age_days", "half_life"):
            value = getattr(self, field)
            if value is None and field != "lower_age_days":
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{field} must be a finite real number")
            # Canonicalize integer/float equivalents and negative zero.
            object.__setattr__(self, field, float(value) if value else 0.0)
        if self.lower_age_days < 0:
            raise ValueError("lower_age_days must be nonnegative")
        if self.upper_age_days is not None and self.upper_age_days <= self.lower_age_days:
            raise ValueError("upper_age_days must exceed lower_age_days")
        if self.half_life is not None:
            if self.half_life <= 0:
                raise ValueError("half_life must be positive")
            if self.aggregation not in {"count", "share"}:
                raise ValueError("Decay is supported only for count and share")

    @property
    def name(self) -> str:
        upper = "all" if self.upper_age_days is None else _number_name(self.upper_age_days)
        name = (f"rel__{self.relation}__{self.aggregation}__age_"
                f"{_number_name(self.lower_age_days)}_{upper}d")
        if self.half_life is not None:
            name += f"__h{_number_name(self.half_life)}d"
        return name

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def kernel_key(self) -> tuple[float, float | None, float | None]:
        return self.lower_age_days, self.upper_age_days, self.half_life


def _mask(ages: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    selected = ages > spec.lower_age_days
    if spec.upper_age_days is not None:
        selected &= ages <= spec.upper_age_days
    return selected


def _weights(ages: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    selected = _mask(ages, spec)
    values = selected.astype(np.float64)
    if spec.half_life is not None:
        values[selected] = np.exp2(-ages[selected] / spec.half_life)
    return values


def _category_values(value) -> frozenset[str]:
    return frozenset(part for part in parse_genres(value) if part != "unknown")


class SnapshotRelationalFeatures:
    """Compile specs without reading labels, query identifiers or watch totals.

    ``transform`` returns numeric feature columns only, preserving candidate
    order and index (including duplicate indices). One compiler represents one
    frozen snapshot, not a rolling-update serving simulation. No user history
    truncation or target-dependent history filtering is performed.
    """

    def __init__(self, events: pd.DataFrame, items: pd.DataFrame, cutoff):
        for frame, required, label in (
            (events, {"user_id", "item_id", "timestamp"}, "events"),
            (items, {"item_id"}, "items"),
        ):
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"{label} missing required columns: {sorted(missing)}")
        if items.item_id.duplicated().any():
            raise ValueError("Duplicate item_id rows make metadata ambiguous")
        cutoff = pd.Timestamp(cutoff)
        if pd.isna(cutoff):
            raise ValueError("cutoff must be a valid timestamp")
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert(None)
        self.cutoff = cutoff.normalize()
        timestamps = pd.to_datetime(events.timestamp, utc=True, errors="raise").dt.tz_convert(None)
        if timestamps.isna().any():
            raise ValueError("Event timestamps must not be missing")
        # Deliberately copy only these columns, so labels and target aggregates
        # cannot enter the compiler through a later generic groupby/aggregation.
        self.history = events.loc[timestamps < self.cutoff, ["user_id", "item_id"]].copy()
        self.history["timestamp"] = timestamps.loc[timestamps < self.cutoff]
        self.history = self.history.reset_index(drop=True)
        self.ages = (self.cutoff - self.history.timestamp).dt.total_seconds().to_numpy() / 86400
        self.days = self.history.timestamp.to_numpy(dtype="datetime64[D]").astype(np.int64)
        self._item_codes, self._item_ids = pd.factorize(self.history.item_id, sort=False)
        self._item_position = {item: pos for pos, item in enumerate(self._item_ids)}
        self._item_cache: dict[FeatureSpec, np.ndarray] = {}
        self._item_counts: dict[tuple, tuple[np.ndarray, float]] = {}
        self._metadata: dict[str, dict] = {}
        for relation, column in _RELATION_COLUMNS.items():
            source = items[column] if column in items else pd.Series(None, index=items.index)
            self._metadata[relation] = {
                item: _category_values(value) for item, value in zip(items.item_id, source)
            }

    def _item_values(self, spec: FeatureSpec) -> np.ndarray:
        if spec in self._item_cache:
            return self._item_cache[spec]
        size = len(self._item_ids)
        if spec.aggregation in {"count", "share"}:
            if spec.kernel_key not in self._item_counts:
                weights = _weights(self.ages, spec)
                counts = np.bincount(self._item_codes, weights=weights, minlength=size)
                self._item_counts[spec.kernel_key] = counts, float(weights.sum())
            counts, denominator = self._item_counts[spec.kernel_key]
            values = counts if spec.aggregation == "count" else (
                counts / denominator if denominator > 0 else np.zeros(size))
        else:
            selected = _mask(self.ages, spec)
            codes = self._item_codes[selected]
            if spec.aggregation == "recency":
                values = np.full(size, 365.0)
                np.minimum.at(values, codes, self.ages[selected])
            else:
                pairs = pd.DataFrame({"item": codes, "day": self.days[selected]}).drop_duplicates()
                values = np.bincount(pairs.item.to_numpy(dtype=np.int64), minlength=size).astype(float)
        self._item_cache[spec] = values
        return values

    @staticmethod
    def _match_matrix(candidate_ids: np.ndarray, history_ids: np.ndarray, metadata: dict) -> np.ndarray:
        # Invert only this user's short history, then OR all overlapping tokens.
        # This avoids explode/join row multiplication for multi-valued metadata.
        inverted: dict[str, list[int]] = {}
        for j, item in enumerate(history_ids):
            for value in metadata.get(item, ()):
                inverted.setdefault(value, []).append(j)
        unique_ids, inverse = np.unique(candidate_ids, return_inverse=True)
        matches = np.zeros((len(unique_ids), len(history_ids)), dtype=bool)
        for row, item in enumerate(unique_ids):
            for value in metadata.get(item, ()):
                indices = inverted.get(value)
                if indices:
                    matches[row, indices] = True
        return matches[inverse]

    def transform(self, candidates: pd.DataFrame, specs) -> pd.DataFrame:
        specs = list(specs)
        if not all(isinstance(spec, FeatureSpec) for spec in specs):
            raise TypeError("Every specification must be a FeatureSpec")
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate feature specifications are not allowed")
        missing = {"user_id", "item_id"}.difference(candidates.columns)
        if missing:
            raise ValueError(f"candidates missing required columns: {sorted(missing)}")
        if candidates[["user_id", "item_id"]].isna().any().any():
            raise ValueError("Candidate user/item IDs must not be missing")
        values = np.zeros((len(candidates), len(specs)), dtype=np.float32)
        if not specs or candidates.empty:
            return pd.DataFrame(values, columns=names, index=candidates.index)

        local_specs = []
        candidate_items = candidates.item_id.to_numpy()
        item_positions = np.array([self._item_position.get(item, -1) for item in candidate_items])
        known = item_positions >= 0
        for column, spec in enumerate(specs):
            if spec.relation != "item":
                local_specs.append((column, spec))
                continue
            source = self._item_values(spec)
            if spec.aggregation == "recency":
                values[:, column] = 365
            values[known, column] = source[item_positions[known]]

        if local_specs:
            relevant_positions = np.flatnonzero(self.history.user_id.isin(candidates.user_id.unique()))
            relevant = self.history.iloc[relevant_positions]
            history_groups = {
                uid: relevant_positions[indices]
                for uid, indices in relevant.groupby("user_id", sort=False).indices.items()
            }
            for uid, rows in candidates.groupby("user_id", sort=False).indices.items():
                positions = history_groups.get(uid, np.array([], dtype=np.int64))
                ages, days = self.ages[positions], self.days[positions]
                history_ids = self.history.item_id.to_numpy()[positions]
                current_ids = candidate_items[rows]
                match_cache = {}
                kernel_cache = {}
                for column, spec in local_specs:
                    if spec.relation not in match_cache:
                        match_cache[spec.relation] = (
                            np.ones((1, len(ages)), dtype=bool) if spec.relation == "user"
                            else self._match_matrix(current_ids, history_ids, self._metadata[spec.relation]))
                    matches = match_cache[spec.relation]
                    if spec.kernel_key not in kernel_cache:
                        kernel_cache[spec.kernel_key] = _weights(ages, spec)
                    weights = kernel_cache[spec.kernel_key]
                    if spec.aggregation in {"count", "share"}:
                        result = matches @ weights
                        if spec.aggregation == "share":
                            denominator = weights.sum()
                            result = result / denominator if denominator > 0 else np.zeros_like(result)
                    else:
                        selected = _mask(ages, spec)
                        if spec.aggregation == "recency":
                            result = np.minimum(
                                np.where(matches & selected, ages, 365).min(axis=1, initial=365), 365)
                        else:
                            # Per-day OR counts the day once across multiple events.
                            result = np.zeros(matches.shape[0])
                            for day in np.unique(days[selected]):
                                result += matches[:, selected & (days == day)].any(axis=1)
                    values[rows, column] = result
        return pd.DataFrame(values, columns=names, index=candidates.index)


def manual_specs() -> list[FeatureSpec]:
    """Fixed 21-feature relational control chosen without validation feedback."""
    result = []
    for relation in ("genre", "director", "country"):
        for aggregation in ("count", "share"):
            result.extend(FeatureSpec(relation, aggregation, upper_age_days=window) for window in (None, 7, 30))
        result.append(FeatureSpec(relation, "recency"))
    return result


def candidate_specs() -> list[FeatureSpec]:
    """Deterministic 157-spec search bank, including ``manual_specs``.

    Search covers relations, aggregation, nonstandard cumulative windows,
    disjoint age intervals and exponential kernels. The bank is a human-defined
    grammar, not an unconstrained discovery claim. Selection happens elsewhere.
    No labels, validation scores or sampled users are used to construct it.
    """
    result = manual_specs()
    intervals = ((0, 1), (1, 3), (3, 7), (7, 14), (14, 30), (30, 60), (60, None))
    for aggregation in ("count", "share"):
        result.extend(FeatureSpec("content_type", aggregation, upper_age_days=w) for w in (None, 7, 30))
    for relation in _RELATION_COLUMNS:
        for aggregation in ("count", "share"):
            result.extend(FeatureSpec(relation, aggregation, upper_age_days=w) for w in (3, 10, 21))
            result.extend(FeatureSpec(relation, aggregation, lo, hi) for lo, hi in intervals)
            result.extend(FeatureSpec(relation, aggregation, half_life=h) for h in (3, 7, 21))
        result.append(FeatureSpec(relation, "active_days"))
    result.append(FeatureSpec("content_type", "recency"))
    for relation in ("user", "item"):
        result.extend(FeatureSpec(relation, "count", upper_age_days=w) for w in (None, 3, 10, 21, 60))
        result.extend(FeatureSpec(relation, aggregation) for aggregation in ("recency", "active_days"))
    result.extend(FeatureSpec("item", "count", lo, hi) for lo, hi in intervals)
    if len({spec.name for spec in result}) != len(result):
        raise AssertionError("Feature bank must contain distinct specifications")
    return result
