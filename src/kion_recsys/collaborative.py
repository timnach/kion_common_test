"""Causal, memory-bounded item-based collaborative filtering on CPU.

The model fits a binary user--item matrix from every available history row
strictly before a frozen cutoff.  It never reads labels or event intensity.
For two items with ``c`` common users and frequencies ``n_i`` and ``n_j``::

    similarity(i, j) = c / sqrt(n_i * n_j) * c / (c + shrinkage)

Each source item retains only its strongest neighbours.  The raw cosine is
symmetric, but pruning rows independently generally makes the stored matrix
asymmetric.  A user's recommendation score is the sum of similarities from
their distinct historical items to a candidate.  No user history is truncated.

Item Gram products are computed in sparse blocks and immediately pruned; no
full item--item Gram matrix or dense user--item score matrix is constructed.
A block may still become dense *in sparse storage*, so its worst-case size is
``block_size * n_items``.  This bounds peak memory, not quadratic total work.
"""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse


class CollaborativeRetriever:
    """Fit an item-KNN snapshot using past, metadata-backed items only.

    ``neighbors`` bounds retained neighbours per source item.  ``shrinkage``
    discounts similarities supported by few users; zero gives plain cosine.
    Item IDs break score ties in ascending order, including at the pruning
    boundary.  Times are interpreted in UTC and the cutoff is exclusive.
    Empty histories and cold users produce empty retrievals; a separate
    popularity source should supply their fallback candidates.
    """

    def __init__(
        self,
        events: pd.DataFrame,
        items: pd.DataFrame,
        cutoff: str | pd.Timestamp,
        neighbors: int = 80,
        shrinkage: float = 10,
        block_size: int = 128,
    ) -> None:
        started = perf_counter()
        for name, value in (("neighbors", neighbors), ("block_size", block_size)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isscalar(shrinkage) or not np.isfinite(shrinkage) or shrinkage < 0:
            raise ValueError("shrinkage must be finite and nonnegative")
        if not {"timestamp", "user_id", "item_id"}.issubset(events.columns):
            raise ValueError("events must contain timestamp, user_id and item_id")
        if "item_id" not in items or items["item_id"].isna().any() or items["item_id"].duplicated().any():
            raise ValueError("items must contain unique, nonmissing item_id values")
        self.cutoff = pd.Timestamp(cutoff)
        if pd.isna(self.cutoff):
            raise ValueError("cutoff must be a valid timestamp")
        if self.cutoff.tzinfo is not None:
            self.cutoff = self.cutoff.tz_convert("UTC").tz_localize(None)
        self.neighbors = int(neighbors)
        self.shrinkage = float(shrinkage)
        self.block_size = int(block_size)
        timestamps = pd.to_datetime(events["timestamp"], errors="raise", utc=True).dt.tz_convert(None)
        if timestamps.isna().any():
            raise ValueError("events must not contain missing timestamps")
        history = events.loc[timestamps < self.cutoff, ["user_id", "item_id"]]
        # Future-only metadata items do not establish past catalogue availability.
        history = history.loc[history["item_id"].isin(items["item_id"])]
        for name in ("user_id", "item_id"):
            values = history[name]
            if len(values) and (values.isna().any() or not pd.api.types.is_integer_dtype(values.dtype)):
                raise ValueError(f"past {name} values must have an integer dtype without missing values")
        self.item_ids = np.sort(history["item_id"].unique()).astype(np.int64)
        self.user_ids = np.sort(history["user_id"].unique()).astype(np.int64)
        self.n_items, self.n_users = len(self.item_ids), len(self.user_ids)
        self.eligible_items = frozenset(int(i) for i in self.item_ids)
        self._item_index = pd.Index(self.item_ids)
        self._user_index = pd.Index(self.user_ids)
        row = self._user_index.get_indexer(history["user_id"])
        col = self._item_index.get_indexer(history["item_id"])
        matrix = sparse.csr_matrix(
            (np.ones(len(history), dtype=np.float32), (row, col)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        )
        # Repeated events still mean one binary relation, rather than extra mass.
        matrix.sum_duplicates()
        matrix.data.fill(1.0)
        matrix.sort_indices()
        self.user_item_csr = matrix
        self.n_interactions = int(matrix.nnz)
        self.n_history_rows = len(history)
        self.max_block_nnz = 0
        self.similarity = self._fit_similarity()
        self.fit_seconds = perf_counter() - started

    @staticmethod
    def _top_positions(indices: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
        """Indices of strongest entries, resolving even boundary ties by ID."""
        if k <= 0 or len(scores) == 0:
            return np.empty(0, dtype=np.int64)
        if len(scores) > k:
            threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
            eligible = np.flatnonzero(scores >= threshold)
        else:
            eligible = np.arange(len(scores))
        order = np.lexsort((indices[eligible], -scores[eligible]))
        return eligible[order[:k]]

    def _fit_similarity(self) -> sparse.csr_matrix:
        if self.n_items == 0:
            return sparse.csr_matrix((0, 0), dtype=np.float32)
        item_user = self.user_item_csr.T.tocsr()
        frequencies = np.diff(item_user.indptr).astype(np.float64)
        # Allocate only the final upper bound; each block is discarded after use.
        capacity = self.n_items * min(self.neighbors, self.n_items - 1)
        columns = np.empty(capacity, dtype=np.int32)
        values = np.empty(capacity, dtype=np.float32)
        indptr = np.zeros(self.n_items + 1, dtype=np.int64)
        filled = 0
        for start in range(0, self.n_items, self.block_size):
            stop = min(start + self.block_size, self.n_items)
            common = (item_user[start:stop] @ self.user_item_csr).tocsr()
            self.max_block_nnz = max(self.max_block_nnz, int(common.nnz))
            for local_row, source in enumerate(range(start, stop)):
                begin, end = common.indptr[local_row : local_row + 2]
                neighbours = common.indices[begin:end]
                counts = common.data[begin:end].astype(np.float64)
                keep = (neighbours != source) & (counts > 0)
                neighbours, counts = neighbours[keep], counts[keep]
                scores = counts / np.sqrt(frequencies[source] * frequencies[neighbours])
                if self.shrinkage:
                    scores *= counts / (counts + self.shrinkage)
                selected = self._top_positions(neighbours, scores, self.neighbors)
                size = len(selected)
                columns[filled : filled + size] = neighbours[selected]
                values[filled : filled + size] = scores[selected]
                filled += size
                indptr[source + 1] = filled
            del common
        result = sparse.csr_matrix(
            (values[:filled].copy(), columns[:filled].copy(), indptr),
            shape=(self.n_items, self.n_items),
        )
        result.sort_indices()
        return result

    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame({
            "user_id": pd.Series(dtype=np.int64),
            "item_id": pd.Series(dtype=np.int64),
            "knn_score": pd.Series(dtype=np.float64),
            "knn_rank": pd.Series(dtype=np.int64),
        })

    def _scores_for_user(self, user_position: int) -> sparse.csr_matrix:
        scores = (self.user_item_csr.getrow(user_position) @ self.similarity).tocsr()
        scores.sum_duplicates()
        scores.sort_indices()
        return scores

    def candidates_for(self, user_ids: Iterable[int], k: int = 80) -> pd.DataFrame:
        """Return positive-score unseen candidates, sorted by user then rank.

        ``knn_rank`` is one-based.  Short lists are left short instead of padding
        with zero-score items; combining retrieval sources is the caller's job.
        """
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 0:
            raise ValueError("k must be a nonnegative integer")
        requested = np.asarray(sorted({int(uid) for uid in user_ids}), dtype=np.int64)
        if not len(requested) or not self.n_items or k == 0:
            return self._empty()
        positions = self._user_index.get_indexer(requested)
        rows = []
        for uid, position in zip(requested, positions):
            if position < 0:
                continue
            scores = self._scores_for_user(int(position))
            seen = self.user_item_csr.indices[
                self.user_item_csr.indptr[position] : self.user_item_csr.indptr[position + 1]
            ]
            keep = (scores.data > 0) & ~np.isin(scores.indices, seen, assume_unique=True)
            indices, values = scores.indices[keep], scores.data[keep]
            selected = self._top_positions(indices, values, int(k))
            rows.extend(
                (int(uid), int(self.item_ids[indices[p]]), float(values[p]), rank)
                for rank, p in enumerate(selected, start=1)
            )
        if not rows:
            return self._empty()
        return pd.DataFrame(rows, columns=self._empty().columns).astype({
            "user_id": "int64", "item_id": "int64", "knn_score": "float64", "knn_rank": "int64",
        })

    def scores_for(self, candidates: pd.DataFrame) -> np.ndarray:
        """Score every supplied pair, preserving row position and duplicates.

        This scores pairs beyond the retrieved top-k as well.  Unknown users,
        unavailable items, and disconnected pairs receive zero.  Seen items are
        not filtered here: scoring and eligibility are separate operations.
        Each requested user's sparse score vector is computed once, then freed.
        """
        if not {"user_id", "item_id"}.issubset(candidates.columns):
            raise ValueError("candidates must contain user_id and item_id")
        output = np.zeros(len(candidates), dtype=np.float64)
        if not len(candidates) or not self.n_items:
            return output
        user_positions = self._user_index.get_indexer(candidates["user_id"])
        item_positions = self._item_index.get_indexer(candidates["item_id"])
        valid_rows = np.flatnonzero((user_positions >= 0) & (item_positions >= 0))
        # Group by positional indices so arbitrary/duplicate DataFrame indexes
        # do not reorder results or cause label-based assignment surprises.
        order = np.argsort(user_positions[valid_rows], kind="stable")
        grouped_rows = valid_rows[order]
        if not len(grouped_rows):
            return output
        boundaries = np.r_[0, np.flatnonzero(np.diff(user_positions[grouped_rows])) + 1, len(grouped_rows)]
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            requested_rows = grouped_rows[begin:end]
            scores = self._scores_for_user(int(user_positions[requested_rows[0]]))
            if scores.nnz == 0:
                continue
            sought = item_positions[requested_rows]
            locations = np.searchsorted(scores.indices, sought)
            in_range = locations < scores.nnz
            matches = np.zeros(len(sought), dtype=bool)
            matches[in_range] = scores.indices[locations[in_range]] == sought[in_range]
            output[requested_rows[matches]] = scores.data[locations[matches]]
        return output
