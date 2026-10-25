"""Deterministic candidate retrieval using only a frozen past snapshot."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .data import parse_genres


class SnapshotRetriever:
    """Union popularity and genre sources, then cap by a fixed retrieval score.

    The eligible catalogue is metadata items observed before the cutoff.  This
    conservative choice needs no guessed release/availability date.  Subsequent
    evaluation must report positives outside this catalogue separately.
    """

    def __init__(
        self,
        events: pd.DataFrame,
        items: pd.DataFrame,
        cutoff: str | pd.Timestamp,
        global_k: int = 60,
        recent_k: int = 60,
        genre_k: int = 20,
        max_candidates: int = 180,
    ) -> None:
        for name, value in (("global_k", global_k), ("recent_k", recent_k), ("genre_k", genre_k)):
            if not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(max_candidates, (int, np.integer)) or max_candidates <= 0:
            raise ValueError("max_candidates must be a positive integer")
        if not {"timestamp", "user_id", "item_id"}.issubset(events.columns):
            raise ValueError("events must contain timestamp, user_id and item_id")
        if "item_id" not in items or items["item_id"].duplicated().any():
            raise ValueError("items must contain unique item_id values")
        self.cutoff = pd.Timestamp(cutoff)
        if self.cutoff.tzinfo is not None:
            self.cutoff = self.cutoff.tz_convert("UTC").tz_localize(None)
        self.cutoff = self.cutoff.normalize()
        self.global_k, self.recent_k, self.genre_k = global_k, recent_k, genre_k
        self.max_candidates = max_candidates
        self.history = events.loc[events["timestamp"] < self.cutoff].copy()
        genres = items["genres"] if "genres" in items else pd.Series(None, index=items.index, dtype=object)
        self.genre_map = {int(i): parse_genres(g) for i, g in zip(items["item_id"], genres)}
        eligible = set(self.genre_map).intersection(self.history["item_id"].unique())
        self.eligible_items = frozenset(int(i) for i in eligible)
        self.item_counts = self._counts(self.history)
        recent = self.history.loc[self.history["timestamp"] >= self.cutoff - pd.Timedelta(days=7)]
        self.recent_counts = self._counts(recent)
        self._global_top = self._top(self.item_counts, global_k)
        self._recent_top = self._top(self.recent_counts, recent_k)
        self._global_set, self._recent_set = set(self._global_top), set(self._recent_top)
        self._global_scores = self._normalize(self.item_counts)
        self._recent_scores = self._normalize(self.recent_counts)
        genre_items: dict[str, list[int]] = defaultdict(list)
        # The list is already sorted by count desc, then item_id asc.
        for item in self._top(self.item_counts, len(self.item_counts)):
            for genre in self.genre_map[item]:
                if len(genre_items[genre]) < genre_k:
                    genre_items[genre].append(item)
        self._genre_items = dict(genre_items)

    def _counts(self, frame: pd.DataFrame) -> pd.Series:
        counts = frame.loc[frame["item_id"].isin(self.eligible_items)].groupby("item_id").size()
        counts.index.name = "item_id"
        return counts.astype(np.int64).sort_index()

    @staticmethod
    def _top(counts: pd.Series, k: int) -> list[int]:
        return [int(item) for item, _ in sorted(counts.items(), key=lambda row: (-int(row[1]), int(row[0])))[:k]]

    @staticmethod
    def _normalize(counts: pd.Series) -> dict[int, float]:
        maximum = float(counts.max()) if len(counts) else 1.0
        return {int(i): float(c) / maximum for i, c in counts.items()}

    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame({
            "user_id": pd.Series(dtype=np.int64),
            "item_id": pd.Series(dtype=np.int64),
            "retrieval_score": pd.Series(dtype=np.float64),
            "source_global": pd.Series(dtype=np.uint8),
            "source_recent": pd.Series(dtype=np.uint8),
            "source_genre": pd.Series(dtype=np.uint8),
        })

    def candidates_for(self, user_ids: Iterable[int]) -> pd.DataFrame:
        """Generate candidates without reading any evaluation labels.

        Source flags indicate membership before capping.  Previously consumed
        items are excluded for both warm and cold-start evaluations.
        """
        requested = sorted({int(u) for u in user_ids})
        if not requested or not self.eligible_items:
            return self._empty()
        histories = {
            int(uid): group["item_id"].tolist()
            for uid, group in self.history.loc[self.history["user_id"].isin(requested)].groupby("user_id", sort=False)
        }
        rows = []
        common = self._global_set | self._recent_set
        for uid in requested:
            previous = histories.get(uid, [])
            seen = set(previous)
            genre_counts: Counter[str] = Counter()
            for item in previous:
                genre_counts.update(self.genre_map.get(int(item), ()))
            top_genres = sorted(genre_counts, key=lambda g: (-genre_counts[g], g))[:3]
            genre_pool = {item for genre in top_genres for item in self._genre_items.get(genre, ())}
            total_genres = sum(genre_counts.values()) or 1
            genre_shares = {g: n / total_genres for g, n in genre_counts.items()}
            scores = []
            for item in (common | genre_pool) - seen:
                genre_score = sum(genre_shares.get(g, 0.0) for g in self.genre_map[item])
                score = self._global_scores.get(item, 0.0) + self._recent_scores.get(item, 0.0) + genre_score
                scores.append((item, float(score)))
            selected = sorted(scores, key=lambda row: (-row[1], row[0]))[: self.max_candidates]
            rows.extend((uid, item, score, int(item in self._global_set), int(item in self._recent_set), int(item in genre_pool))
                        for item, score in selected)
        if not rows:
            return self._empty()
        result = pd.DataFrame(rows, columns=self._empty().columns)
        result = result.astype({"user_id": "int64", "item_id": "int64", "retrieval_score": "float64",
                                "source_global": "uint8", "source_recent": "uint8", "source_genre": "uint8"})
        return result.sort_values(["user_id", "item_id"], kind="stable").reset_index(drop=True)
