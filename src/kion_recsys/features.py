"""Point-in-time snapshot features. A cutoff is the start of a forecast horizon.

Every aggregate uses timestamp < cutoff, including ties. KION timestamps are
last-watch proxies, not the original event stream. No target column is consumed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .data import parse_genres


@dataclass(frozen=True)
class DecayConfig:
    half_life: float = 7.0  # days, lambda = log(2) / half_life
    alpha: float = 0.0
    beta: float = 0.0
    genre_weight: float = 1.0
    adaptive_power: float = 0.0

    def __post_init__(self):
        if self.half_life <= 0 or self.alpha < 0 or self.beta < 0:
            raise ValueError("half_life > 0 and alpha,beta >= 0 required")
        if not 0 <= self.genre_weight <= 1 or not 0 <= self.adaptive_power <= 1:
            raise ValueError("genre_weight and adaptive_power must be in [0, 1]")

    def to_dict(self):
        return asdict(self)


def decay_affinity(ages, similarity, regularity, config: DecayConfig, median_gap=7.):
    """Return affinity, log similarity mass, and ESS for each candidate.

    similarity: candidates x history, in [0,1]. Unlike mean(y_s), this remains
    candidate-dependent when all historical implicit events have y_s=1.
    R varies with the historical item's categories and shrinks for short histories.
    """
    ages, similarity = np.asarray(ages, float), np.asarray(similarity, float)
    if len(ages) == 0:
        return np.zeros((similarity.shape[0], 3))
    if np.any(ages <= 0):
        raise ValueError("History must strictly precede cutoff")
    n = len(ages)
    adaptation = ((median_gap + 1) / 8) ** (config.adaptive_power * n / (n + 10))
    half_life = np.clip(config.half_life * adaptation, .25, 180)
    a = np.exp2(-ages / half_life) * (1 + config.alpha * np.asarray(regularity))
    weights = a[None, :] * (1 + config.beta * similarity)
    denom = weights.sum(axis=1)
    mass = (weights * similarity).sum(axis=1)
    affinity = np.divide(mass, denom, out=np.zeros_like(mass), where=denom > 1e-30)
    ess = np.divide(denom**2, (weights**2).sum(axis=1),
                    out=np.zeros_like(mass), where=(weights**2).sum(axis=1) > 1e-30)
    return np.column_stack([affinity, np.log1p(mass), ess])


class SnapshotFeatures:
    def __init__(self, events, items, users, cutoff):
        self.cutoff = pd.Timestamp(cutoff).normalize()
        self.history = events.loc[events.timestamp < self.cutoff].copy()
        self.items = items.set_index("item_id", drop=False)
        self.users = users.set_index("user_id", drop=False)
        genres = items.get("genres", pd.Series("", index=items.index))
        self.genre_map = {int(i): parse_genres(g) for i, g in zip(items.item_id, genres)}
        vocab = sorted({g for gs in self.genre_map.values() for g in gs})
        self.genre_idx = {g: i for i, g in enumerate(vocab)}
        self.ids = items.item_id.to_numpy()
        self.idx = {int(item): i for i, item in enumerate(self.ids)}
        self.genre_matrix = np.zeros((len(items), len(vocab)), dtype=np.float32)
        for item, genres in self.genre_map.items():
            for g in genres:
                self.genre_matrix[self.idx[item], self.genre_idx[g]] = 1
        self.types = self.items.reindex(self.ids).get(
            "content_type", pd.Series("unknown", index=self.ids)).fillna("unknown").astype(str).to_numpy()
        self.windows = sorted(set([1, 7, 30, *range(2, 15), 21, 45, 60, 90]))
        h = self.history
        self.age = (self.cutoff - h.timestamp).dt.total_seconds().to_numpy() / 86400
        self.counts = {w: h.loc[self.age <= w].groupby("item_id").size() for w in self.windows}
        self.total_counts = h.groupby("item_id").size()
        self.first = h.groupby("item_id").timestamp.min()
        self.last = h.groupby("item_id").timestamp.max()
        # Integer day lag d=1 means [cutoff-1d, cutoff), not the current day.
        self.lags = [(d, d) for d in range(1, 15)] + [(15, 21), (22, 30), (31, 45), (46, 60), (61, 90)]
        self.lag_counts = {(lo, hi): h.loc[(self.age > lo - 1) & (self.age <= hi)].groupby("item_id").size()
                           for lo, hi in self.lags}
        self.groups = {}
        self.manual_columns = []
        self.bank_columns = []
        self.trend_columns = []
        self.cat_columns = []

    def _history_arrays(self, uid, candidate_ids):
        hist = self.groups.get(int(uid))
        if hist is None:
            return np.array([]), np.empty((len(candidate_ids), 0)), np.empty((len(candidate_ids), 0)), np.array([]), 7.
        hist = hist.sort_values(["timestamp", "item_id"])
        ages = (self.cutoff - hist.timestamp).dt.total_seconds().to_numpy() / 86400
        hidx = [self.idx[int(i)] for i in hist.item_id]
        cidx = [self.idx[int(i)] for i in candidate_ids]
        hg, cg = self.genre_matrix[hidx], self.genre_matrix[cidx]
        intersection = cg @ hg.T
        union = cg.sum(axis=1)[:, None] + hg.sum(axis=1)[None, :] - intersection
        sim = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        type_sim = (self.types[cidx, None] == self.types[hidx][None, :]).astype(float)
        # Missing type is not evidence of similarity.
        type_sim *= (self.types[cidx, None] != "unknown") & (self.types[hidx][None, :] != "unknown")
        unique_ages = np.sort(np.unique(ages))
        gaps = np.diff(unique_ages)
        median_gap = float(np.median(gaps)) if len(gaps) else 7.
        category_r = np.zeros(hg.shape[1])
        for j in np.flatnonzero(hg.sum(axis=0) >= 3):
            ca = np.unique(ages[hg[:, j] > 0])
            if len(ca) >= 3:
                gap = np.diff(np.sort(ca))
                category_r[j] = len(ca) / (len(ca) + 5) / (1 + gap.std() / max(gap.mean(), 1e-9))
        regularity = np.divide(hg @ category_r, hg.sum(axis=1),
                               out=np.zeros(len(hg)), where=hg.sum(axis=1) > 0)
        return ages, sim, type_sim, regularity, median_gap

    def transform(self, candidates, decay_configs: dict[str, DecayConfig]):
        """Keep row order exactly. labels/query IDs are carried but never read."""
        out = candidates.copy().reset_index(drop=True)
        relevant = self.history[self.history.user_id.isin(out.user_id.unique())]
        self.groups = {int(u): g for u, g in relevant.groupby("user_id", sort=False)}
        manual = [c for c in ["retrieval_score", "source_global", "source_recent", "source_genre"] if c in out]
        cats = ["item_id"]
        for col in ["content_type", "age_rating", "for_kids", "release_year"]:
            if col in self.items:
                name = "item_" + col
                out[name] = out.item_id.map(self.items[col]).fillna("unknown").astype(str)
                cats.append(name)
        manual += cats
        for w in [1, 7, 30]:
            out[f"item_count_{w}d"] = out.item_id.map(self.counts[w]).fillna(0).astype(float)
            manual.append(f"item_count_{w}d")
        out["item_count_all"] = out.item_id.map(self.total_counts).fillna(0).astype(float)
        first_dates = pd.to_datetime(out.item_id.map(self.first.to_dict()))
        last_dates = pd.to_datetime(out.item_id.map(self.last.to_dict()))
        out["item_age"] = (self.cutoff - first_dates).dt.total_seconds().fillna(0) / 86400
        out["item_recency"] = ((self.cutoff - last_dates).dt.total_seconds() / 86400).fillna(365)
        manual += ["item_count_all", "item_age", "item_recency"]
        bank = []
        for w in self.windows:
            if w in [1, 7, 30]:
                continue
            name = f"auto_item_count_{w}d"
            out[name] = out.item_id.map(self.counts[w]).fillna(0).astype(float)
            bank.append(name)
        for lo, hi in self.lags:
            name = f"auto_item_lag_{lo}_{hi}d"
            out[name] = out.item_id.map(self.lag_counts[lo, hi]).fillna(0).astype(float)
            bank.append(name)
        # Normalise by exposure proxy (total platform activity), not just window length.
        c7 = out.item_id.map(self.counts[7]).fillna(0).to_numpy()
        previous = out.item_id.map(self.counts[30] - self.counts[7].reindex(self.counts[30].index, fill_value=0)).fillna(0).to_numpy()
        n7 = max(1, int((self.age <= 7).sum()))
        nprev = max(1, int(((self.age > 7) & (self.age <= 30)).sum()))
        out["trend_item_share"] = np.log((c7 + 1) / (n7 + len(self.items))) - np.log((previous + 1) / (nprev + len(self.items)))
        out["trend_item_rate"] = np.log1p(c7 / 7) - np.log1p(previous / 23)
        trends = ["trend_item_share", "trend_item_rate", "trend_user_rate", "trend_genre_drift"]
        user_manual = ["user_count_all", "user_recency", "user_span", "user_gap_median", "user_genre_diversity",
                       *[f"user_count_{w}d" for w in [1, 7, 30]],
                       *[f"genre_affinity_{w}" for w in ["all", "7d", "30d"]], "genre_match_recency"]
        manual += user_manual
        user_bank = []
        for w in self.windows:
            if w not in [1, 7, 30]:
                user_bank += [f"auto_user_count_{w}d", f"auto_genre_sum_{w}d"]
        for lo, hi in self.lags:
            user_bank += [f"auto_user_lag_{lo}_{hi}d", f"auto_genre_lag_{lo}_{hi}d"]
        bank += user_bank
        extra = user_manual + user_bank + ["trend_user_rate", "trend_genre_drift"]
        decay_cols = [f"{name}_{c}" for name in decay_configs for c in ["affinity", "mass", "ess"]]
        # Allocate a dense block once to avoid pandas fragmentation in the inner loop.
        values = np.zeros((len(out), len(extra) + len(decay_cols)), dtype=np.float32)
        col_idx = {c: j for j, c in enumerate(extra + decay_cols)}
        for uid, indices in out.groupby("user_id", sort=False).indices.items():
            ids = out.iloc[indices].item_id.to_numpy()
            ages, sim, type_sim, regularity, gap = self._history_arrays(uid, ids)
            n = len(ages)
            v = {"user_count_all": n, "user_recency": ages.min() if n else 365,
                 "user_span": ages.max() - ages.min() if n else 0,
                 "user_gap_median": gap, "user_genre_diversity": 0}
            if n:
                hids = self.groups[int(uid)].item_id
                v["user_genre_diversity"] = len({g for i in hids for g in self.genre_map[int(i)]})
            for w in [1, 7, 30]:
                v[f"user_count_{w}d"] = int((ages <= w).sum())
            v["genre_affinity_all"] = sim.mean(axis=1) if n else 0.
            for w in [7, 30]:
                selected = ages <= w
                v[f"genre_affinity_{w}d"] = sim[:, selected].mean(axis=1) if selected.any() else np.zeros(len(ids))
            v["genre_match_recency"] = np.where(sim > 0, ages, 365).min(axis=1) if n else 365.
            v["trend_user_rate"] = np.log1p((ages <= 7).sum() / 7) - np.log1p(((ages > 7) & (ages <= 30)).sum() / 23)
            v["trend_genre_drift"] = v["genre_affinity_7d"] - v["genre_affinity_30d"]
            for w in self.windows:
                if w in [1, 7, 30]:
                    continue
                mask = ages <= w
                v[f"auto_user_count_{w}d"] = mask.sum()
                v[f"auto_genre_sum_{w}d"] = sim[:, mask].sum(axis=1)
            for lo, hi in self.lags:
                mask = (ages > lo - 1) & (ages <= hi)
                v[f"auto_user_lag_{lo}_{hi}d"] = mask.sum()
                v[f"auto_genre_lag_{lo}_{hi}d"] = sim[:, mask].sum(axis=1)
            for name, cfg in decay_configs.items():
                mixed = cfg.genre_weight * sim + (1 - cfg.genre_weight) * type_sim
                dec = decay_affinity(ages, mixed, regularity, cfg, gap)
                for j, suffix in enumerate(["affinity", "mass", "ess"]):
                    v[f"{name}_{suffix}"] = dec[:, j]
            for c, val in v.items():
                values[indices, col_idx[c]] = val
        out = pd.concat([out, pd.DataFrame(values, columns=extra + decay_cols)], axis=1)
        out["item_id"] = out.item_id.astype(str)
        self.manual_columns, self.bank_columns, self.trend_columns, self.cat_columns = manual, bank, trends, cats
        return out
