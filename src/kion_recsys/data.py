"""Read KION's aggregated interaction tables without inventing event history.

``last_watch_dt`` is the last observation of a user--item pair.  Duration and
watch percentage describe all observations of that pair, rather than a single
view.  They are retained for analysis; loading does not filter on either value.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_genres(value: Any) -> tuple[str, ...]:
    """Return a deterministic, unique genre tuple; missing metadata is empty.

    KION uses comma-separated strings.  Lists and stringified lists are also
    accepted to make small fixtures and metadata exports straightforward.
    """
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        parts = list(value)
    else:
        if value is None or pd.isna(value):
            return ()
        text = str(value).strip()
        if text.casefold() in {"", "nan", "none", "null", "<na>"}:
            return ()
        if text.startswith(("[", "(")):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                parts = list(parsed)
            else:
                parts = re.split(r"[,;|]", text)
        else:
            parts = re.split(r"[,;|]", text)
    cleaned = set()
    for part in parts:
        if part is None or pd.isna(part):
            continue
        text = str(part).strip().casefold()
        if text not in {"", "nan", "none", "null", "<na>"}:
            cleaned.add(text)
    return tuple(sorted(cleaned))


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns: {', '.join(missing)}")


def _integer_ids(series: pd.Series, name: str) -> pd.Series:
    """Validate IDs before conversion, including missing and fractional IDs."""
    values = pd.to_numeric(series, errors="coerce")
    invalid = values.isna() | ~np.isfinite(values) | (values < 0)
    invalid |= values != np.floor(values)
    # KION IDs are small; do not silently wrap an unsigned or out-of-range ID.
    invalid |= values > np.iinfo(np.int64).max
    if invalid.any():
        examples = series.loc[invalid].head(3).astype(str).tolist()
        raise ValueError(f"{name}: expected nonnegative integer IDs; invalid values: {examples}")
    try:
        return values.astype(np.int64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name}: IDs must fit in signed 64-bit integers") from exc


def load_data(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load interactions, items and users from a KION download directory.

    Both ``raw_dir/data_original/*.csv`` and ``raw_dir/*.csv`` layouts work.
    A missing user metadata row is allowed: it is a property of the source
    dataset, and should not silently remove interactions from evaluation.
    """
    root = Path(raw_dir)
    expected = ("interactions.csv", "items.csv", "users.csv")
    locations = (root / "data_original", root)
    source = next((p for p in locations if all((p / name).is_file() for name in expected)), None)
    if source is None:
        searched = ", ".join(str(p) for p in locations)
        raise FileNotFoundError(f"Expected {', '.join(expected)} together in one directory; searched {searched}")
    events, items, users = [pd.read_csv(source / name) for name in expected]
    _require_columns(events, ("user_id", "item_id", "last_watch_dt"), "interactions.csv")
    _require_columns(items, ("item_id",), "items.csv")
    _require_columns(users, ("user_id",), "users.csv")
    for frame, name, columns in (
        (events, "interactions.csv", ("user_id", "item_id")),
        (items, "items.csv", ("item_id",)),
        (users, "users.csv", ("user_id",)),
    ):
        for column in columns:
            frame[column] = _integer_ids(frame[column], f"{name}.{column}")
    parsed = pd.to_datetime(events["last_watch_dt"], errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError(f"interactions.csv.last_watch_dt: {int(parsed.isna().sum())} missing or invalid dates")
    events["timestamp"] = parsed.dt.tz_convert(None).dt.normalize()
    duplicate_pairs = events.duplicated(["user_id", "item_id"])
    if duplicate_pairs.any():
        raise ValueError(
            f"interactions.csv: {int(duplicate_pairs.sum())} duplicate user-item pairs. "
            "Expected KION's one-row-per-pair aggregate; refusing to guess how to combine events."
        )
    for frame, key, name in ((items, "item_id", "items.csv"), (users, "user_id", "users.csv")):
        if frame[key].duplicated().any():
            raise ValueError(f"{name}: duplicate {key} values are ambiguous metadata")
    missing_items = pd.Index(events["item_id"].unique()).difference(items["item_id"])
    if len(missing_items):
        raise ValueError(
            f"items.csv: missing metadata for {len(missing_items)} interaction item IDs "
            f"(examples: {missing_items[:3].tolist()}); restore the complete item table"
        )
    if "genres" not in items:
        items["genres"] = None
    events = events.sort_values(["timestamp", "user_id", "item_id"], kind="stable").reset_index(drop=True)
    items = items.sort_values("item_id", kind="stable").reset_index(drop=True)
    users = users.sort_values("user_id", kind="stable").reset_index(drop=True)
    return events, items, users


def data_summary(events: pd.DataFrame, items: pd.DataFrame, users: pd.DataFrame) -> dict[str, Any]:
    """Summarize the observed table, not a reconstructed stream of views."""
    lengths = events.groupby("user_id", sort=False).size()
    dates = pd.to_datetime(events["timestamp"])
    first, last = dates.min(), dates.max()
    missingness = {
        name: {str(col): float(frame[col].isna().mean()) if len(frame) else 0.0 for col in frame.columns}
        for name, frame in (("events", events), ("items", items), ("users", users))
    }
    quantiles = {
        str(q): float(lengths.quantile(q)) if len(lengths) else 0.0
        for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    }
    result: dict[str, Any] = {
        "n_events": int(len(events)),
        "n_users": int(events["user_id"].nunique()),
        "n_items": int(events["item_id"].nunique()),
        "n_metadata_users": int(len(users)),
        "n_metadata_items": int(len(items)),
        "date_min": first.date().isoformat() if pd.notna(first) else None,
        "date_max": last.date().isoformat() if pd.notna(last) else None,
        "n_observed_days": int(dates.nunique()),
        "n_calendar_days": int((last - first).days + 1) if pd.notna(first) else 0,
        "sequence_length_mean": float(lengths.mean()) if len(lengths) else 0.0,
        "sequence_length_quantiles": quantiles,
        "duplicate_user_item_pairs": int(events.duplicated(["user_id", "item_id"]).sum()),
        "duplicate_item_metadata_ids": int(items["item_id"].duplicated().sum()),
        "duplicate_user_metadata_ids": int(users["user_id"].duplicated().sum()),
        "event_users_without_metadata": int((~pd.Index(events["user_id"].unique()).isin(users["user_id"])).sum()),
        "event_items_without_metadata": int((~pd.Index(events["item_id"].unique()).isin(items["item_id"])).sum()),
        "missing_fraction": missingness,
        "semantics": "One aggregate per user-item; timestamp is the last watch date, not every view.",
    }
    # Keep the public result safe to write with json.dump(..., allow_nan=False).
    json.dumps(result, allow_nan=False)
    return result
