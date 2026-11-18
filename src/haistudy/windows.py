"""Windowing and feature extraction for HAI telemetry.

Every detector in the study receives the same input, built here:

* the recording is cut into 60-second windows that advance in 10-second steps;
* each window is summarised, tag by tag, by five features: mean, standard
  deviation, minimum, maximum and slope;
* a window never spans a gap in the recording.

Alongside the features, each window keeps its start time, end time and attack
label in a separate table (``meta``). The detectors never see ``meta``; it is
used only to score their alerts.

Constant tags are kept (pilot decision P1, Option B). ``fit_scaler`` therefore
guards against a zero spread, so that a tag which never moves in training is
scaled to 0 and any later movement shows up in its raw units.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIME_COLUMN = "time"
LABEL_PREFIX = "attack"
STATS = ("mean", "std", "min", "max", "slope")


def tag_columns(df: pd.DataFrame) -> list[str]:
    """Return the sensor tag columns: everything except the time and label columns.

    This is the only place that decides what a detector may see, so the
    label columns can never leak into the features by accident.
    """
    return [c for c in df.columns if c != TIME_COLUMN and not c.startswith(LABEL_PREFIX)]


def continuous_segments(times: pd.Series, step_seconds: int = 1) -> list[tuple[int, int]]:
    """Split a recording into runs of rows exactly ``step_seconds`` apart.

    Returns a list of (first_row, last_row_plus_one) pairs. A window is only
    ever cut from inside one segment, so no window can span a jump in time.
    """
    t = pd.to_datetime(times).to_numpy()
    gaps = np.flatnonzero(np.diff(t) != np.timedelta64(step_seconds, "s"))
    starts = np.concatenate(([0], gaps + 1))
    ends = np.concatenate((gaps + 1, [len(t)]))
    return list(zip(starts.tolist(), ends.tolist()))


def _window_stats(block: np.ndarray) -> np.ndarray:
    """Five features for a block of windows.

    block has shape (windows, tags, seconds). Returns shape (windows, tags, 5)
    holding mean, standard deviation, minimum, maximum and slope, in that order.
    """
    length = block.shape[-1]
    mean = block.mean(axis=-1)
    std = block.std(axis=-1)               # population standard deviation (ddof=0)
    mn = block.min(axis=-1)
    mx = block.max(axis=-1)

    # Least-squares slope, in units per second, of each tag against time.
    # With centred time tc, slope = sum(tc * x) / sum(tc ** 2).
    tc = np.arange(length, dtype=float) - (length - 1) / 2.0
    slope = block @ tc / np.dot(tc, tc)

    # A tag that does not move within the window gets exactly zero spread and
    # zero slope. Without this, rounding in the mean can leave values of about
    # 1e-14, which the scaler would later blow up.
    flat = mx == mn
    std[flat] = 0.0
    slope[flat] = 0.0
    mean[flat] = mn[flat]

    return np.stack([mean, std, mn, mx, slope], axis=-1)


def make_windows(
    df: pd.DataFrame,
    tags: list[str] | None = None,
    length: int = 60,
    step: int = 10,
    chunk: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut one HAI file into windows and summarise each one.

    Parameters
    ----------
    df : one HAI file as read by pandas, rows in time order, one row per second.
    tags : tag columns to use. Defaults to every tag column (``tag_columns``).
    length : window length in rows (seconds). Fixed at 60 in the study.
    step : distance between window starts, in rows. Fixed at 10.
    chunk : windows processed at a time. Affects memory use only, not results.

    Returns
    -------
    features : one row per window, ``len(tags) * 5`` columns named
        ``<tag>__<stat>``, grouped by tag (all five features of a tag sit
        together, which keeps tag-level SHAP grouping simple).
    meta : one row per window with ``start``, ``end`` and ``attack``
        (True if any second in the window is labelled as an attack). Absent
        label columns, as in the training files, give False throughout.
    """
    if tags is None:
        tags = tag_columns(df)
    values = df[tags].to_numpy(dtype=float)
    times = pd.to_datetime(df[TIME_COLUMN]).reset_index(drop=True)
    labels = (df["attack"].to_numpy() > 0) if "attack" in df.columns else np.zeros(len(df), bool)

    feats, starts, ends, attack = [], [], [], []
    for seg_start, seg_end in continuous_segments(times):
        n = seg_end - seg_start
        if n < length:
            continue                       # segment too short for one window
        firsts = np.arange(seg_start, seg_end - length + 1, step)
        # Every window of this segment, as a view: (all_windows, tags, length).
        view = np.lib.stride_tricks.sliding_window_view(values[seg_start:seg_end], length, axis=0)
        lab = np.lib.stride_tricks.sliding_window_view(labels[seg_start:seg_end], length)
        picked = firsts - seg_start        # keep every step-th window
        for i in range(0, len(picked), chunk):
            idx = picked[i:i + chunk]
            feats.append(_window_stats(view[idx]))
            attack.append(lab[idx].any(axis=1))
        starts.append(times.iloc[firsts].to_numpy())
        ends.append(times.iloc[firsts + length - 1].to_numpy())

    columns = [f"{t}__{s}" for t in tags for s in STATS]
    if not feats:
        return pd.DataFrame(columns=columns, dtype=float), pd.DataFrame(columns=["start", "end", "attack"])
    stacked = np.concatenate(feats)        # (windows, tags, 5)
    features = pd.DataFrame(stacked.reshape(len(stacked), -1), columns=columns)
    meta = pd.DataFrame({
        "start": np.concatenate(starts),
        "end": np.concatenate(ends),
        "attack": np.concatenate(attack),
    })
    return features, meta


def fit_scaler(features: pd.DataFrame) -> pd.DataFrame:
    """Learn per-feature centre and scale from TRAINING windows only.

    Returns a two-row frame (``centre``, ``scale``). Where a feature has zero
    spread in training, its scale is set to 1 (the guard for constant tags),
    so that feature is 0 in normal operation and shows any later change in
    its raw units instead of dividing by zero.
    """
    centre = features.mean(axis=0)
    scale = features.std(axis=0, ddof=0)
    scale = scale.where(scale > 0, 1.0)
    return pd.DataFrame([centre, scale], index=["centre", "scale"])


def apply_scaler(features: pd.DataFrame, scaler: pd.DataFrame) -> pd.DataFrame:
    """Standardise features with a scaler from ``fit_scaler``."""
    missing = set(scaler.columns) - set(features.columns)
    if missing:
        raise ValueError(f"features lack {len(missing)} columns the scaler expects, e.g. {sorted(missing)[:3]}")
    f = features[scaler.columns]
    return (f - scaler.loc["centre"]) / scaler.loc["scale"]
