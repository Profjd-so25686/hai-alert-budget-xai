"""Unit tests for windowing, on small synthetic recordings with known answers."""
import numpy as np
import pandas as pd
import pytest

from haistudy.windows import (
    apply_scaler,
    continuous_segments,
    fit_scaler,
    make_windows,
    tag_columns,
)


def recording(seconds=200, start="2020-07-11 00:00:00", attack_at=None):
    """Synthetic HAI-like file: a rising ramp, a constant tag, a noisy tag."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "time": pd.date_range(start, periods=seconds, freq="s"),
        "P1_RAMP": 2.0 * np.arange(seconds) + 5.0,     # slope exactly 2 per second
        "P2_CONST": np.full(seconds, 397.63785),         # never moves
        "P3_NOISE": rng.normal(size=seconds),
        "attack": np.zeros(seconds, dtype=int),
        "attack_P1": np.zeros(seconds, dtype=int),
    })
    if attack_at is not None:
        df.loc[attack_at, ["attack", "attack_P1"]] = 1
    return df


def test_label_and_time_columns_are_never_features():
    assert tag_columns(recording()) == ["P1_RAMP", "P2_CONST", "P3_NOISE"]


def test_window_count_and_column_layout():
    feats, meta = make_windows(recording(200))
    # starts 0, 10, ..., 140 (the last start that still fits 60 rows)
    assert len(feats) == len(meta) == 15
    assert feats.shape[1] == 3 * 5
    assert list(feats.columns[:5]) == ["P1_RAMP__mean", "P1_RAMP__std", "P1_RAMP__min", "P1_RAMP__max", "P1_RAMP__slope"]
    assert meta["end"].iloc[0] - meta["start"].iloc[0] == pd.Timedelta(seconds=59)
    assert meta["start"].iloc[1] - meta["start"].iloc[0] == pd.Timedelta(seconds=10)


def test_features_match_hand_calculation():
    df = recording(200)
    feats, _ = make_windows(df)
    w = df["P1_RAMP"].to_numpy()[10:70]                 # second window
    row = feats.iloc[1]
    assert row["P1_RAMP__mean"] == pytest.approx(w.mean())
    assert row["P1_RAMP__std"] == pytest.approx(w.std())
    assert row["P1_RAMP__min"] == w.min()
    assert row["P1_RAMP__max"] == w.max()
    assert row["P1_RAMP__slope"] == pytest.approx(2.0)


def test_constant_tag_has_exactly_zero_spread_and_slope():
    feats, _ = make_windows(recording(200))
    assert (feats["P2_CONST__std"] == 0).all()
    assert (feats["P2_CONST__slope"] == 0).all()
    assert (feats["P2_CONST__mean"] == 397.63785).all()


def test_no_window_spans_a_time_gap():
    a = recording(100, start="2020-07-11 00:00:00")
    b = recording(100, start="2020-07-11 01:00:00")      # one-hour jump
    df = pd.concat([a, b], ignore_index=True)
    assert continuous_segments(df["time"]) == [(0, 100), (100, 200)]
    feats, meta = make_windows(df)
    assert len(feats) == 2 * 5                            # starts 0..40 in each segment
    assert ((meta["end"] - meta["start"]) == pd.Timedelta(seconds=59)).all()


def test_attack_label_marks_any_overlapping_window():
    _, meta = make_windows(recording(200, attack_at=[65]))
    # Second 65 lies in every window that starts between second 6 and second 65,
    # so in the windows starting at 10, 20, 30, 40, 50 and 60.
    offsets = (meta["start"] - meta["start"].iloc[0]).dt.total_seconds().astype(int)
    assert offsets[meta["attack"]].tolist() == [10, 20, 30, 40, 50, 60]


def test_results_do_not_depend_on_chunk_size():
    df = recording(500)
    f1, _ = make_windows(df, chunk=3)
    f2, _ = make_windows(df, chunk=10_000)
    pd.testing.assert_frame_equal(f1, f2)


def test_scaler_guards_constant_features():
    feats, _ = make_windows(recording(200))
    scaler = fit_scaler(feats)
    assert scaler.loc["scale", "P2_CONST__mean"] == 1.0
    scaled = apply_scaler(feats, scaler)
    assert np.isfinite(scaled.to_numpy()).all()
    assert (scaled["P2_CONST__mean"] == 0).all()
    # If the constant tag later moves by 3 units, the scaled feature shows 3.
    moved = recording(200)
    moved["P2_CONST"] += 3.0
    f_moved, _ = make_windows(moved)
    assert apply_scaler(f_moved, scaler)["P2_CONST__mean"].iloc[0] == pytest.approx(3.0)
