"""Pilot: P7 (where extreme values occur in normal validation windows) and correlated tag groups.

Part A, P7. Some tags barely vary in training, so in validation their
standardised features can reach hundreds of standard deviations even in normal
operation. This part finds every NORMAL validation window whose largest
standardised feature exceeds a threshold, and classifies it by where it lies:

  recovery   within --recovery-minutes after the end of an attack
  lead-in    within --recovery-minutes before the start of an attack
  start-up   within --recovery-minutes of the start of a file
  other      anywhere else in normal operation

It also lists the tags responsible. The pattern decides the remedy: mostly
recovery points to excluding post-attack periods from normal windows; mostly
start-up points to excluding file starts; mostly other points to clipping or
robust scaling.

Part B, tag groups. The fidelity measure deletes tags in groups, so that a
deleted tag cannot be "replaced" by a near-copy that stays in place. Tags whose
absolute Pearson correlation exceeds 0.8 in the normal training data are
linked, and each connected set of linked tags forms one group. Constant tags
have no correlation and stay on their own.

Uses train1 to train3 and the validation files test1 and test2 only. It never
opens test3 to test5.

Usage (from the project folder, environment active):

    python scripts/pilot_p7_groups.py
    python scripts/pilot_p7_groups.py --threshold 50 --recovery-minutes 30
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from haistudy.windows import apply_scaler, fit_scaler, make_windows, tag_columns  # noqa: E402

DATA = ROOT / "data" / "raw" / "hai-21.03"
TRAIN = ["train1", "train2", "train3"]
VALIDATION = ["test1", "test2"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=50.0,
                    help="a normal window is 'extreme' if any standardised feature exceeds this, in absolute value")
    ap.add_argument("--recovery-minutes", type=float, default=30.0)
    ap.add_argument("--corr", type=float, default=0.8, help="absolute correlation that links two tags")
    return ap.parse_args()


def attack_intervals(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Start and end time of each contiguous run of attack-labelled seconds."""
    t = pd.to_datetime(df["time"]).reset_index(drop=True)
    a = df["attack"].to_numpy() > 0
    edges = np.flatnonzero(np.diff(np.concatenate(([0], a.astype(int), [0]))))
    return [(t[s], t[e - 1]) for s, e in zip(edges[::2], edges[1::2])]


def tag_groups(train: pd.DataFrame, tags: list[str], threshold: float) -> list[list[str]]:
    """Connected components of tags linked by |Pearson r| > threshold."""
    X = train[tags]
    varying = [c for c in tags if X[c].std() > 0]
    r = np.abs(np.corrcoef(X[varying].to_numpy(dtype=float), rowvar=False))
    parent = list(range(len(varying)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(varying)):
        for j in range(i + 1, len(varying)):
            if r[i, j] > threshold:
                parent[find(i)] = find(j)
    groups: dict[int, list[str]] = {}
    for i, name in enumerate(varying):
        groups.setdefault(find(i), []).append(name)
    out = [sorted(g) for g in groups.values()] + [[c] for c in tags if c not in varying]
    return sorted(out, key=lambda g: (-len(g), g[0]))


def main() -> int:
    a = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    # Training: windows for the scaler, raw rows for the correlations.
    train_raw = [pd.read_csv(DATA / f"{n}.csv.gz") for n in TRAIN]
    tags = tag_columns(train_raw[0])
    scaler = fit_scaler(pd.concat([make_windows(d)[0] for d in train_raw], ignore_index=True))

    # ---------------- Part A: P7 ----------------
    rows = []
    gap = pd.Timedelta(minutes=a.recovery_minutes)
    n_normal = 0
    for name in VALIDATION:
        df = pd.read_csv(DATA / f"{name}.csv.gz")
        feats, meta = make_windows(df)
        X = apply_scaler(feats, scaler).abs()
        normal = ~meta["attack"].to_numpy()
        n_normal += int(normal.sum())
        peak = X.max(axis=1).to_numpy()
        top = X.idxmax(axis=1).str.split("__").str[0].to_numpy()
        intervals = attack_intervals(df)
        file_start = pd.to_datetime(df["time"]).iloc[0]
        for i in np.flatnonzero(normal & (peak > a.threshold)):
            s, e = meta["start"].iloc[i], meta["end"].iloc[i]
            after = [s - end for _, end in intervals if s > end]
            before = [start - e for start, _ in intervals if start > e]
            since_end = min(after) if after else None
            to_start = min(before) if before else None
            if since_end is not None and since_end <= gap:
                where = "recovery"
            elif to_start is not None and to_start <= gap:
                where = "lead-in"
            elif s - file_start <= gap:
                where = "start-up"
            else:
                where = "other"
            rows.append({"file": name, "start": s, "peak": round(float(peak[i]), 1), "tag": top[i], "where": where,
                         "minutes_since_attack_end": None if since_end is None else round(since_end.total_seconds() / 60, 1)})

    ext = pd.DataFrame(rows)
    print(f"[A] Normal validation windows: {n_normal:,}. With a standardised feature above {a.threshold:g}: {len(ext):,}")
    if len(ext):
        print("\n    By location (within " f"{a.recovery_minutes:g} minutes):")
        print(ext["where"].value_counts().to_string())
        print("\n    By responsible tag (top 10):")
        print(ext["tag"].value_counts().head(10).to_string())
        print("\n    By file:")
        print(ext["file"].value_counts().to_string())
        ext.to_csv(out / f"pilot_p7_extremes_{stamp}.csv", index=False)

    # ---------------- Part B: tag groups ----------------
    groups = tag_groups(pd.concat(train_raw, ignore_index=True), tags, a.corr)
    multi = [g for g in groups if len(g) > 1]
    print(f"\n[B] Tag groups at |r| > {a.corr:g} on {sum(len(d) for d in train_raw):,} training rows: "
          f"{len(groups)} groups from {len(tags)} tags ({len(multi)} with more than one tag)")
    for g in multi:
        print(f"    {len(g):>2} tags: {', '.join(g)}")

    (out / f"pilot_tag_groups_{stamp}.json").write_text(json.dumps({
        "date_utc": stamp, "train": TRAIN, "abs_correlation_threshold": a.corr,
        "n_groups": len(groups), "groups": groups,
    }, indent=2) + "\n")
    print(f"\nSaved results/pilot_tag_groups_{stamp}.json" + (f" and results/pilot_p7_extremes_{stamp}.csv" if len(ext) else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())