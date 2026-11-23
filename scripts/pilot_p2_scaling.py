"""Pilot decision P2: how to scale features of tags that never move in training.

22 tags are constant throughout train1 to train3, so their features have zero
spread and cannot be standardised in the usual way. Two rules are compared:

  raw       scale = 1, so a change shows in the tag's own units. A tag that sits
            at 2,880 and moves to 2,000 scores 880.
  relative  scale = max(1, |the tag's normal level|), applied to all five
            features of the tag, so a change is measured as a fraction of the
            tag's normal value. The same move scores about 0.31, and a flag
            going from 1 to 0 still scores 1.

The script fits the scaler on ALL training files, then looks at the VALIDATION
files (test1 and test2) under each rule. For each rule it reports:

  1. which constant tags move at all in validation, and whether in attack or
     normal windows (movement in normal windows would cause false alerts);
  2. how large the moves are, in scaled units;
  3. in attack windows, the share of each window's total squared size that
     comes from constant-tag features. A share near 1 means those features
     would swamp reconstruction error and SHAP explanations.

It never opens test3 to test5.

Usage (from the project folder, environment active):

    python scripts/pilot_p2_scaling.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from haistudy.windows import apply_scaler, fit_scaler, make_windows  # noqa: E402

DATA = ROOT / "data" / "raw" / "hai-21.03"
TRAIN = ["train1", "train2", "train3"]
VALIDATION = ["test1", "test2"]


def windows_of(names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Window several files separately (no window spans two files) and stack them."""
    feats, metas = [], []
    for name in names:
        f, m = make_windows(pd.read_csv(DATA / f"{name}.csv.gz"))
        m["file"] = name
        feats.append(f)
        metas.append(m)
    return pd.concat(feats, ignore_index=True), pd.concat(metas, ignore_index=True)


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    train_f, _ = windows_of(TRAIN)
    raw = fit_scaler(train_f)                               # current guard: zero spread -> scale 1
    const_cols = [c for c in train_f.columns if train_f[c].std(ddof=0) == 0]
    const_tags = sorted({c.split("__")[0] for c in const_cols})

    # Relative rule: every feature of a constant tag (mean, std, min, max, slope) is
    # divided by the tag's normal level, since all five are in the tag's own units.
    relative = raw.copy()
    for c in const_cols:
        level = abs(raw.loc["centre", f"{c.split('__')[0]}__mean"])
        relative.loc["scale", c] = max(1.0, level)

    val_f, val_m = windows_of(VALIDATION)
    attack = val_m["attack"].to_numpy()
    print(f"Training: {len(train_f):,} windows from {', '.join(TRAIN)}. "
          f"Constant features: {len(const_cols)} from {len(const_tags)} tags.")
    print(f"Validation: {len(val_f):,} windows from {', '.join(VALIDATION)}, {int(attack.sum())} attack windows.\n")

    # 1. Which constant tags move in validation, measured on the raw window mean.
    moves = []
    for t in const_tags:
        dev = (val_f[f"{t}__mean"] - raw.loc["centre", f"{t}__mean"]).abs()
        rng = (val_f[f"{t}__max"] - val_f[f"{t}__min"]).abs()
        moved = (dev > 0) | (rng > 0)
        moves.append({
            "tag": t, "training_value": raw.loc["centre", f"{t}__mean"],
            "attack_windows_moved": int((moved & attack).sum()),
            "normal_windows_moved": int((moved & ~attack).sum()),
            "max_change_raw_units": float(dev.max()),
        })
    moves = pd.DataFrame(moves)
    moved = moves[(moves.attack_windows_moved > 0) | (moves.normal_windows_moved > 0)]
    print("[1] Constant tags that move in validation")
    print(moved.to_string(index=False) if len(moved) else "    none")

    # 2 and 3. Size of the moves and share of squared size, under each rule.
    results = {}
    print("\n[2] Under each rule, in ATTACK windows where a constant tag moved")
    for name, scaler in (("raw", raw), ("relative", relative)):
        X = apply_scaler(val_f, scaler)
        sq = X.to_numpy() ** 2
        cmask = X.columns.isin(const_cols)
        share = sq[:, cmask].sum(axis=1) / sq.sum(axis=1).clip(min=1e-12)
        touched = (np.abs(X.loc[:, cmask].to_numpy()) > 0).any(axis=1)
        sel = attack & touched
        largest = float(np.abs(X.loc[sel, cmask].to_numpy()).max()) if sel.any() else 0.0
        other = float(np.abs(X.loc[attack, ~cmask].to_numpy()).max()) if attack.any() else 0.0
        results[name] = {
            "attack_windows_with_constant_move": int(sel.sum()),
            "median_share_constant": round(float(np.median(share[sel])), 3) if sel.any() else None,
            "max_share_constant": round(float(share[sel].max()), 3) if sel.any() else None,
            "largest_constant_value": round(largest, 2),
            "largest_other_value_in_attacks": round(other, 2),
            "normal_windows_with_constant_move": int((~attack & touched).sum()),
        }
        r = results[name]
        print(f"    {name:<8} windows {r['attack_windows_with_constant_move']:>4} | share of squared size from "
              f"constant tags: median {r['median_share_constant']}, max {r['max_share_constant']} | "
              f"largest constant-tag value {r['largest_constant_value']}, largest other feature "
              f"{r['largest_other_value_in_attacks']}")

    print("\nReading the result: a median share near 1 means constant-tag features would dominate "
          "reconstruction error and SHAP attributions in those windows. Movement in normal windows would mean "
          "false alerts from those tags under either rule.")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    moves.to_csv(out / f"pilot_p2_moves_{stamp}.csv", index=False)
    (out / f"pilot_p2_scaling_{stamp}.json").write_text(json.dumps({
        "date_utc": stamp, "train": TRAIN, "validation": VALIDATION,
        "constant_tags": const_tags, "rules": results,
    }, indent=2) + "\n")
    print(f"Saved results/pilot_p2_moves_{stamp}.csv and results/pilot_p2_scaling_{stamp}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())