"""Pilot: SHAP additivity, KernelSHAP cost and KernelSHAP-TreeSHAP agreement.

Runs on ONE training file only (train1 by default). It opens no validation or
test data. It answers three questions that must be settled before the
analysis plan is pre-registered:

1. Additivity. Do exact TreeSHAP attributions for the isolation forest sum to
   its mean path length on real HAI windows? If so, TreeSHAP and KernelSHAP
   can be pointed at the same output and compared fairly (H3).
2. Cost. How long does KernelSHAP take per window at each candidate setting,
   and does the projected full workload fit within 24 CPU hours?
3. Agreement. How closely does KernelSHAP rank features and tags the same way
   as TreeSHAP at each setting? This is a preview only; H3 is tested later
   on held-out attack events, not on training windows.

Training windows contain no attacks, so the windows explained here are the
most anomalous training windows (shortest mean path length) plus a few
random ones. That is enough for timing and agreement; it says nothing about
detection.

Usage (from the project folder, environment active):

    python scripts/pilot_shap.py
    python scripts/pilot_shap.py --windows-per-setting 20 --workload 1650

Results are printed and saved to results/pilot_shap_<date>.csv and .json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from haistudy.explain import (  # noqa: E402
    mean_path_length,
    mean_path_length_by_trees,
    rank_agreement,
    tag_totals,
)
from haistudy.windows import apply_scaler, fit_scaler, make_windows  # noqa: E402

# Candidate KernelSHAP settings: (background clusters, coalition samples per window).
# KernelSHAP's own default for 395 features would be 2 * 395 + 2048 = 2838 samples.
SETTINGS = [(10, 1000), (10, 2838), (25, 1000), (25, 2838)]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="data/raw/hai-21.03/train1.csv.gz", help="training file to use")
    ap.add_argument("--seed", type=int, default=0, help="seed for the forest and all sampling (pilot only)")
    ap.add_argument("--trees", type=int, default=100, help="isolation forest size")
    ap.add_argument("--windows-per-setting", type=int, default=10, help="windows explained at each KernelSHAP setting")
    ap.add_argument("--workload", type=int, default=1650,
                    help="ASSUMED number of windows to explain per model run in the study "
                         "(attack windows plus the normal-window sample). Replace with the real figure.")
    ap.add_argument("--model-runs", type=int, default=11,
                    help="KernelSHAP model runs in the study: PCA once, isolation forest and autoencoder five seeds each")
    ap.add_argument("--budget-hours", type=float, default=24.0, help="CPU-hour limit from the proposal")
    return ap.parse_args()


def main() -> int:
    a = parse_args()
    if any(f"test{k}" in a.file for k in (3, 4, 5)):
        sys.exit("Refusing to run: test3 to test5 are sealed until week 20.")
    rng = np.random.default_rng(a.seed)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    print(f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}, shap {shap.__version__}, "
          f"{os.cpu_count()} logical CPUs")

    # 1. Windows and scaling (training file only).
    t0 = time.perf_counter()
    df = pd.read_csv(ROOT / a.file)
    feats, meta = make_windows(df)
    scaler = fit_scaler(feats)
    X = apply_scaler(feats, scaler)
    cols = list(X.columns)
    Xv = X.to_numpy()
    print(f"\n[1] {a.file}: {len(df):,} rows -> {Xv.shape[0]:,} windows x {Xv.shape[1]} features "
          f"({time.perf_counter() - t0:.1f} s). Attack windows: {int(meta['attack'].sum())}")

    # 2. Isolation forest.
    t0 = time.perf_counter()
    forest = IsolationForest(n_estimators=a.trees, random_state=a.seed).fit(Xv)
    mpl = mean_path_length(forest, Xv)
    print(f"[2] Isolation forest: {a.trees} trees, max_samples {forest.max_samples_}, "
          f"fitted in {time.perf_counter() - t0:.1f} s. Mean path length {mpl.min():.2f} to {mpl.max():.2f}")

    # Windows to explain: the 2/3 most anomalous, the rest random.
    n = a.windows_per_setting
    order = np.argsort(mpl)
    n_top = max(1, 2 * n // 3)
    pick = np.concatenate([order[:n_top], rng.choice(order[n_top:], n - n_top, replace=False)])
    Xe = Xv[pick]

    # 3. Additivity of exact TreeSHAP.
    check = Xv[rng.choice(len(Xv), min(200, len(Xv)), replace=False)]
    fast, slow = mean_path_length(forest, check), mean_path_length_by_trees(forest, check)
    t0 = time.perf_counter()
    tree_exp = shap.TreeExplainer(forest)
    tree_sv = tree_exp.shap_values(check)
    tree_secs = (time.perf_counter() - t0) / len(check)
    base = float(np.ravel(tree_exp.expected_value)[0])
    gap = np.abs(tree_sv.sum(axis=1) + base - slow).max()
    print(f"[3] Fast and tree-by-tree mean path length agree to {np.abs(fast - slow).max():.1e}")
    print(f"    TreeSHAP additivity on {len(check)} real windows: largest gap {gap:.1e} "
          f"({'PASS' if gap < 1e-6 else 'FAIL'}); {tree_secs * 1000:.1f} ms per window")
    tree_e = tree_exp.shap_values(Xe)
    tree_tags, _ = tag_totals(tree_e, cols)

    # 4. KernelSHAP at each setting: cost and agreement with TreeSHAP.
    f = lambda z: mean_path_length(forest, z)  # noqa: E731
    rows = []
    for k, nsamples in SETTINGS:
        background = shap.kmeans(Xv, k)
        kexp = shap.KernelExplainer(f, background)
        t0 = time.perf_counter()
        ker = np.asarray(kexp.shap_values(Xe, nsamples=nsamples, silent=True))
        secs = (time.perf_counter() - t0) / n
        ker_tags, _ = tag_totals(ker, cols)
        add_gap = np.abs(ker.sum(axis=1) + float(np.ravel(kexp.expected_value)[0]) - f(Xe)).max()
        rho_f = rank_agreement(ker, tree_e)
        rho_t = rank_agreement(ker_tags, tree_tags)
        hours = secs * a.workload * a.model_runs / 3600
        rows.append({
            "background": k, "nsamples": nsamples, "secs_per_window": round(secs, 3),
            "projected_cpu_hours": round(hours, 2), "within_budget": hours <= a.budget_hours,
            "median_rho_features": round(float(np.nanmedian(rho_f)), 3),
            "median_rho_tags": round(float(np.nanmedian(rho_t)), 3),
            "min_rho_tags": round(float(np.nanmin(rho_t)), 3),
            "kernel_additivity_gap": float(f"{add_gap:.1e}"),
        })
        print(f"[4] KernelSHAP background {k:>2}, {nsamples:>4} samples: {secs:6.2f} s/window, "
              f"projected {hours:6.1f} CPU h, median rho features {rows[-1]['median_rho_features']:.3f}, "
              f"tags {rows[-1]['median_rho_tags']:.3f}")

    table = pd.DataFrame(rows)
    print("\nSummary\n" + table.to_string(index=False))
    print(f"\nProjection assumes {a.workload:,} windows per model run and {a.model_runs} model runs, "
          f"timed on the isolation forest only. Times are wall-clock on one process; "
          f"PCA and the autoencoder may cost more or less per evaluation.")

    table.to_csv(out_dir / f"pilot_shap_{stamp}.csv", index=False)
    (out_dir / f"pilot_shap_{stamp}.json").write_text(json.dumps({
        "date_utc": stamp, "file": a.file, "seed": a.seed, "trees": a.trees,
        "windows": int(Xv.shape[0]), "features": int(Xv.shape[1]),
        "windows_per_setting": n, "workload_assumed": a.workload, "model_runs": a.model_runs,
        "treeshap_additivity_gap": float(gap), "treeshap_ms_per_window": round(tree_secs * 1000, 2),
        "python": platform.python_version(), "sklearn": sklearn.__version__, "shap": shap.__version__,
        "cpus": os.cpu_count(), "settings": rows,
    }, indent=2) + "\n")
    print(f"Saved results/pilot_shap_{stamp}.csv and .json")
    return 0 if gap < 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())