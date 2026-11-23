"""Pilot rerun: KernelSHAP against TreeSHAP with a MATCHED background.

The first pilot (pilot_shap.py) compared KernelSHAP with TreeSHAP in its
"tree path dependent" mode. The two then explain against different reference
distributions, so they can disagree even when both are accurate. Agreement was
low and did not improve with more samples, which pointed to that mismatch.

This rerun gives both explainers the SAME background: one random sample of
real training windows, unweighted. TreeSHAP runs in "interventional" mode on
that background, which is the quantity KernelSHAP estimates. Any remaining
disagreement is then approximation error, which is what H3 is meant to test.

KernelSHAP runs with l1_reg=False, so it estimates every feature and does no
automatic feature selection.

Like the first pilot, it reads ONE training file only and opens no validation
or test data.

Usage (from the project folder, environment active):

    python scripts/pilot_shap_matched.py
    python scripts/pilot_shap_matched.py --windows-per-setting 20 --workload 1650

Results are printed and saved to results/pilot_shap_matched_<date>.csv and .json.
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
from haistudy.explain import mean_path_length, rank_agreement, tag_totals  # noqa: E402
from haistudy.windows import apply_scaler, fit_scaler, make_windows  # noqa: E402

# Candidate settings: (background windows, KernelSHAP coalition samples per window).
# Memory grows with background x samples x features (100 x 5000 x 395 is about 1.6 GB); keep it there or below.
SETTINGS = [(50, 2838), (50, 10000), (100, 2838), (100, 5000)]
TOP_K = 10  # tags compared in the top-k overlap measure


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="data/raw/hai-21.03/train1.csv.gz", help="training file to use")
    ap.add_argument("--seed", type=int, default=0, help="seed for the forest and all sampling (pilot only)")
    ap.add_argument("--trees", type=int, default=100, help="isolation forest size")
    ap.add_argument("--windows-per-setting", type=int, default=10, help="windows explained at each setting")
    ap.add_argument("--workload", type=int, default=1650,
                    help="ASSUMED windows explained per model run in the study. Replace with the real figure.")
    ap.add_argument("--model-runs", type=int, default=11,
                    help="KernelSHAP model runs in the study: PCA once, isolation forest and autoencoder five seeds each")
    ap.add_argument("--budget-hours", type=float, default=24.0, help="CPU-hour limit from the proposal")
    return ap.parse_args()


def agreement_without_zeros(kernel: np.ndarray, tree: np.ndarray) -> np.ndarray:
    """Spearman per window, using only features the exact method gives a non-zero value.

    Features TreeSHAP sets exactly to zero (for example constant tags, which no
    tree ever splits on) carry no ranking information; KernelSHAP gives them
    small noisy values, which would create many meaningless ties.
    """
    out = np.full(len(tree), np.nan)
    for i in range(len(tree)):
        keep = tree[i] != 0
        if keep.sum() >= 3:
            out[i] = rank_agreement(kernel[i:i + 1, keep], tree[i:i + 1, keep])[0]
    return out


def top_k_overlap(kernel_tags: np.ndarray, tree_tags: np.ndarray, k: int) -> np.ndarray:
    """Share of the k most important tags (by absolute value) that both methods agree on."""
    out = np.zeros(len(tree_tags))
    for i in range(len(tree_tags)):
        a = set(np.argsort(-np.abs(kernel_tags[i]))[:k])
        b = set(np.argsort(-np.abs(tree_tags[i]))[:k])
        out[i] = len(a & b) / k
    return out


def main() -> int:
    a = parse_args()
    if any(f"test{k}" in a.file for k in (3, 4, 5)):
        sys.exit("Refusing to run: test3 to test5 are sealed until week 20.")
    rng = np.random.default_rng(a.seed)
    np.random.seed(a.seed)  # KernelSHAP draws its coalitions from numpy's global generator
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    print(f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}, shap {shap.__version__}, "
          f"{os.cpu_count()} logical CPUs")

    # 1. Windows and scaling (training file only).
    df = pd.read_csv(ROOT / a.file)
    feats, meta = make_windows(df)
    X = apply_scaler(feats, fit_scaler(feats))
    cols = list(X.columns)
    Xv = X.to_numpy()
    print(f"\n[1] {a.file}: {Xv.shape[0]:,} windows x {Xv.shape[1]} features. Attack windows: {int(meta['attack'].sum())}")

    # 2. Isolation forest and the windows to explain.
    forest = IsolationForest(n_estimators=a.trees, random_state=a.seed).fit(Xv)
    f = lambda z: mean_path_length(forest, z)  # noqa: E731
    mpl = f(Xv)
    n = a.windows_per_setting
    order = np.argsort(mpl)
    n_top = max(1, 2 * n // 3)
    pick = np.concatenate([order[:n_top], rng.choice(order[n_top:], n - n_top, replace=False)])
    Xe = Xv[pick]
    print(f"[2] Isolation forest: {a.trees} trees. Explaining {n} windows ({n_top} most anomalous, {n - n_top} random)")

    rows = []
    for bg_size, nsamples in SETTINGS:
        # One background per size, reused by both explainers.
        background = Xv[np.random.default_rng(a.seed + bg_size).choice(len(Xv), bg_size, replace=False)]

        t0 = time.perf_counter()
        tree_exp = shap.TreeExplainer(forest, data=background, feature_perturbation="interventional")
        tree_sv = np.asarray(tree_exp.shap_values(Xe))
        tree_secs = (time.perf_counter() - t0) / n
        tree_gap = np.abs(tree_sv.sum(axis=1) + float(np.ravel(tree_exp.expected_value)[0]) - f(Xe)).max()

        kexp = shap.KernelExplainer(f, background)
        t0 = time.perf_counter()
        ker = np.asarray(kexp.shap_values(Xe, nsamples=nsamples, l1_reg=False, silent=True))
        secs = (time.perf_counter() - t0) / n

        ker_tags, _ = tag_totals(ker, cols)
        tree_tags, _ = tag_totals(tree_sv, cols)
        rho_all = rank_agreement(ker, tree_sv)
        rho_nz = agreement_without_zeros(ker, tree_sv)
        rho_tag = rank_agreement(ker_tags, tree_tags)
        overlap = top_k_overlap(ker_tags, tree_tags, TOP_K)
        hours = secs * a.workload * a.model_runs / 3600

        rows.append({
            "background": bg_size, "nsamples": nsamples,
            "kernel_secs_per_window": round(secs, 3), "tree_ms_per_window": round(tree_secs * 1000, 1),
            "projected_cpu_hours": round(hours, 2), "within_budget": hours <= a.budget_hours,
            "rho_features_all": round(float(np.nanmedian(rho_all)), 3),
            "rho_features_nonzero": round(float(np.nanmedian(rho_nz)), 3),
            "rho_tags": round(float(np.nanmedian(rho_tag)), 3),
            "rho_tags_min": round(float(np.nanmin(rho_tag)), 3),
            f"top{TOP_K}_tag_overlap": round(float(np.mean(overlap)), 2),
            "nonzero_features_median": int(np.median((tree_sv != 0).sum(axis=1))),
            "tree_additivity_gap": float(f"{tree_gap:.1e}"),
        })
        r = rows[-1]
        print(f"[3] background {bg_size:>3}, {nsamples:>5} samples: KernelSHAP {secs:6.2f} s/window "
              f"({hours:5.1f} CPU h projected) | rho features {r['rho_features_all']:.3f}, "
              f"non-zero {r['rho_features_nonzero']:.3f}, tags {r['rho_tags']:.3f}, "
              f"top-{TOP_K} tags {r[f'top{TOP_K}_tag_overlap']:.2f} | TreeSHAP additivity gap {tree_gap:.1e}")

    table = pd.DataFrame(rows)
    print("\nSummary\n" + table.to_string(index=False))
    print(f"\nProjection assumes {a.workload:,} windows per model run and {a.model_runs} model runs, timed on the "
          f"isolation forest only. Agreement is on training windows, not attacks: a preview of H3, not a test of it.")

    table.to_csv(out_dir / f"pilot_shap_matched_{stamp}.csv", index=False)
    (out_dir / f"pilot_shap_matched_{stamp}.json").write_text(json.dumps({
        "date_utc": stamp, "file": a.file, "seed": a.seed, "trees": a.trees,
        "windows_per_setting": n, "workload_assumed": a.workload, "model_runs": a.model_runs,
        "tree_mode": "interventional", "kernel_l1_reg": False, "background": "random training windows, unweighted",
        "python": platform.python_version(), "sklearn": sklearn.__version__, "shap": shap.__version__,
        "cpus": os.cpu_count(), "settings": rows,
    }, indent=2) + "\n")
    print(f"Saved results/pilot_shap_matched_{stamp}.csv and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())