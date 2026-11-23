"""Pilot: KernelSHAP cost for PCA and the autoencoder, and the total for all three detectors.

The earlier pilots timed KernelSHAP on the isolation forest only. This script
times it on the other two detectors at the proposed setting (100 background
windows, 2,838 samples) and the fallback (50 and 2,838), then adds the three
detectors together for the 24 CPU-hour check in the proposal.

Both detectors here are simple stand-ins built only for timing. Their final
settings (number of PCA components, autoencoder size, training) are chosen
later on validation data and fixed in the pre-registration. What matters here
is how long one model evaluation takes, which depends mainly on input size.

To measure CPU time honestly, PyTorch is limited to one thread, and the
script reports process CPU time as well as wall-clock time.

Reads ONE training file only. Opens no validation or test data.

Usage (from the project folder, environment active):

    python scripts/pilot_cost_all.py
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
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from haistudy.windows import apply_scaler, fit_scaler, make_windows  # noqa: E402

SETTINGS = [(100, 2838), (50, 2838)]  # proposed, fallback
BATCH = 50_000  # rows per model call, to keep memory bounded


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="data/raw/hai-21.03/train1.csv.gz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows", type=int, default=5, help="windows explained per detector and setting")
    ap.add_argument("--workload", type=int, default=1638, help="windows per model run (pilot note P5)")
    ap.add_argument("--iforest-secs", type=float, nargs=2, default=[3.117, 1.737],
                    help="isolation forest KernelSHAP seconds per window at the proposed and fallback settings (P4)")
    ap.add_argument("--seeds", type=int, default=5, help="seeds for the isolation forest and autoencoder")
    ap.add_argument("--budget-hours", type=float, default=24.0)
    return ap.parse_args()


def batched(score, X: np.ndarray) -> np.ndarray:
    """Apply a scoring function in chunks so large KernelSHAP batches fit in memory."""
    return np.concatenate([score(X[i:i + BATCH]) for i in range(0, len(X), BATCH)])


def build_pca(Xv: np.ndarray):
    pca = PCA(n_components=0.95, svd_solver="full").fit(Xv)  # keep 95 per cent of variance (stand-in)

    def score(Z):
        recon = pca.inverse_transform(pca.transform(Z))
        return ((Z - recon) ** 2).sum(axis=1)
    return score, f"PCA, {pca.n_components_} components"


def build_autoencoder(Xv: np.ndarray, seed: int):
    torch.manual_seed(seed)
    d = Xv.shape[1]
    model = torch.nn.Sequential(
        torch.nn.Linear(d, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 16), torch.nn.ReLU(),
        torch.nn.Linear(16, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, d),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = torch.tensor(Xv, dtype=torch.float32)
    t0 = time.perf_counter()
    for _ in range(3):  # a short fit is enough for timing
        for idx in torch.randperm(len(data)).split(256):
            batch = data[idx]
            loss = ((model(batch) - batch) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()

    def score(Z):
        with torch.no_grad():
            z = torch.tensor(Z, dtype=torch.float32)
            return ((model(z) - z) ** 2).sum(dim=1).numpy().astype(float)
    return score, f"autoencoder {d}-64-16-64-{d}, 3 epochs in {time.perf_counter() - t0:.0f} s"


def time_kernelshap(score, Xv, Xe, bg_size, nsamples, seed):
    background = Xv[np.random.default_rng(seed + bg_size).choice(len(Xv), bg_size, replace=False)]
    f = lambda z: batched(score, z)  # noqa: E731
    exp = shap.KernelExplainer(f, background)
    w0, c0 = time.perf_counter(), time.process_time()
    sv = np.asarray(exp.shap_values(Xe, nsamples=nsamples, l1_reg=False, silent=True))
    wall = (time.perf_counter() - w0) / len(Xe)
    cpu = (time.process_time() - c0) / len(Xe)
    gap = np.abs(sv.sum(axis=1) + float(np.ravel(exp.expected_value)[0]) - f(Xe)).max()
    return wall, cpu, gap


def main() -> int:
    a = parse_args()
    if any(f"test{k}" in a.file for k in (3, 4, 5)):
        sys.exit("Refusing to run: test3 to test5 are sealed until week 20.")
    torch.set_num_threads(1)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (ROOT / "results").mkdir(exist_ok=True)

    print(f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}, shap {shap.__version__}, "
          f"torch {torch.__version__} (1 thread), {os.cpu_count()} logical CPUs")

    df = pd.read_csv(ROOT / a.file)
    feats, _ = make_windows(df)
    Xv = apply_scaler(feats, fit_scaler(feats)).to_numpy()
    Xe = Xv[rng.choice(len(Xv), a.windows, replace=False)]
    print(f"[1] {a.file}: {Xv.shape[0]:,} windows x {Xv.shape[1]} features; explaining {a.windows} windows per setting")

    detectors = {"pca": build_pca(Xv), "autoencoder": build_autoencoder(Xv, a.seed)}
    for name, (_, desc) in detectors.items():
        print(f"[2] {desc}")

    rows = []
    for (bg, ns), if_secs in zip(SETTINGS, a.iforest_secs):
        timings = {"isolation_forest": if_secs}
        for name, (score, _) in detectors.items():
            wall, cpu, gap = time_kernelshap(score, Xv, Xe, bg, ns, a.seed)
            timings[name] = max(wall, cpu)  # the larger of wall-clock and CPU time, to stay conservative
            print(f"[3] {name:<12} background {bg:>3}, {ns} samples: {wall:5.2f} s wall, {cpu:5.2f} s CPU per window; "
                  f"additivity gap {gap:.1e}")
        runs = {"pca": 1, "isolation_forest": a.seeds, "autoencoder": a.seeds}
        hours = {k: timings[k] * a.workload * runs[k] / 3600 for k in timings}
        total = sum(hours.values())
        rows.append({
            "background": bg, "nsamples": ns,
            "pca_secs": round(timings["pca"], 3), "iforest_secs": round(timings["isolation_forest"], 3),
            "autoencoder_secs": round(timings["autoencoder"], 3),
            "pca_hours": round(hours["pca"], 2), "iforest_hours": round(hours["isolation_forest"], 2),
            "autoencoder_hours": round(hours["autoencoder"], 2),
            "total_cpu_hours": round(total, 2), "within_budget": total <= a.budget_hours,
        })

    table = pd.DataFrame(rows)
    print("\nSummary\n" + table.to_string(index=False))
    print(f"\nTotal = seconds per window x {a.workload:,} windows x model runs "
          f"(PCA 1, isolation forest {a.seeds}, autoencoder {a.seeds}). Isolation forest times are from P4.")

    table.to_csv(ROOT / "results" / f"pilot_cost_all_{stamp}.csv", index=False)
    (ROOT / "results" / f"pilot_cost_all_{stamp}.json").write_text(json.dumps({
        "date_utc": stamp, "file": a.file, "seed": a.seed, "windows": a.windows, "workload": a.workload,
        "seeds": a.seeds, "torch_threads": 1, "detectors": {k: v[1] for k, v in detectors.items()},
        "python": platform.python_version(), "sklearn": sklearn.__version__, "shap": shap.__version__,
        "torch": torch.__version__, "cpus": os.cpu_count(), "settings": rows,
    }, indent=2) + "\n")
    print(f"Saved results/pilot_cost_all_{stamp}.csv and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())