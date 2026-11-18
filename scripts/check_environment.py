"""Environment check on synthetic data. It uses no HAI data, so it can run before ethics approval.

It prints the library versions, then confirms on random numbers that:
  1. exact TreeSHAP attributions for a scikit-learn isolation forest sum to the
     forest's mean path length (the additivity check planned for the pilot), and
  2. KernelSHAP can explain that same output.
A pass here shows the toolchain works. It is not a study result.
"""
import platform
import sys
import time

import numpy as np
import scipy
import shap
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length


def mean_path_length(forest: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Mean path length over the trees: the quantity both explainers must explain."""
    depth = np.zeros(len(X))
    for tree, feats in zip(forest.estimators_, forest.estimators_features_):
        Xs = X[:, feats]
        leaves = tree.apply(Xs)
        depth += np.ravel(tree.decision_path(Xs).sum(axis=1)) + _average_path_length(tree.tree_.n_node_samples[leaves]) - 1.0
    return depth / len(forest.estimators_)


def main() -> int:
    print("python      ", platform.python_version())
    for name, mod in (("numpy", np), ("scipy", scipy), ("scikit-learn", sklearn), ("shap", shap)):
        print(f"{name:<12}", mod.__version__)
    try:
        import torch
        print("torch       ", torch.__version__)
    except ImportError:
        print("torch        not installed (needed only for the autoencoder)")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 20))
    Xa = X[:30] + rng.normal(3, 1, size=(30, 20)) * (rng.random((30, 20)) < 0.2)
    forest = IsolationForest(n_estimators=100, random_state=0).fit(X)

    tree_exp = shap.TreeExplainer(forest)
    sv = tree_exp.shap_values(Xa)
    total = sv.sum(axis=1) + float(np.ravel(tree_exp.expected_value)[0])
    gap = float(np.abs(total - mean_path_length(forest, Xa)).max())
    print(f"TreeSHAP additivity: largest gap from mean path length = {gap:.2e}")

    t0 = time.perf_counter()
    kernel_exp = shap.KernelExplainer(lambda z: mean_path_length(forest, z), shap.kmeans(X, 10))
    ks = kernel_exp.shap_values(Xa[:3], nsamples=200, silent=True)
    print(f"KernelSHAP ran on 3 windows in {time.perf_counter() - t0:.1f} s, shape {np.shape(ks)}")

    ok = gap < 1e-8
    print("PASS" if ok else "FAIL: TreeSHAP does not sum to the mean path length with these versions")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
