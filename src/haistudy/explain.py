"""Explanation helpers shared by the pilot and the main study.

The isolation forest is explained twice, with exact TreeSHAP and with
sampling-based KernelSHAP. For the comparison to be fair, both must explain
the same number: the forest's mean path length (average depth at which a
window is isolated, over all trees). Short paths mean anomalous windows.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length


def mean_path_length(forest: IsolationForest, X) -> np.ndarray:
    """Mean path length of each row of X, computed from the forest's own score.

    scikit-learn scores a row as s = -2 ** (-E(h) / c), where E(h) is the mean
    path length and c is the average path length of an unsuccessful search in
    a tree built on ``max_samples_`` points. Inverting gives E(h) directly,
    which is far faster than walking every tree.
    """
    c = _average_path_length([forest.max_samples_])[0]
    return -np.log2(-forest.score_samples(np.asarray(X))) * c


def mean_path_length_by_trees(forest: IsolationForest, X) -> np.ndarray:
    """The same quantity computed tree by tree. Slow; used only to check the fast version."""
    X = np.asarray(X)
    depth = np.zeros(len(X))
    for tree, feats in zip(forest.estimators_, forest.estimators_features_):
        Xs = X[:, feats]
        leaves = tree.apply(Xs)
        depth += np.ravel(tree.decision_path(Xs).sum(axis=1)) + _average_path_length(tree.tree_.n_node_samples[leaves]) - 1.0
    return depth / len(forest.estimators_)


def tag_totals(values: np.ndarray, columns: list[str]) -> tuple[np.ndarray, list[str]]:
    """Sum feature attributions to one value per tag (features are named '<tag>__<stat>')."""
    tags = list(dict.fromkeys(c.split("__")[0] for c in columns))
    index = {t: i for i, t in enumerate(tags)}
    out = np.zeros((values.shape[0], len(tags)))
    for j, c in enumerate(columns):
        out[:, index[c.split("__")[0]]] += values[:, j]
    return out, tags


def rank_agreement(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman correlation between two attribution vectors, row by row.

    Rows where either vector is constant (no ranking possible) give NaN.
    """
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        if np.ptp(a[i]) > 0 and np.ptp(b[i]) > 0:
            out[i] = spearmanr(a[i], b[i]).statistic
    return out