"""haistudy: explainable unsupervised anomaly detection on HAI 21.03 under an alert budget.

This package is deliberately empty of analysis code. The pre-registration is lodged
before any detector is built, so the modules below are placeholders that name where
each part of the study will live:

    data        loading HAI files and applying the fixed partition
    windows     60-second windows, 10-second step, per-tag summary features
    detectors   windowed PCA, isolation forest, autoencoder
    alerts      thresholding, alert merging, rolling 12-hour budget, calibration
    explain     TreeSHAP and KernelSHAP, tag-group deletion, stability, latency
    stats       exact McNemar, Holm, Wilcoxon, bootstrap intervals
    custody     SHA-256 manifests for the dataset and the sealed held-out files
"""

__version__ = "0.1.0"
