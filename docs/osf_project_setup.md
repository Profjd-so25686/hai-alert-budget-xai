# Open Science Framework project: what to create

You create the account yourself at https://osf.io (sign in through your institution or with ORCID if you have one). Then create one project with the settings below. Keep the project private until the pre-registration is lodged in week 10.

## Project

- **Title:** Explainable Unsupervised Anomaly Detection for SCADA Telemetry under a Small-Utility Alert Budget: A Design Science Study Using the HAI Industrial Control System Dataset
- **Category:** Project
- **Description:** MSc Cybersecurity research project (University of Essex Online). The study compares three unsupervised anomaly detectors (windowed PCA, isolation forest, autoencoder) on the public HAI 21.03 dataset under a budget of five false alerts in any 12-hour period, and measures the fidelity, stability and computational cost of SHAP explanations, including agreement between KernelSHAP and exact TreeSHAP on the same isolation forest. All data are public secondary data. No human participants are involved.
- **Licence:** MIT for code; CC BY 4.0 for documents
- **Tags:** industrial control systems; SCADA; anomaly detection; explainable AI; SHAP; HAI dataset; alert budget; pre-registration
- **Contributors:** yourself as administrator. Add your supervisor as a read-only contributor if he wishes.

## Components

| Component | Contents |
|---|---|
| Pre-registration | The registration itself, created from the template for secondary data analysis (Van den Akker et al., 2021) |
| Materials | Link to the GitHub repository (use the GitHub add-on); `config/study.yaml` |
| Custody | `docs/hai_manifest.json` once the data are hashed in week 6 |
| Deviations | `docs/deviation_log.md`, updated as needed after lodging |

## Before lodging in week 10

1. Fill every placeholder in the pre-registration pack (seeds, tuning grids, KernelSHAP settings, pilot findings, file hashes).
2. Confirm the pack agrees with the final proposal and with `config/study.yaml`.
3. Tag the repository `v1.0-preregistered` and cite the commit in the registration.
4. Lodge the registration, then add its link to the README and the dissertation.
