# Explainable unsupervised anomaly detection on HAI 21.03 under an alert budget

Code for an MSc Cybersecurity research project (University of Essex Online, Saheed Oyewo).

The study asks two questions on the public HAI 21.03 industrial control system dataset:

1. Within a budget of five false alerts in any 12-hour period, how do windowed PCA, an isolation forest and an autoencoder compare in detecting labelled attacks, measured by the share of attack events detected?
2. How faithful, stable and costly are SHAP explanations of those detectors, and how closely does approximate KernelSHAP agree with exact TreeSHAP on the same isolation forest?

The analysis plan is pre-registered on the Open Science Framework before the held-out data are opened. Link: TODO (add after lodging in week 10).

## Status

Pre-registered on OSF: https://doi.org/10.17605/OSF.IO/85GBC (lodged 25 November 2026). The analysis plan is fixed at tag v1.0-preregistered (commit 813df5f). The repository holds the environment, the windowing and scaling code with tests, the chain-of-custody tool, and the pilot scripts and pilot note (docs/pilot\_note.md) that informed the registered settings. Detector code is added after registration, on validation data only; the held-out files remain sealed until the evaluation stage.

## Layout

|Path|Purpose|
|-|-|
|`config/study.yaml`|Fixed study settings. Must match the pre-registration|
|`src/haistudy/`|Python package. Only `custody.py` is implemented so far|
|`scripts/check\_environment.py`|Checks the toolchain on synthetic data|
|`scripts/custody.py`|Hashes the dataset and verifies the sealed held-out files|
|`docs/deviation\_log.md`|Dated record of any departure from the pre-registration|
|`docs/osf\_project\_setup.md`|Text and structure for the OSF project|
|`data/`, `results/`|Local only. Ignored by Git|

## Set up (Windows, PowerShell)

Python 3.12 is required.

```powershell
py -3.12 -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python scripts\\check\_environment.py
pytest
```

`check\_environment.py` should end with `PASS`. Then record the exact environment on your machine:

```powershell
pip freeze > requirements-lock.txt
```

## Put the repository on GitHub

```powershell
git init -b main
git add .
git commit -m "Initial scaffold: environment, study settings, custody tool"
git tag v0.1-scaffold
```

Create an empty repository on GitHub (no README, no licence, since both are here), then:

```powershell
git remote add origin https://github.com/<your-account>/hai-alert-budget-xai.git
git push -u origin main --tags
```

Keep it private until the dissertation is submitted if you prefer; the proposal commits to releasing it under the MIT Licence.

## Data

HAI 21.03 is published by the Affiliated Institute of ETRI under CC BY-SA 4.0 at https://github.com/icsdataset/hai. The data are not stored in this repository.

Do not download the data until formal ethics approval is in place. Then, in week 6:

```powershell
python scripts\\custody.py hash data\\raw docs\\hai\_manifest.json
git add docs\\hai\_manifest.json
git commit -m "Record SHA-256 manifest of HAI 21.03 at download"
```

Immediately before the held-out files are opened in week 20:

```powershell
python scripts\\custody.py verify data\\raw docs\\hai\_manifest.json --only test3 test4 test5
```

## Milestone tags

|Tag|When|
|-|-|
|`v0.1-scaffold`|Environment and settings in place|
|`v0.2-data-sealed`|Dataset hashed; held-out files sealed (week 6)|
|`v0.3-pilot`|Pilot complete (week 9)|
|`v1.0-preregistered`|Matches the lodged pre-registration (week 10)|
|`v2.0-results`|Held-out evaluation complete (week 22)|

## Licence

Code: MIT (see `LICENSE`). Any derived data that is shared carries CC BY-SA 4.0, as HAI does.

## Use of generative AI

The code in this repository was drafted with Claude, an AI assistant, and reviewed, run and tested by the author. This covers the initial scaffold (environment, settings and custody tool), the windowing and scaling module, the explanation helpers and the pilot scripts. Each study decision was taken by the author and is recorded with its reasons in docs/pilot_note.md.

