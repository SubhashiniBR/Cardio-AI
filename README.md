# Cardio-AI: ML and backend

This repository contains the Member 1 model-training and FastAPI service for the Cardio-AI Track A prototype. Predictions are for educational/decision-support visualization only, are not a substitute for diagnostic imaging, and are **vessel-level probabilities**. They do not locate a patient's plaque or lesion anatomically.

## Dataset and targets

The source workbook is `data/raw/extention of Z-Alizadeh sani dataset.xlsx`; the source file is ignored by Git. The loader reads `Sheet 1 - Table 1` and validates the inspected 303 rows and 59 columns' required outcome labels. The other sheet is empty of records. Raw categorical values remain as supplied, including `Sex` value `Fmale` and `Obesity` values `Y`/`N`. `Exertional CP` is constant in the inspected data and is excluded from model features.

The four binary tasks are:

| API/model key | Source column | Positive label |
|---|---|---|
| `cad` | `Cath` | `CAD` |
| `lad` | `LAD` | `Stenotic` |
| `lcx` | `LCX` | `Stenotic` |
| `rca` | `RCA` | `Stenotic` |

`LAD`, `LCX`, `RCA`, and `Cath` are excluded from every task's feature matrix. Numeric codes are kept as numeric observations; the implementation does not assign undocumented meanings to them.

## Environment and commands

Use Python 3.12 and the pinned dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ml.eda
python -m ml.train
pytest -q -m "not integration"
pytest -q
python -m ml.explain
uvicorn backend.main:app --reload
```

The trainer compares Logistic Regression, Random Forest, and XGBoost with 5-fold nested stratified cross-validation. In each outer fold, algorithm selection uses only that fold's training data and 4-fold inner CV; the chosen algorithm is calibrated using only outer training data and scored once on the untouched outer test fold. Preprocessing stays inside every pipeline. After nested estimates are complete, a separate 4-fold CV over all development records selects the deployable algorithm and the calibrated final model is fit on all records. This full-data selection/fitting does not produce the reported outer-fold scores. Logistic Regression and Random Forest use built-in class weighting; XGBoost is unweighted.

Final deployment selections and nested outer-fold results (mean ± standard deviation across folds):

| Target | Selected model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| CAD | Random Forest | 0.844 ± 0.065 | 0.875 ± 0.060 | 0.917 ± 0.056 | 0.894 ± 0.043 | 0.919 ± 0.054 | 0.107 ± 0.041 |
| LAD | Random Forest | 0.792 ± 0.059 | 0.790 ± 0.052 | 0.882 ± 0.053 | 0.833 ± 0.045 | 0.857 ± 0.061 | 0.153 ± 0.029 |
| LCX | XGBoost | 0.693 ± 0.064 | 0.650 ± 0.131 | 0.504 ± 0.060 | 0.565 ± 0.079 | 0.729 ± 0.075 | 0.205 ± 0.020 |
| RCA | Random Forest | 0.657 ± 0.026 | 0.569 ± 0.033 | 0.369 ± 0.161 | 0.430 ± 0.130 | 0.691 ± 0.017 | 0.213 ± 0.006 |

These are internal cross-validation estimates on 303 records, not external validation or clinical evidence.

`LOW`/`MODERATE`/`HIGH` are prototype display bands, not clinical risk categories or validated cutoffs. Outer-fold display thresholds are estimated using only each outer training split. Deployment thresholds are separately derived from the final development-data fit and do not affect the nested evaluation scores.

## API contract

- `GET /health` returns `{"status":"ok"}` without loading the workbook or model artifacts.
- `GET /schema` returns feature names, display names, types, allowed category values, observed numeric ranges, and defaults. It excludes all target/leakage fields and the constant feature.
- `POST /predict` accepts one flat JSON object containing every feature returned by `/schema`. Pydantic rejects missing fields, extra fields, unobserved categorical labels, wrong types, and numeric values outside the observed training range.

See [the detailed API contract](docs/api_contract.md), including a synthetic sample request and matching response.

Example request construction (uses a real row without including the outcome columns):

```python
import pandas as pd
row = pd.read_excel(
    "data/raw/extention of Z-Alizadeh sani dataset.xlsx",
    sheet_name="Sheet 1 - Table 1",
).iloc[0]
forbidden = {"LAD", "LCX", "RCA", "Cath", "Exertional CP"}
payload = {key: value for key, value in row.items() if key not in forbidden}
```

The prediction response contains `predictions` (four probabilities and prototype display bands), `explanations` (top patient-level SHAP contributions), `measurements` (the submitted values), and `anatomical_localization: false`. Swagger UI at `/docs` documents the typed response.

Set `CARDIO_AI_ARTIFACT_DIR` to the artifact directory and `CARDIO_AI_CORS_ORIGINS` to comma-separated exact frontend origins. Defaults allow only `http://localhost:5173` and `http://127.0.0.1:5173`; wildcard CORS is disabled. Deployment requires four `{target}_model.joblib` files, `inference_schema.json`, `background_features.joblib`, and `risk_thresholds.json`. API startup and inference do not read the workbook. Preprocessing is embedded in each calibrated model artifact; no separate preprocessing file is used.

## Validation and documentation

Tests cover dataset shape and labels, preservation of observed categorical values, forbidden-column exclusion, request validation, health/schema routes, and deterministic repeated predictions. See `docs/validation.md` for the limitations of the small dataset and the interpretation of reported validation metrics.
