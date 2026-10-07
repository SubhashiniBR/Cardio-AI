# Frontend API contract

Base URL during local development: `http://127.0.0.1:8000`.

## `GET /health`

```json
{"status":"ok"}
```

## `GET /schema`

Returns `{"features": [...]}`. Each feature includes its source feature name, display name, type, group, allowed category values when categorical, observed minimum/maximum when numeric, and a default. The schema excludes `LAD`, `LCX`, `RCA`, `Cath`, and constant `Exertional CP`.

## `POST /predict`

Send one flat JSON object with **every** feature from `/schema`, using the workbook's original category spellings and values. The Pydantic request model rejects missing and extra keys, categories absent from the source data, invalid types, and values outside the observed numeric range.

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  --data-binary "@docs/sample_request.json"
```

The example request is a synthetic feature vector made from each numeric field's observed median and each categorical field's mode. It is not a single patient's row. The corresponding example response is saved in [sample_response.json](sample_response.json). It contains four probabilities and risk bands, top patient-level SHAP contributions, the submitted measurements, and `anatomical_localization: false`.

`LOW`, `MODERATE`, and `HIGH` are prototype display bands, not clinical risk categories or cutoffs. The returned vessel-level values do not locate lesions or plaques.

The API reads the saved `inference_schema.json`, four calibrated `{target}_model.joblib` files, `background_features.joblib`, and `risk_thresholds.json` from `CARDIO_AI_ARTIFACT_DIR` (default `models/`). Preprocessing is embedded in each calibrated model artifact. Neither app startup nor `/health` requires the training workbook. Configure frontend access with `CARDIO_AI_CORS_ORIGINS`, a comma-separated list of exact origins; defaults are `http://localhost:5173,http://127.0.0.1:5173` and wildcard origins are disabled.
