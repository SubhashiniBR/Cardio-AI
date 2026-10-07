"""FastAPI prediction interface used by the Cardio-AI frontend."""

from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import BACKGROUND_PATH, CORS_ORIGINS, MODEL_DIR, MODEL_PATHS, classify_risk
from backend.schemas import PatientInput, PredictResponse, get_feature_schema
from ml.constants import TARGETS
from ml.explain import patient_explanation

app = FastAPI(
    title="Cardio-AI Prediction API", version="1.0.0",
    description=("Educational prototype predictions. Risk bands are display bands, not clinical "
                 "categories. Outputs are vessel-level probabilities and do not anatomically localize lesions."),
)
app.add_middleware(
    CORSMiddleware, allow_origins=list(CORS_ORIGINS), allow_credentials=False,
    allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Authorization"],
)


@lru_cache(maxsize=1)
def _background() -> pd.DataFrame:
    if not BACKGROUND_PATH.is_file():
        raise FileNotFoundError(f"Explanation background artifact missing: {BACKGROUND_PATH}")
    return joblib.load(BACKGROUND_PATH)


@lru_cache(maxsize=4)
def _model(target: str):
    path = MODEL_PATHS[target]
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact missing: {path}")
    return joblib.load(path)


@app.get("/health")
def health():
    """Liveness endpoint; works without the source workbook or model artifacts."""
    return {"status": "ok"}


@app.get("/schema")
def schema():
    try:
        return {"features": get_feature_schema()}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictResponse)
def predict(patient: PatientInput):
    measurements = patient.model_dump(by_alias=True)
    try:
        feature_schema = get_feature_schema()
        if set(measurements) != {item["feature_name"] for item in feature_schema}:
            raise HTTPException(status_code=422, detail="Request does not match the saved inference feature schema.")
        for item in feature_schema:
            value = measurements[item["feature_name"]]
            if item.get("allowed_options") is not None and value not in item["allowed_options"]:
                raise HTTPException(status_code=422, detail=f"Invalid value for {item['feature_name']!r}.")
        X = pd.DataFrame([measurements])
        background = _background()
        predictions, explanations = {}, {}
        for target in TARGETS:
            model = _model(target)
            probability = float(model.predict_proba(X)[0, 1])
            predictions[target] = {"probability": probability, "risk": classify_risk(target, probability)}
            explanations[target] = patient_explanation(model, background, X, top_n=8)
    except HTTPException:
        raise
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return {
        "predictions": predictions, "explanations": explanations,
        "measurements": measurements, "anatomical_localization": False,
    }
