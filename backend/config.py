"""Backend paths, CORS settings, and prototype display bands."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ml.constants import ROOT, TARGETS

MODEL_DIR = Path(os.environ.get("CARDIO_AI_ARTIFACT_DIR", ROOT / "models"))
THRESHOLDS_PATH = MODEL_DIR / "risk_thresholds.json"
SCHEMA_PATH = MODEL_DIR / "inference_schema.json"
BACKGROUND_PATH = MODEL_DIR / "background_features.joblib"
MODEL_PATHS = {name: MODEL_DIR / f"{name}_model.joblib" for name in TARGETS}
DEPLOYED_FRONTEND_ORIGIN = "https://cardio-ai-jte7.onrender.com"
_configured_cors_origins = (
    origin.strip()
    for origin in os.environ.get(
        "CARDIO_AI_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
)
CORS_ORIGINS = tuple(dict.fromkeys((*_configured_cors_origins, DEPLOYED_FRONTEND_ORIGIN)))


def load_risk_thresholds() -> dict:
    if not THRESHOLDS_PATH.is_file():
        return {}
    return json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))


RISK_THRESHOLDS = load_risk_thresholds()


def classify_risk(target: str, probability: float) -> str:
    """Return a prototype display band, never a clinical risk category."""
    if target not in RISK_THRESHOLDS:
        raise RuntimeError(f"Risk display thresholds for {target!r} are not available; train models first.")
    thresholds = RISK_THRESHOLDS[target]
    if probability < thresholds["low_max"]:
        return "LOW"
    if probability >= thresholds["high_min"]:
        return "HIGH"
    return "MODERATE"
