"""API request/response contracts loaded from persisted training artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from backend.config import SCHEMA_PATH


def read_feature_contract(path: Path = SCHEMA_PATH) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Inference schema artifact missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("features"), list):
        raise ValueError("Unsupported or invalid inference schema artifact.")
    return data


def get_feature_schema() -> list[dict]:
    return read_feature_contract()["features"]


def create_patient_model():
    """Construct OpenAPI request validation without accessing the source workbook."""
    try:
        features = get_feature_schema()
    except (FileNotFoundError, ValueError):
        # Importing the app and serving /health remain possible before artifacts deploy.
        # Without a schema, let the route return a clear 503 instead of a misleading
        # request-validation 422; no prediction is possible until the contract exists.
        return create_model("PatientInput", __config__=ConfigDict(extra="allow"))
    fields = {}
    for index, item in enumerate(features):
        name = f"feature_{index}"
        description = item["display_name"]
        options = item.get("allowed_options")
        if options is not None and item["type"] == "categorical":
            annotation = Literal.__getitem__(tuple(options))
        elif item["type"] == "integer":
            annotation = int
        else:
            annotation = float
        constraints = {"description": description}
        if item.get("minimum") is not None:
            constraints["ge"] = item["minimum"]
            constraints["le"] = item["maximum"]
        fields[name] = (annotation, Field(..., alias=item["feature_name"], **constraints))
    return create_model("PatientInput", __config__=ConfigDict(extra="forbid", populate_by_name=True), **fields)


PatientInput = create_patient_model()


class TargetPrediction(BaseModel):
    probability: float = Field(ge=0, le=1)
    risk: Literal["LOW", "MODERATE", "HIGH"]


class TargetPredictions(BaseModel):
    cad: TargetPrediction
    lad: TargetPrediction
    lcx: TargetPrediction
    rca: TargetPrediction


class FeatureContribution(BaseModel):
    feature: str
    display_name: str
    value: str | int | float | bool
    contribution: float


class TargetExplanations(BaseModel):
    cad: list[FeatureContribution]
    lad: list[FeatureContribution]
    lcx: list[FeatureContribution]
    rca: list[FeatureContribution]


class PredictResponse(BaseModel):
    predictions: TargetPredictions
    explanations: TargetExplanations
    measurements: dict[str, str | int | float | bool]
    anatomical_localization: Literal[False] = False
