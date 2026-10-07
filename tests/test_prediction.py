import pandas as pd
import pytest

from backend.schemas import PatientInput, get_feature_schema
from ml.constants import CONSTANT_COLUMNS_TO_EXCLUDE, LEAKAGE_COLUMNS
from ml.data import build_features_and_target, load_dataset


def test_schema_matches_safe_model_features():
    frame = load_dataset()
    expected, _ = build_features_and_target(frame, "cad")
    schema_names = {item["feature_name"] for item in get_feature_schema()}
    assert schema_names == set(expected.columns)
    assert not schema_names.intersection(LEAKAGE_COLUMNS)
    assert not schema_names.intersection(CONSTANT_COLUMNS_TO_EXCLUDE)


def test_request_schema_accepts_observed_row_and_rejects_invalid_values():
    frame = load_dataset()
    feature_frame, _ = build_features_and_target(frame, "cad")
    payload = feature_frame.iloc[0].to_dict()
    patient = PatientInput(**payload)
    assert patient.model_dump(by_alias=True)["Sex"] == payload["Sex"]

    invalid_category = dict(payload)
    invalid_category["Sex"] = "Female"
    with pytest.raises(Exception):
        PatientInput(**invalid_category)

    invalid_type = dict(payload)
    invalid_type["Age"] = "not-a-number"
    with pytest.raises(Exception):
        PatientInput(**invalid_type)

    missing_field = dict(payload)
    missing_field.pop("Age")
    with pytest.raises(Exception):
        PatientInput(**missing_field)

    out_of_range = dict(payload)
    out_of_range["Age"] = 1000
    with pytest.raises(Exception):
        PatientInput(**out_of_range)

