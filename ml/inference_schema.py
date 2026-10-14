"""Build the serializable feature contract used by the API at runtime."""

from __future__ import annotations

import pandas as pd

from ml.constants import CONSTANT_COLUMNS_TO_EXCLUDE, LEAKAGE_COLUMNS


def build_inference_schema(frame: pd.DataFrame) -> dict:
    excluded = set(LEAKAGE_COLUMNS) | set(CONSTANT_COLUMNS_TO_EXCLUDE)
    features = frame.drop(columns=[column for column in excluded if column in frame.columns])
    schema = []
    for column in features.columns:
        values = features[column]
        numeric = pd.api.types.is_numeric_dtype(values.dtype)
        item = {
            "feature_name": column,
            "display_name": column,
            "type": "integer" if numeric and pd.api.types.is_integer_dtype(values.dtype)
            else "number" if numeric else "categorical",
            "category": "clinical",
            "allowed_options": None,
            "minimum": None,
            "maximum": None,
            "default": None,
        }
        if numeric:
            item["minimum"] = int(values.min()) if item["type"] == "integer" else float(values.min())
            item["maximum"] = int(values.max()) if item["type"] == "integer" else float(values.max())
            median = values.median()
            item["default"] = int(median) if item["type"] == "integer" else float(median)
            # Preserve observed discrete numeric values as allowed values without
            # assigning undocumented meanings to those codes.
            unique = sorted(values.dropna().unique().tolist())
            if item["type"] == "integer" and len(unique) <= 12:
                item["allowed_options"] = [int(value) for value in unique]
        else:
            options = values.dropna().astype(str).unique().tolist()
            item["allowed_options"] = options
            item["default"] = str(values.mode().iloc[0])
        schema.append(item)
    return {"version": 1, "features": schema}
