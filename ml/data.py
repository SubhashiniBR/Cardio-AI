"""Read and validate the original workbook without altering it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.constants import (
    CONSTANT_COLUMNS_TO_EXCLUDE,
    DATA_PATH,
    DATA_SHEET,
    LEAKAGE_COLUMNS,
    TARGETS,
)


class DatasetValidationError(ValueError):
    """Raised when the workbook differs from the inspected source contract."""


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the verified data sheet; never write to the source workbook."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_excel(path, sheet_name=DATA_SHEET, engine="openpyxl")
    validate_dataset(frame)
    return frame


def validate_dataset(frame: pd.DataFrame) -> None:
    """Check the fields and observed target values needed by this project."""
    if frame.empty:
        raise DatasetValidationError("The verified data sheet has no records.")
    if frame.columns.has_duplicates:
        raise DatasetValidationError("Dataset contains duplicate column names.")

    required = {item["column"] for item in TARGETS.values()}
    required.add("Exertional CP")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DatasetValidationError(f"Required columns are missing: {missing}")

    for name, spec in TARGETS.items():
        column = spec["column"]
        if frame[column].isna().any():
            raise DatasetValidationError(f"Target {column!r} contains missing values.")
        observed = set(frame[column].unique())
        expected = {spec["positive_label"]}
        expected.update({"Normal"} if name != "cad" else {"Normal"})
        if observed != expected:
            raise DatasetValidationError(
                f"Unexpected labels for {column!r}: {sorted(map(str, observed))}; "
                f"expected {sorted(expected)}."
            )

    if frame["Exertional CP"].nunique(dropna=False) != 1:
        raise DatasetValidationError(
            "Exertional CP is no longer constant; review the feature-exclusion rule."
        )


def build_features_and_target(
    frame: pd.DataFrame, target_name: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a leakage-safe feature matrix and 0/1 target for one task."""
    if target_name not in TARGETS:
        raise KeyError(f"Unknown target {target_name!r}; expected one of {tuple(TARGETS)}")
    validate_dataset(frame)
    target_spec = TARGETS[target_name]
    y = frame[target_spec["column"]].eq(target_spec["positive_label"]).astype("int8")
    excluded = set(LEAKAGE_COLUMNS) | set(CONSTANT_COLUMNS_TO_EXCLUDE)
    X = frame.drop(columns=[column for column in excluded if column in frame.columns]).copy()
    leaked = excluded.intersection(X.columns)
    if leaked:
        raise AssertionError(f"Forbidden model input columns: {sorted(leaked)}")
    if target_spec["column"] in X.columns:
        raise AssertionError(f"Target {target_spec['column']} appears in model inputs.")
    return X, y.rename(target_name)
