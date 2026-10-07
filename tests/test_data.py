import pandas as pd
import pytest

from ml.constants import DATA_PATH, LEAKAGE_COLUMNS, TARGETS
from ml.data import DatasetValidationError, build_features_and_target, load_dataset, validate_dataset


def test_verified_workbook_loads_with_expected_shape():
    frame = load_dataset(DATA_PATH)
    assert frame.shape == (303, 59)
    assert frame.duplicated().sum() == 0
    assert frame.isna().sum().sum() == 0


def test_source_categories_are_preserved():
    frame = load_dataset()
    assert set(frame["Sex"]) == {"Male", "Fmale"}
    assert set(frame["Obesity"]) == {"Y", "N"}
    assert set(frame["Cath"]) == {"CAD", "Normal"}


def test_each_task_has_only_the_approved_positive_class():
    frame = load_dataset()
    for name, spec in TARGETS.items():
        X, y = build_features_and_target(frame, name)
        assert set(y.unique()) == {0, 1}
        assert y.sum() == frame[spec["column"]].eq(spec["positive_label"]).sum()
        assert set(LEAKAGE_COLUMNS).isdisjoint(X.columns)


def test_unexpected_target_label_fails_validation():
    frame = load_dataset()
    changed = frame.copy()
    changed.loc[0, "LAD"] = "Unknown"
    with pytest.raises(DatasetValidationError, match="Unexpected labels"):
        validate_dataset(changed)

