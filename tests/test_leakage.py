from ml.constants import CONSTANT_COLUMNS_TO_EXCLUDE, LEAKAGE_COLUMNS, TARGETS
from ml.data import build_features_and_target, load_dataset


def test_outcome_and_constant_columns_never_enter_features():
    frame = load_dataset()
    forbidden = set(LEAKAGE_COLUMNS) | set(CONSTANT_COLUMNS_TO_EXCLUDE)
    for target_name in TARGETS:
        X, y = build_features_and_target(frame, target_name)
        assert forbidden.isdisjoint(X.columns)
        assert len(X) == len(y) == len(frame)
        assert set(y.unique()) == {0, 1}


def test_positive_class_mapping_is_explicit():
    frame = load_dataset()
    for target_name, spec in TARGETS.items():
        _, y = build_features_and_target(frame, target_name)
        assert y.sum() == frame[spec["column"]].eq(spec["positive_label"]).sum()
