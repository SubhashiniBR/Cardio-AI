"""Fold-local preprocessing for numeric and categorical workbook features."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X):
    """Build an unfitted transformer; fit it only inside model CV/training."""
    numeric = list(X.select_dtypes(include="number").columns)
    categorical = [column for column in X.columns if column not in numeric]

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(estimator, X):
    """Pair fold-local preprocessing with a classifier."""
    return Pipeline(
        steps=[("preprocessor", build_preprocessor(X)), ("model", estimator)]
    )
