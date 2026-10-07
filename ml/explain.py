"""SHAP explanations over the fitted preprocessing-plus-model pipeline."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.constants import RANDOM_SEED, ROOT, TARGETS
from ml.data import build_features_and_target, load_dataset


def shap_values(model, X_background: pd.DataFrame, X_patient: pd.DataFrame):
    """Explain the calibrated positive probability in original feature space."""
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import shap
    except ImportError as exc:
        raise RuntimeError("Install the pinned SHAP dependency to enable explanations.") from exc

    columns = list(X_background.columns)
    background = X_background.sample(
        n=min(40, len(X_background)), random_state=42
    ).reset_index(drop=True)
    category_values = {}
    numeric_background = background.copy()
    for column in columns:
        if not pd.api.types.is_numeric_dtype(background[column].dtype):
            values = list(X_background[column].astype(str).unique())
            category_values[column] = values
            code = {value: index for index, value in enumerate(values)}
            numeric_background[column] = background[column].astype(str).map(code)
    numeric_background = numeric_background.astype(float)

    def positive_probability(masked_values):
        masked = pd.DataFrame(masked_values, columns=columns)
        batch = pd.DataFrame(index=masked.index)
        for column in columns:
            if column in category_values:
                codes = np.rint(masked[column].to_numpy()).astype(int)
                codes = np.clip(codes, 0, len(category_values[column]) - 1)
                batch[column] = [category_values[column][code] for code in codes]
            else:
                batch[column] = masked[column].to_numpy()
        return model.predict_proba(batch)[:, 1]

    numeric_patient = X_patient.copy()
    for column, values in category_values.items():
        code = {value: index for index, value in enumerate(values)}
        numeric_patient[column] = X_patient[column].astype(str).map(code).fillna(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        explainer = shap.Explainer(
            positive_probability,
            numeric_background,
            algorithm="permutation",
            feature_names=columns,
            seed=RANDOM_SEED,
        )
        explanation = explainer(
            numeric_patient.astype(float),
            max_evals=2 * len(columns) + 1,
        )
    return columns, np.asarray(explanation.values)


def patient_explanation(model, X_background, patient, top_n: int = 8) -> list[dict]:
    names, values = shap_values(model, X_background, patient)
    ranked = sorted(
        ((name, float(value)) for name, value in zip(names, values[0], strict=True)),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:top_n]
    return [
        {
            "feature": feature,
            "display_name": feature,
            "value": patient.iloc[0][feature].item()
            if hasattr(patient.iloc[0][feature], "item")
            else patient.iloc[0][feature],
            "contribution": contribution,
        }
        for feature, contribution in ranked
    ]


def global_importance(model, X_background, top_n: int | None = None) -> list[dict]:
    explain_rows = X_background.sample(
        n=min(30, len(X_background)), random_state=42
    ).reset_index(drop=True)
    names, values = shap_values(model, X_background, explain_rows)
    ranked = sorted(
        ((name, float(np.mean(np.abs(values[:, index])))) for index, name in enumerate(names)),
        key=lambda item: item[1],
        reverse=True,
    )
    if top_n is not None:
        ranked = ranked[:top_n]
    return [
        {"feature": feature, "display_name": feature, "mean_abs_contribution": value}
        for feature, value in ranked
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    frame = load_dataset()
    out = ROOT / "results" / "shap"
    out.mkdir(parents=True, exist_ok=True)
    for name in TARGETS:
        X, _ = build_features_and_target(frame, name)
        model = joblib.load(ROOT / "models" / f"{name}_model.joblib")
        importance = global_importance(model, X, args.top)
        pd.DataFrame(importance).to_csv(out / f"{name}_global_importance.csv", index=False)
        pd.DataFrame(patient_explanation(model, X, X.iloc[[0]])).to_csv(
            out / f"{name}_example_explanation.csv", index=False
        )
        print(f"Wrote SHAP summaries for {name}")


if __name__ == "__main__":
    main()
