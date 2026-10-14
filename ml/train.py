"""Train, select, and evaluate leakage-safe classifiers with nested CV."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from ml.constants import RANDOM_SEED, ROOT, TARGETS
from ml.data import build_features_and_target, load_dataset
from ml.evaluate import choose_risk_thresholds, classification_metrics
from ml.inference_schema import build_inference_schema
from ml.preprocessing import build_pipeline


def candidate_estimators():
    estimators = {
        "logistic_regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample", min_samples_leaf=2,
            n_jobs=1, random_state=RANDOM_SEED,
        ),
    }
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("XGBoost is required; install requirements.txt before training.") from exc
    estimators["xgboost"] = XGBClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.04, subsample=0.9,
        colsample_bytree=0.9, reg_lambda=1.0, scale_pos_weight=1.0,
        eval_metric="logloss", n_jobs=1, random_state=RANDOM_SEED, verbosity=0,
    )
    return estimators


def _select_algorithm(X, y, estimators, splits=4):
    """Choose by mean ROC-AUC, then F1, using only the supplied training data."""
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_SEED)
    scores = {}
    for name, estimator in estimators.items():
        pipeline = build_pipeline(clone(estimator), X)
        scored = cross_validate(pipeline, X, y, cv=cv, scoring=("roc_auc", "f1"), n_jobs=1)
        scores[name] = {
            "roc_auc_mean": float(scored["test_roc_auc"].mean()),
            "f1_mean": float(scored["test_f1"].mean()),
        }
    winner = max(scores, key=lambda name: (scores[name]["roc_auc_mean"], scores[name]["f1_mean"]))
    return winner, scores


def _calibrated(estimator, X):
    pipeline = build_pipeline(clone(estimator), X)
    return CalibratedClassifierCV(
        estimator=pipeline, method="sigmoid",
        cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED),
        ensemble=True,
    )


def _plot_evaluation(target, y, probabilities, output_dir):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

    for name in ("confusion_matrices", "roc_curves", "calibration_curves"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y, probabilities >= .5, labels=[0, 1], ax=ax)
    ax.set_title(f"{target.upper()} nested outer-fold confusion matrix")
    fig.tight_layout(); fig.savefig(output_dir / "confusion_matrices" / f"{target}.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_predictions(y, probabilities, ax=ax, name=target.upper())
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_title(f"{target.upper()} nested outer-fold ROC")
    fig.tight_layout(); fig.savefig(output_dir / "roc_curves" / f"{target}.png", dpi=160); plt.close(fig)
    observed, predicted = calibration_curve(y, probabilities, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(predicted, observed, marker="o", label=target.upper())
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction")
    ax.set_title(f"{target.upper()} nested outer-fold calibration curve"); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "calibration_curves" / f"{target}.png", dpi=160); plt.close(fig)


def run_training(data_path: Path = None, model_dir: Path = None, results_dir: Path = None):
    data_path = data_path or ROOT / "data" / "raw" / "extention of Z-Alizadeh sani dataset.xlsx"
    model_dir = model_dir or ROOT / "models"
    results_dir = results_dir or ROOT / "results"
    model_dir.mkdir(parents=True, exist_ok=True); results_dir.mkdir(parents=True, exist_ok=True)
    frame = load_dataset(data_path)
    try:
        manifest_dataset = str(Path(data_path).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        manifest_dataset = Path(data_path).name
    manifest = {"seed": RANDOM_SEED, "dataset": manifest_dataset,
                "evaluation": "5-fold nested stratified CV; inner 4-fold selection and calibration",
                "targets": {}}
    comparison_rows, fold_rows, final_rows = [], [], []
    thresholds, fold_thresholds = {}, {}
    estimators = candidate_estimators()

    for target in TARGETS:
        X, y = build_features_and_target(frame, target)
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        probabilities = np.full(len(y), np.nan)
        selections = []
        fold_thresholds[target] = {}
        for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), 1):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            selected, inner_scores = _select_algorithm(X_train, y_train, estimators)
            selections.append(selected)
            model = _calibrated(estimators[selected], X_train)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model.fit(X_train, y_train)
            fold_prob = model.predict_proba(X_test)[:, 1]
            probabilities[test_idx] = fold_prob
            metrics = classification_metrics(y_test, fold_prob)
            fold_rows.append({"target": target, "fold": fold, "selected_model": selected, **metrics})
            # These display boundaries are estimated exclusively from outer training data.
            train_prob = model.predict_proba(X_train)[:, 1]
            fold_thresholds[target][str(fold)] = choose_risk_thresholds(
                y_train, train_prob, method="outer-training calibrated probability"
            )

        if np.isnan(probabilities).any():
            raise RuntimeError(f"Outer CV did not produce predictions for all {target} records.")
        metrics_by_fold = [row for row in fold_rows if row["target"] == target]
        pooled = classification_metrics(y, probabilities)
        summary = {}
        for metric, pooled_value in pooled.items():
            values = [row[metric] for row in metrics_by_fold]
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1))
            summary[f"{metric}_outer_oof"] = pooled_value
        counts = {name: selections.count(name) for name in estimators}

        # Select the deployable algorithm with all available development data only
        # after the unbiased outer-fold estimate has been computed.
        selected_final, full_selection_scores = _select_algorithm(X, y, estimators)
        comparison_rows.extend({"target": target, "model": name, **scores,
                                "selected_in_outer_folds": counts[name]}
                               for name, scores in full_selection_scores.items())
        final_rows.append({"target": target, "outer_fold_selection_counts": json.dumps(counts), **summary})
        _plot_evaluation(target, y, probabilities, results_dir)
        pd.DataFrame({"target": target, "actual": y.to_numpy(),
                      "probability": probabilities}).to_csv(results_dir / f"{target}_oof_predictions.csv", index=False)

        deploy_model = _calibrated(estimators[selected_final], X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            deploy_model.fit(X, y)
        joblib.dump(deploy_model, model_dir / f"{target}_model.joblib")
        deploy_prob = deploy_model.predict_proba(X)[:, 1]
        thresholds[target] = choose_risk_thresholds(
            y, deploy_prob, method="final development-fit calibrated probability"
        )
        manifest["targets"][target] = {
            "selected_model_for_deployment": selected_final,
            "outer_fold_selection_counts": counts,
            "inner_full_data_selection_scores": full_selection_scores,
            "nested_outer_metrics": summary,
            "risk_thresholds": thresholds[target],
        }
        print(f"{target.upper()}: deployment selection {selected_final}; nested outer ROC-AUC="
              f"{summary['roc_auc_mean']:.3f} (+/- {summary['roc_auc_std']:.3f})")

    # The API schema is serialized from the training workbook; API startup does not read it.
    (model_dir / "inference_schema.json").write_text(
        json.dumps(build_inference_schema(frame), indent=2), encoding="utf-8")
    (model_dir / "risk_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    (model_dir / "outer_fold_risk_thresholds.json").write_text(json.dumps(fold_thresholds, indent=2), encoding="utf-8")
    (model_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    Xcad, _ = build_features_and_target(frame, "cad")
    joblib.dump(Xcad, model_dir / "background_features.joblib")
    pd.DataFrame(comparison_rows).to_csv(results_dir / "model_comparison.csv", index=False)
    pd.DataFrame(final_rows).to_csv(results_dir / "final_calibrated_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(results_dir / "fold_metrics.csv", index=False)
    # Kept as a results convenience copy; API loads the deployment artifact from models/.
    (results_dir / "risk_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    return comparison_rows, final_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    run_training(data_path=args.data)


if __name__ == "__main__":
    main()
