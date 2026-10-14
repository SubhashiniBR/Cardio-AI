"""Read-only dataset profile and label distribution report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml.constants import DATA_PATH, ROOT, TARGETS
from ml.data import load_dataset


def profile(frame: pd.DataFrame) -> dict:
    return {
        "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
        "columns": [
            {
                "name": column,
                "dtype": str(frame[column].dtype),
                "missing": int(frame[column].isna().sum()),
                "unique_values": int(frame[column].nunique(dropna=False)),
                "values": [str(value) for value in frame[column].dropna().unique()],
            }
            for column in frame.columns
        ],
        "duplicate_rows": int(frame.duplicated().sum()),
        "targets": {
            name: {
                "column": spec["column"],
                "positive_label": spec["positive_label"],
                "counts": {
                    str(key): int(value)
                    for key, value in frame[spec["column"]].value_counts(dropna=False).items()
                },
            }
            for name, spec in TARGETS.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "data_profile.json")
    args = parser.parse_args()

    frame = load_dataset(args.data)
    result = profile(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Rows x columns: {frame.shape[0]} x {frame.shape[1]}")
    print(f"Duplicate rows: {result['duplicate_rows']}")
    print(f"Total missing values: {sum(item['missing'] for item in result['columns'])}")
    for name, target in result["targets"].items():
        print(f"{name} ({target['column']}): {target['counts']}")
    print(f"Profile written to {args.output}")


if __name__ == "__main__":
    main()
