#!/usr/bin/env python3
"""Validate the prediction and leaderboard files from reproduce_tg.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PREDICTION_COLUMNS = ["true_values", "predicted_values"]
LEADERBOARD_COLUMNS = [
    "Dataset",
    "Descriptor",
    "Algorithm",
    "Train R²",
    "Train MSE",
    "Train MAE",
    "Test R²",
    "Test MSE",
    "Test MAE",
    "Source_File",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reproduction_output")
    )
    return parser.parse_args()


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float]:
    values = frame[PREDICTION_COLUMNS].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Prediction files contain non-finite values")
    reference = frame["true_values"].to_numpy(float)
    prediction = frame["predicted_values"].to_numpy(float)
    return {
        "R²": float(r2_score(reference, prediction)),
        "MSE": float(mean_squared_error(reference, prediction)),
        "MAE": float(mean_absolute_error(reference, prediction)),
    }


def assert_close(label: str, observed: float, expected: float) -> None:
    if not np.isclose(observed, expected, rtol=1e-7, atol=1e-8):
        raise AssertionError(
            f"{label} mismatch: predictions give {observed}, record gives {expected}"
        )


def main() -> None:
    output_dir = args.output_dir.expanduser().resolve()
    metadata_path = output_dir / "run_metadata.json"
    leaderboard_path = output_dir / "leaderboard_record.csv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    leaderboard = pd.read_csv(leaderboard_path)
    if len(leaderboard) != 1:
        raise ValueError("leaderboard_record.csv must contain exactly one result row")
    missing = [column for column in LEADERBOARD_COLUMNS if column not in leaderboard]
    if missing:
        raise ValueError(f"Leaderboard record is missing columns: {missing}")

    files = metadata["outputs"]
    train_path = output_dir / files["train_predictions"]
    test_path = output_dir / files["test_predictions"]
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    for name, frame in (("train", train), ("test", test)):
        if list(frame.columns) != PREDICTION_COLUMNS:
            raise ValueError(
                f"{name} prediction schema must be exactly {PREDICTION_COLUMNS}"
            )
    if len(train) != metadata["n_train"] or len(test) != metadata["n_test"]:
        raise AssertionError("Prediction row counts do not match run_metadata.json")
    if len(train) + len(test) != metadata["n_total"]:
        raise AssertionError("Train and test rows do not sum to the total row count")

    row = leaderboard.iloc[0]
    train_metrics = calculate_metrics(train)
    test_metrics = calculate_metrics(test)
    for metric, value in train_metrics.items():
        assert_close(f"Train {metric}", value, float(row[f"Train {metric}"]))
    for metric, value in test_metrics.items():
        assert_close(f"Test {metric}", value, float(row[f"Test {metric}"]))
    if row["Source_File"] != test_path.name:
        raise AssertionError("Source_File does not name the test prediction file")
    if row["Dataset"] != metadata["dataset"]:
        raise AssertionError("Dataset label differs between result and metadata")
    if row["Descriptor"] != metadata["descriptor"]:
        raise AssertionError("Descriptor label differs between result and metadata")
    if row["Algorithm"] != metadata["algorithm"]:
        raise AssertionError("Algorithm label differs between result and metadata")

    print(
        "Validated: "
        f"{metadata['n_total']} rows, Test R2={test_metrics['R²']:.6f}, "
        f"Test MAE={test_metrics['MAE']:.6f}"
    )


if __name__ == "__main__":
    args = parse_args()
    main()
