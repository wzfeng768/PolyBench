#!/usr/bin/env python3
"""Run a documented Tg benchmark from raw CSV to PolyBench-compatible outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


DATASET_LABEL = "Glass transition temperature"
DESCRIPTOR_LABEL = "Morgan_2048"
ALGORITHM_LABEL = "XGBoost"
SMILES_CANDIDATES = ("SMILES", "smiles", "CSMILES")
TARGET_CANDIDATES = ("Tg(℃)", "Tg(°C)", "Tg", "property")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Raw Tg CSV")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reproduction_output")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--smiles-column")
    parser.add_argument("--target-column")
    return parser.parse_args()


def resolve_column(
    frame: pd.DataFrame,
    explicit: str | None,
    candidates: tuple[str, ...],
    kind: str,
) -> str:
    if explicit:
        if explicit not in frame.columns:
            raise ValueError(f"Requested {kind} column {explicit!r} is absent")
        return explicit
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        f"Could not identify the {kind} column. Available columns: "
        f"{list(frame.columns)}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_fingerprints(smiles_values: list[str]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    features = np.zeros((len(smiles_values), 2048), dtype=np.uint8)
    invalid_rows: list[int] = []
    for row_index, smiles in enumerate(smiles_values):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            invalid_rows.append(row_index)
            continue
        fingerprint = generator.GetFingerprint(molecule)
        DataStructs.ConvertToNumpyArray(fingerprint, features[row_index])
    if invalid_rows:
        preview = ", ".join(str(value) for value in invalid_rows[:10])
        raise ValueError(f"Invalid SMILES at zero-based rows: {preview}")
    return features


def calculate_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(reference, prediction)),
        "mse": float(mean_squared_error(reference, prediction)),
        "mae": float(mean_absolute_error(reference, prediction)),
    }


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between zero and one")
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    frame = pd.read_csv(input_path)
    smiles_column = resolve_column(
        frame, args.smiles_column, SMILES_CANDIDATES, "SMILES"
    )
    target_column = resolve_column(
        frame, args.target_column, TARGET_CANDIDATES, "target"
    )
    smiles = frame[smiles_column].astype(str).tolist()
    target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(float)
    invalid_target_rows = np.flatnonzero(~np.isfinite(target))
    if len(invalid_target_rows):
        preview = ", ".join(str(int(value)) for value in invalid_target_rows[:10])
        raise ValueError(f"Non-numeric or missing targets at zero-based rows: {preview}")

    features = make_fingerprints(smiles)
    all_indices = np.arange(len(frame), dtype=int)
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )

    model_settings = {
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": args.seed,
        "n_jobs": 4,
        "verbosity": 0,
    }
    model = XGBRegressor(**model_settings)
    model.fit(features[train_indices], target[train_indices])
    train_prediction = model.predict(features[train_indices])
    test_prediction = model.predict(features[test_indices])
    train_metrics = calculate_metrics(target[train_indices], train_prediction)
    test_metrics = calculate_metrics(target[test_indices], test_prediction)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{DESCRIPTOR_LABEL}_{ALGORITHM_LABEL}_{DATASET_LABEL}"
    train_name = f"{stem}_train_predictions.csv"
    test_name = f"{stem}_test_predictions.csv"
    leaderboard_name = "leaderboard_record.csv"
    metadata_name = "run_metadata.json"

    pd.DataFrame(
        {
            "true_values": target[train_indices],
            "predicted_values": train_prediction,
        }
    ).to_csv(output_dir / train_name, index=False, float_format="%.10g")
    pd.DataFrame(
        {
            "true_values": target[test_indices],
            "predicted_values": test_prediction,
        }
    ).to_csv(output_dir / test_name, index=False, float_format="%.10g")

    leaderboard_record = {
        "Dataset": DATASET_LABEL,
        "Descriptor": DESCRIPTOR_LABEL,
        "Algorithm": ALGORITHM_LABEL,
        "Train R²": train_metrics["r2"],
        "Train MSE": train_metrics["mse"],
        "Train MAE": train_metrics["mae"],
        "Test R²": test_metrics["r2"],
        "Test MSE": test_metrics["mse"],
        "Test MAE": test_metrics["mae"],
        "Source_File": test_name,
    }
    pd.DataFrame([leaderboard_record]).to_csv(
        output_dir / leaderboard_name, index=False, float_format="%.10g"
    )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_columns": {
            "smiles": smiles_column,
            "target": target_column,
        },
        "dataset": DATASET_LABEL,
        "descriptor": DESCRIPTOR_LABEL,
        "algorithm": ALGORITHM_LABEL,
        "n_total": int(len(all_indices)),
        "n_train": int(len(train_indices)),
        "n_test": int(len(test_indices)),
        "split": {
            "type": "random row split",
            "test_fraction": args.test_size,
            "seed": args.seed,
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist(),
        },
        "fingerprint": {"type": "Morgan", "radius": 2, "n_bits": 2048},
        "model_settings": model_settings,
        "metrics": {"train": train_metrics, "test": test_metrics},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdBase.rdkitVersion,
            "scikit-learn": package_version("scikit-learn"),
            "xgboost": package_version("xgboost"),
        },
        "outputs": {
            "train_predictions": train_name,
            "test_predictions": test_name,
            "leaderboard_record": leaderboard_name,
        },
    }
    (output_dir / metadata_name).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    print(f"Rows: {len(all_indices)} ({len(train_indices)} train / {len(test_indices)} test)")
    print(f"Train R2={train_metrics['r2']:.6f}, MAE={train_metrics['mae']:.6f}")
    print(f"Test  R2={test_metrics['r2']:.6f}, MAE={test_metrics['mae']:.6f}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
