# Minimal Reproduction Example

This protocol runs one documented PolyBench workflow for glass-transition temperature from the raw task CSV to prediction files and a leaderboard-compatible result row. It uses a Morgan 2,048-bit fingerprint and XGBoost so that descriptor generation is self-contained in the pinned environment.

It does not recreate a historical stored score whose original split indices are unavailable. The generated result is a new, auditable open-test run with its own saved indices and metadata.

## 1. Create the environment

```bash
conda env create -f environment.yml
conda activate PolyBench
```

Confirm the principal versions:

```bash
python -c "import rdkit, sklearn, xgboost; print(rdkit.__version__, sklearn.__version__, xgboost.__version__)"
```

Expected versions are RDKit 2025.03.5, scikit-learn 1.7.1, and XGBoost 3.0.5. `environment.yml` has no machine-specific `prefix`, so Conda chooses the installation location.

## 2. Obtain the raw task CSV

Download version 2 of the PolyBench dataset from [Figshare](https://doi.org/10.6084/m9.figshare.30917717) and locate:

```text
Glass transition temperature.csv
```

The deposited file has 8,055 data rows and the columns `polymer_name`, `SMILES`, and `Tg(℃)`. The audited version-2 file has SHA-256 digest `37ef7d910c7e1e6a0a0ee7a613acb981433c535c026e1ac4da740cb0389fc83d`. The script records the digest it actually reads and also accepts explicit `--smiles-column` and `--target-column` arguments if a compatible copy uses different headers.

## 3. Run the documented model

From the repository root:

```bash
python reproducibility/reproduce_tg.py \
  --input "/path/to/Glass transition temperature.csv" \
  --output-dir reproduction_output \
  --seed 42
```

The script performs the following operations:

1. Verifies that every SMILES and target value is usable.
2. Generates Morgan fingerprints with radius 2 and 2,048 bits.
3. Creates a 70/30 row split with `random_state=42`.
4. Fits `XGBRegressor` with 250 trees, depth 6, learning rate 0.05, `subsample=0.9`, and `colsample_bytree=0.9`.
5. Writes metrics, exact row indices, package versions, and the input SHA-256 digest.

With the pinned environment and audited file, the reference run produced `Test R² = 0.851233` and `Test MAE = 30.350916 °C`. Exact values are printed and stored rather than hard-coded as validator requirements because numerical libraries and hardware can introduce small differences.

## 4. Validate the outputs

```bash
python reproducibility/validate_outputs.py \
  --output-dir reproduction_output
```

The validator recomputes all six train/test metrics from the prediction files and checks them against `leaderboard_record.csv`. It also verifies the required schemas and train/test row counts.

The output directory contains:

| File | Purpose |
|------|---------|
| `Morgan_2048_XGBoost_Glass transition temperature_train_predictions.csv` | Platform prediction schema: `true_values,predicted_values` |
| `Morgan_2048_XGBoost_Glass transition temperature_test_predictions.csv` | Platform prediction schema: `true_values,predicted_values` |
| `leaderboard_record.csv` | `Dataset`, `Descriptor`, `Algorithm`, and train/test R², MSE, and MAE |
| `run_metadata.json` | Input digest, row indices, versions, settings, and output filenames |

## 5. Contribute the result

The generated test-prediction CSV can be submitted through the PolyBench [upload page](http://polybench.ciac.jl.cn/upload) as type `Predictions`. Uploads enter a review queue; generating a compatible file does not automatically publish or add a score to the live leaderboard. Include the generated `leaderboard_record.csv` and `run_metadata.json` as supporting material or provide a repository/archive link.

## Checklist

- [ ] Record the Figshare version and DOI.
- [ ] Preserve the raw input SHA-256 digest from `run_metadata.json`.
- [ ] Preserve the generated train/test row indices.
- [ ] Report the representation, model settings, random seed, and package versions.
- [ ] Run `validate_outputs.py` before submission.
- [ ] State that the evaluation uses an open 70/30 test split, not a hidden test.

## Other scripts

The root-level training scripts support additional model families and pre-generated feature directories. Their historical stored outputs used script-specific data layouts and are not all reconstructed by this minimal example. See `HARDWARE.md` for the execution environment and reproducibility limitations.
