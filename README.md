# PolyBench

PolyBench is an open collection of polymer property-prediction tasks, stored open-test reference results, and model-training scripts based on repeat-unit representations.

## Public Resources

- [Web interface](http://polybench.ciac.jl.cn/)
- [Leaderboard](http://polybench.ciac.jl.cn/leaderboard)
- [Algorithm catalog](http://polybench.ciac.jl.cn/ai)
- [Version-2 task CSV archive](https://doi.org/10.6084/m9.figshare.30917717) (CC BY 4.0)

The website is a live HTTP service. The Figshare record is the versioned data archive. Current leaderboard values are open-test reference results, not hidden-test competition scores.

## Reproduce One Task

Create the pinned environment, download `Glass transition temperature.csv` from the Figshare record, and run:

```bash
conda env create -f environment.yml
conda activate PolyBench

python reproducibility/reproduce_tg.py \
  --input "/path/to/Glass transition temperature.csv" \
  --output-dir reproduction_output

python reproducibility/validate_outputs.py \
  --output-dir reproduction_output
```

The example starts from the raw task CSV, creates Morgan fingerprints, uses one documented 70/30 split, trains XGBoost, and writes train/test prediction files plus a leaderboard-compatible summary row. See [REPRODUCE.md](REPRODUCE.md) for the complete protocol and limitations.

## Repository Layout

- Root-level `*.py`: original model-training and clustering scripts.
- `reproducibility/`: self-contained raw-CSV-to-result example and output validator.
- `environment.yml`: pinned Conda/Python environment without a machine-specific prefix.
- `HARDWARE.md`: training hardware, key package versions, and stochasticity notes.
- `REPRODUCE.md`: step-by-step reproduction checklist and output schema.

## Scope

The public Figshare version contains 39 task CSVs. It does not currently contain the original split indices for every stored score, all generated feature matrices, fitted models, row-level source manifests, or curation logs. Consequently, the example above reproduces a documented benchmark-compatible workflow; it does not recreate every historical leaderboard value.

## Licences

Code in this repository is released under the [MIT License](LICENSE). The version-2 Figshare dataset is released under CC BY 4.0; cite and attribute the dataset record when using its files.
