# Reproduce

## Environment Setup

```bash
conda env create -f environment.yml
conda activate PolyBench
```

## Data Layout

Place your data under the working directory before running any script:

```
<working_dir>/
├── train_data/
│   └── <feature_method>/   # e.g. Fingers_/RDKit_1024
│       └── *.csv
└── test_data/
    └── <feature_method>/
        └── *.csv
```

## Running Models

All scripts are run from the repository root. Examples:

**Classical ML (CPU)**
```bash
python GBDT.py
python XGBoost.py
python LightGBM.py
python CatBoost.py          # interactive GPU selection
python CatBoost.py --gpu 0  # specify GPU
python CatBoost.py --cpu    # force CPU
```

**Deep Learning (GPU auto-detected)**
```bash
python MLP.py
python GCN.py
python RNN.py
python dmpnn_predictor.py
```

**Other ML models**
```bash
python Linear.py
python Ridge.py
python ElasticNet.py
python KNN.py
python SVM.py
python DecisionTree.py
python RF.py
python ExtraTrees.py
python HistGBDT.py
```

## Clustering

```bash
python Cluster_with_save.py
```

## Output

Each script writes results (metrics, plots, saved models) to its own output directory under the working directory.
