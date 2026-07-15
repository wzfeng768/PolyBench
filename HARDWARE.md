# Hardware and Software Environment

## Training hardware

The reported PolyBench benchmark runs were produced on a server with the following configuration:

| Component | Specification |
|-----------|---------------|
| CPU | 2 x Intel Xeon Gold 6130 at 2.10 GHz (32 physical cores, 64 threads) |
| RAM | 251 GB available DDR4 (256 GB installed) |
| GPU | 2 x NVIDIA GeForce RTX 4090 (24 GB VRAM each) |
| GPU Driver | 535.216.01 |
| OS | Ubuntu 22.04 LTS (Linux kernel 5.19) |

Classical machine-learning scripts run on CPU by default. Deep-learning scripts use a GPU when available and support CPU fallback; CatBoost can be selected explicitly for CPU or GPU execution.

## Key software versions

The complete dependency set is pinned in `environment.yml`. Versions directly used by the reproducibility example and principal model scripts include:

| Package | Version |
|---------|---------|
| Python | 3.11.13 |
| NumPy | 1.26.4 |
| pandas | 2.3.1 |
| scikit-learn | 1.7.1 |
| RDKit | 2025.03.5 |
| XGBoost | 3.0.5 |
| CatBoost | 1.2.8 |
| PyTorch | 2.5.1+cu121 |
| torch-geometric | 2.6.1 |

Mordred v1.2.0 was used to generate the archived physicochemical descriptor matrices discussed in the manuscript, but it is not required by the public Morgan-fingerprint reproduction example and is not part of the current Python 3.11 environment file.

## Stochasticity

Random partitions and stochastic estimators are seeded in the relevant scripts. The public example uses random seed 42 by default and stores the seed, input SHA-256 digest, package versions, row indices, and model settings in `run_metadata.json`. Small numerical differences can still occur across operating systems, CPU instruction sets, thread schedules, and GPU/CUDA libraries.
