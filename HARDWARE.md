# Hardware

All experiments were conducted on a local server with the following configuration:

| Component | Specification |
|-----------|---------------|
| CPU | Intel Xeon Gold 6130 × 2 (32 cores / 64 threads, 2.10 GHz) |
| RAM | 256 GB DDR4 |
| GPU | NVIDIA GeForce RTX 4090 × 2 (24 GB VRAM each) |
| GPU Driver | 535.216.01 |
| OS | Ubuntu (Linux 5.19) |

## Notes

- Classical ML models (Linear, Ridge, ElasticNet, KNN, SVM, Decision Tree, Random Forest, Extra Trees, GBDT, HistGBDT, XGBoost, LightGBM, CatBoost) run on CPU by default; CatBoost also supports GPU acceleration via `--gpu` flag.
- Deep learning models (MLP, GCN, RNN, DMPNN) use GPU automatically when available; CPU fallback is supported.
