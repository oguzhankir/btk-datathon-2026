# Notebooks

`final_solution.ipynb` — the end-to-end writeup notebook. It loads the artifacts produced by the
pipeline (`src/` + `scripts/`) and walks through: data, key EDA findings, the CV protocol, the full
results table, the Ridge-stack blend (reproduced from saved OOFs), the signal-floor analysis, and the
final submission. Run it from the repo root or from `notebooks/` (the setup cell handles both paths).

Pipeline code lives in `src/` and `scripts/`; committed EDA/post-EDA scripts live in `reports/eda/`
(`deep_eda.py`, `residual_analysis.py`, `floor_analysis.py`). Notebooks are for the final writeup only.
