# Experiment Log

One entry per experiment: hypothesis → result → decision. Metric rows are
auto-appended by `scripts/run_experiment.py`; the hypothesis/decision text is curated.

Reference baselines (prior session, 5-fold seed 42 LGBM): numeric+cats RMSE 9.18,
+ TF-IDF SVD128 RMSE 9.13. Everything below must beat these.

## CV ↔ LB mapping
(to be filled after first leaderboard results: compare LB MSE vs cv_mse and vs
rmse_year_2024plus² — whichever tracks better drives remaining decisions)

### exp001 — Baseline LGBM: raw numerics + native cats, no FE (submission anchor)
- CV MSE **85.0537** | RMSE 9.2225 (±0.2140) | 2024+ RMSE 10.2696 | y<100 RMSE 9.3013 | 44 features
