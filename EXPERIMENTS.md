# Experiment Log

One entry per experiment: hypothesis → result → decision. Metric rows are
auto-appended by `scripts/run_experiment.py`; the hypothesis/decision text is curated.

Reference baselines (prior session, 5-fold seed 42 LGBM): numeric+cats RMSE 9.18,
+ TF-IDF SVD128 RMSE 9.13. Everything below must beat these.

> **Environment note (2026-06-11):** the build environment could not reach
> huggingface.co, so embedding features (e5-large SVD/kNN) were skipped for
> exp005–exp008 and exp011 (logged in results.csv `notes`), and exp004/exp009/exp010
> were not run. On a machine with HF access + GPU: re-run with
> `python scripts/run_all.py --force` (or delete the affected rows from results.csv),
> then re-blend. CVs below are therefore lower bounds for those configs.

## CV ↔ LB mapping (filled 2026-06-11, first 3 submissions)

| sub | cv_mse | rmse_2024+² | LB MSE | LB − cv_mse |
|---|---|---|---|---|
| exp001 | 85.05 | 105.46 | 93.28 | +8.2 |
| exp005 | 76.25 | 95.10 | 86.04 | +9.8 |
| blend | 74.93 | 93.27 | 84.74 | +9.8 |

- **Ranking fully preserved; deltas transfer ~1:1** (blend−exp005: −1.33 CV vs −1.30 LB).
- `cv_mse` + ~9–10 offset is the best LB predictor; `rmse_year_2024plus²` systematically
  OVER-estimates LB (predicted 93.3 for blend, actual 84.7) — use it only as a drift check.
- **Decision: trust cv_mse for all remaining decisions.** The constant offset comes from the
  test set's late-year skew and doesn't affect choices.

### exp001 — Baseline LGBM: raw numerics + native cats, no FE (submission anchor)
- CV MSE **85.0537** | RMSE 9.2225 (±0.2140) | 2024+ RMSE 10.2696 | y<100 RMSE 9.3013 | 44 features
- Hypothesis: reproduce the prior-session 9.18 reference on the fixed folds.
- Result: 9.22 — matches within fold-scheme noise. Decision: anchor submission #1; calibrates CV↔LB.

### exp002 — exp001 + tabular FE (NA flags, role-skill alignment, ratios, time, target encoding)
- CV MSE **83.9734** | RMSE 9.1637 (±0.1501) | 2024+ RMSE 10.1882 | y<100 RMSE 9.2600 | 85 features
- Hypothesis: NA-informativeness + role-skill alignment + ratios add signal trees can't synthesize alone.
- Result: −1.08 MSE vs exp001. Modest but real. Decision: keep all FE for downstream configs.

### exp003 — exp002 + classic text features + ridge/logistic TF-IDF meta-features
- CV MSE **78.5095** | RMSE 8.8606 (±0.1151) | 2024+ RMSE 9.8387 | y<100 RMSE 9.0439 | 109 features
- Hypothesis: the text verbalizes the score (fact #6) → OOF ridge/logit meta-features are the biggest single win.
- Result: −5.46 MSE vs exp002, by far the largest jump. Decision: text meta is mandatory in every config.

### exp005 — CatBoost on exp004 feature set (native cats, GPU-capable)
- CV MSE **76.2535** | RMSE 8.7323 (±0.1469) | 2024+ RMSE 9.7521 | y<100 RMSE 8.9009 | 109 features
- Hypothesis: ordered target statistics + native cats give CatBoost an edge at 10k rows.
- Result: best single model so far (−2.26 MSE vs LGBM on same features). Decision: submission #2; tune with Optuna next.
- Ran without embedding features (HF unreachable — see note above).

### exp006 — XGBoost on exp004 feature set
- CV MSE **78.9979** | RMSE 8.8881 (±0.1354) | 2024+ RMSE 9.8907 | y<100 RMSE 9.0634 | 109 features
- Result: between LGBM and CatBoost; keeps the blend diverse. Decision: keep as blend member only.
- Ran without embedding features (HF unreachable).

### exp007 — Two-stage: P(y==100) classifier blended with regressor (all-rows vs y<100), tuned on OOF
- CV MSE **78.2534** | RMSE 8.8461 (±0.1226) | 2024+ RMSE 9.8105 | y<100 RMSE 9.0593 | 109 features
- Hypothesis: explicit ceiling handling (AUC ~0.93 classifier) beats a single regressor.
- Result: best combo = all-rows regressor, gamma=1.0 → −0.26 MSE vs its own plain regressor (78.51).
  Helps, but less than hoped — the regressor already nearly saturates the ceiling signal via text meta.
  Decision: keep as blend member; revisit with embeddings + BERT meta on GPU.
- Ran without embedding features (HF unreachable).

### exp008 — exp004 + year-ratio sample weights (importance weighting toward test year mix)
- CV MSE **79.3281** | RMSE 8.9066 (±0.1331) | 2024+ RMSE 9.8553 | y<100 RMSE 9.0992 | 109 features
- Hypothesis: weighting train toward the test year mix improves the 2024+ slice (LB proxy).
- Result: WORSE overall (+0.82 vs exp003) and 2024+ RMSE not better (9.855 vs 9.839).
  Decision: year-reweighting rejected for now; the year feature itself already carries the drift.

### exp011 — Torch MLP on tabular (standardized, imputed + NA flags) + raw embeddings (blend diversity)
- CV MSE **87.5505** | RMSE 9.3568 (±0.1926) | 2024+ RMSE 10.4961 | y<100 RMSE 9.4843 | 109 features
- Result: weak alone (no raw embeddings available here), but decorrelated → got 13% weight in the blend.
  Decision: keep; re-run with 1024-d e5 embeddings on GPU.

### blend — ridge stacker over exp001/002/003/005/006/007/008/011 OOFs
- CV MSE **74.9273** | RMSE 8.6561 | 2024+ RMSE 9.6574
- Weight-blend gave 75.07; fold-safe positive-Ridge stacker won (74.93). Decision: submission #3.

## Pending (need GPU + HF access)
- exp004 (embeddings SVD64 + kNN target), exp009 (BERT fine-tune — likely strongest single
  component since text verbalizes the score), exp010 (kitchen sink + Optuna `--hpo-trials 200`),
  re-runs of exp005–008/011 with embeddings, then re-blend.

