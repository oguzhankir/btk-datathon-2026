# BTK Datathon 2026 — Career Success Score Prediction

Predict `career_success_score` (continuous, 0–100) for 10,000 students from
40 numeric features, 5 categoricals, and a Turkish free-text mentor feedback field.
**Metric: MSE.** This repo is a full, reproducible experiment pipeline:
fixed 5-fold CV, leakage-safe OOF feature stacking, per-experiment artifacts and an
append-only results log.

## How to run

Everything goes through `./run.sh`; every knob (model, features, embeddings,
HPO trials, sample weighting, seeds) lives in the experiment's `configs/<exp_id>.yaml`.

```bash
./run.sh setup          # CPU deps    (setup-gpu adds torch/sentence-transformers/transformers)
./run.sh exp005         # run ONE experiment — no need for run_all
./run.sh exp010 --hpo-trials 200   # CLI overrides the config's hpo.trials
./run.sh all [--force]  # every config, skipping exp_ids already in results.csv
./run.sh blend          # OOF blend + ridge stacker (or: ./run.sh blend exp005 exp007)
./run.sh submit exp005  # write submissions/sub_exp005.csv + sanity report
./run.sh eda            # regenerate reports/figures/
./run.sh adv            # adversarial validation
```

Per-config knobs: `model`, `model_params` (inline overrides), `model_params_file`
(e.g. Optuna output), `features:` (tabular_fe / text_classic / ridge_meta /
target_encoding / embeddings {model, svd, knn, raw} / extra_oof_features),
`sample_weighting`, `n_seeds`, `hpo: {trials, timeout}`, and for exp009 `bert: {...}`.

## Key findings

1. **The text verbalizes the score.** The Turkish `mentor_feedback_text` is LLM-generated
   and encodes the target: "mükemmel" → mean 91.8, "sınırlı" → 67.0. OOF TF-IDF
   ridge/logistic meta-features are the single biggest win (−5.5 CV MSE).
2. **Ceiling effect**: 7.7% of train has y=100 exactly; a P(y==100) classifier reaches
   AUC ~0.93. Two-stage blending helps (exp007), modestly once text meta is present.
3. **Covariate shift is temporal only**: adversarial AUC 0.65 with year features,
   0.52 without (verified: `scripts/adversarial_validation.py`). Test skews to 2024–2026,
   where target means are lower — `rmse_year_2024plus` is tracked for every run as the LB proxy.
   Naive year-reweighting of samples does NOT help (exp008).
4. **Missingness is signal**: NA rows score lower; per-column NA flags + row NA count
   are features, no imputation for GBMs.
5. **Top interaction** (SHAP): `application_year × project_quality_score` — the value of a
   strong project portfolio shifts across years.

## Results (CV, fixed 5-fold, seed 42 — see results.csv / EXPERIMENTS.md)

| exp | description | CV MSE | LB MSE |
|---|---|---|---|
| exp001 | LGBM baseline, no FE | 85.05 | 93.28 |
| exp003 | + tabular FE + text features & TF-IDF meta | 78.51 | — |
| exp005 | CatBoost on the same features | 76.25 | 86.04 |
| exp009 | BERT Turkish fine-tune (fixed: y/100 + warmup + best-epoch) | 135.07 | — |
| exp012 | XLM-RoBERTa-large fine-tune | 131.16 | — |
| exp013 | kitchen-sink v2: FE + BERT/XLM-R OOFs, Optuna-tuned LGBM | **75.14** | — |
| exp019 | CatBoost on kitchen-sink v3 (4 text OOFs) | 75.28 | — |
| **blend (best)** | ridge stacker over 21 OOFs | **73.61** | **82.96** |

CV↔LB offset ≈ +9.3–9.8 with ranking fully preserved (full mapping in EXPERIMENTS.md).
Rejected on evidence: year-reweighting, embedding SVD/kNN features, multi-seed text
averaging, pseudo-labeling, isotonic/snap-to-100 post-processing. The day-by-day
narrative (decisions, failures, LB audit trail) lives in `docs/progress/`.

### Signal-floor analysis (`reports/eda/floor_analysis.py`)

The modelling frontier is closed — proven by 7 independent diagnostics, all reproducible
from saved artifacts (no fitting, leakage-free). Run `python reports/eda/floor_analysis.py`:

1. **The CV→LB offset is 100% the test set's late-year skew**, not overfitting: re-weighting
   the OOF MSE by the test's year mix gives 83.4 ≈ the actual LB. Per-year bias ≈ 0.
2. **Within-year R² is flat at ~0.69** (incl. 2025-26): late years are harder only because
   their target *variance* is larger → no hidden late-year signal we fail to model.
3. **Ceiling & dispersion oracles** recover ~0 / hurt → neither is a lever.
4. **Blend members are 0.95 correlated** → diversity exhausted (blend beats best single by ~1.5 MSE).
5. **Text describes the profile, not the score** (highest-error rows: glowing text ↔ low score)
   → `target = g(features) + irreducible noise`; text is a noisier view of the same features.
6. **No forgotten signal**: every raw column is used; text-stated numbers vs the columns have
   −0.01 correlation with the residual.
7. **The gap to the public #1 (≈2.25 MSE) is inside the public-subset noise band** (±2-3 MSE):
   the live ranking is largely chance → don't overfit the public LB; submit a robust CV-backed blend.

Figures: `reports/figures/{year_error_decomposition,r2_by_year,blend_member_corr,public_lb_noise}.png`.

### Blend variants (`scripts/blend.py --method {auto,ridge,weights,equal} --tag <name>`)

Five robustness-profile variants, all at the floor (CV 73.6–74.0) but from different member
sets/methods, used to confirm the public-LB noise and pick the private bet: `blend_full_ridge`
(73.61, main), `blend_strong_ridge` (73.78), `blend_strong_wts` (73.88), `blend_core_ridge`
(73.85), `blend_core_equal` (74.02).

## Figure gallery (`reports/figures/`, generated by `reports/eda/deep_eda.py`)

| | |
|---|---|
| ![target](reports/figures/target_distribution.png) Left-skewed target with a hard ceiling at 100 (7.7%). | ![drift](reports/figures/role_year_drift.png) The drop in late years hits every role → temporal drift, not role mix. |
| ![ridge](reports/figures/text_ridge_coefficients.png) TF-IDF terms by Ridge coefficient: the feedback text literally spells out the score. | ![ceiling](reports/figures/ceiling_feature_profile.png) What makes a y=100 row: interview/project/portfolio scores, not GPA. |
| ![overlay](reports/figures/train_test_overlays.png) Train-vs-test overlays: only year features shift. | ![residuals](reports/figures/baseline_residuals.png) Baseline residuals concentrate at the ceiling and late years. |
| ![shap](reports/figures/shap_interaction_heatmaps.png) Top-5 SHAP interaction pairs. | ![keywords](reports/figures/ceiling_text_keywords.png) Keyword prevalence: ceiling rows vs rest. |

## Repo layout

```
data/raw/            competition CSVs        data/processed/   folds.csv, caches
src/                 pipeline (data, cv, models, features/, hpo, two_stage)
configs/             one YAML per experiment (exp001–exp011)
scripts/             run_experiment, run_all, make_submission, blend, adversarial_validation
artifacts/           oof_/test_ predictions, params, importances per experiment
reports/eda|figures  committed EDA code + PNGs       submissions/  sub_*.csv
results.csv          append-only experiment log      EXPERIMENTS.md  curated log
```

## Protocol (non-negotiable)

- One fixed fold file (`data/processed/folds.csv`), stratified on target deciles, seed 42 — never re-split.
- Every fitted transform (TF-IDF, target encoding, kNN-target, scalers) fits inside the training fold only.
- All predictions clipped to [0, 100]; MSE and RMSE logged, plus year-sliced and non-ceiling RMSE.
- Every run appends to `results.csv` and persists `artifacts/{oof,test}_{exp}.npy` for stacking.
