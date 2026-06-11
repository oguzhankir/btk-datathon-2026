# BTK Datathon 2026 — Repo Build Instructions

You are setting up and implementing a complete, competition-grade ML pipeline in this repo
(`oguzhankir/btk-datathon-2026`). Work autonomously, commit in small logical steps.
The user will run experiments locally (possibly on a rented GPU) and submit to Kaggle.

## 1. Competition context

- Task: predict `career_success_score` (continuous, 0–100) for 10,000 test students.
- **Metric: MSE** (optimize squared error; always report both MSE and RMSE in logs).
- Files already in repo root: `train.csv` (10,000 × 47), `test_x.csv` (10,000 × 46),
  `sample_submission.csv` (columns: `student_id`, `career_success_score`). Move them to `data/raw/`.
- Deadline: Sunday. 3 submissions/day. Today's goal: 1 baseline + 2 advanced submission candidates.
- Mixed feature types: 40 numeric, 5 categorical (`department`, `university_tier`, `target_role`,
  `hobby`, `preferred_social_media_platform`), 1 Turkish free-text field (`mentor_feedback_text`),
  1 id (`student_id`).

## 2. Established EDA findings (verified — build on these, do NOT rediscover from scratch)

These were verified in a prior analysis session. Treat as facts; your job is to go deeper (§6).

1. **Ceiling effect**: 773 train rows (7.7%) have target exactly 100; only 1 row at 0.
   Distribution left-skewed (mean 76.9, std 15.2). Share of 100s grows by year: 3.7% (2019) → 13% (2025).
   A LightGBM classifier for `P(y==100)` reaches **AUC 0.929** on numerics alone; text-only
   logistic on TF-IDF reaches AUC 0.904. → Two-stage modeling is mandatory to try (§7, exp007).
2. **Covariate shift is ONLY temporal**: adversarial validation train-vs-test AUC = 0.65 with year
   features, **0.49 without them** (i.e. indistinguishable). Test is skewed to late years:
   2024–2026 = 62% of test vs ~33% of train. Target mean drops in late years (77.9 → 74.2).
   → Keep year features; try year-reweighting of training samples (exp008); report year-sliced CV.
3. **Missingness is informative, not random**: 7 columns have NAs (`english_exam_score`,
   `internship_duration_months`, `portfolio_score`, `github_avg_stars`,
   `open_source_contribution_count`, `linkedin_profile_score`, `hr_interview_score`).
   NA rows have lower target (e.g. linkedin NA: 73.9 vs 77.2). `github_avg_stars` and
   `open_source_contribution_count` are always NA together (a "data unavailable" pattern —
   repo_count can still be > 0). → Add per-column NA flags + row NA count; do NOT impute for GBMs.
4. **Feature structure**: `project_quality_score` is the strongest single feature (ρ≈0.56),
   then `technical_interview_score` (0.34). The 9 technical skill scores form a correlated block
   (~0.55 pairwise). `cgpa`, `attendance_rate`, `english_exam_score`, `age` ≈ zero marginal
   correlation (revive via interactions only). `hobby` and `preferred_social_media_platform`
   are near-pure noise (likely intentional distractors).
5. **Role–skill alignment exists**: per-role, the most predictive skill differs
   (Cloud Engineer→cloud_score ρ=0.39; MLOps→problem_solving 0.44; Backend→devops/cloud 0.35;
   Data Scientist→problem_solving/ML 0.33). Role mean targets range 74.0 (Cybersecurity) to 80.4 (Cloud).
6. **Text is high-signal, LLM-generated Turkish** (all 10,000 unique, ~33 words each, no NAs).
   It verbalizes the score: keyword "mükemmel" → mean target 91.8 (vs 76.2 without);
   "olağanüstü" → 89.7; "eksik" → 69.9; "sınırlı" → 67.0; contrastive "ancak" appears in 58% of
   texts → 73.9 vs 81.3. Text-only Ridge on full word(1-3)+char_wb(3-5) TF-IDF: **RMSE 12.26**
   (vs 15.19 for predict-mean; vs 12.82 for SVD128+LGBM — full-vocab linear beats SVD).
7. Reference baselines (5-fold, seed 42, LGBM 2000 trees lr=0.05 + early stopping, preds clipped):
   numeric+cats **RMSE 9.18**, + TF-IDF SVD128 **RMSE 9.13**. Anything you build must beat these.
8. No duplicate rows/ids, no train–test id overlap. `pandas>=3.0` note: string columns load as
   dtype `str`, not `object` — use `df[c].dtype.kind not in 'ifu'` style checks.

## 3. Repo structure (create exactly this)

```
├── data/
│   ├── raw/                  # train.csv, test_x.csv, sample_submission.csv (move here)
│   └── processed/            # folds.csv, cached feature matrices (parquet), text embeddings (.npy)
├── src/
│   ├── data.py               # load_raw(), make_folds(), load_folds()
│   ├── features/
│   │   ├── tabular.py        # all tabular FE (§7.1), returns df + feature list
│   │   ├── text_classic.py   # keyword/lexicon + stats + TF-IDF builders
│   │   ├── text_embed.py     # sentence-transformer embeddings w/ disk cache
│   │   └── meta.py           # OOF meta-features (ridge-on-tfidf, knn-on-embeddings) — fold-safe
│   ├── cv.py                 # run_cv(model, X, y, folds) → oof, test_pred, per-fold + year-sliced metrics
│   ├── models.py             # registry: lgbm, xgb, catboost, ridge, mlp (device-aware)
│   ├── hpo.py                # GenericOptunaSearch class (§8)
│   ├── two_stage.py          # ceiling classifier + regressor blend (§7.3)
│   └── utils.py              # seed_everything, timer, device detection, logging helpers
├── configs/                  # one YAML per experiment (exp001.yaml, ...)
├── scripts/
│   ├── run_experiment.py     # python scripts/run_experiment.py -c configs/exp004.yaml
│   ├── run_all.py            # runs every config sequentially, skips already-completed exp_ids
│   ├── make_submission.py    # -e exp_id → submissions/sub_{exp_id}.csv + sanity report
│   ├── blend.py              # OOF-weight optimization over selected exp_ids → blended submission
│   └── adversarial_validation.py
├── artifacts/                # oof_{exp}.npy, test_{exp}.npy, params_{exp}.json, importance_{exp}.csv
├── reports/
│   ├── eda/                  # committed EDA scripts/notebooks
│   └── figures/              # committed PNGs (referenced from README)
├── notebooks/                # final_solution.ipynb assembled later (leave placeholder README)
├── submissions/
├── results.csv               # append-only experiment log (§5)
├── EXPERIMENTS.md            # human-readable: one entry per exp (hypothesis → result → decision)
├── requirements.txt          # pinned versions
└── README.md                 # project overview, findings summary, how-to-run, figure gallery
```

## 4. Non-negotiable protocol (every experiment)

1. **Fixed folds**: generate ONCE via `make_folds()` → `data/processed/folds.csv`
   (`student_id`, `fold`). Use `StratifiedKFold(5, shuffle=True, random_state=42)` stratified on
   `pd.qcut(y, 10, duplicates='drop')` bins (stabilizes folds given the ceiling mass).
   Every experiment loads this file. Never re-split. Stacking validity depends on this.
2. **Determinism**: `seed_everything(42)` everywhere; fixed seeds in model params.
3. **No leakage**: every fitted transform (TF-IDF, target encoding, OOF meta-features,
   kNN-target features, scalers) is fit on the training part of each fold only.
4. **Clip** all predictions to [0, 100] before scoring and before writing submissions.
5. **Persist artifacts** per experiment: `artifacts/oof_{exp}.npy`, `artifacts/test_{exp}.npy`
   (test pred = mean over 5 fold models), `params_{exp}.json`, `importance_{exp}.csv` (if tree model).
6. **Log** every run by appending one row to `results.csv` AND a short entry to `EXPERIMENTS.md`.
7. **Device-aware**: `utils.detect_device()` → use GPU for xgboost (`device='cuda'`),
   catboost (`task_type='GPU'`), torch models when available; silently fall back to CPU.
   LightGBM stays CPU (fast enough at 10k rows). Everything must run CPU-only without code changes.

## 5. results.csv schema

`exp_id, timestamp, description, n_features, cv_mse, cv_rmse, cv_rmse_std, fold_rmses (json),
rmse_year_2024plus, rmse_y_lt_100, runtime_s, device, config_path, notes`

`rmse_year_2024plus` = OOF RMSE restricted to rows with application_year ≥ 2024 (proxy for LB).
`rmse_y_lt_100` = OOF RMSE on non-ceiling rows (diagnoses where errors come from).

## 6. Deeper EDA (commit code + figures to reports/)

Extend the prior analysis. Produce polished figures (consistent seaborn theme, titled, captioned in
README). Required investigations beyond what's established:

- Interaction discovery: SHAP interaction values (or H-statistic) on a strong LGBM; visualize top 5
  interaction pairs as 2D partial dependence heatmaps.
- Per-role analysis: target distribution per role per year (is role mix driving the temporal drift?).
- Ceiling-row profiling: what distinguishes y==100 rows (feature means, text keyword prevalence)?
- Text deep-dive: top TF-IDF terms by Ridge coefficient (most positive / most negative — this is a
  jury-friendly figure); sentence count stats; does the text explicitly mention the student's
  target_role and does role-mention consistency matter?
- Residual analysis of baseline model: residual vs prediction, vs year, vs role → where is the model
  weakest? (Feeds tomorrow's iteration.)
- Train-vs-test overlay distributions for top-15 features (confirm fact #2 visually).

## 7. Experiments to implement (one config each — all runnable tonight)

### 7.1 Tabular feature engineering (features/tabular.py — used by exp002+)

- NA flags per column + `na_count` per row.
- Role–skill alignment: a `ROLE_SKILL_MAP` dict (role → ordered list of relevant skill cols, write it
  from domain sense, e.g. Cloud Engineer → [cloud, devops, backend]); features: `matched_skill`
  (mean of mapped skills), `matched_skill_rank` (rank of matched skill among the student's 9 skills),
  `matched_minus_mean`.
- Skill-block aggregates: mean/max/min/std of the 9 skill scores, top2-mean, (max−mean) specialization gap.
- Ratios & combos: `interview_conversion = interviews_attended / (applications_sent+1)`,
  `award_rate = hackathon_awards / (hackathon_count+1)`, `intern_months_per = internship_duration_months
  / (internship_count+1)`, `stars_total = github_repo_count * github_avg_stars`,
  `oss_per_repo`, `experience_total = real_client + freelance + internship counts`.
- Time: `years_since_grad = application_year − graduation_year` (negative = applied while student),
  `grad_age = age − (application_year − graduation_year)`.
- Pairwise products/ratios among the top block: project_quality × technical_interview,
  project_quality × portfolio, etc. (keep it modest, ~10 hand-picked; trees find the rest).
- OOF target encoding for the 5 cats and for (target_role × university_tier) cross
  (smoothed, fit within folds via meta.py utilities).

### 7.2 Text features

- `text_classic.py`: char/word/sentence counts; Turkish lexicon flags & counts — positives
  (mükemmel, olağanüstü, etkileyici, güçlü, başarılı, donanımlı, dikkat çekici), negatives
  (eksik, sınırlı, zorluk, engelliyor, yetersiz, geliştirmesi gereken), contrast markers
  (ancak, fakat, ama, ne yazık ki) + position of first contrast marker (early "ancak" ≈ worse);
  pos_count − neg_count; whether text mentions the student's own target_role keywords.
- `meta.py` → `ridge_tfidf_oof()`: word(1-3) + char_wb(3-5) TF-IDF (sublinear, min_df=2) → Ridge,
  fit per fold → returns OOF vector + test prediction. This becomes a single dense meta-feature
  for GBMs (established text-only strength: RMSE 12.26). Also produce a LogisticRegression variant
  for `P(y==100)` as a second meta-feature.
- `text_embed.py`: sentence-transformers, models configurable:
  `intfloat/multilingual-e5-large` (prefix "query: "), `BAAI/bge-m3`,
  `paraphrase-multilingual-mpnet-base-v2`. Cache to
  `data/processed/emb_{model_slug}.npy` (train+test stacked) so extraction runs once.
  Usage modes: (a) SVD-64 of embeddings as features, (b) `knn_target_feature()` in meta.py —
  mean/std target of k∈{5,25} nearest train neighbors by cosine, computed fold-safely.
- **exp009 (GPU)**: fine-tune `dbmdz/bert-base-turkish-cased` (regression head, MSE loss, 3 epochs,
  lr 2e-5, max_len 128) per fold → OOF + test predictions saved as artifacts like any experiment.
  Since the text verbalizes the score, this is plausibly the single strongest component. Also save
  its predictions for use as a meta-feature in exp010. Runtime ~minutes/fold on GPU; skip gracefully
  on CPU with a clear message.

### 7.3 Model experiments (configs)

- **exp001 baseline**: LGBM, raw numerics + native cats, NO FE. (= submission anchor, CV ref 9.18.)
- **exp002**: exp001 + §7.1 tabular FE.
- **exp003**: exp002 + classic text features + ridge/logistic TF-IDF meta-features.
- **exp004**: exp003 + embedding SVD + kNN-target features (best embedding model).
- **exp005**: CatBoost on exp004 feature set (native categorical handling, GPU-capable).
- **exp006**: XGBoost on exp004 feature set.
- **exp007 two-stage**: `two_stage.py` — classifier P(y==100) (LGBM on exp004 features incl. text
  meta) + regressor (trained on all rows); final = `p*100 + (1−p)*reg`; ALSO try regressor trained
  on y<100 rows only; pick by OOF MSE. Tune the blend exponent/threshold on OOF.
- **exp008**: exp004 + sample weights = test/train year-frequency ratio (importance weighting).
  Verify via `rmse_year_2024plus` whether it helps where it matters.
- **exp009**: BERT fine-tune (see §7.2).
- **exp010**: kitchen sink — exp004 features + exp009 OOF as feature, Optuna-tuned LGBM (§8).
- **exp011 (GPU/CPU)**: MLP (torch): tabular (standardized, NA-imputed w/ flags) + raw 1024-d
  embeddings; 2–3 hidden layers, dropout, AdamW, early stopping on fold val. Diversity for the blend.
- **blend**: `scripts/blend.py` — load chosen OOFs, optimize non-negative weights on OOF MSE
  (scipy SLSQP + hill-climbing from equal weights), print per-model weight & blended CV, write
  blended test submission. Also implement a Ridge stacker variant (fit on OOF matrix, fold-safe).

## 8. Generic HPO (src/hpo.py)

`GenericOptunaSearch(model_name, feature_set, n_trials, timeout, device)`:
- Search spaces defined per model in one dict (lgbm: num_leaves 31–255, lr log 0.01–0.1, min_child,
  feature/bagging fractions, lambda_l1/l2, max_bin; xgb & catboost equivalents; mlp: width/depth/
  dropout/lr/wd).
- Objective = mean CV MSE using THE fixed folds; MedianPruner with per-fold reporting
  (prune after fold 2 if clearly bad).
- Saves study to `artifacts/optuna_{name}.db` (sqlite, resumable), best params to
  `artifacts/params_{name}.json`; run_experiment.py can reference a params file in its config.
- CLI: `python scripts/run_experiment.py -c configs/exp010.yaml --hpo-trials 200`.

## 9. Config format (YAML) — example

```yaml
exp_id: exp004
description: "FE + classic text + ridge meta + e5 embeddings (SVD64 + knn)"
model: lgbm
model_params_file: null          # or artifacts/params_lgbm.json
features:
  tabular_fe: true
  text_classic: true
  ridge_meta: true
  embeddings: { model: intfloat/multilingual-e5-large, svd: 64, knn: [5, 25] }
target_transform: none           # none | as-is; clip always applied
sample_weighting: none           # none | year_ratio
n_seeds: 1                       # exp010 final: 3 (seed-average fold models)
```

## 10. Submission flow & today's plan

`make_submission.py -e <exp_id|blend>`: reads `artifacts/test_{exp}.npy`, clips to [0,100], writes
`submissions/sub_{exp_id}.csv` matching sample_submission format exactly; prints sanity stats
(mean/std/min/max, % at bounds) next to train-target stats and the experiment's CV.

Today's three submissions (in order):
1. `sub_exp001` — baseline anchor → calibrates CV↔LB.
2. Best single advanced model by `cv_mse` (likely exp007 or exp010).
3. `sub_blend` over all available OOFs.

After LB results: compare LB MSE vs `cv_mse` AND vs `rmse_year_2024plus`² to learn which local
metric tracks LB. Record the mapping in EXPERIMENTS.md — it drives all remaining decisions.

## 11. Quality bar

- `requirements.txt` pinned (pandas>=3 quirk from §2.8 handled), `python -m` style imports, type
  hints, docstrings on public functions, no notebooks for pipeline code (notebooks only for EDA/final).
- README: problem summary, key EDA figures with one-line insights, how to reproduce
  (3 commands: install, run_all, blend), results table auto-generatable from results.csv.
- Small commits with clear messages as you build (structure → data/cv → features → experiments → docs).
- Time-box: prefer getting exp001–exp008 + blend runnable over polishing; exp009/011 are GPU bonuses.
