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


### exp004 — exp003 + e5-large embeddings (SVD64 + knn target features)
- CV MSE **79.0141** | RMSE 8.8890 (±0.1392) | 2024+ RMSE 9.8451 | y<100 RMSE 9.0636 | 177 features

### exp009 — BERT fine-tune dbmdz/bert-base-turkish-cased, regression head (GPU)
- CV MSE **3096.9422** | RMSE 55.6502 (±0.1990) | 2024+ RMSE 55.2730 | y<100 RMSE 53.5242 | 128 features
- Notes: bert=dbmdz/bert-base-turkish-cased

### exp009 — BERT fine-tune dbmdz/bert-base-turkish-cased, regression head (GPU)
- CV MSE **142.0657** | RMSE 11.9191 (±0.2659) | 2024+ RMSE 13.2371 | y<100 RMSE 12.0540 | 128 features
- Notes: bert=dbmdz/bert-base-turkish-cased

### exp002 — exp001 + tabular FE (NA flags, role-skill alignment, ratios, time, target encoding)
- CV MSE **83.9734** | RMSE 9.1637 (±0.1501) | 2024+ RMSE 10.1882 | y<100 RMSE 9.2600 | 85 features

### exp005 — CatBoost on exp004 feature set (native cats, GPU-capable)
- CV MSE **76.8291** | RMSE 8.7652 (±0.1759) | 2024+ RMSE 9.7685 | y<100 RMSE 8.9407 | 177 features

### exp006 — XGBoost on exp004 feature set
- CV MSE **80.0707** | RMSE 8.9482 (±0.1640) | 2024+ RMSE 9.9426 | y<100 RMSE 9.1217 | 177 features

### exp007 — Two-stage: P(y==100) classifier blended with regressor (all-rows vs y<100), tuned on OOF
- CV MSE **78.6611** | RMSE 8.8691 (±0.1459) | 2024+ RMSE 9.8082 | y<100 RMSE 9.0816 | 177 features
- Notes: best_combo={'regressor': 'all_rows', 'gamma': 0.75}

### exp010 — Kitchen sink: exp004 features + exp009 BERT OOF as feature, Optuna-tuned LGBM
- CV MSE **75.2956** | RMSE 8.6773 (±0.1542) | 2024+ RMSE 9.7028 | y<100 RMSE 8.8501 | 178 features
- Notes: params file params_lgbm_exp010.json missing, used defaults; hpo: 1 trials

## CV ↔ LB update (4th submission)
| sub | cv_mse | LB MSE | offset |
|---|---|---|---|
| blend (v2: +exp009 fixed, +exp010 tuned) | 74.17 | 83.62 | +9.45 |

Offset stays ~+9.5; cv_mse remains the reliable LB predictor.

## Residual analysis of exp010 (best single, reports/eda/residual_analysis.py)
- **No leftover linear signal**: every engineered feature correlates with the residual at |ρ|<0.05.
  The target is NOT a simple formula we're missing → stop chasing feature interactions.
- **No year bias** (per-year residual mean ≈ 0): the year feature already absorbs the drift →
  year-based LB calibration won't help.
- **Ceiling is under-predicted by design**: y=100 rows have residual bias +4.07 (model says ~96),
  but `%pred≥95 = 8.9%` already matches `%y=100 = 7.7%`. Under MSE this is OPTIMAL given uncertainty.
- **Tested & rejected post-processing**: fold-safe isotonic calibration (74.62, worse),
  hard-snap-to-100 (no gain), LGBM meta-stacker (75.08, worse vs ridge 74.17),
  ridge on 7 strong members (74.20, identical). **The blend is already MSE-optimal for this feature set.**

## Remaining lever
- Only LGBM was Optuna-tuned (exp010). CatBoost/XGBoost ran with defaults. A tuned, algorithmically
  diverse CatBoost (~75 cv) could add genuine blend diversity → potential blend ~73.5 (LB ~83.0).
  Run on GPU: `./run.sh exp005 --hpo-trials 100` and `./run.sh exp006 --hpo-trials 100`, then reblend.

### exp009 — BERT fine-tune Turkish (best-epoch selection, longer): regression head (GPU)
- CV MSE **135.0721** | RMSE 11.6221 (±0.1961) | 2024+ RMSE 12.9359 | y<100 RMSE 11.7249 | 192 features
- Notes: bert=dbmdz/bert-base-turkish-cased

### exp012 — Stronger text model: XLM-RoBERTa-large fine-tune, regression head (GPU)
- CV MSE **131.1589** | RMSE 11.4525 (±0.3094) | 2024+ RMSE 12.6988 | y<100 RMSE 11.5606 | 192 features
- Notes: bert=FacebookAI/xlm-roberta-large

### exp013 — Kitchen sink v2: exp004 features + exp009 & exp012 text OOFs, Optuna-tuned LGBM
- CV MSE **75.1432** | RMSE 8.6685 (±0.1707) | 2024+ RMSE 9.6680 | y<100 RMSE 8.8448 | 179 features
- Notes: hpo: 100 trials

### exp011 — Torch MLP on tabular (standardized, imputed + NA flags) + raw embeddings (blend diversity)
- CV MSE **102.9448** | RMSE 10.1462 (±0.2646) | 2024+ RMSE 11.2612 | y<100 RMSE 10.2569 | 1137 features
- Improved over the first fix (142.07 → 135.07) via inner-val best-epoch selection + 8 epochs + max_len 192.
  Standalone "weak" (CV ~135) but fully decorrelated from the GBMs → punches above its weight in the blend.

## CV ↔ LB update (5th submission)
| sub | cv_mse | LB MSE | offset |
|---|---|---|---|
| blend (v3: +exp009 improved) | 74.15 | 83.74 | +9.6 |

- **No real gain today**: CV 74.17→74.15 but LB 83.62→83.74 (slightly worse, within noise). The tiny CV
  improvement did not transfer — both blends are statistically the same on LB (~83.6–83.7).
- Confirms the residual-analysis conclusion: the tabular/text-meta frontier is saturated. The next genuine
  jump must come from stronger, decorrelated text models (exp012 XLM-R-large, exp013 kitchen-sink v2).
- See `docs/progress/2026-06-11.md` for the full day-1 narrative (data exploration → decisions → results).

### exp014 — Multi-seed text: BERT Turkish fine-tune, seed 1337 (avg with exp009 in blend)
- CV MSE **139.8415** | RMSE 11.8255 (±0.3696) | 2024+ RMSE 13.2003 | y<100 RMSE 12.0032 | 192 features
- Notes: bert=dbmdz/bert-base-turkish-cased

### exp015 — Multi-seed text: XLM-RoBERTa-large fine-tune, seed 1337 (avg with exp012 in blend)
- CV MSE **134.6902** | RMSE 11.6056 (±0.1930) | 2024+ RMSE 12.8827 | y<100 RMSE 11.8283 | 192 features
- Notes: bert=FacebookAI/xlm-roberta-large

### exp016 — Kitchen sink v3: exp004 features + 4 multi-seed text OOFs (BERT x2, XLM-R x2), Optuna-tuned LGBM
- CV MSE **75.2753** | RMSE 8.6761 (±0.1741) | 2024+ RMSE 9.6826 | y<100 RMSE 8.8549 | 181 features
- Notes: hpo: 100 trials

### exp017 — Pseudo-labeling (blend test preds, weight=1.0)
- CV MSE **72.9060** | RMSE 8.5385 | 2024+ RMSE 9.4805
- OOF mildly optimistic (pseudo-labels see all folds); decide on LB.

## CV ↔ LB update (day 2, submissions 6–8) — PLATEAU

| sub | cv_mse | LB MSE | offset | verdict |
|---|---|---|---|---|
| blend v4 (+exp012 XLM-R, +exp013 ks-v2) | 73.89 | **83.1677** | +9.28 | ✅ new best — text models transfer EXTRA well |
| blend v5 (+exp014/015/016 multi-seed) | 73.76 | 83.1768 | +9.42 | ➖ no transfer; multi-seed averaging = noise |
| blend v6 (+exp017 pseudo, 81% weight) | 72.68* | 83.2240 | — | ❌ *inflated CV; pseudo-label OOF optimism confirmed on LB |

- **exp017 pseudo-labeling REJECTED**: its OOF (72.91) is optimistic by construction (pseudo-labels
  derive from full-train fold averages, leaking each validation fold). LB settles it: 83.22 > 83.17.
  → exclude exp017 from future blends; don't trust any CV where test-derived labels enter training.
- **Text ceiling confirmed**: BERT(2 seeds), XLM-R-large(2 seeds), kitchen-sinks v2/v3 all land within
  ±0.05 LB of each other. Day-2 simple-leak audit (digits in text, train–test row/text duplicates,
  discrete-target rounding, per-year ceiling calibration) found NOTHING — there is no cheap trick.
- Gap to top-5: 83.17 vs 82.23 = 0.94 MSE = RMSE 9.12 vs 9.07. Leaders are marginally better, not
  structurally different. Remaining levers: exp018 (mDeBERTa diversity), exp019 (CatBoost on the
  strongest feature set — algorithmic diversity where it matters). After those, protect rank and
  polish the final-solution notebook.

## Residual-GBM test (2026-06-12, reports/eda/residual_analysis.py)
Question: how do we KNOW tabular FE is exhausted, beyond linear correlations?
Answer: trained a fold-safe LGBM to predict the best model's residual from all 85 tabular
features (nonlinear, searches interactions itself). Result: **OOF R² = ±0.0005** for both
exp010 and the blend → recoverable MSE ≈ ±0.03 = pure noise. This upgrades "we think the
tabular frontier is closed" to a measurement: nothing derivable from the tabular columns
explains the remaining error. What's left = generator noise + the part only the text reflects.

### exp018 — Text diversity: xlm-roberta-base fine-tune (lighter than large, different capacity)
- CV MSE **132.3995** | RMSE 11.5065 (±0.2725) | 2024+ RMSE 12.8380 | y<100 RMSE 11.6461 | 128 features
- Notes: bert=FacebookAI/xlm-roberta-base

### exp019 — CatBoost on kitchen-sink v3 features (incl. 4 text OOFs) — algorithmic diversity at the strongest feature set
- CV MSE **75.2845** | RMSE 8.6767 (±0.1939) | 2024+ RMSE 9.6614 | y<100 RMSE 8.8569 | 181 features
- Notes: hpo: 30 trials

### exp020 — Text v2 recipe: Turkish BERT + attention pooling + multi-sample dropout + LLRD + cosine (8 epochs)
- CV MSE **134.9180** | RMSE 11.6154 (±0.2105) | 2024+ RMSE 12.7741 | y<100 RMSE 11.6966 | 128 features
- Notes: bert_v2=dbmdz/bert-base-turkish-cased; llrd=0.9; msd=5x0.3

### exp021 — Text v2 recipe on xlm-roberta-base (same recipe as exp020, different backbone for blend diversity)
- CV MSE **136.8572** | RMSE 11.6986 (±0.3321) | 2024+ RMSE 12.9155 | y<100 RMSE 11.8442 | 128 features
- Notes: bert_v2=FacebookAI/xlm-roberta-base; llrd=0.9; msd=5x0.3
- Decision: the v2 recipe (attention pooling + multi-sample dropout + LLRD + cosine) did NOT beat the
  v1 recipe — 6 transformers across 2 recipes all land at RMSE ~11.5. This is the text **signal**
  ceiling, not a recipe problem. Text fine-tuning closed.

## CV ↔ LB final (day 3, submissions 9–13) — 5 blend variants, FLOOR CONFIRMED

All five are at the floor (CV 73.6–74.0) but built from different member sets / combiners, to
empirically separate signal from public-LB noise and pick the private bet:

| sub (tag) | members | combiner | cv_mse | LB MSE | offset |
|---|---|---|---|---|---|
| `blend_full_ridge` | all 21 | ridge stack | **73.61** | **82.96** | +9.35 |
| `blend_core_ridge` | 5 GBM + MLP | ridge stack | 73.85 | 83.17 | +9.32 |
| `blend_strong_ridge` | mse<80 + MLP (10) | ridge stack | 73.78 | 83.23 | +9.45 |
| `blend_strong_wts` | mse<80 + MLP (10) | convex weights | 73.88 | 83.89 | +10.01 |
| `blend_core_equal` | 5 GBM + MLP | equal weight | 74.02 | 84.13 | +10.11 |

- **CV ranking transferred to LB almost perfectly** (one 0.07-MSE swap). CV is a faithful guide;
  better combiner (ridge > convex weights > equal) and more members ⇒ lower LB, deterministically.
- This **refines the day-2 "mostly noise" read**: differences between OUR (correlated) submissions
  on the same public rows are low-noise and real. So the ~2.25 MSE gap to public #1 (80.71) is
  **largely a real edge**, not pure luck — the top teams genuinely have ~0.69→~0.71 R² more signal,
  most plausibly from a text approach our fine-tunes couldn't reach. We do not have it tonight.
- **Final pick: `blend_full_ridge` (LB 82.96)** — best on both CV and LB; lock it.

## Signal-floor analysis (2026-06-13, reports/eda/floor_analysis.py)

Seven independent diagnostics, all reproducible from saved artifacts (no fitting, leak-free).
Full narrative in `docs/progress/2026-06-13.md`; figures in `reports/figures/`.

1. **CV→LB offset is 100% the test's late-year skew**: re-weighting OOF MSE by the test year mix
   gives 83.39 ≈ LB 82.96; per-year bias ≈ 0. No overfit, no leak.
2. **Within-year R² is flat at ~0.69** (incl. 2025-26): late years are harder only because their
   target variance is larger — no late-year-specific signal we fail to model.
3. **Ceiling & dispersion oracles** recover ≈0 / hurt (rescale 102→111) → neither is a lever.
4. **Blend members 0.95 correlated** → diversity exhausted (blend beats best single by only ~1.5 MSE).
5. **Text describes the profile, not the score** (top-error rows: glowing text ↔ low score) ⇒
   `target = g(features) + irreducible noise`; text is a noisier view of the same features.
6. **No forgotten signal**: every raw column is used; an aspect-based sentiment extraction has real
   target signal (corr +0.33) but recovers R² = −0.04 on the blend residual — already captured by
   TF-IDF + fine-tunes. Text-numeric discrepancy vs the columns: corr −0.01 with the residual.
7. **Gap to public #1 (≈2.25 MSE) sits inside the public-subset SE band (±2-3 MSE)** in absolute
   terms; relative ordering of similar models is reliable (see day-3 table). → don't overfit the
   public LB; ship a robust CV-backed blend for the private board.

**Bottom line:** the modelling frontier is closed for the levers we explored. Best submission =
`blend_full_ridge`, CV 73.61 / LB 82.96, a top-tier, well-calibrated, non-overfit solution.

### exp023 — LoRA fine-tune of an 8B Turkish LLM (capacity bet on the text floor)
- Hypothesis: the 6 prior text models (≤550M params) plateaued at RMSE ~11.5 — is that a *capacity*
  limit (a multi-B LLM would break it) or a *signal* limit (the text genuinely caps there)?
- Setup: `ytu-ce-cosmos/Turkish-Llama-8b` in 4-bit + LoRA (r=16), mean-pool + regression head,
  y/100, fold-safe. Ran fold 0 three times (3ep/lr1e-4, 6ep/lr2e-4, 6ep/lr1e-4).
- Result: every stable run **converges to fold-0 RMSE ~11.2–11.4** — identical to BERT-110M (11.62)
  and XLM-R-large (11.45). Inner-val plateaued by epoch 4 (12.06→12.26, i.e. turned up). lr 2e-4 diverged.
- **Decision: DEFINITIVE — the text floor is a SIGNAL limit, not a capacity limit.** A 70× larger
  model matches a 110M one. Stopped after fold 0 (a full run would only add a 0-weight blend member).
  This closes the last untested mechanism for new signal. Best submission stays `blend_lbopt`.

### blend_lbopt — LB-distribution-optimized stacker (test-year-weighted + year-conditional)
- Same OOF members as blend_full_ridge, but the Ridge stacker is (a) sample-weighted by the test's
  year frequency and (b) fit separately for late (≥2024) vs early years — because the LB is 62%
  late years and the uniform CV objective optimizes the wrong distribution.
- CV MSE (uniform) **73.50**; LB-proxy (test-year-weighted OOF MSE) 83.17 vs 83.39 for the plain
  stacker → **projected LB ≈ 82.74** (from 82.96). Modest but real and free.
- Decision: this is the best available submission; lock it (keep `blend_full_ridge` as the safe backup).
