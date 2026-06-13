# Progress Logs

This folder holds the **daily progress reports** for the competition. Each file tells that day's
story end to end: data inspection, the decisions made and their rationale, the models tried, what
worked and what didn't, LB results, and next steps.

For the per-experiment record see `EXPERIMENTS.md` (repo root); for the numeric log see `results.csv`.

## Logs

| Date | Report | Summary |
|---|---|---|
| 2026-06-11 | [2026-06-11.md](2026-06-11.md) | Day 1: pipeline built, exp001–011 + blend; best LB ~83.6. Tabular frontier closed (residual analysis); BERT fixed. exp012/exp013 running overnight. |
| 2026-06-12 | [2026-06-12.md](2026-06-12.md) | Day 2: the text push held → **LB 83.17** (new best). Multi-seed and pseudo-labeling rejected; simple-leak audit clean. Evening: exp018 mDeBERTa + exp019 CatBoost@ks-v3. |
| 2026-06-13 | [2026-06-13.md](2026-06-13.md) | Day 3: **signal floor proven** (7 independent tests, all reproducible via `reports/eda/floor_analysis.py` + 4 figures). The CV→LB gap is 100% year distribution; the gap to the top is largely public-LB noise. Decision: don't chase the public LB, lock a robust blend for the private board. 5 final blend variants prepared. |
