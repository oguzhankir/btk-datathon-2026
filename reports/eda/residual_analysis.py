"""Residual analysis of the best single model (exp010): where is signal left?

Loads exp010 OOF, computes residuals, and ranks every engineered feature by how
much it still correlates with the error — if the synthetic target is a formula,
leftover structure shows up here and tells us what to feature-engineer next.
    python reports/eda/residual_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data import TARGET, fold_array, load_raw  # noqa: E402
from src.features.tabular import build_tabular  # noqa: E402
from src.utils import ARTIFACTS, get_logger  # noqa: E402

log = get_logger()


def main() -> None:
    train, test, _ = load_raw()
    y = train[TARGET].to_numpy(dtype=float)
    oof = np.load(ARTIFACTS / "oof_exp010.npy")
    res = y - oof
    years = train["application_year"].to_numpy()

    print(f"exp010 OOF: rmse={np.sqrt((res**2).mean()):.4f}  bias={res.mean():+.4f}")
    print(f"  ceiling (y=100): rmse={np.sqrt((res[y==100]**2).mean()):.3f}  bias={res[y==100].mean():+.3f}")
    print(f"  middle  (y<100): rmse={np.sqrt((res[y<100]**2).mean()):.3f}  bias={res[y<100].mean():+.3f}")

    # systematic bias by year (calibration target)
    print("\nresidual mean by application_year (positive = model under-predicts):")
    for yr in sorted(np.unique(years)):
        m = years == yr
        print(f"  {yr}: n={m.sum():5d}  res_mean={res[m].mean():+.3f}  |res|={np.abs(res[m]).mean():.3f}")

    # leftover correlation between residual and every feature
    X, _, feats = build_tabular(train, test, fe=True)
    num = [c for c in feats if X[c].dtype.kind in "ifu"]
    cors = {c: np.corrcoef(X[c].fillna(X[c].median()), res)[0, 1] for c in num}
    s = pd.Series(cors).dropna().sort_values(key=np.abs, ascending=False)
    print("\ntop-20 features still correlated with the residual (leftover signal):")
    for c, v in s.head(20).items():
        print(f"  {v:+.4f}  {c}")

    # residual on the hardest decile of predictions
    q = pd.qcut(oof, 10, labels=False, duplicates="drop")
    print("\nrmse by predicted decile (where the model is weakest):")
    for d in sorted(np.unique(q)):
        m = q == d
        print(f"  decile {d}: pred~{oof[m].mean():6.2f}  rmse={np.sqrt((res[m]**2).mean()):.3f}")


if __name__ == "__main__":
    main()
