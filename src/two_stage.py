"""Two-stage ceiling model: P(y==100) classifier blended with a regressor.

final = p^gamma * 100 + (1 - p^gamma) * reg
Variants tried on OOF: regressor trained on all rows vs. on y<100 rows only,
and a small grid of blend exponents gamma. Best combo picked by OOF MSE.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.cv import mse, rmse
from src.data import N_FOLDS
from src.models import get_model
from src.utils import clip_preds, get_logger

log = get_logger()

GAMMAS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def run_two_stage(
    X: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    X_test: pd.DataFrame,
    reg_params: dict | None = None,
    clf_params: dict | None = None,
    device: str = "cpu",
    years: np.ndarray | None = None,
) -> dict:
    """Fit classifier + two regressor variants per fold; pick blend by OOF MSE."""
    y = np.asarray(y, dtype=float)
    is100 = (y == 100).astype(int)
    t0 = time.time()

    p_oof = np.zeros(len(X))
    reg_all_oof = np.zeros(len(X))
    reg_lt_oof = np.zeros(len(X))
    p_test = np.zeros(len(X_test))
    reg_all_test = np.zeros(len(X_test))
    reg_lt_test = np.zeros(len(X_test))

    for fold in range(N_FOLDS):
        tr, va = np.where(folds != fold)[0], np.where(folds == fold)[0]
        X_tr, X_va = X.iloc[tr], X.iloc[va]

        clf = get_model("lgbm_clf", clf_params, device)
        clf.fit(X_tr, is100[tr], X_va, is100[va])
        p_oof[va] = clf.predict(X_va)
        p_test += clf.predict(X_test) / N_FOLDS

        reg_a = get_model("lgbm", reg_params, device)
        reg_a.fit(X_tr, y[tr], X_va, y[va])
        reg_all_oof[va] = reg_a.predict(X_va)
        reg_all_test += reg_a.predict(X_test) / N_FOLDS

        lt = y[tr] < 100
        # validate the y<100 regressor on the non-ceiling part of the val fold
        va_lt = va[y[va] < 100]
        reg_l = get_model("lgbm", reg_params, device)
        reg_l.fit(X_tr.iloc[lt], y[tr][lt], X.iloc[va_lt], y[va_lt])
        reg_lt_oof[va] = reg_l.predict(X_va)
        reg_lt_test += reg_l.predict(X_test) / N_FOLDS
        log.info(f"  fold {fold}: clf+2 regs done")

    best = {"mse": np.inf}
    for reg_name, (r_oof, r_test) in {
        "all_rows": (reg_all_oof, reg_all_test),
        "lt100_rows": (reg_lt_oof, reg_lt_test),
    }.items():
        for gamma in GAMMAS:
            w = p_oof**gamma
            blend_oof = clip_preds(w * 100 + (1 - w) * r_oof)
            m = mse(y, blend_oof)
            if m < best["mse"]:
                wt = p_test**gamma
                best = {
                    "mse": m,
                    "regressor": reg_name,
                    "gamma": gamma,
                    "oof": blend_oof,
                    "test_pred": clip_preds(wt * 100 + (1 - wt) * r_test),
                }
    # plain regressor (gamma -> inf equivalent baseline) for reference
    plain = mse(y, clip_preds(reg_all_oof))
    log.info(
        f"two-stage best: reg={best['regressor']} gamma={best['gamma']} "
        f"mse={best['mse']:.4f} (plain reg mse={plain:.4f})"
    )

    oof = best["oof"]
    fold_rmses = [rmse(y[folds == f], oof[folds == f]) for f in range(N_FOLDS)]
    result = {
        "oof": oof,
        "test_pred": best["test_pred"],
        "cv_mse": best["mse"],
        "cv_rmse": rmse(y, oof),
        "cv_rmse_std": float(np.std(fold_rmses)),
        "fold_rmses": fold_rmses,
        "runtime_s": time.time() - t0,
        "importance": None,
        "rmse_y_lt_100": rmse(y[y < 100], oof[y < 100]),
        "best_combo": {"regressor": best["regressor"], "gamma": best["gamma"]},
    }
    if years is not None:
        m2024 = np.asarray(years) >= 2024
        result["rmse_year_2024plus"] = rmse(y[m2024], oof[m2024])
    return result
