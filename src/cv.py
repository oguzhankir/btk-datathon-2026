"""Cross-validation runner producing OOF/test predictions and sliced metrics."""
from __future__ import annotations

import json
import time
from typing import Callable

import numpy as np
import pandas as pd

from src.data import N_FOLDS
from src.utils import clip_preds, get_logger

log = get_logger()


def mse(y, p) -> float:
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def rmse(y, p) -> float:
    return float(np.sqrt(mse(y, p)))


def run_cv(
    model_factory: Callable[[], object],
    X: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    X_test: pd.DataFrame | None = None,
    sample_weight: np.ndarray | None = None,
    years: np.ndarray | None = None,
    n_seeds: int = 1,
) -> dict:
    """Run the fixed 5-fold CV.

    model_factory must return a fresh wrapper from src.models per call.
    Returns dict with oof, test_pred, per-fold and year/ceiling-sliced metrics,
    and mean feature importance when the model exposes it.
    """
    y = np.asarray(y, dtype=float)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test)) if X_test is not None else None
    fold_rmses: list[float] = []
    importances: list[pd.Series] = []
    t0 = time.time()

    for fold in range(N_FOLDS):
        tr, va = np.where(folds != fold)[0], np.where(folds == fold)[0]
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        w_tr = sample_weight[tr] if sample_weight is not None else None
        fold_va = np.zeros(len(va))
        for s in range(n_seeds):
            model = model_factory()
            if n_seeds > 1 and hasattr(model, "params"):
                for k in ("random_state", "random_seed"):
                    if k in model.params:
                        model.params[k] = model.params[k] + s
            model.fit(X_tr, y[tr], X_va, y[va], sample_weight=w_tr)
            fold_va += model.predict(X_va) / n_seeds
            if test_pred is not None:
                test_pred += model.predict(X_test) / (N_FOLDS * n_seeds)
            imp = model.feature_importance() if hasattr(model, "feature_importance") else None
            if imp is not None:
                importances.append(imp)
        oof[va] = fold_va
        fr = rmse(y[va], clip_preds(fold_va))
        fold_rmses.append(fr)
        log.info(f"  fold {fold}: rmse={fr:.4f}")

    oof = clip_preds(oof)
    if test_pred is not None:
        test_pred = clip_preds(test_pred)

    result = {
        "oof": oof,
        "test_pred": test_pred,
        "cv_mse": mse(y, oof),
        "cv_rmse": rmse(y, oof),
        "cv_rmse_std": float(np.std(fold_rmses)),
        "fold_rmses": fold_rmses,
        "runtime_s": time.time() - t0,
        "importance": pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
        if importances
        else None,
    }
    if years is not None:
        m = np.asarray(years) >= 2024
        result["rmse_year_2024plus"] = rmse(y[m], oof[m]) if m.any() else float("nan")
    lt = y < 100
    result["rmse_y_lt_100"] = rmse(y[lt], oof[lt])
    log.info(
        f"CV mse={result['cv_mse']:.4f} rmse={result['cv_rmse']:.4f} "
        f"(±{result['cv_rmse_std']:.4f}) 2024+={result.get('rmse_year_2024plus', float('nan')):.4f} "
        f"y<100={result['rmse_y_lt_100']:.4f}"
    )
    return result


def append_results_row(row: dict) -> None:
    """Append one experiment row to results.csv (creates header if missing)."""
    from src.utils import RESULTS_CSV

    cols = [
        "exp_id", "timestamp", "description", "n_features", "cv_mse", "cv_rmse",
        "cv_rmse_std", "fold_rmses", "rmse_year_2024plus", "rmse_y_lt_100",
        "runtime_s", "device", "config_path", "notes",
    ]
    df = pd.DataFrame([{c: row.get(c, "") for c in cols}])
    header = not RESULTS_CSV.exists()
    df.to_csv(RESULTS_CSV, mode="a", header=header, index=False)


def fold_rmses_json(fold_rmses: list[float]) -> str:
    """Serialize per-fold RMSEs for the results.csv row."""
    return json.dumps([round(f, 4) for f in fold_rmses])
