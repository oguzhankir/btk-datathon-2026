"""Blend experiment OOFs: non-negative weight optimization + Ridge stacker.

Usage:
    python scripts/blend.py -e exp004 exp005 exp006 exp007
    python scripts/blend.py            # auto: every exp with saved artifacts
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cv import append_results_row, fold_rmses_json, mse, rmse  # noqa: E402
from src.data import N_FOLDS, TARGET, fold_array, load_raw  # noqa: E402
from src.utils import ARTIFACTS, SEED, clip_preds, get_logger, seed_everything  # noqa: E402

log = get_logger()


def discover_exp_ids() -> list[str]:
    """All exp ids with both oof_ and test_ artifacts saved."""
    ids = []
    for p in sorted(ARTIFACTS.glob("oof_*.npy")):
        exp = p.stem.removeprefix("oof_")
        if exp != "blend" and (ARTIFACTS / f"test_{exp}.npy").exists():
            ids.append(exp)
    return ids


def optimize_weights(oofs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative, sum-1 weights minimizing OOF MSE (SLSQP + hill climbing)."""
    n = oofs.shape[1]

    def obj(w: np.ndarray) -> float:
        return mse(y, clip_preds(oofs @ w))

    res = minimize(
        obj,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )
    w = np.maximum(res.x, 0)
    w /= w.sum()
    # hill-climb refinement
    rng = np.random.default_rng(SEED)
    best = obj(w)
    for _ in range(2000):
        cand = np.maximum(w + rng.normal(0, 0.02, n), 0)
        if cand.sum() == 0:
            continue
        cand /= cand.sum()
        s = obj(cand)
        if s < best:
            best, w = s, cand
    return w


def ridge_stack(
    oofs: np.ndarray, y: np.ndarray, tests: np.ndarray, folds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fold-safe Ridge stacker on the OOF matrix."""
    from sklearn.linear_model import Ridge

    stack_oof = np.zeros(len(y))
    for fold in range(N_FOLDS):
        tr, va = folds != fold, folds == fold
        m = Ridge(alpha=1.0, positive=True, fit_intercept=True).fit(oofs[tr], y[tr])
        stack_oof[va] = m.predict(oofs[va])
    m = Ridge(alpha=1.0, positive=True, fit_intercept=True).fit(oofs, y)
    return clip_preds(stack_oof), clip_preds(m.predict(tests))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--exp-ids", nargs="+", default=None)
    args = ap.parse_args()
    seed_everything()

    exp_ids = args.exp_ids or discover_exp_ids()
    if len(exp_ids) < 2:
        sys.exit(f"need >=2 experiments with artifacts, found {exp_ids}")
    log.info(f"blending: {exp_ids}")

    train, _, _ = load_raw()
    y = train[TARGET].to_numpy(dtype=float)
    folds = fold_array(train)
    years = train["application_year"].to_numpy()

    oofs = np.column_stack([np.load(ARTIFACTS / f"oof_{e}.npy") for e in exp_ids])
    tests = np.column_stack([np.load(ARTIFACTS / f"test_{e}.npy") for e in exp_ids])
    for e, col in zip(exp_ids, oofs.T):
        log.info(f"  {e}: oof mse={mse(y, col):.4f}")

    w = optimize_weights(oofs, y)
    w_oof = clip_preds(oofs @ w)
    w_test = clip_preds(tests @ w)
    s_oof, s_test = ridge_stack(oofs, y, tests, folds)

    log.info("weights: " + ", ".join(f"{e}={wi:.3f}" for e, wi in zip(exp_ids, w)))
    log.info(f"weight-blend mse={mse(y, w_oof):.4f} | ridge-stack mse={mse(y, s_oof):.4f}")

    if mse(y, s_oof) < mse(y, w_oof):
        oof, test, method = s_oof, s_test, "ridge_stack"
    else:
        oof, test, method = w_oof, w_test, "weights"

    np.save(ARTIFACTS / "oof_blend.npy", oof)
    np.save(ARTIFACTS / "test_blend.npy", test)
    fold_rmses = [rmse(y[folds == f], oof[folds == f]) for f in range(N_FOLDS)]
    m2024 = years >= 2024
    append_results_row(
        {
            "exp_id": "blend",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "description": f"{method} blend of {'+'.join(exp_ids)}",
            "n_features": len(exp_ids),
            "cv_mse": round(mse(y, oof), 4),
            "cv_rmse": round(rmse(y, oof), 4),
            "cv_rmse_std": round(float(np.std(fold_rmses)), 4),
            "fold_rmses": fold_rmses_json(fold_rmses),
            "rmse_year_2024plus": round(rmse(y[m2024], oof[m2024]), 4),
            "rmse_y_lt_100": round(rmse(y[y < 100], oof[y < 100]), 4),
            "runtime_s": 0,
            "device": "cpu",
            "config_path": "",
            "notes": "weights: " + ", ".join(f"{e}={wi:.3f}" for e, wi in zip(exp_ids, w)),
        }
    )
    log.info(f"blend ({method}) saved: cv_mse={mse(y, oof):.4f}")


if __name__ == "__main__":
    main()
