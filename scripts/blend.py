"""Blend experiment OOFs: non-negative weight optimization + Ridge stacker.

Usage:
    python scripts/blend.py                                  # auto: best of ridge/weights, all artifacts
    python scripts/blend.py -e exp010 exp013 exp019          # explicit members
    python scripts/blend.py -e ... --method equal --tag blend_equal
    python scripts/blend.py --method ridge --tag blend_full

--method: auto (pick ridge-vs-weights by OOF MSE) | ridge | weights | equal
--tag:    output name -> artifacts/{oof,test}_{tag}.npy, a results.csv row, and
          (via make_submission -e {tag}) submissions/sub_{tag}.csv
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

# Members whose OOF cannot be trusted (e.g. test-derived labels leak into it):
# exp017 pseudo-labeling showed CV 72.9 but LB 83.22 — auto-discovery skips these.
# Pass ids explicitly (`-e exp017 ...`) only to inspect them on purpose.
EXCLUDED = {"exp017"}


def discover_exp_ids() -> list[str]:
    """All exp ids with both oof_ and test_ artifacts saved (minus EXCLUDED, blend tags)."""
    ids = []
    for p in sorted(ARTIFACTS.glob("oof_*.npy")):
        exp = p.stem.removeprefix("oof_")
        if exp.startswith("blend") or exp in EXCLUDED:
            continue
        if (ARTIFACTS / f"test_{exp}.npy").exists():
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
    oofs: np.ndarray, y: np.ndarray, tests: np.ndarray, folds: np.ndarray,
    sample_weight: np.ndarray | None = None,
    segment: np.ndarray | None = None,
    test_segment: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold-safe Ridge stacker on the OOF matrix.

    sample_weight: per-row weight (e.g. test-year frequency) so the stacker is
        optimized for the LB's year distribution rather than the uniform train one.
    segment / test_segment: if given, fit a SEPARATE stacker per segment value
        (e.g. late-year vs early-year) — the LB is 62% late years, and the optimal
        member mix differs there.
    """
    from sklearn.linear_model import Ridge

    def _fit_predict(Xtr, ytr, Xpred, w):
        return Ridge(alpha=1.0, positive=True, fit_intercept=True).fit(
            Xtr, ytr, sample_weight=w
        ).predict(Xpred)

    segs = np.unique(segment) if segment is not None else [None]
    stack_oof = np.zeros(len(y))
    test_pred = np.zeros(len(tests))
    for s in segs:
        tr_seg = np.ones(len(y), bool) if s is None else (segment == s)
        te_seg = np.ones(len(tests), bool) if s is None else (test_segment == s)
        for fold in range(N_FOLDS):
            tr = (folds != fold) & tr_seg
            va = (folds == fold) & tr_seg
            w = sample_weight[tr] if sample_weight is not None else None
            stack_oof[va] = _fit_predict(oofs[tr], y[tr], oofs[va], w)
        w_all = sample_weight[tr_seg] if sample_weight is not None else None
        test_pred[te_seg] = _fit_predict(oofs[tr_seg], y[tr_seg], tests[te_seg], w_all)
    return clip_preds(stack_oof), clip_preds(test_pred)


def test_year_weights(train_years: np.ndarray, test_years: np.ndarray) -> np.ndarray:
    """Per-train-row weight = test/train year frequency ratio, normalized to mean 1.

    Makes the stacker minimize the test set's (late-year-heavy) MSE — the LB proxy.
    """
    te = np.bincount(test_years) / len(test_years)
    tr = np.bincount(train_years, minlength=len(te)) / len(train_years)
    w = np.where(tr[train_years] > 0, te[train_years] / np.maximum(tr[train_years], 1e-9), 1.0)
    return w / w.mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--exp-ids", nargs="+", default=None)
    ap.add_argument("--method", choices=["auto", "ridge", "weights", "equal"], default="auto")
    ap.add_argument("--tag", default="blend", help="output artifact/results name")
    ap.add_argument("--weight-by-test-year", action="store_true",
                    help="optimize the stacker for the test's year distribution (LB proxy)")
    ap.add_argument("--year-conditional", action="store_true",
                    help="fit separate stackers for late (>=2024) vs early years")
    args = ap.parse_args()
    seed_everything()

    exp_ids = args.exp_ids or discover_exp_ids()
    if len(exp_ids) < 2:
        sys.exit(f"need >=2 experiments with artifacts, found {exp_ids}")
    log.info(f"blending ({args.method}, tag={args.tag}): {exp_ids}")

    train, test_df, _ = load_raw()
    y = train[TARGET].to_numpy(dtype=float)
    folds = fold_array(train)
    years = train["application_year"].to_numpy()
    test_years = test_df["application_year"].to_numpy()

    oofs = np.column_stack([np.load(ARTIFACTS / f"oof_{e}.npy") for e in exp_ids])
    tests = np.column_stack([np.load(ARTIFACTS / f"test_{e}.npy") for e in exp_ids])
    for e, col in zip(exp_ids, oofs.T):
        log.info(f"  {e}: oof mse={mse(y, col):.4f}")

    sw = test_year_weights(years, test_years) if args.weight_by_test_year else None
    seg = (years >= 2024).astype(int) if args.year_conditional else None
    test_seg = (test_years >= 2024).astype(int) if args.year_conditional else None

    n = len(exp_ids)
    candidates: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    if args.method in ("auto", "weights"):
        w = optimize_weights(oofs, y)
        candidates["weights"] = (clip_preds(oofs @ w), clip_preds(tests @ w),
                                 "weights: " + ", ".join(f"{e}={wi:.3f}" for e, wi in zip(exp_ids, w)))
    if args.method in ("auto", "ridge"):
        s_oof, s_test = ridge_stack(oofs, y, tests, folds, sample_weight=sw,
                                    segment=seg, test_segment=test_seg)
        extra = (" +test-year-wt" if sw is not None else "") + (" +year-cond" if seg is not None else "")
        candidates["ridge_stack"] = (s_oof, s_test, f"ridge alpha=1.0 positive{extra}")
    if args.method == "equal":
        we = np.full(n, 1.0 / n)
        candidates["equal"] = (clip_preds(oofs @ we), clip_preds(tests @ we),
                               f"equal weight 1/{n} each")

    for name, (o, _, _) in candidates.items():
        log.info(f"  {name} mse={mse(y, o):.4f}")
    method = min(candidates, key=lambda k: mse(y, candidates[k][0]))
    oof, test, note = candidates[method]

    np.save(ARTIFACTS / f"oof_{args.tag}.npy", oof)
    np.save(ARTIFACTS / f"test_{args.tag}.npy", test)
    fold_rmses = [rmse(y[folds == f], oof[folds == f]) for f in range(N_FOLDS)]
    m2024 = years >= 2024
    append_results_row(
        {
            "exp_id": args.tag,
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
            "notes": note,
        }
    )
    log.info(f"{args.tag} ({method}) saved: cv_mse={mse(y, oof):.4f}  -> ./run.sh submit {args.tag}")


if __name__ == "__main__":
    main()
