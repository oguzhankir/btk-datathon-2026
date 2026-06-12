"""exp017: pseudo-labeling — adapt to the test year-mix via blend test predictions.

The test set skews to 2024–2026 (62% vs train's ~33%). We take the blend's test
predictions as soft labels, append the test rows (down-weighted) to every fold's
training part, and retrain a tuned LGBM on the kitchen-sink feature set. OOF is
still computed on real train labels only, so the CV stays comparable.

Caveat: pseudo-labels come from fold-model averages, so a little information from
each validation fold leaks into the pseudo-labels — the OOF here is mildly
optimistic. Trust the LB to confirm.

    python scripts/pseudo_label.py [--weight 0.3 0.5 1.0] [--source blend]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.cv import append_results_row, fold_rmses_json, rmse  # noqa: E402
from src.data import N_FOLDS, TARGET, fold_array, load_raw  # noqa: E402
from src.features import build_features  # noqa: E402
from src.models import get_model  # noqa: E402
from src.utils import ARTIFACTS, EXPERIMENTS_MD, clip_preds, detect_device, get_logger, seed_everything  # noqa: E402

log = get_logger()

EXP_ID = "exp017"
# kitchen-sink v3 feature block (same as configs/exp016.yaml)
FEATURES = {
    "tabular_fe": True,
    "target_encoding": True,
    "text_classic": True,
    "ridge_meta": True,
    "embeddings": {"model": "intfloat/multilingual-e5-large", "svd": 64, "knn": [5, 25]},
    "extra_oof_features": {
        "bert_pred": "exp009", "bert_pred2": "exp014",
        "xlmr_pred": "exp012", "xlmr_pred2": "exp015",
    },
}


def load_params() -> dict:
    """Best tuned LGBM params, preferring the latest kitchen-sink study."""
    for name in ("params_lgbm_exp016.json", "params_lgbm_exp013.json", "params_exp013.json"):
        p = ARTIFACTS / name
        if p.exists():
            log.info(f"params: {name}")
            return json.loads(p.read_text())
    log.warning("no tuned params found — using LGBM defaults")
    return {}


def run(weight: float, X_tr: pd.DataFrame, X_te: pd.DataFrame, y: np.ndarray,
        pseudo_y: np.ndarray, folds: np.ndarray, params: dict, device: str) -> tuple[np.ndarray, np.ndarray]:
    """One pseudo-label pass: test rows (weight w) joined to each fold's train part."""
    oof = np.zeros(len(X_tr))
    test_pred = np.zeros(len(X_te))
    for fold in range(N_FOLDS):
        tr = np.where(folds != fold)[0]
        va = np.where(folds == fold)[0]
        X_aug = pd.concat([X_tr.iloc[tr], X_te], axis=0, ignore_index=True)
        y_aug = np.concatenate([y[tr], pseudo_y])
        w_aug = np.concatenate([np.ones(len(tr)), np.full(len(X_te), weight)])
        model = get_model("lgbm", params or None, device)
        model.fit(X_aug, y_aug, X_tr.iloc[va], y[va], sample_weight=w_aug)
        oof[va] = model.predict(X_tr.iloc[va])
        test_pred += model.predict(X_te) / N_FOLDS
    return clip_preds(oof), clip_preds(test_pred)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", type=float, nargs="+", default=[0.3, 0.5, 1.0],
                    help="pseudo-label sample weights to try (picked by OOF MSE)")
    ap.add_argument("--source", default="blend", help="experiment whose test preds are the pseudo-labels")
    args = ap.parse_args()
    seed_everything()
    device = detect_device()
    t0 = time.time()

    src_p = ARTIFACTS / f"test_{args.source}.npy"
    if not src_p.exists():
        sys.exit(f"missing {src_p} — run the blend first")
    pseudo_y = clip_preds(np.load(src_p))

    train, test, _ = load_raw()
    y = train[TARGET].to_numpy(dtype=float)
    folds = fold_array(train)
    years = train["application_year"].to_numpy()
    X_tr, X_te, notes = build_features(FEATURES, train, test, y, folds)
    params = load_params()

    best = None
    for w in args.weight:
        oof, test_pred = run(w, X_tr, X_te, y, pseudo_y, folds, params, device)
        m = float(np.mean((y - oof) ** 2))
        log.info(f"pseudo weight={w}: oof mse={m:.4f}")
        if best is None or m < best[0]:
            best = (m, w, oof, test_pred)

    m, w, oof, test_pred = best
    fold_rmses = [rmse(y[folds == f], oof[folds == f]) for f in range(N_FOLDS)]
    m2024 = years >= 2024
    np.save(ARTIFACTS / f"oof_{EXP_ID}.npy", oof)
    np.save(ARTIFACTS / f"test_{EXP_ID}.npy", test_pred)
    notes.append(f"pseudo_source={args.source}; best_weight={w}; oof mildly optimistic (see docstring)")
    append_results_row({
        "exp_id": EXP_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": "Pseudo-labeling: blend test preds as soft labels, tuned LGBM on kitchen-sink v3 features",
        "n_features": X_tr.shape[1],
        "cv_mse": round(m, 4),
        "cv_rmse": round(float(np.sqrt(m)), 4),
        "cv_rmse_std": round(float(np.std(fold_rmses)), 4),
        "fold_rmses": fold_rmses_json(fold_rmses),
        "rmse_year_2024plus": round(rmse(y[m2024], oof[m2024]), 4),
        "rmse_y_lt_100": round(rmse(y[y < 100], oof[y < 100]), 4),
        "runtime_s": round(time.time() - t0, 1),
        "device": device,
        "config_path": "scripts/pseudo_label.py",
        "notes": "; ".join(notes),
    })
    with EXPERIMENTS_MD.open("a") as f:
        f.write(
            f"\n### {EXP_ID} — Pseudo-labeling (blend test preds, weight={w})\n"
            f"- CV MSE **{m:.4f}** | RMSE {np.sqrt(m):.4f} | 2024+ RMSE {rmse(y[m2024], oof[m2024]):.4f}\n"
            f"- OOF mildly optimistic (pseudo-labels see all folds); decide on LB.\n"
        )
    log.info(f"{EXP_ID} complete: best weight={w}, oof mse={m:.4f} (runtime {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
