"""Adversarial validation: how distinguishable are train and test rows?

Reproduces EDA fact #2: AUC ≈ 0.65 with year features, ≈ 0.50 without.
Usage:
    python scripts/adversarial_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_raw  # noqa: E402
from src.features.tabular import build_tabular  # noqa: E402
from src.utils import SEED, get_logger, seed_everything  # noqa: E402

log = get_logger()

YEAR_FEATURES = ["application_year", "graduation_year", "years_since_grad", "grad_age", "age"]


def adv_auc(X: pd.DataFrame, is_test: np.ndarray) -> float:
    """5-fold OOF AUC of an LGBM telling train rows from test rows."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    oof = np.zeros(len(X))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, va in skf.split(X, is_test):
        m = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63, random_state=SEED, verbosity=-1
        )
        m.fit(X.iloc[tr], is_test[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return float(roc_auc_score(is_test, oof))


def main() -> None:
    seed_everything()
    train, test, _ = load_raw()
    X_tr, X_te, _ = build_tabular(train, test, fe=True)
    X = pd.concat([X_tr, X_te], ignore_index=True)
    is_test = np.r_[np.zeros(len(X_tr)), np.ones(len(X_te))].astype(int)

    auc_full = adv_auc(X, is_test)
    no_year = [c for c in X.columns if c not in YEAR_FEATURES]
    auc_noyear = adv_auc(X[no_year], is_test)
    log.info(f"adversarial AUC with year features:    {auc_full:.4f}")
    log.info(f"adversarial AUC without year features: {auc_noyear:.4f}")
    log.info("shift is temporal-only if the second number is ~0.50")


if __name__ == "__main__":
    main()
