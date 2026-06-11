"""Feature assembly: turn a config `features:` block into model matrices."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import CAT_COLS, TEXT_COL
from src.utils import get_logger

log = get_logger()


def build_features(
    feat_cfg: dict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Assemble the feature matrices for an experiment config.

    feat_cfg keys (all optional): tabular_fe, text_classic, ridge_meta,
    target_encoding, embeddings {model, svd, knn}.
    Returns (X_train, X_test, notes). Missing optional deps (e.g.
    sentence-transformers) are skipped with a note instead of failing.
    """
    from src.features.tabular import build_tabular

    notes: list[str] = []
    X_tr, X_te, _ = build_tabular(train, test, fe=bool(feat_cfg.get("tabular_fe", False)))

    if feat_cfg.get("text_classic", False):
        from src.features.text_classic import build_text_classic

        tc_tr, _ = build_text_classic(train[TEXT_COL], train["target_role"])
        tc_te, _ = build_text_classic(test[TEXT_COL], test["target_role"])
        X_tr = pd.concat([X_tr, tc_tr.reset_index(drop=True)], axis=1)
        X_te = pd.concat([X_te, tc_te.reset_index(drop=True)], axis=1)

    if feat_cfg.get("ridge_meta", False):
        from src.features.meta import logit_tfidf_ceiling_oof, ridge_tfidf_oof

        r_oof, r_test = ridge_tfidf_oof(train[TEXT_COL], y, test[TEXT_COL], folds)
        l_oof, l_test = logit_tfidf_ceiling_oof(train[TEXT_COL], y, test[TEXT_COL], folds)
        X_tr["meta_ridge_tfidf"], X_te["meta_ridge_tfidf"] = r_oof, r_test
        X_tr["meta_logit_ceiling"], X_te["meta_logit_ceiling"] = l_oof, l_test

    if feat_cfg.get("target_encoding", False):
        from src.features.meta import target_encode_oof

        cross_tr = train["target_role"].astype(str) + "_x_" + train["university_tier"].astype(str)
        cross_te = test["target_role"].astype(str) + "_x_" + test["university_tier"].astype(str)
        for name, (ctr, cte) in {
            **{c: (train[c], test[c]) for c in CAT_COLS},
            "role_x_tier": (cross_tr, cross_te),
        }.items():
            te_oof, te_test = target_encode_oof(ctr, y, cte, folds)
            X_tr[f"te_{name}"], X_te[f"te_{name}"] = te_oof, te_test

    emb_cfg = feat_cfg.get("embeddings") or {}
    if emb_cfg:
        try:
            from src.features.meta import knn_target_feature
            from src.features.text_embed import get_embeddings, svd_features

            model_name = emb_cfg.get("model", "intfloat/multilingual-e5-large")
            emb_tr, emb_te = get_embeddings(train[TEXT_COL], test[TEXT_COL], model_name)
            if emb_cfg.get("raw", False):
                raw_tr = pd.DataFrame(emb_tr, columns=[f"emb_{i}" for i in range(emb_tr.shape[1])])
                raw_te = pd.DataFrame(emb_te, columns=[f"emb_{i}" for i in range(emb_te.shape[1])])
                X_tr = pd.concat([X_tr, raw_tr], axis=1)
                X_te = pd.concat([X_te, raw_te], axis=1)
            n_svd = int(emb_cfg.get("svd", 64))
            if n_svd:
                s_tr, s_te = svd_features(emb_tr, emb_te, n_svd)
                for i in range(n_svd):
                    X_tr[f"emb_svd_{i}"], X_te[f"emb_svd_{i}"] = s_tr[:, i], s_te[:, i]
            for k in emb_cfg.get("knn", []):
                om, os_, tm, ts = knn_target_feature(emb_tr, y, emb_te, folds, k=int(k))
                X_tr[f"knn{k}_mean"], X_te[f"knn{k}_mean"] = om, tm
                X_tr[f"knn{k}_std"], X_te[f"knn{k}_std"] = os_, ts
        except (ImportError, OSError) as e:
            msg = f"embeddings skipped ({type(e).__name__}: {e})"
            log.warning(msg)
            notes.append(msg)

    extra = feat_cfg.get("extra_oof_features") or {}
    for name, exp_id in extra.items():
        from src.utils import ARTIFACTS

        oof_p, test_p = ARTIFACTS / f"oof_{exp_id}.npy", ARTIFACTS / f"test_{exp_id}.npy"
        if oof_p.exists() and test_p.exists():
            X_tr[name], X_te[name] = np.load(oof_p), np.load(test_p)
        else:
            msg = f"extra OOF feature '{name}' skipped (artifacts for {exp_id} missing)"
            log.warning(msg)
            notes.append(msg)

    return X_tr, X_te, notes
