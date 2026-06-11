"""Fold-safe OOF meta-features: TF-IDF linear models, target encoding, kNN-on-embeddings.

Every function fits its transform/model inside each training fold only and
returns (oof_values_for_train, values_for_test). Test values come from a model
fit on the full train set (standard stacking practice with fixed folds).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.data import N_FOLDS
from src.utils import DATA_PROCESSED, SEED, get_logger

log = get_logger()


def _cache_path(name: str) -> "Path":
    from pathlib import Path

    return DATA_PROCESSED / f"meta_{name}.npz"


def _cached(name: str, key: str):
    p = _cache_path(name)
    if p.exists():
        z = np.load(p, allow_pickle=False)
        if "key" in z and str(z["key"]) == key:
            return z["oof"], z["test"]
    return None


def _save_cache(name: str, key: str, oof: np.ndarray, test: np.ndarray) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez(_cache_path(name), key=np.array(key), oof=oof, test=test)


def _fold_key(folds: np.ndarray, extra: str = "") -> str:
    return hashlib.md5(folds.tobytes() + extra.encode()).hexdigest()[:12]


def ridge_tfidf_oof(
    texts_train: pd.Series,
    y: np.ndarray,
    texts_test: pd.Series,
    folds: np.ndarray,
    alpha: float = 1.0,
    cache_name: str | None = "ridge_tfidf",
) -> tuple[np.ndarray, np.ndarray]:
    """OOF Ridge prediction on word(1-3)+char_wb(3-5) TF-IDF (text-only RMSE ≈ 12.26)."""
    from sklearn.linear_model import Ridge

    from src.features.text_classic import make_tfidf_vectorizers

    key = _fold_key(folds, f"ridge{alpha}{len(texts_train)}")
    if cache_name and (hit := _cached(cache_name, key)) is not None:
        return hit
    oof = np.zeros(len(texts_train))
    for fold in range(N_FOLDS):
        tr, va = folds != fold, folds == fold
        vec = make_tfidf_vectorizers()
        Xtr = vec.fit_transform(texts_train[tr])
        model = Ridge(alpha=alpha, random_state=SEED).fit(Xtr, y[tr])
        oof[va] = model.predict(vec.transform(texts_train[va]))
        log.info(f"  ridge_tfidf fold {fold} done")
    vec = make_tfidf_vectorizers()
    model = Ridge(alpha=alpha, random_state=SEED).fit(vec.fit_transform(texts_train), y)
    test = model.predict(vec.transform(texts_test))
    if cache_name:
        _save_cache(cache_name, key, oof, test)
    return oof, test


def logit_tfidf_ceiling_oof(
    texts_train: pd.Series,
    y: np.ndarray,
    texts_test: pd.Series,
    folds: np.ndarray,
    cache_name: str | None = "logit_tfidf_ceiling",
) -> tuple[np.ndarray, np.ndarray]:
    """OOF P(y==100) from logistic regression on TF-IDF (text-only AUC ≈ 0.904)."""
    from sklearn.linear_model import LogisticRegression

    from src.features.text_classic import make_tfidf_vectorizers

    key = _fold_key(folds, f"logit{len(texts_train)}")
    if cache_name and (hit := _cached(cache_name, key)) is not None:
        return hit
    target = (np.asarray(y) == 100).astype(int)
    oof = np.zeros(len(texts_train))
    for fold in range(N_FOLDS):
        tr, va = folds != fold, folds == fold
        vec = make_tfidf_vectorizers()
        Xtr = vec.fit_transform(texts_train[tr])
        model = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit(Xtr, target[tr])
        oof[va] = model.predict_proba(vec.transform(texts_train[va]))[:, 1]
        log.info(f"  logit_tfidf fold {fold} done")
    vec = make_tfidf_vectorizers()
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit(
        vec.fit_transform(texts_train), target
    )
    test = model.predict_proba(vec.transform(texts_test))[:, 1]
    if cache_name:
        _save_cache(cache_name, key, oof, test)
    return oof, test


def target_encode_oof(
    cat_train: pd.Series,
    y: np.ndarray,
    cat_test: pd.Series,
    folds: np.ndarray,
    smoothing: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed OOF mean-target encoding for one categorical series."""
    y = np.asarray(y, dtype=float)
    prior = y.mean()
    oof = np.full(len(cat_train), prior)
    cat_train = cat_train.astype(str).reset_index(drop=True)
    cat_test = cat_test.astype(str).reset_index(drop=True)
    for fold in range(N_FOLDS):
        tr, va = folds != fold, folds == fold
        grp = pd.DataFrame({"c": cat_train[tr], "y": y[tr]}).groupby("c")["y"].agg(["mean", "count"])
        enc = (grp["mean"] * grp["count"] + prior * smoothing) / (grp["count"] + smoothing)
        oof[va] = cat_train[va].map(enc).fillna(prior).to_numpy()
    grp = pd.DataFrame({"c": cat_train, "y": y}).groupby("c")["y"].agg(["mean", "count"])
    enc = (grp["mean"] * grp["count"] + prior * smoothing) / (grp["count"] + smoothing)
    test = cat_test.map(enc).fillna(prior).to_numpy()
    return oof, test


def knn_target_feature(
    emb_train: np.ndarray,
    y: np.ndarray,
    emb_test: np.ndarray,
    folds: np.ndarray,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean/std target of the k nearest train neighbors by cosine, fold-safe.

    Returns (oof_mean, oof_std, test_mean, test_std).
    """
    from sklearn.neighbors import NearestNeighbors

    y = np.asarray(y, dtype=float)
    oof_mean = np.zeros(len(emb_train))
    oof_std = np.zeros(len(emb_train))
    for fold in range(N_FOLDS):
        tr, va = np.where(folds != fold)[0], np.where(folds == fold)[0]
        nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(emb_train[tr])
        _, idx = nn.kneighbors(emb_train[va])
        neigh_y = y[tr][idx]
        oof_mean[va] = neigh_y.mean(axis=1)
        oof_std[va] = neigh_y.std(axis=1)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(emb_train)
    _, idx = nn.kneighbors(emb_test)
    neigh_y = y[idx]
    return oof_mean, oof_std, neigh_y.mean(axis=1), neigh_y.std(axis=1)
