"""Sentence-transformer embeddings with a disk cache (extracted once, reused)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import DATA_PROCESSED, get_logger

log = get_logger()

E5_PREFIX_MODELS = ("intfloat/multilingual-e5-large", "intfloat/multilingual-e5-base")


def model_slug(model_name: str) -> str:
    """Filesystem-safe slug for a HF model name."""
    return model_name.replace("/", "_").replace("-", "_")


def get_embeddings(
    texts_train: pd.Series,
    texts_test: pd.Series,
    model_name: str = "intfloat/multilingual-e5-large",
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed train+test texts, caching the stacked matrix to data/processed.

    Returns (emb_train, emb_test). Raises ImportError with a clear message if
    sentence-transformers is unavailable (callers may skip gracefully).
    """
    cache = DATA_PROCESSED / f"emb_{model_slug(model_name)}.npy"
    n_train = len(texts_train)
    if cache.exists():
        emb = np.load(cache)
        return emb[:n_train], emb[n_train:]

    from sentence_transformers import SentenceTransformer  # noqa: deferred heavy import

    from src.utils import detect_device

    texts = pd.concat([texts_train, texts_test], ignore_index=True).fillna("").tolist()
    if model_name in E5_PREFIX_MODELS:
        texts = [f"query: {t}" for t in texts]
    log.info(f"Embedding {len(texts)} texts with {model_name} ...")
    model = SentenceTransformer(model_name, device=detect_device())
    emb = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    return emb[:n_train], emb[n_train:]


def svd_features(
    emb_train: np.ndarray, emb_test: np.ndarray, n_components: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """TruncatedSVD of stacked embeddings (unsupervised → safe to fit on all rows)."""
    from sklearn.decomposition import TruncatedSVD

    from src.utils import SEED

    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    stacked = svd.fit_transform(np.vstack([emb_train, emb_test]))
    return stacked[: len(emb_train)], stacked[len(emb_train):]
