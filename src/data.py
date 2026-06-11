"""Data loading and the single fixed CV split used by every experiment."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.utils import DATA_PROCESSED, DATA_RAW, SEED

ID_COL = "student_id"
TARGET = "career_success_score"
TEXT_COL = "mentor_feedback_text"
CAT_COLS = [
    "department",
    "university_tier",
    "target_role",
    "hobby",
    "preferred_social_media_platform",
]
SKILL_COLS = [
    "coding_score",
    "problem_solving_score",
    "data_structures_score",
    "sql_score",
    "machine_learning_score",
    "backend_score",
    "frontend_score",
    "cloud_score",
    "devops_score",
]
FOLDS_PATH = DATA_PROCESSED / "folds.csv"
N_FOLDS = 5


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test and sample submission from data/raw."""
    train = pd.read_csv(DATA_RAW / "train.csv")
    test = pd.read_csv(DATA_RAW / "test_x.csv")
    sub = pd.read_csv(DATA_RAW / "sample_submission.csv")
    return train, test, sub


def make_folds(overwrite: bool = False) -> pd.DataFrame:
    """Create the fixed 5-fold split, stratified on target deciles.

    Written ONCE to data/processed/folds.csv; every experiment loads that file.
    """
    if FOLDS_PATH.exists() and not overwrite:
        return load_folds()
    train, _, _ = load_raw()
    y = train[TARGET]
    bins = pd.qcut(y, 10, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.full(len(train), -1, dtype=int)
    for i, (_, val_idx) in enumerate(skf.split(train, bins)):
        fold[val_idx] = i
    df = pd.DataFrame({ID_COL: train[ID_COL], "fold": fold})
    FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FOLDS_PATH, index=False)
    return df


def load_folds() -> pd.DataFrame:
    """Load the fixed fold assignment (student_id, fold)."""
    if not FOLDS_PATH.exists():
        return make_folds()
    return pd.read_csv(FOLDS_PATH)


def fold_array(train: pd.DataFrame) -> np.ndarray:
    """Fold id per train row, aligned to the given train dataframe order."""
    folds = load_folds().set_index(ID_COL)["fold"]
    return folds.loc[train[ID_COL]].to_numpy()
