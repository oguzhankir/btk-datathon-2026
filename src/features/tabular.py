"""Tabular feature engineering (§7.1): NA flags, role-skill alignment, ratios, time."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import CAT_COLS, ID_COL, SKILL_COLS, TARGET, TEXT_COL

# Role -> ordered list of most-relevant skill columns (domain sense, backed by EDA fact #5).
ROLE_SKILL_MAP: dict[str, list[str]] = {
    "AI Engineer": ["machine_learning_score", "problem_solving_score", "coding_score"],
    "Backend Developer": ["backend_score", "devops_score", "cloud_score", "sql_score"],
    "Cloud Engineer": ["cloud_score", "devops_score", "backend_score"],
    "Cybersecurity Analyst": ["problem_solving_score", "devops_score", "coding_score"],
    "Data Analyst": ["sql_score", "problem_solving_score", "machine_learning_score"],
    "Data Scientist": ["problem_solving_score", "machine_learning_score", "sql_score"],
    "DevOps Engineer": ["devops_score", "cloud_score", "backend_score"],
    "Frontend Developer": ["frontend_score", "coding_score", "problem_solving_score"],
    "MLOps Engineer": ["problem_solving_score", "machine_learning_score", "devops_score", "cloud_score"],
    "Product Analyst": ["sql_score", "problem_solving_score", "communication_score"],
    "Software Developer": ["coding_score", "data_structures_score", "problem_solving_score", "backend_score"],
}

# Hand-picked pairwise interactions among the strongest features (EDA fact #4).
INTERACTION_PAIRS: list[tuple[str, str]] = [
    ("project_quality_score", "technical_interview_score"),
    ("project_quality_score", "portfolio_score"),
    ("project_quality_score", "cv_quality_score"),
    ("technical_interview_score", "hr_interview_score"),
    ("technical_interview_score", "communication_score"),
    ("project_quality_score", "linkedin_profile_score"),
    ("portfolio_score", "github_repo_count"),
    ("cgpa", "project_quality_score"),
    ("english_exam_score", "communication_score"),
    ("attendance_rate", "cgpa"),
]


def _base_columns(df: pd.DataFrame) -> list[str]:
    """All raw model columns: numerics + categoricals (no id/target/text)."""
    drop = {ID_COL, TARGET, TEXT_COL}
    return [c for c in df.columns if c not in drop]


def build_tabular(
    train: pd.DataFrame, test: pd.DataFrame, fe: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build the tabular feature frame for train and test.

    Returns (X_train, X_test, feature_names). Categoricals are pandas
    `category` dtype with categories shared across train/test. With fe=False
    only raw columns are returned (exp001 baseline).
    """
    n_train = len(train)
    df = pd.concat([train, test], ignore_index=True, sort=False)
    feats = _base_columns(test)  # test has every column except the target

    if fe:
        # --- NA indicators (missingness is informative — EDA fact #3) ---
        na_cols = [c for c in feats if df[c].isna().any()]
        for c in na_cols:
            df[f"na_{c}"] = df[c].isna().astype(int)
        df["na_count"] = df[na_cols].isna().sum(axis=1)
        feats += [f"na_{c}" for c in na_cols] + ["na_count"]

        # --- skill block aggregates ---
        sk = df[SKILL_COLS]
        df["skill_mean"] = sk.mean(axis=1)
        df["skill_max"] = sk.max(axis=1)
        df["skill_min"] = sk.min(axis=1)
        df["skill_std"] = sk.std(axis=1)
        top2 = np.sort(sk.to_numpy(), axis=1)[:, -2:]
        df["skill_top2_mean"] = top2.mean(axis=1)
        df["skill_specialization_gap"] = df["skill_max"] - df["skill_mean"]
        feats += ["skill_mean", "skill_max", "skill_min", "skill_std",
                  "skill_top2_mean", "skill_specialization_gap"]

        # --- role–skill alignment ---
        skill_ranks = sk.rank(axis=1, ascending=False)
        matched = np.full(len(df), np.nan)
        matched_rank = np.full(len(df), np.nan)
        for role, cols in ROLE_SKILL_MAP.items():
            mask = (df["target_role"] == role).to_numpy()
            cols_in = [c for c in cols if c in SKILL_COLS]
            matched[mask] = df.loc[mask, cols_in].mean(axis=1)
            matched_rank[mask] = skill_ranks.loc[mask, cols_in[0]]
        df["matched_skill"] = matched
        df["matched_skill_rank"] = matched_rank
        df["matched_minus_mean"] = df["matched_skill"] - df["skill_mean"]
        feats += ["matched_skill", "matched_skill_rank", "matched_minus_mean"]

        # --- ratios & combos ---
        df["interview_conversion"] = df["interviews_attended"] / (df["applications_sent"] + 1)
        df["award_rate"] = df["hackathon_awards"] / (df["hackathon_count"] + 1)
        df["intern_months_per"] = df["internship_duration_months"] / (df["internship_count"] + 1)
        df["stars_total"] = df["github_repo_count"] * df["github_avg_stars"]
        df["oss_per_repo"] = df["open_source_contribution_count"] / (df["github_repo_count"] + 1)
        df["experience_total"] = (
            df["real_client_project_count"] + df["freelance_project_count"] + df["internship_count"]
        )
        feats += ["interview_conversion", "award_rate", "intern_months_per",
                  "stars_total", "oss_per_repo", "experience_total"]

        # --- time ---
        df["years_since_grad"] = df["application_year"] - df["graduation_year"]
        df["grad_age"] = df["age"] - df["years_since_grad"]
        feats += ["years_since_grad", "grad_age"]

        # --- hand-picked products ---
        for a, b in INTERACTION_PAIRS:
            name = f"x_{a.replace('_score', '')}_{b.replace('_score', '')}"
            df[name] = df[a] * df[b]
            feats.append(name)

    # categorical dtype with shared categories
    for c in CAT_COLS:
        df[c] = df[c].astype("category")

    X = df[feats]
    return X.iloc[:n_train].reset_index(drop=True), X.iloc[n_train:].reset_index(drop=True), feats
