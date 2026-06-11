"""Deeper EDA (§6): interactions, role/year drift, ceiling profiling, text, residuals.

Produces all PNGs into reports/figures/. Run after exp001 (residual plot uses its OOF):
    python reports/eda/deep_eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data import SKILL_COLS, TARGET, TEXT_COL, fold_array, load_raw  # noqa: E402
from src.features.text_classic import POSITIVE_WORDS, NEGATIVE_WORDS, ROLE_KEYWORDS  # noqa: E402
from src.utils import ARTIFACTS, ROOT, SEED, get_logger, seed_everything  # noqa: E402

FIG = ROOT / "reports" / "figures"
log = get_logger()
sns.set_theme(style="whitegrid", context="talk", palette="deep")

TOP_FEATURES = [
    "project_quality_score", "technical_interview_score", "portfolio_score",
    "cv_quality_score", "linkedin_profile_score", "hr_interview_score",
    "communication_score", "coding_score", "problem_solving_score",
    "machine_learning_score", "application_year", "graduation_year",
    "github_repo_count", "internship_count", "applications_sent",
]


def _save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {name}")


def fig_target_distribution(train: pd.DataFrame) -> None:
    """Target histogram with the ceiling mass highlighted."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(train[TARGET], bins=60, ax=ax)
    ax.axvline(100, color="crimson", ls="--")
    n100 = int((train[TARGET] == 100).sum())
    ax.set_title(f"career_success_score: left-skewed with {n100} rows ({n100 / len(train):.1%}) exactly at 100")
    _save(fig, "target_distribution.png")


def fig_role_year_drift(train: pd.DataFrame) -> None:
    """Per-role target mean by year — is role mix driving temporal drift?"""
    g = (
        train.groupby(["application_year", "target_role"], observed=True)[TARGET]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(g, x="application_year", y=TARGET, hue="target_role", ax=ax, marker="o")
    overall = train.groupby("application_year")[TARGET].mean()
    ax.plot(overall.index, overall.values, color="black", lw=3, label="ALL")
    ax.legend(bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.set_title("Target mean per role per year: drift hits every role, not a mix effect")
    _save(fig, "role_year_drift.png")

    share = train.groupby("application_year")["target_role"].value_counts(normalize=True).unstack()
    fig, ax = plt.subplots(figsize=(12, 5))
    share.plot.area(ax=ax, legend=False)
    ax.set_title("Role mix by year (shares are stable → mix does not explain the drift)")
    ax.set_ylabel("share")
    _save(fig, "role_mix_by_year.png")


def fig_ceiling_profile(train: pd.DataFrame) -> None:
    """What distinguishes y==100 rows: standardized feature-mean gaps + text keywords."""
    num = [c for c in train.columns if train[c].dtype.kind in "ifu" and c != TARGET]
    is100 = train[TARGET] == 100
    gap = ((train.loc[is100, num].mean() - train.loc[~is100, num].mean()) / train[num].std()).sort_values()
    top = pd.concat([gap.head(7), gap.tail(7)])
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["crimson" if v < 0 else "seagreen" for v in top]
    ax.barh(top.index, top.values, color=colors)
    ax.set_title("Ceiling rows (y=100): standardized feature-mean gap vs rest")
    ax.set_xlabel("(mean_100 − mean_rest) / std")
    _save(fig, "ceiling_feature_profile.png")

    t = train[TEXT_COL].str.lower()
    rows = []
    for w in POSITIVE_WORDS + NEGATIVE_WORDS + ["ancak"]:
        has = t.str.contains(w, regex=False)
        rows.append({"keyword": w, "P(kw | y=100)": has[is100].mean(), "P(kw | y<100)": has[~is100].mean()})
    kw = pd.DataFrame(rows).set_index("keyword")
    fig, ax = plt.subplots(figsize=(10, 6))
    kw.plot.barh(ax=ax)
    ax.set_title("Text keyword prevalence: ceiling rows vs rest")
    _save(fig, "ceiling_text_keywords.png")


def fig_text_ridge_coefficients(train: pd.DataFrame) -> None:
    """Top word TF-IDF terms by Ridge coefficient (jury-friendly figure)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=5)
    X = vec.fit_transform(train[TEXT_COL].str.lower())
    model = Ridge(alpha=1.0).fit(X, train[TARGET])
    coefs = pd.Series(model.coef_, index=vec.get_feature_names_out())
    top = pd.concat([coefs.nsmallest(15).sort_values(), coefs.nlargest(15).sort_values()])
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(top.index, top.values, color=["crimson" if v < 0 else "seagreen" for v in top])
    ax.set_title("mentor_feedback_text verbalizes the score:\ntop ±15 TF-IDF terms by Ridge coefficient")
    _save(fig, "text_ridge_coefficients.png")


def fig_text_role_mention(train: pd.DataFrame) -> None:
    """Does mentioning the student's own target role in the text matter?"""
    t = train[TEXT_COL].str.lower()
    mentions = np.array(
        [any(kw in txt for kw in ROLE_KEYWORDS.get(role, [])) for txt, role in zip(t, train["target_role"])]
    )
    sents = t.str.count(r"[.!?]+").clip(lower=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x=mentions, y=train[TARGET], ax=axes[0])
    axes[0].set_title(
        f"Own-role mentioned in text ({mentions.mean():.0%} of rows)\n"
        f"mean target: {train[TARGET][mentions].mean():.1f} vs {train[TARGET][~mentions].mean():.1f}"
    )
    axes[0].set_xlabel("text mentions target_role")
    sns.histplot(sents, discrete=True, ax=axes[1])
    axes[1].set_title("Sentences per feedback text")
    _save(fig, "text_role_mention_and_sentences.png")


def fig_residuals(train: pd.DataFrame) -> None:
    """Residual analysis of the exp001 baseline OOF (where is the model weakest?)."""
    oof_path = ARTIFACTS / "oof_exp001.npy"
    if not oof_path.exists():
        log.warning("oof_exp001.npy missing — run exp001 first; skipping residual figures")
        return
    res = train[TARGET].to_numpy() - np.load(oof_path)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(np.load(oof_path), res, s=4, alpha=0.3)
    axes[0].axhline(0, color="crimson", ls="--")
    axes[0].set(xlabel="OOF prediction", ylabel="residual", title="Residual vs prediction")
    sns.boxplot(x=train["application_year"], y=res, ax=axes[1], color="steelblue")
    axes[1].set(title="Residual vs application_year", ylabel="")
    sns.boxplot(y=train["target_role"], x=res, ax=axes[2], color="steelblue")
    axes[2].set(title="Residual vs role", xlabel="residual", ylabel="")
    fig.suptitle("exp001 baseline residuals: errors concentrate at the ceiling and late years", y=1.02)
    _save(fig, "baseline_residuals.png")


def fig_train_test_overlays(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Train-vs-test overlay distributions for the top-15 features (fact #2 visually)."""
    fig, axes = plt.subplots(5, 3, figsize=(16, 18))
    for ax, col in zip(axes.ravel(), TOP_FEATURES):
        sns.kdeplot(train[col].dropna(), ax=ax, label="train", fill=True, alpha=0.3)
        sns.kdeplot(test[col].dropna(), ax=ax, label="test", fill=True, alpha=0.3)
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
        ax.legend(fontsize=8)
    fig.suptitle("Train vs test distributions: only year features shift", y=1.005)
    _save(fig, "train_test_overlays.png")


def fig_shap_interactions(train: pd.DataFrame) -> None:
    """SHAP interaction discovery on a strong LGBM + 2D PDP heatmaps for top-5 pairs."""
    import lightgbm as lgb
    import shap

    from src.features.tabular import build_tabular

    _, test, _ = load_raw()
    X, _, _ = build_tabular(train, test, fe=True)
    # SHAP's tree explainer needs an all-numeric matrix → category codes
    X = X.copy()
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].cat.codes
    y = train[TARGET].to_numpy()
    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=63, random_state=SEED, verbosity=-1
    ).fit(X, y)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), 1500, replace=False)
    Xs = X.iloc[idx]
    inter = shap.TreeExplainer(model).shap_interaction_values(Xs)
    strength = np.abs(inter).mean(axis=0)
    np.fill_diagonal(strength, 0)
    cols = X.columns.to_numpy()
    iu = np.triu_indices_from(strength, k=1)
    order = np.argsort(strength[iu])[::-1][:5]
    pairs = [(cols[iu[0][o]], cols[iu[1][o]]) for o in order]
    log.info(f"top SHAP interaction pairs: {pairs}")

    fig, axes = plt.subplots(1, 5, figsize=(26, 5))
    for ax, (a, b) in zip(axes, pairs):
        if isinstance(X[a].dtype, pd.CategoricalDtype) or isinstance(X[b].dtype, pd.CategoricalDtype):
            ax.set_title(f"{a} × {b} (categorical)", fontsize=10)
            continue
        ga = pd.qcut(Xs[a], 8, duplicates="drop")
        gb = pd.qcut(Xs[b], 8, duplicates="drop")
        grid = pd.Series(model.predict(Xs), index=Xs.index).groupby([ga, gb], observed=True).mean().unstack()
        sns.heatmap(grid, ax=ax, cmap="viridis", cbar=True)
        ax.set_title(f"{a}\n× {b}", fontsize=10)
        ax.set_xticklabels([]), ax.set_yticklabels([])
    fig.suptitle("Top-5 SHAP interaction pairs: mean prediction over quantile bins", y=1.03)
    _save(fig, "shap_interaction_heatmaps.png")


def main() -> None:
    seed_everything()
    train, test, _ = load_raw()
    fig_target_distribution(train)
    fig_role_year_drift(train)
    fig_ceiling_profile(train)
    fig_text_ridge_coefficients(train)
    fig_text_role_mention(train)
    fig_residuals(train)
    fig_train_test_overlays(train, test)
    fig_shap_interactions(train)
    log.info("deep EDA complete")


if __name__ == "__main__":
    main()
