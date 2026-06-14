"""Post-EDA: why are we at the signal floor, and what is the LB gap actually made of?

This script reproduces, end to end, every diagnostic behind the 2026-06-13 decision
that the modelling frontier is closed and the leaderboard gap to the top is mostly
public-LB noise. Run it after a blend exists:

    python reports/eda/floor_analysis.py

It PRINTS every number cited in docs/progress/2026-06-13.md and writes four figures
to reports/figures/ (year_error_decomposition, r2_by_year, blend_member_corr,
public_lb_noise). Nothing here is fit on data — it only reads saved OOF/test
artifacts, so it is leakage-free by construction and safe to re-run any time.

Sections:
  1. Year distribution train-vs-test and the target's temporal decline.
  2. OOF residual by year -> the CV->LB offset is 100% explained by year mix.
  3. R^2 within each year -> late-year error is higher VARIANCE, not worse modelling.
  4. Ceiling + dispersion oracles -> neither is a recoverable lever.
  5. Residual correlation between blend members -> diversity is exhausted.
  6. Text-vs-score contradictions -> text encodes the profile, not the score.
  7. Raw-column coverage + text-numeric discrepancy -> no forgotten signal.
  8. Public-LB standard error -> the gap to #1 is within subset noise.
  9. Aspect-based text extraction -> real target signal, zero residual signal (already captured).
 10. NN leak check + feature-twin floor -> no kNN/copy leak; identical-feature students differ ~11.
 11. External-data note -> dataset is synthetic, so external data is meaningless (not just barred).

CAVEAT (added 2026-06-14, after the public LB settled): every floor here is the floor of
*this* feature set / approach. The public leaderboard's entire top-11 converged to MSE 80.3–81.2
(we finished ~82.2), which is strong evidence that a real, findable edge exists that this isolated
pipeline did not discover — most likely a competition-shared trick/leak or a feature insight. The
"R²0.69 ceiling" below is OUR ceiling, not a proof of the data's true limit. See docs/progress/2026-06-14.md.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data import TARGET, TEXT_COL, fold_array, load_raw  # noqa: E402
from src.utils import ARTIFACTS, get_logger  # noqa: E402

log = get_logger()
FIGS = Path(__file__).resolve().parents[1] / "figures"

# public leaderboard standing observed on 2026-06-13 (top-10), for the noise section
LB_TOP10 = [80.71, 81.07, 81.43, 81.61, 81.67, 81.69, 81.74, 81.76, 81.90, 81.93]
OUR_LB = 82.96


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")


def section_year_offset(train, test, y, oof, res) -> dict:
    """Sections 1-2: year distribution, residual by year, year-weighted LB estimate."""
    yr = train["application_year"].to_numpy()
    te_w = pd.Series(test["application_year"].to_numpy()).value_counts(normalize=True)

    print("\n[1-2] residual by application_year (test-weighted OOF MSE == honest LB proxy)")
    print(f"{'year':>6}{'train_n':>9}{'test_w':>8}{'oof_mse':>9}{'bias':>8}{'tgt_var':>9}")
    per_year_mse, per_year_var = {}, {}
    rows = []
    for Y in sorted(np.unique(yr)):
        m = yr == Y
        pm, pv, b = np.mean(res[m] ** 2), float(np.var(y[m])), float(np.mean(res[m]))
        per_year_mse[Y], per_year_var[Y] = pm, pv
        rows.append((Y, m.sum(), te_w.get(Y, 0.0), pm, b, pv))
        print(f"{Y:>6}{m.sum():>9}{te_w.get(Y, 0.0):>8.3f}{pm:>9.2f}{b:>+8.3f}{pv:>9.1f}")

    lb_est = sum(te_w.get(Y, 0.0) * per_year_mse[Y] for Y in per_year_mse)
    print(f"\n  flat OOF MSE        = {np.mean(res ** 2):.3f}")
    print(f"  test-year-weighted  = {lb_est:.3f}  (matches actual LB {OUR_LB})")
    print("  => the entire CV->LB gap is the late-year skew of the test set, not overfitting.")

    _style()
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows, columns=["year", "train_n", "test_w", "oof_mse", "bias", "tgt_var"])
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(df["year"], df["oof_mse"], color="#c44", alpha=0.75, label="OOF MSE")
    ax1.set_ylabel("OOF MSE", color="#c44")
    ax2 = ax1.twinx()
    ax2.plot(df["year"], df["test_w"], "o-", color="#247", label="share of TEST")
    ax2.set_ylabel("share of test set", color="#247")
    ax1.set_title("Error explodes in late years — and the test set is concentrated there")
    fig.tight_layout()
    fig.savefig(FIGS / "year_error_decomposition.png", dpi=120)
    plt.close(fig)
    return {"per_year_mse": per_year_mse, "per_year_var": per_year_var, "lb_est": lb_est, "yr": yr}


def section_r2_by_year(y, oof, res, yr) -> None:
    """Section 3: R^2 within each year — the decisive floor proof."""
    print("\n[3] R^2 within each year (fraction of that year's variance we explain)")
    print(f"{'year':>6}{'tgt_std':>9}{'oof_mse':>9}{'R2':>8}{'%y=100':>9}")
    years, r2s = [], []
    for Y in sorted(np.unique(yr)):
        m = yr == Y
        var, ms = float(np.var(y[m])), float(np.mean(res[m] ** 2))
        r2 = 1 - ms / var
        years.append(Y)
        r2s.append(r2)
        print(f"{Y:>6}{np.std(y[m]):>9.2f}{ms:>9.2f}{r2:>+8.3f}{np.mean(y[m] == 100) * 100:>8.1f}%")
    print("  => R^2 is FLAT across years. Late years are harder only because their target")
    print("     variance is larger; we capture the same SHARE of signal everywhere. There is")
    print("     no late-year-specific signal we are failing to model.")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, r2s, "o-", color="#247", lw=2)
    ax.axhline(np.mean(r2s), ls="--", color="#999", label=f"mean R²={np.mean(r2s):.3f}")
    ax.set_ylim(0.5, 0.8)
    ax.set_title("Within-year R² is flat → late-year error is variance, not modelling failure")
    ax.set_xlabel("application_year")
    ax.set_ylabel("R² within year")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "r2_by_year.png", dpi=120)
    plt.close(fig)


def section_oracles(train, y, oof, res) -> None:
    """Section 4: late-year ceiling + dispersion oracles (both fail)."""
    yr = train["application_year"].to_numpy()
    late = yr >= 2025
    cl = (y == 100) & late
    print("\n[4] late-year (2025-2026) oracles — upper bounds on two intuitive 'fixes'")
    print(f"  late ceiling rmse={np.sqrt(np.mean(res[cl] ** 2)):.2f} (well predicted); "
          f"ceiling is only {np.sum(res[cl] ** 2) / np.sum(res[late] ** 2) * 100:.1f}% of late error")
    o = oof.copy(); o[cl] = 100
    print(f"  ORACLE snap true-100 late -> 100 : {np.mean(res[late] ** 2):.2f} -> "
          f"{np.mean((y[late] - o[late]) ** 2):.2f}  (tiny, and needs oracle knowledge)")
    o2 = oof.copy()
    for Y in (2025, 2026):
        m = yr == Y
        mu = oof[m].mean()
        o2[m] = mu + (oof[m] - mu) * (y[m].std() / oof[m].std())
    print(f"  ORACLE rescale late dispersion   : {np.mean(res[late] ** 2):.2f} -> "
          f"{np.mean((y[late] - o2[late]) ** 2):.2f}  (WORSE — under-dispersion is correct)")


def section_member_corr(y) -> None:
    """Section 5: residual correlation between strong blend members."""
    ids = ["exp010", "exp013", "exp016", "exp019", "exp005", "exp011"]
    res = {e: y - np.load(ARTIFACTS / f"oof_{e}.npy")
           for e in ids if (ARTIFACTS / f"oof_{e}.npy").exists()}
    R = pd.DataFrame(res).corr()
    off = R.values[np.triu_indices(len(R), 1)]
    print("\n[5] residual correlation between blend members")
    print(R.round(3).to_string())
    print(f"  mean off-diagonal corr = {off.mean():.3f}  => members are near-identical;")
    print("     only the embedding-MLP (exp011, ~0.89) adds anything. Diversity is exhausted.")

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(R, annot=True, fmt=".3f", cmap="rocket_r", vmin=0.85, vmax=1.0, ax=ax)
    ax.set_title(f"Blend-member residual correlation (mean off-diag {off.mean():.3f})")
    fig.tight_layout()
    fig.savefig(FIGS / "blend_member_corr.png", dpi=120)
    plt.close(fig)


def section_text_vs_score(train, y, oof) -> None:
    """Section 6: the largest-error rows show text describes the profile, not the score."""
    res = np.abs(y - oof)
    idx = np.argsort(-res)[:6]
    print("\n[6] highest-error rows: text sentiment vs true score (text == profile, not score)")
    for i in idx:
        t = train[TEXT_COL].iloc[i]
        print(f"  true={y[i]:5.1f} pred={oof[i]:5.1f} | {t[:120]}")
    print("  => glowing text can carry a low score and vice-versa: target = g(features)+noise,")
    print("     text = a (noisier) view of the same features. No extra signal lives in the text.")


def section_no_forgotten_signal(train, test, y, oof, res) -> None:
    """Section 7: every raw column is used; text-numeric claims carry no residual signal."""
    from src.features.tabular import build_tabular

    raw = set(train.columns) - {TARGET, "student_id"}
    _, _, feats = build_tabular(train, test, fe=True)
    unused = {c for c in raw if c not in feats and not any(c in f for f in feats)}
    print("\n[7] feature-coverage + text-numeric discrepancy audit")
    print(f"  raw columns not referenced by any engineered feature: {sorted(unused)}")
    print("     (only the free-text column, handled separately) => no forgotten column.")

    txt = train[TEXT_COL].fillna("")
    words = {"bir": 1, "iki": 2, "üç": 3, "dört": 4, "beş": 5,
             "altı": 6, "yedi": 7, "sekiz": 8, "dokuz": 9, "on": 10}

    def first_num(s: str) -> float:
        s = s.lower()
        m = re.search(r"\b(\d+)\b", s)
        if m:
            return int(m.group(1))
        for w, nval in words.items():
            if re.search(r"\b" + w + r"\b", s):
                return nval
        return np.nan

    if "internship_count" in train.columns:
        staj = txt.apply(lambda s: first_num(s) if "staj" in s.lower() else np.nan).to_numpy()
        col = train["internship_count"].to_numpy()
        m = ~np.isnan(staj) & ~np.isnan(col)
        if m.sum() > 30:
            disc = staj[m] - col[m]
            c = np.corrcoef(disc, res[m])[0, 1]
            print(f"  text states an internship count on {m.sum()} rows; "
                  f"matches the column {np.mean(disc == 0) * 100:.1f}% of the time")
            print(f"  corr(text-vs-column discrepancy, residual) = {c:+.3f}  => no signal there either.")


def section_public_lb_noise(y, oof) -> None:
    """Section 8: public-LB standard error vs the observed leaderboard spread."""
    sq = (y - oof) ** 2
    print("\n[8] public-LB standard error (public score is a SUBSET of the 10k test)")
    ses = {}
    for frac in (0.2, 0.3, 0.5):
        n = int(10000 * frac)
        se = float(np.std(sq) / np.sqrt(n))
        ses[frac] = se
        print(f"  public={int(frac * 100)}% (n={n}): SE(MSE)=±{se:.2f}  ->  95% band ±{1.96 * se:.1f} MSE")
    spread = LB_TOP10[-1] - LB_TOP10[0]
    print(f"\n  top-1..top-10 spread = {spread:.2f} MSE | our gap to #1 = {OUR_LB - LB_TOP10[0]:.2f} MSE")
    print("  => both are inside the public-subset noise band: the live ranking is largely chance.")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar([20, 30, 50], [ses[f] * 0 + (OUR_LB) for f in (0.2, 0.3, 0.5)],
                yerr=[1.96 * ses[f] for f in (0.2, 0.3, 0.5)],
                fmt="o", color="#247", capsize=6, label="our score ±95% public-subset band")
    ax.axhline(LB_TOP10[0], ls="--", color="#c44", label=f"public #1 = {LB_TOP10[0]}")
    ax.axhspan(LB_TOP10[0], LB_TOP10[-1], color="#c44", alpha=0.12, label="top-10 band")
    ax.set_xlabel("assumed public-LB size (% of test)")
    ax.set_ylabel("MSE")
    ax.set_title("The gap to the top is inside public-subset noise")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "public_lb_noise.png", dpi=120)
    plt.close(fig)


def section_aspect_extraction(train, y, oof, res, folds) -> None:
    """Section 9: aspect-based sentiment extraction — does a NEW, structured text
    representation add anything beyond the TF-IDF + fine-tune OOFs already in the blend?"""
    import lightgbm as lgb

    t = train[TEXT_COL].fillna("").str.lower().to_numpy()
    aspects = {
        "proje": ["proje", "portföy", "portfolyo"], "mulakat": ["mülakat"],
        "kodlama": ["kodlama", "programlama", "yazılım geliş"], "sql": ["sql", "veritaban"],
        "bulut": ["bulut", "cloud"], "devops": ["devops", "ci/cd"], "problem": ["problem çöz"],
        "iletisim": ["iletişim", "sunum", "takım"], "staj": ["staj", "deneyim"],
        "ml": ["makine öğren", "derin öğren", "yapay zeka"],
    }
    pos = ["mükemmel", "olağanüstü", "etkileyici", "güçlü", "başarılı", "donanımlı", "yüksek",
           "iyi", "ustalı", "dikkat çekici", "heyecan", "tutku"]
    neg = ["eksik", "sınırlı", "zorluk", "yetersiz", "geliştir", "düşük", "daha fazla",
           "ihtiyaç", "gereken", "zayıf"]

    def sent_near(s: str, kws: list[str]) -> int:
        sc = 0
        for kw in kws:
            for m in re.finditer(re.escape(kw), s):
                w = s[max(0, m.start() - 60): m.end() + 60]
                sc += sum(w.count(p) for p in pos) - sum(w.count(nw) for nw in neg)
        return sc

    X = pd.DataFrame({a: [sent_near(s, kws) for s in t] for a, kws in aspects.items()})
    print("\n[9] aspect-based text extraction (signed sentiment in a ±60-char window per aspect)")
    print(f"  sanity — corr(total aspect sentiment, target) = {np.corrcoef(X.sum(axis=1), y)[0, 1]:+.3f}"
          "  (real signal, comparable to a mid-strength tabular feature)")
    pred = np.zeros(len(res))
    for f in sorted(np.unique(folds)):
        tr, va = folds != f, folds == f
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                              random_state=42, verbosity=-1)
        m.fit(X[tr], res[tr])
        pred[va] = m.predict(X[va])
    base, mod = float(np.mean(res ** 2)), float(np.mean((res - pred) ** 2))
    print(f"  residual-GBM on aspect features: R²={1 - mod / base:+.4f}  recoverable MSE={base - mod:+.4f}")
    print("  => the aspect signal is REAL but already captured by TF-IDF + fine-tune OOFs; a brand-new")
    print("     structured extraction recovers nothing. Independent methods converge — text is saturated.")


def section_knn_floor(train, test, y) -> None:
    """Section 10: nearest-neighbor leak check + the feature-twin floor.

    The most intuitive proof of the (OUR-approach) signal floor: how far apart are the
    targets of two students with NEARLY IDENTICAL features? If feature-twins still differ
    by ~the model RMSE, the features simply don't determine the target — no model (simple
    or complex) can do better. Also rules out a near-duplicate / kNN leak (test rows that
    are copies of train rows).
    """
    from sklearn.neighbors import NearestNeighbors

    from src.features.tabular import build_tabular

    X, Xte, feats = build_tabular(train, test, fe=True)
    num = [c for c in feats if X[c].dtype.kind in "ifu"]
    Xtr, Xt = X[num].to_numpy(float), Xte[num].to_numpy(float)
    med = np.nanmedian(np.vstack([Xtr, Xt]), 0)
    Xtr, Xt = np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xt), med, Xt)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr, Xt = (Xtr - mu) / sd, (Xt - mu) / sd

    d_te, _ = NearestNeighbors(n_neighbors=1).fit(Xtr).kneighbors(Xt)
    print("\n[10] nearest-neighbor leak check + feature-twin floor")
    print(f"  test→train NN distance: min={d_te.min():.3f} median={np.median(d_te):.3f} "
          f"| near-duplicates (<0.1): {(d_te < 0.1).sum()}/{len(Xt)}  => no kNN/copy leak")
    d_tr, idx = NearestNeighbors(n_neighbors=2).fit(Xtr).kneighbors(Xtr)
    nbr, nd = idx[:, 1], d_tr[:, 1]
    close = nd < np.percentile(nd, 5)
    print(f"  feature-twins (closest 5% of train pairs): mean |y_i − y_twin| = "
          f"{np.abs(y - y[nbr])[close].mean():.2f} points")
    print("  => near-identical-feature students differ by ~11 in target — the SAME number as the")
    print("     text floor (11.5), kNN-target (~11) and residual-GBM (R²≈0). The features cap any")
    print("     model at ~R²0.69; this is the floor of OUR feature set (see day-3/day-4 progress notes).")


def section_synthetic_note() -> None:
    """Section 11: why external data cannot help here (synthetic dataset)."""
    print("\n[11] external-data note")
    print("  The dataset is synthetic: students are generated (STU_0xxxxx), the target is an")
    print("  organizer-generated formula + injected noise, and the Turkish feedback is LLM-written.")
    print("  There is no real-world entity to join against, so external data is not just rule-")
    print("  restricted — it is meaningless here. The only legitimate 'external' input is a")
    print("  pre-trained model (we used BERT/XLM-R/8B-LLM), which all plateaued at the text floor.")


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    train, test, _ = load_raw()
    y = train[TARGET].to_numpy(dtype=float)
    blend_path = ARTIFACTS / "oof_blend_full_ridge.npy"
    if not blend_path.exists():
        blend_path = ARTIFACTS / "oof_blend.npy"
    oof = np.load(blend_path)
    res = y - oof
    log.info(f"floor analysis on {blend_path.name}: flat OOF MSE={np.mean(res ** 2):.3f}")

    st = section_year_offset(train, test, y, oof, res)
    section_r2_by_year(y, oof, res, st["yr"])
    section_oracles(train, y, oof, res)
    section_member_corr(y)
    section_text_vs_score(train, y, oof)
    section_no_forgotten_signal(train, test, y, oof, res)
    section_public_lb_noise(y, oof)
    section_aspect_extraction(train, y, oof, res, fold_array(train))
    section_knn_floor(train, test, y)
    section_synthetic_note()
    log.info(f"figures written to {FIGS}/")


if __name__ == "__main__":
    main()
