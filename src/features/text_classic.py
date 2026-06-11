"""Classic text features: stats + Turkish lexicon counts (EDA fact #6)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

POSITIVE_WORDS = [
    "mükemmel", "olağanüstü", "etkileyici", "güçlü", "başarılı", "donanımlı", "dikkat çekici",
]
NEGATIVE_WORDS = [
    "eksik", "sınırlı", "zorluk", "engelliyor", "yetersiz", "geliştirmesi gereken",
]
CONTRAST_MARKERS = ["ancak", "fakat", "ama", "ne yazık ki"]

# Keywords that indicate the text talks about the student's own target role.
ROLE_KEYWORDS: dict[str, list[str]] = {
    "AI Engineer": ["yapay zeka", "ai"],
    "Backend Developer": ["backend", "arka uç"],
    "Cloud Engineer": ["bulut", "cloud"],
    "Cybersecurity Analyst": ["siber", "güvenlik"],
    "Data Analyst": ["veri analiz", "analiz"],
    "Data Scientist": ["veri bilim", "makine öğren"],
    "DevOps Engineer": ["devops"],
    "Frontend Developer": ["frontend", "ön yüz", "arayüz"],
    "MLOps Engineer": ["mlops"],
    "Product Analyst": ["ürün"],
    "Software Developer": ["yazılım"],
}


def _count_any(text: str, words: list[str]) -> int:
    return sum(text.count(w) for w in words)


def _first_pos(text: str, words: list[str]) -> float:
    """Relative position (0–1) of the first occurrence of any word; 1.0 if absent."""
    idxs = [text.find(w) for w in words if w in text]
    return min(idxs) / max(len(text), 1) if idxs else 1.0


def build_text_classic(
    texts: pd.Series, target_roles: pd.Series
) -> tuple[pd.DataFrame, list[str]]:
    """Per-row text statistics and lexicon features. Stateless → leakage-free."""
    t = texts.fillna("").str.lower()
    out = pd.DataFrame(index=texts.index)
    out["txt_char_count"] = t.str.len()
    out["txt_word_count"] = t.str.split().str.len()
    out["txt_sent_count"] = t.apply(lambda s: max(len(re.findall(r"[.!?]+", s)), 1))
    out["txt_pos_count"] = t.apply(lambda s: _count_any(s, POSITIVE_WORDS))
    out["txt_neg_count"] = t.apply(lambda s: _count_any(s, NEGATIVE_WORDS))
    out["txt_contrast_count"] = t.apply(lambda s: _count_any(s, CONTRAST_MARKERS))
    out["txt_first_contrast_pos"] = t.apply(lambda s: _first_pos(s, CONTRAST_MARKERS))
    out["txt_pos_minus_neg"] = out["txt_pos_count"] - out["txt_neg_count"]
    for w in POSITIVE_WORDS + NEGATIVE_WORDS:
        out[f"txt_has_{w.replace(' ', '_')}"] = t.str.contains(w, regex=False).astype(int)
    out["txt_mentions_role"] = [
        int(any(kw in txt for kw in ROLE_KEYWORDS.get(role, [])))
        for txt, role in zip(t, target_roles)
    ]
    return out, list(out.columns)


def make_tfidf_vectorizers(min_df: int = 2):
    """Word(1-3) + char_wb(3-5) sublinear TF-IDF union (the proven recipe)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=min_df)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=min_df
                ),
            ),
        ]
    )
