"""Feature engineering shared by train_model.py, evaluate.py, predict.py and app.py."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

TEXT_COLS = [
    "title", "location", "department", "company_profile",
    "description", "requirements", "benefits",
    "employment_type", "required_experience", "required_education",
    "industry", "function",
]

CAT_COLS = [
    "employment_type", "required_experience", "required_education",
    "industry", "function", "telecommuting", "has_company_logo", "has_questions",
]

LENGTH_COLS = ["company_profile", "description", "requirements", "benefits"]


def _fill(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("")
    return df


def make_text(df):
    df = _fill(df)
    parts = [
        df[c].fillna("").astype(str) if c in df.columns
        else pd.Series([""] * len(df), index=df.index)
        for c in TEXT_COLS
    ]
    return parts[0].str.cat(parts[1:], sep=" ")


def _numeric_features(df):
    """Hand-crafted numeric/flag features from structured fields."""
    df = _fill(df)
    n = len(df)
    feats = []

    def col(name):
        return df[name] if name in df.columns else pd.Series([np.nan] * n)

    # Binary flags
    for name in ["telecommuting", "has_company_logo", "has_questions"]:
        s = col(name)
        feats.append(pd.to_numeric(s, errors="coerce").fillna(0).clip(0, 1).values.reshape(-1, 1))

    # Presence flags: empty fields are common in scam postings
    for name in ["company_profile", "description", "requirements", "benefits", "location", "title"]:
        s = col(name).astype("string").fillna("").astype(str).str.strip()
        feats.append(((s == "") | (s.str.lower() == "nan")).astype(float).values.reshape(-1, 1))

    feats.append(col("salary_range").notna().astype(float).values.reshape(-1, 1))

    # Text-length features (log1p)
    for name in LENGTH_COLS + ["title", "location"]:
        s = col(name).astype("string").fillna("").astype(str).str.len().clip(lower=0)
        feats.append(np.log1p(s.values).reshape(-1, 1))

    return sparse.csr_matrix(np.hstack(feats))


_REGEX_PATTERNS = {
    # Direct-contact / off-platform recruitment is a strong scam signal
    "has_url": r"https?://|www\.",
    "has_email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "has_phone": r"\+?\d[\d\s\-()]{8,}\d",
    "has_messenger": r"whatsapp|telegram|signal|wechat|viber|skype|imo\b",
    # Money bait
    "has_currency_amount": r"[$€£₹]\s?\d|\b\d+\s?(?:usd|dollars?|rupees?|euros?)\b",
    "earn_per_period": r"\b(?:earn|make|get|paid)\b[^.]{0,40}\b(?:per|a|each|/)\s*(?:day|week|month|hour)\b",
    # Pressure / too-good-to-be-true language
    "urgency": r"act now|limited time|urgent|apply immediately|hurry|last chance|act fast",
    "no_interview": r"no interview|without (?:an? )?interview|no experience (?:is )?required",
    "guarantee": r"guarantee[ds]?|100%|risk[- ]free|easy money|passive income",
}


def _regex_features(text):
    """Suspicious-pattern flags computed on the concatenated posting text."""
    t = text.astype("string").fillna("").astype(str).str.lower()
    feats = [t.str.contains(pat, regex=True, na=False).astype(float)
             for pat in _REGEX_PATTERNS.values()]
    # Raw counts of currency amounts and exclamation marks
    feats.append(t.str.count(r"[$€£₹]\s?\d").clip(upper=5).astype(float))
    feats.append(t.str.count("!").clip(upper=10).astype(float))
    return sparse.csr_matrix(np.column_stack([f.values for f in feats]))


def _cat_matrix(df, encoder=None):
    df = _fill(df)
    cats = df.reindex(columns=CAT_COLS).fillna("").astype(str)
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=True)
        return encoder.fit_transform(cats), encoder
    return encoder.transform(cats), encoder


def _fit_vectorizer(texts, **kw):
    vec = TfidfVectorizer(**kw)
    return vec.fit(texts)


def build_features(df, bundle=None):
    """Return sparse feature matrix. If bundle is None, fit and return (X, bundle)."""
    text = make_text(df)

    if bundle is None:
        bundle = {}
        bundle["word_vec"] = _fit_vectorizer(
            text, stop_words="english", ngram_range=(1, 2),
            min_df=2, max_features=20000, sublinear_tf=True,
        )
        bundle["char_vec"] = _fit_vectorizer(
            text, analyzer="char_wb", ngram_range=(3, 5),
            min_df=3, max_features=20000, sublinear_tf=True,
        )
        _, bundle["cat_encoder"] = _cat_matrix(df)

    parts = [
        bundle["word_vec"].transform(text),
        bundle["char_vec"].transform(text),
        _cat_matrix(df, bundle["cat_encoder"])[0],
        _numeric_features(df),
        _regex_features(text),
    ]
    X = sparse.hstack(parts, format="csr")

    if bundle.get("_fitted") is None:
        pass
    return X, bundle


def fit_bundle(df):
    X, bundle = build_features(df, None)
    bundle["_fitted"] = True
    return X, bundle


def transform(df, bundle):
    X, _ = build_features(df, bundle)
    return X
