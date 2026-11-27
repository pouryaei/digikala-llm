"""
ML service: predict recommendation_status (recommended / not_recommended / no_idea)
from the text of a review. Model: TF-IDF (char 2-5 grams) + Logistic Regression (M2).

    from src.ml_service import predict_recommendation
    predict_recommendation("کیفیت عالیه، پیشنهاد می‌کنم")
    predict_recommendation("خوب بود", title="راضیم")     # title is optional

The input is normalized EXACTLY like M1 normalized the training text
(normalize_persian_text below is copied from M1's preprocessing), then the
saved sklearn Pipeline is applied to the raw string.

Edge cases never reach the model. They return valid=False, confidence=0.0 and a
`reason` (empty_text / symbols_or_emoji_only / digits_only / spam_repetition):
the label is INVALID_LABEL ("no_idea"; set it to None if you prefer no label).
"""
import html
import os
import re
import threading
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Text normalization (identical to M1)
# --------------------------------------------------------------------------- #
CHAR_MAP = str.maketrans("يىك٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "ییک01234567890123456789")
WEB_MARKUP = re.compile(r"https?://\S+|www\.\S+|</?[a-z][^>]*>", re.IGNORECASE)
REMOVE_CHARS = re.compile(
    r"[ـ\u064b-\u065f\u0670\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]"
)
NULL_TEXT = {"", "none", "nan", "null", "n/a", "-"}


def normalize_persian_text(value):
    if value is None:
        return None
    if "&" in value:
        value = html.unescape(value)
    value = unicodedata.normalize("NFKC", value).translate(CHAR_MAP)
    value = WEB_MARKUP.sub(" ", value)
    value = REMOVE_CHARS.sub("", value)
    value = " ".join(value.replace("\u200b", " ").split())
    return value if value.lower() not in NULL_TEXT else None


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
LABELS = ("recommended", "not_recommended", "no_idea")
INVALID_LABEL = "no_idea"        # label returned for invalid inputs (None = no label)
LOW_CONFIDENCE = 0.5             # below this, result["low_confidence"] is True
MAX_CHARS = 5000                 # longer texts are truncated, not rejected
SPAM_MIN_LEN = 6                 # min length for the "only 1-2 distinct characters" rule
SPAM_MIN_TOKENS = 5              # min tokens for the "one token repeated" rule
SPAM_TOKEN_SHARE = 0.8           # share of the most frequent token that marks spam

_LETTER = re.compile(r"[^\W\d_]")   # any letter (Persian/Latin), no digits/underscore
_ALNUM = re.compile(r"[^\W_]")      # any letter or digit


def _find_model_path() -> Path:
    env = os.getenv("DIGIKALA_MODEL_PATH")
    if env:
        return Path(env)
    candidates = ("models/sentiment_model.joblib", "m2_outputs/sentiment_model.joblib",
                  "m2_outputs/best_model.joblib")
    for parent in Path(__file__).resolve().parents:
        for rel in candidates:
            if (parent / rel).exists():
                return parent / rel
    return Path.cwd() / candidates[0]


MODEL_PATH = _find_model_path()
_model = None
_lock = threading.Lock()


def load_model():
    """Load the pipeline once (thread-safe)."""
    global _model
    with _lock:
        if _model is None:
            import joblib
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"{MODEL_PATH} not found. Put sentiment_model.joblib in models/ "
                    f"or set DIGIKALA_MODEL_PATH.")
            _model = joblib.load(MODEL_PATH)
        return _model


# --------------------------------------------------------------------------- #
# Input checks
# --------------------------------------------------------------------------- #
def _prepare(text, title=None):
    """Return (clean_text, None) or (None, reason)."""
    parts = []
    for p in (title, text):
        if isinstance(p, str):
            n = normalize_persian_text(p)
            if n:
                parts.append(n)
    clean = " ".join(parts)
    if not clean:
        return None, "empty_text"
    if not _ALNUM.search(clean):
        return None, "symbols_or_emoji_only"
    if not _LETTER.search(clean):
        return None, "digits_only"
    compact = clean.replace(" ", "")
    if len(compact) >= SPAM_MIN_LEN and len(set(compact)) <= 2:
        return None, "spam_repetition"
    tokens = clean.split()
    if len(tokens) >= SPAM_MIN_TOKENS:
        top = Counter(tokens).most_common(1)[0][1]
        if top / len(tokens) >= SPAM_TOKEN_SHARE:
            return None, "spam_repetition"
    return clean[:MAX_CHARS], None


def _invalid(reason: str) -> Dict:
    return {"label": INVALID_LABEL, "confidence": 0.0,
            "probabilities": {k: 0.0 for k in LABELS},
            "valid": False, "reason": reason, "low_confidence": True}


def _probabilities(model, texts: List[str]):
    import numpy as np
    classes = list(model.classes_)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(texts)
    else:  # e.g. LinearSVC: softmax over decision scores (not calibrated)
        s = model.decision_function(texts)
        e = np.exp(s - s.max(axis=1, keepdims=True))
        proba = e / e.sum(axis=1, keepdims=True)
    return classes, proba


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def predict_recommendation_batch(texts: List[str], titles: Optional[List[str]] = None) -> List[Dict]:
    """Predict many reviews at once (one model call for all valid inputs)."""
    titles = titles or [None] * len(texts)
    results: List[Optional[Dict]] = [None] * len(texts)
    valid_idx, valid_texts = [], []
    for i, (t, ti) in enumerate(zip(texts, titles)):
        clean, reason = _prepare(t, ti)
        if reason:
            results[i] = _invalid(reason)
        else:
            valid_idx.append(i)
            valid_texts.append(clean)
    if valid_texts:
        classes, proba = _probabilities(load_model(), valid_texts)
        for i, row in zip(valid_idx, proba):
            j = int(row.argmax())
            results[i] = {
                "label": classes[j], "confidence": round(float(row[j]), 4),
                "probabilities": {c: round(float(p), 4) for c, p in zip(classes, row)},
                "valid": True, "reason": None,
                "low_confidence": float(row[j]) < LOW_CONFIDENCE,
            }
    return results


def predict_recommendation(text, title=None) -> Dict:
    """Predict the recommendation status of ONE review.

    Returns {label, confidence, probabilities, valid, reason, low_confidence}.
    Never raises on bad input (None, numbers, empty, emoji-only, spam).
    """
    return predict_recommendation_batch([text], [title])[0]


if __name__ == "__main__":
    samples = ["کیفیت عالیه، واقعاً پیشنهاد می‌کنم", "افتضاح بود اصلا نخرید", "بد نیست، معمولیه",
               "", "   ", "😍😍😍", "12345", "ااااااااا", "خوب خوب خوب خوب خوب خوب", None]
    for s, r in zip(samples, predict_recommendation_batch(samples)):
        print(repr(s), "->", r["label"], r["confidence"], r["valid"], r["reason"])
