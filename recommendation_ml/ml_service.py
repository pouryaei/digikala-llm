"""
ML service: predict recommendation_status (recommended / not_recommended / no_idea)
from the text of a review. Default model: validated adapted Logistic Regression head over frozen character
TF-IDF. The original, tuned LR and XGBoost remain available for comparison.

    from src.ml_service import predict_recommendation
    predict_recommendation("کیفیت عالیه، پیشنهاد می‌کنم")
    predict_recommendation("خوب بود", title="راضیم")     # title is optional

The input follows the EDA's text_title_body feature: normalize the title and
body separately, remove identical fields and join with " . ". The saved
sklearn pipeline then receives that same representation.

Edge cases never reach the model. They return valid=False, confidence=0.0 and a
`reason` (empty_text / symbols_or_emoji_only / digits_only / spam_repetition):
the label is INVALID_LABEL ("no_idea"; set it to None if you prefer no label).
"""
import html
import json
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


def compose_model_text(text, title=None):
    """The EDA training feature: distinct normalized title and body."""
    parts = [normalize_persian_text(p) for p in (title, text) if isinstance(p, str)]
    return " . ".join(dict.fromkeys(p for p in parts if p))


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


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "m2_outputs").is_dir():
            return parent
    return Path(__file__).resolve().parent


PROJECT_ROOT = _project_root()
LEGACY_MODEL_PATH = PROJECT_ROOT / "m2_outputs" / "best_model.joblib"
XGBOOST_REPORT_PATH = PROJECT_ROOT / "m2_outputs" / "xgboost" / "report.json"
ADAPTED_MODEL_PATH = PROJECT_ROOT / "bonus_artifacts" / "adapted_head.npz"


def _find_model_path() -> Path:
    # An explicit caller override is still supported, including historical replay.
    env = os.getenv("DIGIKALA_MODEL_PATH")
    if env:
        return Path(env)
    active = PROJECT_ROOT / "m2_outputs" / "active_model.json"
    if active.is_file():
        choice = json.loads(active.read_text(encoding="utf-8"))
        if choice.get("kind") == "lr_adapted" and choice.get("artifact_path") == "bonus_artifacts/adapted_head.npz":
            return ADAPTED_MODEL_PATH
        if choice.get("kind") == "xgboost" and choice.get("artifact_path") == "xgboost/report.json":
            # Fail clearly if the configured active model is missing; never quietly fall back.
            return XGBOOST_REPORT_PATH
        if choice.get("kind") == "logistic_regression" and choice.get("artifact_path") == "best_model.joblib":
            return LEGACY_MODEL_PATH
        raise ValueError("Invalid m2_outputs/active_model.json")
    candidates = ("models/sentiment_model.joblib", "m2_outputs/sentiment_model.joblib",
                  "m2_outputs/best_model.joblib")
    for parent in Path(__file__).resolve().parents:
        for rel in candidates:
            if (parent / rel).exists():
                return parent / rel
    return Path.cwd() / candidates[0]


MODEL_PATH = _find_model_path()
_model = None
_legacy_model = None
_lock = threading.Lock()


def active_model_kind():
    if MODEL_PATH.resolve() == ADAPTED_MODEL_PATH.resolve():
        return "lr_adapted"
    if MODEL_PATH.name == "report.json" or MODEL_PATH.is_dir():
        return "xgboost"
    return "lr_original"


def is_xgboost_active() -> bool:
    return active_model_kind() == "xgboost"


def load_model():
    """Load the configured local model once (thread-safe)."""
    global _model
    with _lock:
        if _model is None:
            if active_model_kind() == "lr_adapted":
                from recommendation_ml.model_registry import load_candidate
                _model = load_candidate("lr_adapted")
            elif is_xgboost_active():
                from recommendation_ml.xgb_service import load_xgboost
                _model = load_xgboost(MODEL_PATH)
            else:
                import joblib
                if not MODEL_PATH.exists():
                    raise FileNotFoundError(f"Model missing: {MODEL_PATH}")
                _model = joblib.load(MODEL_PATH)
        return _model


def load_legacy_model():
    """Historical LR artifact, independent of the active-model configuration."""
    global _legacy_model
    with _lock:
        if _legacy_model is None:
            from recommendation_ml.model_registry import load_candidate
            _legacy_model = load_candidate("lr_original")
        return _legacy_model


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
    # Keep the original invalid-input checks above (they are not model features).
    # The actual vectorizer must see the same separator/deduplication as training.
    return " . ".join(dict.fromkeys(parts))[:MAX_CHARS], None


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
def _predict_batch(texts, titles, model_provider):
    titles = [None] * len(texts) if titles is None else titles
    if len(titles) != len(texts):
        raise ValueError("Texts and titles must have equal lengths")
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
        model = model_provider()
        classes, proba = _probabilities(model, valid_texts)
        for i, row in zip(valid_idx, proba):
            j = int(row.argmax())
            results[i] = {
                "label": str(classes[j]), "confidence": round(float(row[j]), 4),
                "probabilities": {str(c): round(float(p), 4) for c, p in zip(classes, row)},
                "valid": True, "reason": None,
                "low_confidence": float(row[j]) < LOW_CONFIDENCE,
                "model_version": getattr(model, "model_version", "Character TF-IDF + Logistic Regression"),
            }
    return results


def predict_recommendation_batch(texts: List[str], titles: Optional[List[str]] = None) -> List[Dict]:
    """Predict using the configured active model."""
    return _predict_batch(texts, titles, load_model)


def predict_model_batch(kind, texts, titles=None):
    """Explicit candidate comparison without changing the default predictor."""
    from recommendation_ml.model_registry import load_candidate
    return _predict_batch(texts, titles, lambda: load_candidate(kind))


def predict_model(kind, text, title=None):
    return predict_model_batch(kind, [text], [title])[0]


def predict_legacy_recommendation_batch(texts: List[str], titles: Optional[List[str]] = None) -> List[Dict]:
    """Replay the historical LR, not the active XGBoost."""
    return _predict_batch(texts, titles, load_legacy_model)


def predict_legacy_recommendation(text, title=None) -> Dict:
    return predict_legacy_recommendation_batch([text], [title])[0]


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
