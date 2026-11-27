"""Four verified local text classifiers, with a single reusable prediction contract.

The weights, vocabulary and historical evaluation files are kept separate.
All models are loaded lazily from trusted artifacts shipped in this release.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import joblib
import numpy as np

from recommendation_ml.train_final import ROOT, LABELS, sha

NAMES = {
    "lr_original": "Logistic Regression اصلی",
    "lr_tuned": "Logistic Regression + Hyperparameter Search",
    "lr_adapted": "Logistic Regression + Head Fine-tuning",
    "xgboost": "XGBoost + Hyperparameter Search",
}
MODEL_IDS = tuple(NAMES)
LEGACY_PATH = ROOT / "m2_outputs/best_model.joblib"
ADAPTED_PATH = ROOT / "bonus_artifacts/adapted_head.npz"
ADAPT_REPORT = ROOT / "reports/bonus/adaptation.json"
TUNED_DIR = ROOT / "reports/bonus/hyperparameter_search"
XGB_DIR = ROOT / "m2_outputs/xgboost"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _verify(path, digest):
    if not path.is_file() or sha(path) != digest:
        raise ValueError(f"Missing or inconsistent model artifact: {path}")


class AdaptedLogisticModel:
    """Frozen historical char TF-IDF + independently validated LR head/bias."""

    model_version = "Character TF-IDF + Logistic Regression (Head Fine-tuning)"

    def __init__(self, vectorizer, coef, intercept, labels):
        self.vectorizer = vectorizer
        self.coef_ = np.asarray(coef, dtype=np.float32)
        self.intercept_ = np.asarray(intercept, dtype=np.float32)
        self.classes_ = np.asarray([str(label) for label in labels], dtype=object)
        if (tuple(self.classes_) != LABELS or self.coef_.shape != (len(LABELS), len(vectorizer.vocabulary_))
                or self.intercept_.shape != (len(LABELS),)):
            raise ValueError("Adapted head is incompatible with its frozen vectorizer")
        if not np.isfinite(self.coef_).all() or not np.isfinite(self.intercept_).all():
            raise ValueError("Adapted head contains invalid weights")

    def predict_proba(self, texts):
        X = self.vectorizer.transform(texts).astype(np.float32)
        logits = X @ self.coef_.T + self.intercept_
        logits -= logits.max(axis=1, keepdims=True)
        proba = np.exp(logits)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]


@lru_cache(maxsize=len(MODEL_IDS))
def load_candidate(kind):
    if kind == "lr_original":
        report = _read(ADAPT_REPORT)
        _verify(LEGACY_PATH, report["original_model_sha256"])
        return joblib.load(LEGACY_PATH)
    if kind == "lr_adapted":
        report = _read(ADAPT_REPORT)
        _verify(ADAPTED_PATH, report["adapted_head_sha256"])
        base = load_candidate("lr_original")
        with np.load(ADAPTED_PATH, allow_pickle=False) as artifact:
            return AdaptedLogisticModel(base["tfidf"], artifact["coef"],
                                        artifact["intercept"], artifact["labels"])
    if kind == "lr_tuned":
        report = _read(TUNED_DIR / "report.json")
        path = TUNED_DIR / report["candidate_file"]
        _verify(path, report["candidate_sha256"])
        model = joblib.load(path)
        model.model_version = "Character TF-IDF + Logistic Regression (Hyperparameter Search)"
        return model
    if kind == "xgboost":
        from recommendation_ml.xgb_service import load_xgboost
        return load_xgboost(XGB_DIR)
    raise ValueError(f"Unknown model: {kind}")
