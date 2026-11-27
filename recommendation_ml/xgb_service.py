"""Load the native XGBoost booster plus its train-fitted text preprocessor."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

from recommendation_ml.train_final import LABELS, sha


class XGBoostTextModel:
    """The same string-label predict/predict_proba contract as the saved LR."""

    model_version = "XGBoost + Character TF-IDF (tuned)"

    def __init__(self, preprocessor, booster, labels):
        self.preprocessor = preprocessor
        self.booster = booster
        self.classes_ = np.asarray([str(label) for label in labels], dtype=object)

    def predict_proba(self, texts):
        X = self.preprocessor.transform(texts).tocsr()
        values = np.asarray(self.booster.inplace_predict(X), dtype=np.float64)
        if (values.shape != (X.shape[0], len(self.classes_)) or
                not np.isfinite(values).all() or (values < 0).any() or
                not np.allclose(values.sum(axis=1), 1, atol=1e-5, rtol=0)):
            raise ValueError("Invalid XGBoost class probabilities")
        return values

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]


def load_xgboost(directory):
    directory = Path(directory)
    if directory.is_file() or directory.name == "report.json":
        directory = directory.parent
    report_path = directory / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing XGBoost report: {report_path}. Train or restore the complete XGBoost release.")
    report = json.loads(report_path.read_text(encoding="utf8"))
    if report.get("schema_version") != 1 or report.get("model") != "XGBoost" or tuple(report.get("labels", [])) != LABELS:
        raise ValueError("Invalid or incompatible XGBoost artifact manifest")
    for filename in ("preprocessor.joblib", "model.ubj"):
        path = directory / filename
        if not path.is_file() or sha(path) != report["artifacts"][filename]:
            raise ValueError(f"Missing/mismatched XGBoost artifact: {filename}")
    preprocessor = joblib.load(directory / "preprocessor.joblib")
    booster = xgb.Booster()
    booster.load_model(str(directory / "model.ubj"))
    booster.set_param({"device": "cpu", "nthread": 4})
    if (booster.num_boosted_rounds() != report["best_params"]["n_estimators"]
            or booster.num_features() != report["preprocessing"]["selected_features"]
            or int(preprocessor["select"].get_support().sum()) != booster.num_features()):
        raise ValueError("Booster and train-fitted preprocessor do not match the report")
    return XGBoostTextModel(preprocessor, booster, report["labels"])
