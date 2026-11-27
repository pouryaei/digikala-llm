"""Train and tune the replacement XGBoost text classifier on fixed product splits.

All features and model weights are fitted exclusively on TRAIN. For each
(max_depth, learning_rate) pair, train a single boosting history and score
two prespecified boosting checkpoints on VALIDATION. This is an efficient
grid over all three hyperparameters. Evaluate TEST only after selection.
The historical LR artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import sklearn
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from recommendation_ml.train_final import (
    ROOT, LABELS, SPLITS, _metrics, audit_splits, load_splits, resolve_split_dir, sha,
)
from recommendation_ml.hyperparameter_search import _audit_rows, _recorded_references, PROTECTED_ARTIFACTS, _write_json

DEFAULT_OUTPUT = ROOT / "m2_outputs" / "xgboost"
GRID = {"max_depth": [4, 6], "learning_rate": [0.10, 0.20], "n_estimators": [120, 240]}
MAX_FEATURES = 4096
FIXED = {"objective": "multi:softprob", "num_class": len(LABELS),
         "tree_method": "hist", "device": "cpu", "max_bin": 32,
         "subsample": 0.85, "colsample_bytree": 0.5, "min_child_weight": 3.0,
         "reg_lambda": 2.0, "reg_alpha": 0.0, "seed": 42, "verbosity": 0}
FIXED_TFIDF = {"analyzer": "char", "ngram_range": (2, 5), "max_features": 200000,
               "min_df": 2, "sublinear_tf": True}


def _encoded(labels):
    mapping = {label: i for i, label in enumerate(LABELS)}
    return np.asarray([mapping[label] for label in labels], dtype=np.int32)


def _predict(booster, X, rounds=None):
    kwargs = {"iteration_range": (0, rounds)} if rounds is not None else {}
    proba = booster.inplace_predict(X, **kwargs)
    return np.asarray(LABELS)[np.asarray(proba).argmax(axis=1)]


def run_search(rows, output_dir=DEFAULT_OUTPUT, *, provenance=None, threads=5,
               grid=None):
    start = perf_counter()
    audit = _audit_rows(rows)
    output_dir = Path(output_dir).expanduser().resolve()
    protected = {(ROOT / "m2_outputs").resolve(), (ROOT / "bonus_artifacts").resolve(), ROOT.resolve()}
    if output_dir in protected or (output_dir.exists() and
                                   (not output_dir.is_dir() or any(output_dir.iterdir()))):
        raise FileExistsError("Use a new empty model directory. Do not overwrite historical artifacts.")
    if threads < 1:
        raise ValueError("threads must be positive")
    grid = grid or GRID
    if set(grid) != set(GRID) or any(not values for values in grid.values()):
        raise ValueError("Provide non-empty max_depth, learning_rate and n_estimators grids")
    if min(grid["n_estimators"]) < 1:
        raise ValueError("Boosting rounds must be positive")
    initial_hashes = {str(p.relative_to(ROOT)): sha(p) for p in PROTECTED_ARTIFACTS if p.is_file()}
    texts = {s: [r[2] for r in rows[s]] for s in SPLITS}
    labels = {s: np.asarray([r[3] for r in rows[s]]) for s in SPLITS}
    y_train = _encoded(labels["train"])

    with threadpool_limits(limits=min(threads, 4)):
        print(f"Fit character TF-IDF and chi2 selector on {len(texts['train']):,} TRAIN reviews", flush=True)
        stage = perf_counter()
        tfidf = TfidfVectorizer(**FIXED_TFIDF, dtype=np.float32)
        train_features = tfidf.fit_transform(texts["train"])
        selector = SelectKBest(score_func=chi2, k=min(MAX_FEATURES, train_features.shape[1]))
        X_train = selector.fit_transform(train_features, y_train).tocsr()
        del train_features
        preprocess = Pipeline([("tfidf", tfidf), ("select", selector)])
        X_val = preprocess.transform(texts["validation"]).tocsr()
        preprocessing_seconds = perf_counter() - stage
        print(f"Prepared matrices: train {X_train.shape}, validation {X_val.shape}; nnz={X_train.nnz:,}; {preprocessing_seconds:.1f}s", flush=True)

    dtrain = xgb.QuantileDMatrix(X_train, label=y_train, max_bin=FIXED["max_bin"], nthread=threads)
    trials = []
    best = None
    winning_booster = None
    rounds = sorted(set(grid["n_estimators"]))
    for depth in grid["max_depth"]:
        for rate in grid["learning_rate"]:
            params = {**FIXED, "max_depth": int(depth), "learning_rate": float(rate), "nthread": threads}
            print(f"Training depth={depth}, learning_rate={rate}, max rounds={rounds[-1]}", flush=True)
            fit_start = perf_counter()
            booster = xgb.train(params, dtrain, num_boost_round=rounds[-1], verbose_eval=False)
            fit_seconds = perf_counter() - fit_start
            for n in rounds:
                metrics = _metrics(labels["validation"], _predict(booster, X_val, n))
                trial = {"max_depth": depth, "learning_rate": rate, "n_estimators": n,
                         "validation_macro_f1": metrics["macro_f1"],
                         "validation_accuracy": metrics["accuracy"],
                         "history_fit_seconds": fit_seconds, "history_max_rounds": rounds[-1]}
                trials.append(trial)
                # Selection uses only validation Macro F1; exact ties favor fewer rounds,
                # then shallower tree depth, then lower learning rate.
                key = (-trial["validation_macro_f1"], n, depth, rate)
                print(json.dumps(trial, ensure_ascii=False), flush=True)
                if best is None or key < best[0]:
                    best = (key, trial)
                    winning_booster = booster[:n]
            del booster
    ranked = sorted(trials, key=lambda t: (-t["validation_macro_f1"], t["n_estimators"],
                                           t["max_depth"], t["learning_rate"]))
    for rank, trial in enumerate(ranked, 1):
        trial["rank"] = rank
    chosen = ranked[0]
    best_params = {k: chosen[k] for k in GRID}
    assert winning_booster.num_boosted_rounds() == best_params["n_estimators"]
    validation = _metrics(labels["validation"], _predict(winning_booster, X_val))
    if not np.isclose(validation["macro_f1"], chosen["validation_macro_f1"], atol=1e-12, rtol=0):
        raise ValueError("Winning booster does not reproduce its selected validation checkpoint")
    print("Selected on validation:", best_params, validation, flush=True)

    # The held-out data is transformed and predicted only now.
    del X_train, X_val, dtrain
    pred_start = perf_counter()
    X_test = preprocess.transform(texts["test"]).tocsr()
    predictions = _predict(winning_booster, X_test)
    prediction_seconds = perf_counter() - pred_start
    test = _metrics(labels["test"], predictions)
    class_report = classification_report(labels["test"], predictions, labels=LABELS,
                                         output_dict=True, zero_division=0)
    matrix = confusion_matrix(labels["test"], predictions, labels=LABELS).tolist()
    if initial_hashes != {name: sha(ROOT / name) for name in initial_hashes}:
        raise ValueError("Historical model changed during search")

    output_dir.mkdir(parents=True, exist_ok=True)
    winning_booster.save_model(str(output_dir / "model.ubj"))
    joblib.dump(preprocess, output_dir / "preprocessor.joblib", compress=3)
    with (output_dir / "trials.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
        writer.writeheader()
        writer.writerows(ranked)
    with (output_dir / "test_predictions.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["comment_id", "recommendation_status", "predicted_status"])
        writer.writerows((r[0], r[3], pred) for r, pred in zip(rows["test"], predictions))
    reference = _recorded_references(rows["test"])
    report = {
        "schema_version": 1, "model": "XGBoost", "method": "Fixed validation grid over boosting histories and checkpoints",
        "selection_metric": "macro_f1", "selected_on": "validation",
        "test_used_for_selection": False,
        "tie_break": "fewer boosting rounds, shallower depth, lower learning rate",
        "param_grid": grid, "n_trials": len(trials), "n_training_histories": len(grid["max_depth"]) * len(grid["learning_rate"]),
        "fixed_params": FIXED, "best_params": best_params, "best_validation": validation, "trials": ranked,
        "test": test, "test_classification_report": class_report, "test_confusion_matrix": {"labels": list(LABELS), "matrix": matrix},
        "labels": list(LABELS), "preprocessing": {"tfidf": {**FIXED_TFIDF, "dtype": "float32"},
            "selector": "SelectKBest(chi2)", "max_selected_features": MAX_FEATURES,
            "fitted_vocabulary_size": len(tfidf.vocabulary_), "selected_features": int(selector.get_support().sum()),
            "fitted_on": "train", "text_feature": "text_title_body"},
        "audit": audit, "provenance": provenance or {}, "recorded_test_reference": reference,
        "historical_artifacts_sha256": initial_hashes,
        "artifacts": {"model.ubj": sha(output_dir / "model.ubj"),
                      "preprocessor.joblib": sha(output_dir / "preprocessor.joblib")},
        "xgboost_version": xgb.__version__, "scikit_learn_version": sklearn.__version__,
        "preprocessing_seconds": preprocessing_seconds, "test_prediction_seconds": prediction_seconds,
        "elapsed_seconds": perf_counter() - start,
        "limitations": "A new XGBoost fit on the fixed sampled train/validation data. The historical LR was not necessarily fitted on the same number of training rows. The recorded test is reused for comparison, not a new blind benchmark. Test scores were not used to select hyperparameters.",
    }
    _write_json(output_dir / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-archive", type=Path, help="Full cleaned Forouhar_M1_Complete.zip; deterministic sample")
    source.add_argument("--data-dir", type=Path, help="Directory of three fixed EDA recommendation Parquets")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--threads", type=int, default=5)
    a = p.parse_args()
    if a.source_archive:
        from bonus.adapt_classifier import sample
        print("Read and verify full cleaned source; preserve recorded test IDs", flush=True)
        rows, provenance, _ = sample(a.source_archive)
        provenance = {"mode": "deterministic_sample", **provenance}
    else:
        splits, files = load_splits(resolve_split_dir(a.data_dir), "text_title_body")
        provenance = {"mode": "full_eda_splits", "split_audit": audit_splits(splits),
                      "data_sha256": {name: sha(path) for name, path in files.items()}}
        rows = {s: [(r.comment_id, r.product_id, r.text, r.label, r.text_key) for r in items]
                for s, items in splits.items()}
    report = run_search(rows, a.output_dir, provenance=provenance, threads=a.threads)
    print(json.dumps({k: report[k] for k in ("best_params", "best_validation", "test", "elapsed_seconds")},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
