"""Tune the existing character TF-IDF + Logistic Regression recipe.

The vectorizer is fitted on TRAIN ONLY, and all grid candidates use the
same predefined train/validation split. The test set is scored after
selection. Results and optional candidate are isolated from m2_outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from time import perf_counter
import warnings

import joblib
import numpy as np
import sklearn
from scipy.sparse import vstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from recommendation_ml.train_final import (
    ROOT, LABELS, SPLITS, _metrics, audit_splits, character_model,
    load_splits, resolve_split_dir, sha,
)

DEFAULT_OUTPUT = ROOT / "reports" / "bonus" / "hyperparameter_search"
PARAM_GRID = {"C": [0.25, 1.0, 4.0, 10.0], "class_weight": [None, "balanced"]}
PROTECTED_ARTIFACTS = (
    ROOT / "m2_outputs" / "best_model.joblib",
    ROOT / "bonus_artifacts" / "adapted_head.npz",
)


def _write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _audit_rows(rows):
    """Also enforce the split guarantees when using the sampled source."""
    if set(rows) != set(SPLITS):
        raise ValueError("Expected train, validation and test")
    ids, keys, owners = set(), set(), {}
    stats = {}
    for split in SPLITS:
        counts, products = Counter(), set()
        for cid, pid, text, label, key in rows[split]:
            if (not cid or not pid or not isinstance(text, str) or not text.strip()
                    or not key or label not in LABELS):
                raise ValueError(f"Invalid review in {split}")
            if cid in ids or key in keys:
                raise ValueError("Duplicate review ID or text key")
            if owners.setdefault(pid, split) != split:
                raise ValueError("Product shared across splits")
            ids.add(cid); keys.add(key); products.add(pid); counts[label] += 1
        if set(counts) != set(LABELS):
            raise ValueError(f"A class is missing from {split}")
        stats[split] = {"rows": len(rows[split]), "products": len(products), "classes": dict(counts)}
    return {"splits": stats, "shared_products": 0, "duplicate_review_ids": 0,
            "shared_text_keys": 0}


def _recorded_references(test_rows):
    """Only compare historical results if the exact IDs and labels agree."""
    predictions = ROOT / "m2_outputs" / "test_predictions.csv"
    if not predictions.is_file():
        return None
    with predictions.open(encoding="utf-8-sig", newline="") as handle:
        recorded = {r["comment_id"]: r for r in csv.DictReader(handle)}
    if (len(recorded) != len(test_rows) or
            any(cid not in recorded or recorded[cid]["recommendation_status"] != label
                for cid, _, _, label, _ in test_rows)):
        return None
    y = [r[3] for r in test_rows]
    original = _metrics(y, [recorded[r[0]]["predicted_status"] for r in test_rows])
    return {"matched_recorded_test_ids_and_labels": len(test_rows),
            "original": original, "original_predictions_sha256": sha(predictions)}


def run_search(rows, output_dir=DEFAULT_OUTPUT, *, provenance=None, threads=2):
    start = perf_counter()
    audit = _audit_rows(rows)
    output_dir = Path(output_dir).expanduser().resolve()
    protected = [(ROOT / "m2_outputs").resolve(), (ROOT / "bonus_artifacts").resolve()]
    if (any(output_dir == p or p in output_dir.parents for p in protected)
            or (output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())))):
        raise FileExistsError("Use a new empty output directory outside the deployed model directories.")
    if threads < 1:
        raise ValueError("threads must be positive")
    initial_hashes = {str(p.relative_to(ROOT)): sha(p) for p in PROTECTED_ARTIFACTS if p.is_file()}

    texts = {split: [r[2] for r in rows[split]] for split in SPLITS}
    labels = {split: np.array([r[3] for r in rows[split]]) for split in SPLITS}
    recipe = character_model()
    vectorizer, estimator = recipe["tfidf"], recipe["model"]
    with threadpool_limits(limits=threads), warnings.catch_warnings():
        # Do not silently select a candidate whose optimizer did not converge.
        warnings.simplefilter("error", ConvergenceWarning)
        print(f"Vectorizing {len(texts['train']):,} training and {len(texts['validation']):,} validation reviews",
              flush=True)
        X_train = vectorizer.fit_transform(texts["train"])
        X_val = vectorizer.transform(texts["validation"])
        X = vstack([X_train, X_val], format="csr")
        y = np.concatenate([labels["train"], labels["validation"]])
        fold = PredefinedSplit(np.r_[np.full(X_train.shape[0], -1, dtype=int),
                                     np.zeros(X_val.shape[0], dtype=int)])
        scorer = make_scorer(f1_score, labels=LABELS, average="macro", zero_division=0, pos_label=None)
        search = GridSearchCV(estimator, PARAM_GRID,
                              scoring={"macro_f1": scorer, "accuracy": "accuracy"},
                              cv=fold, refit=False, n_jobs=1, verbose=2,
                              error_score="raise", return_train_score=False)
        search.fit(X, y)
        results = search.cv_results_
        trials = []
        for i, params in enumerate(results["params"]):
            trials.append({
                "C": float(params["C"]), "class_weight": params["class_weight"],
                "validation_macro_f1": float(results["mean_test_macro_f1"][i]),
                "validation_accuracy": float(results["mean_test_accuracy"][i]),
                "fit_seconds": float(results["mean_fit_time"][i]),
            })
        # Strict Macro F1 selection. Exact ties: lower C, then balanced.
        ranked = sorted(trials, key=lambda t: (-t["validation_macro_f1"], t["C"],
                                                t["class_weight"] != "balanced"))
        for rank, trial in enumerate(ranked, 1):
            trial["rank"] = rank
        best = ranked[0]
        params = {k: best[k] for k in PARAM_GRID}
        baseline = next(t for t in trials if t["C"] == 1.0 and t["class_weight"] == "balanced")
        print("Selected on validation:", params, "Macro F1:", best["validation_macro_f1"], flush=True)
        del X, y
        # A separate TRAIN-only fit; no refit on validation or test.
        start_fit = perf_counter()
        estimator.set_params(**params).fit(X_train, labels["train"])
        train_seconds = perf_counter() - start_fit
        validation = _metrics(labels["validation"], estimator.predict(X_val))
        if not np.isclose(validation["macro_f1"], best["validation_macro_f1"], rtol=0, atol=1e-10):
            raise ValueError("Refit did not reproduce selected validation score")
        del X_train, X_val
        # Test features and predictions are evaluated only now, exactly once.
        start_test = perf_counter()
        candidate = Pipeline([("tfidf", vectorizer), ("model", estimator)])
        predicted = candidate.predict(texts["test"])
        prediction_seconds = perf_counter() - start_test
        test = _metrics(labels["test"], predicted)
        classes = classification_report(labels["test"], predicted, labels=LABELS,
                                        output_dict=True, zero_division=0)
        matrix = confusion_matrix(labels["test"], predicted, labels=LABELS).tolist()

    if initial_hashes != {name: sha(ROOT / name) for name in initial_hashes}:
        raise ValueError("Protected production artifact changed during search")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "tuned_candidate.joblib"
    joblib.dump(candidate, model_path, compress=3)
    with (output_dir / "trials.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
        writer.writeheader()
        writer.writerows({**t, "class_weight": t["class_weight"] or "None"} for t in ranked)
    with (output_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["comment_id", "recommendation_status", "predicted_status"])
        writer.writerows((r[0], r[3], p) for r, p in zip(rows["test"], predicted))
    references = _recorded_references(rows["test"])
    report = {
        "method": "GridSearchCV with PredefinedSplit (one fixed validation fold)",
        "model": "Character TF-IDF + LogisticRegression",
        "selection_metric": "macro_f1",
        "tie_break": "lower C, then balanced class weight",
        "param_grid": PARAM_GRID, "n_trials": len(trials), "trials": ranked,
        "fixed_params": {"tfidf": {"analyzer": "char", "ngram_range": [2, 5], "max_features": 200000,
                                    "min_df": 2, "sublinear_tf": True},
                         "logistic_regression": {"solver": "lbfgs", "max_iter": 1000, "tol": 1e-4}},
        "best_params": params, "best_validation": validation, "baseline_validation": baseline,
        "validation_delta_macro_f1": validation["macro_f1"] - baseline["validation_macro_f1"],
        "test": test, "recorded_test_reference": references,
        "test_classification_report": classes, "test_confusion_matrix": {"labels": list(LABELS), "matrix": matrix},
        "audit": audit, "provenance": provenance or {},
        "vectorizer_fitted_on": "train", "selected_on": "validation", "candidate_fitted_on": "train",
        "fitted_vocabulary_size": len(vectorizer.vocabulary_),
        "test_used_for_selection": False, "production_model_replaced": False,
        "protected_artifacts_sha256": initial_hashes, "candidate_sha256": sha(model_path),
        "candidate_file": model_path.name, "scikit_learn": sklearn.__version__,
        "fit_seconds": train_seconds, "test_prediction_seconds": prediction_seconds,
        "elapsed_seconds": perf_counter() - start,
        "limitations": (
            "This is a new fit of the same architecture, not a replacement of the deployed weights. "
            "If sampled, train and validation sizes differ from historical training, so only the grid "
            "trials share an identical training/validation sample. The recorded test is a reused holdout, "
            "not a new blind benchmark. Use the full fixed EDA splits for a full-scale rerun."
        ),
    }
    _write_json(output_dir / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-archive", type=Path, help="Forouhar_M1_Complete.zip, deterministic product-group sample")
    source.add_argument("--data-dir", type=Path, help="Directory containing all three fixed EDA recommendation Parquets")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=2, help="CPU BLAS/OpenMP threads (one grid fit at a time)")
    args = parser.parse_args()
    if args.source_archive:
        from bonus.adapt_classifier import sample
        print("Reading and verifying full cleaned source; preserving recorded test IDs.", flush=True)
        rows, provenance, _ = sample(args.source_archive)
        provenance = {"mode": "deterministic_sample", **provenance}
    else:
        split_rows, files = load_splits(resolve_split_dir(args.data_dir), "text_title_body")
        provenance = {"mode": "full_eda_splits", "split_audit": audit_splits(split_rows),
                      "data_sha256": {name: sha(path) for name, path in files.items()}}
        rows = {split: [(r.comment_id, r.product_id, r.text, r.label, r.text_key) for r in rr]
                for split, rr in split_rows.items()}
    result = run_search(rows, args.output_dir, provenance=provenance, threads=args.threads)
    print(json.dumps({k: result[k] for k in ("best_params", "best_validation", "test",
                                              "recorded_test_reference", "elapsed_seconds")},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
