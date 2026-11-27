"""Reproduce the saved character TF-IDF + Logistic Regression recipe from EDA splits.

The supplied historical model is never overwritten. New training results go
to a separate directory and are selected/assessed without looking at test data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("no_idea", "not_recommended", "recommended")
SPLITS = ("train", "validation", "test")
FEATURES = ("text_title_body", "text_clean")


@dataclass(frozen=True)
class Example:
    comment_id: str
    product_id: str
    text: str
    label: str
    text_key: str
    core_text_key: str


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def resolve_split_dir(data_dir=None):
    paths = [Path(data_dir).expanduser()] if data_dir is not None else [
        ROOT / folder for folder in ("processed_repo_based", "data/processed_repo_based",
                                     "digikala/processed_repo_based", "docs/processed_repo_based")
    ]
    for path in paths:
        if all((path / f"recommendation_{split}.parquet").is_file() for split in SPLITS):
            return path.resolve()
    raise FileNotFoundError("Need recommendation_train.parquet, recommendation_validation.parquet and recommendation_test.parquet from the full EDA notebook. Pass --data-dir.")


def load_splits(data_dir, feature):
    if feature not in FEATURES:
        raise ValueError("Unsupported text feature.")
    import duckdb
    files = {name: data_dir / f"recommendation_{name}.parquet" for name in SPLITS}
    with duckdb.connect(config={"threads": 4, "memory_limit": "2GB"}) as con:
        splits = {}
        for name, file in files.items():
            raw = con.execute(f"""
                SELECT CAST(comment_id AS VARCHAR), CAST(product_id AS VARCHAR),
                       {feature}, recommendation_status, CAST(text_key AS VARCHAR),
                       CAST(core_text_key AS VARCHAR)
                  FROM read_parquet(?) ORDER BY comment_id
            """, [str(file)]).fetchall()
            splits[name] = [Example(*row) for row in raw]
    return splits, files


def audit_splits(splits):
    """Reject duplicate IDs/text and shared products before fitting anything."""
    if set(splits) != set(SPLITS):
        raise ValueError("All three fixed splits are required.")
    ids, full_keys, core_keys, texts, group_owner = set(), set(), set(), set(), {}
    stats = {}
    for name in SPLITS:
        counts = Counter()
        groups = set()
        for row in splits[name]:
            if (row.label not in LABELS or not row.comment_id or not row.product_id
                    or not isinstance(row.text, str) or not row.text.strip()
                    or not row.text_key or not row.core_text_key):
                raise ValueError(f"Invalid input in {name}.")
            if row.comment_id in ids or row.text_key in full_keys or row.core_text_key in core_keys or row.text.casefold() in texts:
                raise ValueError("Duplicate comment or normalized text across/within fixed splits.")
            previous = group_owner.setdefault(row.product_id, name)
            if previous != name:
                raise ValueError("Product shared by training, validation or test.")
            ids.add(row.comment_id); full_keys.add(row.text_key)
            core_keys.add(row.core_text_key); texts.add(row.text.casefold())
            groups.add(row.product_id); counts[row.label] += 1
        if set(counts) != set(LABELS):
            raise ValueError(f"A class is missing from {name}.")
        stats[name] = {"rows": len(splits[name]), "products": len(groups),
                       "label_counts": dict(counts)}
    return {"splits": stats, "duplicate_comment_ids": 0, "duplicate_model_texts": 0,
            "shared_products": 0, "features_are_text_only": True}


def character_model():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    # The vectorizer/classifier parameters were read from best_model.joblib.
    # lbfgs is deterministic; no undocumented tuning or transformer is added.
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                                 max_features=200000, min_df=2, sublinear_tf=True)),
        ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                                     solver="lbfgs", tol=1e-4)),
    ])


def _metrics(y, predicted):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return {"precision": precision_score(y, predicted, labels=LABELS, average="macro", zero_division=0),
            "recall": recall_score(y, predicted, labels=LABELS, average="macro", zero_division=0),
            "macro_f1": f1_score(y, predicted, labels=LABELS, average="macro", zero_division=0),
            "accuracy": accuracy_score(y, predicted)}


def fit(splits, output_dir, *, feature="text_title_body", files=None):
    import joblib
    import sklearn
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    audit = audit_splits(splits)
    output_dir = Path(output_dir).resolve()
    if output_dir == (ROOT / "m2_outputs").resolve() or (output_dir.exists() and any(output_dir.iterdir())):
        raise FileExistsError("Use a new empty output directory; never overwrite the submitted model.")
    output_dir.mkdir(parents=True, exist_ok=True)
    xy = {name: ([r.text for r in rows], [r.label for r in rows]) for name, rows in splits.items()}
    train_x, train_y = xy["train"]
    val_x, val_y = xy["validation"]
    comparison = []
    for name, model in (("Majority", DummyClassifier(strategy="most_frequent")),
                        ("LR - Character", character_model())):
        started = perf_counter()
        model.fit(train_x, train_y)
        train_time = perf_counter() - started
        started = perf_counter()
        predictions = model.predict(val_x)
        pred_time = perf_counter() - started
        result = {"model": name, **_metrics(val_y, predictions), "train_time": train_time,
                  "prediction_time": pred_time}
        comparison.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    # This is the preselected saved-model recipe; the test set is touched once,
    # only after the training and validation steps. Do not tune on these scores.
    test_x, test_y = xy["test"]
    started = perf_counter()
    predicted = model.predict(test_x)
    pred_time = perf_counter() - started
    test_result = {"model": "LR - Character", **_metrics(test_y, predicted),
                   "training_time": train_time, "prediction_time": pred_time}
    confusion = confusion_matrix(test_y, predicted, labels=LABELS).tolist()
    report = classification_report(test_y, predicted, labels=LABELS, output_dict=True, zero_division=0)
    with (output_dir / "model_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader(); writer.writerows(comparison)
    with (output_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["comment_id", "recommendation_status", "predicted_status"])
        writer.writerows((r.comment_id, r.label, label) for r, label in zip(splits["test"], predicted))
    dumps = {
        "test_results.json": test_result,
        "classification_report.json": report,
        "confusion_matrix.json": {"labels": LABELS, "matrix": confusion},
    }
    for filename, body in dumps.items():
        (output_dir / filename).write_text(json.dumps(body, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    joblib.dump(model, output_dir / "best_model.joblib", compress=3)
    manifest = {
        "feature": feature, "audit": audit, "validation": comparison, "test": test_result,
        "data_sha256": {name: sha(file) for name, file in (files or {}).items()},
        "model_sha256": sha(output_dir / "best_model.joblib"),
        "scikit_learn": sklearn.__version__,
        "pipeline": {"tfidf": {"analyzer": "char", "ngram_range": [2,5], "max_features": 200000,
                                 "min_df": 2, "sublinear_tf": True},
                     "logistic_regression": {"C": 1, "class_weight": "balanced", "max_iter": 1000,
                                             "solver": "lbfgs", "tol": 0.0001}},
        "note": "Reproduction of the observed historical artifact configuration on the EDA's fixed splits; new training results are not claimed identical until run and compared. Submitted model remains unchanged."
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=ROOT / "m2_outputs" / "retrained")
    p.add_argument("--feature", choices=FEATURES, default="text_title_body")
    p.add_argument("--dry-run", action="store_true", help="Only verify the exact splits and leakage checks.")
    a = p.parse_args()
    splits, files = load_splits(resolve_split_dir(a.data_dir), a.feature)
    if a.dry_run:
        print(json.dumps(audit_splits(splits), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(fit(splits, a.output_dir, feature=a.feature, files=files), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
