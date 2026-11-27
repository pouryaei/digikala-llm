"""Recalculate supplied test metrics and replay the delivered M2 service on raw reviews."""

from argparse import ArgumentParser
from collections import Counter
import csv
import hashlib
import json
from math import isclose
from pathlib import Path
from time import perf_counter

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from analytics_service import open_analytics_tables, resolve_analytics_files


OUTPUTS = Path(__file__).resolve().parents[1] / "m2_outputs"
LABELS = ("no_idea", "not_recommended", "recommended")


def verify_recorded_test(outputs: Path = OUTPUTS):
    with (outputs / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    ids = [row["comment_id"] for row in rows]
    y_true = [row["recommendation_status"] for row in rows]
    y_pred = [row["predicted_status"] for row in rows]
    if not rows or len(ids) != len(set(ids)) or set(y_true) - set(LABELS) or set(y_pred) - set(LABELS):
        raise ValueError("Invalid IDs or labels in supplied test predictions")
    computed = classification_report(y_true, y_pred, labels=list(LABELS), output_dict=True, zero_division=0)
    recorded = json.loads((outputs / "classification_report.json").read_text(encoding="utf-8-sig"))
    summary = json.loads((outputs / "test_results.json").read_text(encoding="utf-8-sig"))
    for key in (*LABELS, "macro avg", "weighted avg"):
        for metric in ("precision", "recall", "f1-score", "support"):
            if not isclose(computed[key][metric], recorded[key][metric], rel_tol=1e-9, abs_tol=1e-10):
                raise ValueError(f"Stored class report differs: {key} / {metric}")
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)
    if not isclose(accuracy, summary["accuracy"], rel_tol=1e-9) or not isclose(macro_f1, summary["macro_f1"], rel_tol=1e-9):
        raise ValueError("Stored test metrics differ from recorded predictions")
    return rows, {
        "test_count": len(rows), "accuracy": accuracy, "macro_f1": macro_f1, "classes": list(LABELS),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist(),
        "matches_supplied_reports": True,
    }


def verify_integration(data_dir=None):
    import joblib
    import sklearn
    import ml_service

    rows, recorded = verify_recorded_test()
    model_file = OUTPUTS / "best_model.joblib"
    if ml_service.LEGACY_MODEL_PATH.resolve() != model_file.resolve():
        raise ValueError(f"Historical LR artifact must be {model_file}")
    with open_analytics_tables(resolve_analytics_files(data_dir)) as con:
        matches = con.execute("""
            SELECT CAST(r.comment_id AS VARCHAR), r.recommendation_status, r.title_raw, r.body_raw
            FROM reviews r
            INNER JOIN read_csv_auto(?, all_varchar=true) p
                ON CAST(r.comment_id AS VARCHAR) = p.comment_id
        """, [str(OUTPUTS / "test_predictions.csv")]).fetchall()
    by_id = {identifier: (label, title, body) for identifier, label, title, body in matches}
    if len(matches) != len(rows) or len(by_id) != len(rows):
        raise ValueError("Could not match all test IDs to one cleaned review each")
    ordered = [by_id[row["comment_id"]] for row in rows]
    if any(record[0] != row["recommendation_status"] for record, row in zip(ordered, rows)):
        raise ValueError("Test labels differ from the cleaned review data")

    start = perf_counter()
    ml_service.load_legacy_model()
    load_seconds = perf_counter() - start
    start = perf_counter()
    predictions = ml_service.predict_legacy_recommendation_batch([row[2] for row in ordered], [row[1] for row in ordered])
    batch_seconds = perf_counter() - start
    valid = [(row, result) for row, result in zip(rows, predictions) if result["valid"]]
    if not valid:
        raise ValueError("No valid reviews after M2 input validation")
    y_true = [row["recommendation_status"] for row, result in valid]
    y_pred = [result["label"] for row, result in valid]
    with model_file.open("rb") as model:
        digest = hashlib.file_digest(model, "sha256").hexdigest()
    return {
        "model_sha256": digest, "runtime": {"scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "recorded_test": recorded,
        "deployment_replay": {
            "input": "Raw title and body joined to recorded test comment IDs; M2 service normalization and validation",
            "test_count": len(rows), "valid_count": len(valid), "rejected_count": len(rows) - len(valid),
            "rejected_reasons": dict(Counter(result["reason"] for result in predictions if not result["valid"])),
            "same_label_as_recorded_count": sum(result["label"] == row["predicted_status"] for row, result in valid),
            "accuracy_on_valid": accuracy_score(y_true, y_pred),
            "macro_f1_on_valid": f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0),
            "model_load_seconds": load_seconds, "batch_seconds": batch_seconds,
            "note": "Original training-time texts were not supplied. This raw title/body replay is a separate deployment check, not an exact reproduction of training-time preprocessing. Rejected inputs are not counted as no_idea predictions.",
        },
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Directory containing products/reviews Parquet files")
    parser.add_argument("--output", type=Path, default=OUTPUTS / "integration_check.json")
    args = parser.parse_args()
    result = verify_integration(args.data_dir)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
