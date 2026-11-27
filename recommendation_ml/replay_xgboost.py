"""Independently verify the saved native XGBoost model on the recorded test IDs.

This is an artifact replay AFTER selection, not another tuning step.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

from bonus.adapt_classifier import sample
from recommendation_ml.train_final import _metrics
from recommendation_ml.train_xgboost import DEFAULT_OUTPUT
from recommendation_ml.xgb_service import load_xgboost


def replay(source_archive, model_dir=DEFAULT_OUTPUT):
    start = perf_counter()
    model_dir = Path(model_dir)
    report = json.loads((model_dir / "report.json").read_text(encoding="utf8"))
    pools, source, _ = sample(source_archive)
    rows = pools["test"]
    if report["provenance"].get("source_payload_sha256") != source["source_payload_sha256"]:
        raise ValueError("The source is not the one used for this sampled experiment")
    with (model_dir / "test_predictions.csv").open(encoding="utf8", newline="") as handle:
        recorded = {r["comment_id"]: r for r in csv.DictReader(handle)}
    if len(recorded) != len(rows) or any(r[0] not in recorded or
            r[3] != recorded[r[0]]["recommendation_status"] for r in rows):
        raise ValueError("Saved test provenance does not match the full source")
    model = load_xgboost(model_dir)
    prediction = []
    for i in range(0, len(rows), 512):
        prediction.extend(model.predict([r[2] for r in rows[i:i+512]]).tolist())
    matches = sum(p == recorded[r[0]]["predicted_status"] for p, r in zip(prediction, rows))
    result = {
        "test_rows":len(rows), "exact_prediction_matches":matches,
        "all_match":matches == len(rows),
        "source_payload_sha256":source["source_payload_sha256"],
        "metrics":_metrics([r[3] for r in rows], prediction),
        "elapsed_seconds":perf_counter()-start,
        "note":"Replay of the already-selected model; no refitting or hyperparameter selection.",
    }
    if not result["all_match"]:
        raise ValueError("Saved XGBoost model failed exact replay")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-archive", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    r = replay(a.source_archive, a.model_dir)
    body = json.dumps(r, ensure_ascii=False, indent=2)+"\n"
    if a.output:
        a.output.write_text(body, encoding="utf8")
    print(body, flush=True)


if __name__=="__main__":
    main()
