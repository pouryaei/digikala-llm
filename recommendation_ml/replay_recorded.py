"""Recheck the saved model against all 15K recorded labels in the prior full-data ZIP.

No training or model replacement. This reads actual clean review IDs/raw title
and body from Forouhar_M1_Complete.zip and verifies the source payload checksum.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

from sklearn.metrics import accuracy_score, f1_score
import sklearn
import zstandard

import ml_service
from scripts.build_review_store import _HashedStream

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "m2_outputs"
LABELS = ("no_idea", "not_recommended", "recommended")


def replay(archive):
    with (OUTPUTS / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as f:
        recorded = list(csv.DictReader(f))
    wanted = {r["comment_id"]: r for r in recorded}
    if len(wanted) != len(recorded):
        raise ValueError("Recorded test IDs are not unique.")
    matches = {}
    started = perf_counter()
    with ZipFile(archive) as z:
        manifests = [n for n in z.namelist() if n.endswith("data/analytics/reviews_payload.json")]
        if len(manifests) != 1:
            raise ValueError("Expected the full cleaned-review payload.")
        manifest = json.loads(z.read(manifests[0]))
        member = manifests[0].replace("reviews_payload.json", "reviews_payload.csv.zst")
        with z.open(member) as raw:
            hashed = _HashedStream(raw)
            with zstandard.ZstdDecompressor().stream_reader(hashed, read_size=1024*1024) as stream:
                with io.TextIOWrapper(stream, encoding="utf-8", newline="") as text:
                    reader = csv.reader(text)
                    header = next(reader)
                    c, y, t, b = (header.index(name) for name in ("comment_id", "recommendation_status", "title_raw", "body_raw"))
                    count = 0
                    for row in reader:
                        count += 1
                        if row[c] in wanted:
                            if row[c] in matches:
                                raise ValueError("Duplicate test comment in source.")
                            matches[row[c]] = (row[y], row[t], row[b])
            source_sha = hashed.digest.hexdigest()
    if count != manifest["row_count"] or source_sha != manifest["sha256"]:
        raise ValueError("Source payload count/checksum mismatch.")
    if len(matches) != len(recorded):
        raise ValueError("Some recorded test IDs are missing.")
    ordered = [matches[r["comment_id"]] for r in recorded]
    truth = [r["recommendation_status"] for r in recorded]
    original = [r["predicted_status"] for r in recorded]
    if any(y != expected for (y, _, _), expected in zip(ordered, truth)):
        raise ValueError("Source labels disagree with recorded test labels.")
    model_path = OUTPUTS / "best_model.joblib"
    with model_path.open("rb") as handle:
        model_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    expected = json.loads((OUTPUTS / "integration_check.json").read_text(encoding="utf-8-sig"))["model_sha256"]
    if model_sha != expected or ml_service.LEGACY_MODEL_PATH.resolve() != model_path.resolve():
        raise ValueError("Saved model hash/path mismatch.")
    model = ml_service.load_legacy_model()
    training_text = [ml_service.compose_model_text(body, title) for _, title, body in ordered]
    prediction = model.predict(training_text).tolist()
    same = sum(a == b for a, b in zip(prediction, original))
    deployed = ml_service.predict_legacy_recommendation_batch([r[2] for r in ordered], [r[1] for r in ordered])
    valid = [i for i, v in enumerate(deployed) if v["valid"]]
    if not valid:
        raise ValueError("All service inputs were rejected.")
    service_truth = [truth[i] for i in valid]
    service_pred = [deployed[i]["label"] for i in valid]
    return {
        "source_archive": Path(archive).name, "source_payload_sha256": source_sha,
        "source_clean_rows": count,
        "all_test_ids_and_labels_match": True, "model_sha256": model_sha,
        "runtime_scikit_learn": sklearn.__version__, "training_feature": "text_title_body",
        "model_replay": {"n": len(recorded), "same_as_recorded": same,
                         "exact_match": same == len(recorded),
                         "macro_f1": f1_score(truth, prediction, labels=LABELS, average="macro"),
                         "accuracy": accuracy_score(truth, prediction)},
        "service_replay": {"valid": len(valid), "rejected": len(recorded)-len(valid),
                           "rejected_reasons": dict(Counter(v["reason"] for v in deployed if not v["valid"])),
                           "same_as_recorded_on_valid": sum(service_pred[j] == original[i] for j,i in enumerate(valid)),
                           "macro_f1_valid": f1_score(service_truth, service_pred, labels=LABELS, average="macro"),
                           "accuracy_valid": accuracy_score(service_truth, service_pred)},
        "elapsed_seconds": round(perf_counter()-started, 3),
        "note": "Actual artifact replay on the full recorded test set. Invalid-input rejection is kept separate. This verifies the feature and existing model, not a new full training run."
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-archive", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "reports" / "model_replay_complete.json")
    a = p.parse_args()
    result = replay(a.source_archive)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["model_replay"]["exact_match"]:
        raise SystemExit(1)
