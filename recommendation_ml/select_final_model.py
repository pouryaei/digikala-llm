"""Verify all saved candidates on fixed development splits and select by Macro F1.

The existing, previously observed test is replayed after the choice; it is
not a new blind test. No weights are fit or changed by this procedure.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from bonus.adapt_classifier import sample
from recommendation_ml.model_registry import (
    MODEL_IDS, NAMES, load_candidate, LEGACY_PATH, ADAPTED_PATH, ADAPT_REPORT, TUNED_DIR, XGB_DIR,
)
from recommendation_ml.train_final import ROOT, LABELS, _metrics, sha

DEFAULT_OUTPUT = ROOT / "reports/final_model"
RELEASE_CONFIG = ROOT / "m2_outputs/active_model.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def predict(rows, kind, batch_size=512):
    model = load_candidate(kind)
    predictions = []
    started = perf_counter()
    for i in range(0, len(rows), batch_size):
        predictions.extend(model.predict([r[2] for r in rows[i:i+batch_size]]).tolist())
    return predictions, perf_counter()-started


def assert_exact(actual, expected, description):
    if not np.isclose(actual, expected, atol=1e-10, rtol=0):
        raise ValueError(f"{description}: {actual} != historical {expected}")


def run(source_archive, output_dir=DEFAULT_OUTPUT):
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new empty comparison output directory")
    start = perf_counter()
    rows, audit, wanted = sample(source_archive)
    yv = [r[3] for r in rows["validation"]]
    yt = [r[3] for r in rows["test"]]
    adaptation = read(ADAPT_REPORT)
    lr_tuned = read(TUNED_DIR/"report.json")
    xgb = read(XGB_DIR/"report.json")
    for report in (adaptation["audit"],lr_tuned["provenance"],xgb["provenance"]):
        if report["source_payload_sha256"] != audit["source_payload_sha256"] or report["splits"] != audit["splits"]:
            raise ValueError("Candidate splits cannot be compared")
    recorded_artifacts = {
        "lr_original": {str(LEGACY_PATH.relative_to(ROOT)):sha(LEGACY_PATH)},
        "lr_adapted": {str(LEGACY_PATH.relative_to(ROOT)):sha(LEGACY_PATH),
                       str(ADAPTED_PATH.relative_to(ROOT)):sha(ADAPTED_PATH)},
        "lr_tuned": {str((TUNED_DIR/lr_tuned["candidate_file"]).relative_to(ROOT)):lr_tuned["candidate_sha256"]},
        "xgboost": {str((XGB_DIR/k).relative_to(ROOT)):digest for k,digest in xgb["artifacts"].items()},
    }
    expected_validation = {
        "lr_original": adaptation["validation_trials"][0]["validation"],
        "lr_adapted": adaptation["selected_on_validation_only"]["validation"],
        "lr_tuned": lr_tuned["best_validation"],
        "xgboost": xgb["best_validation"],
    }
    expected_test = {
        "lr_original": adaptation["test"]["baseline"],
        "lr_adapted": adaptation["test"]["adapted"],
        "lr_tuned": lr_tuned["test"],
        "xgboost": xgb["test"],
    }
    results = {}
    for kind in MODEL_IDS:
        pred, elapsed = predict(rows["validation"],kind)
        validation = _metrics(yv,pred)
        for metric in ("macro_f1","accuracy"):
            assert_exact(validation[metric],expected_validation[kind][metric],kind+" validation "+metric)
        results[kind] = {"name":NAMES[kind],"validation":validation,"validation_prediction_seconds":elapsed,
                         "artifact_sha256":recorded_artifacts[kind]}
        print(kind,"validation",validation,flush=True)
    ranked = sorted(MODEL_IDS,key=lambda kind:(-results[kind]["validation"]["macro_f1"],kind))
    chosen = ranked[0]
    print("Selected on replayed validation only:",chosen,flush=True)

    # The choice is frozen above. Historical test comparisons cannot change it.
    out.mkdir(parents=True,exist_ok=True)
    for rank,kind in enumerate(ranked,1):
        pred, elapsed = predict(rows["test"],kind)
        metrics = _metrics(yt,pred)
        for metric in ("macro_f1","accuracy"):
            assert_exact(metrics[metric],expected_test[kind][metric],kind+" test "+metric)
        r=results[kind]
        r.update(rank=rank, test=metrics, test_prediction_seconds=elapsed,
                 classification_report=classification_report(yt,pred,labels=LABELS,
                     output_dict=True,zero_division=0),
                 confusion_matrix=confusion_matrix(yt,pred,labels=LABELS).tolist())
        if kind=="lr_original":
            if any(p!=wanted[row[0]]["predicted_status"] for p,row in zip(pred,rows["test"])):
                raise ValueError("Original test predictions differ")
        elif kind in ("lr_tuned","xgboost"):
            directory=TUNED_DIR if kind=="lr_tuned" else XGB_DIR
            with (directory/"test_predictions.csv").open(encoding="utf8",newline="") as f:
                saved={item["comment_id"]:item for item in csv.DictReader(f)}
            if len(saved)!=len(rows["test"]) or any(p!=saved[row[0]]["predicted_status"] or
                       row[3]!=saved[row[0]]["recommendation_status"] for p,row in zip(pred,rows["test"])):
                raise ValueError(f"{kind} differs from recorded predictions")
        if kind==chosen:
            with (out/"test_predictions.csv").open("w",encoding="utf8",newline="") as f:
                w=csv.writer(f);w.writerow(["comment_id","recommendation_status","predicted_status"])
                w.writerows((row[0],row[3],p) for row,p in zip(rows["test"],pred))
        print(kind,"test",metrics,flush=True)
    report={
        "schema_version":1,"selected_model":chosen,"selection_metric":"validation_macro_f1",
        "selection_rule":"maximum replayed Macro F1 on the fixed development validation split; exact ties by model ID",
        "test_used_for_selection":False,"new_blind_test":False,
        "models":[{"id":kind,**results[kind]} for kind in ranked],
        "audit":audit,"labels":list(LABELS),
        "limitations":[
            "This retrospective final selection uses the same previously observed validation and test pools; the test is not a fresh blind holdout.",
            "The original LR training sample and vocabulary differ from the new sampled fits. Historical full training text was unavailable for auditing against every new validation text.",
            "Zero overlap checks apply to the reconstructed train/validation/test pools, not to all historically seen text.",
            "Head fine-tuning is supervised adaptation of a linear head, not transformer or LLM fine-tuning.",
        ],
        "fixed_test_count":len(yt),"elapsed_seconds":perf_counter()-start,
    }
    with (out/"model_comparison.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["rank","id","model","selected","validation_macro_f1","validation_accuracy",
                                    "test_macro_f1","test_accuracy","macro_precision","macro_recall","test_prediction_seconds"])
        w.writeheader()
        for r in report["models"]:
            w.writerow({"rank":r["rank"],"id":r["id"],"model":r["name"],"selected":r["id"]==chosen,
                        "validation_macro_f1":r["validation"]["macro_f1"],"validation_accuracy":r["validation"]["accuracy"],
                        "test_macro_f1":r["test"]["macro_f1"],"test_accuracy":r["test"]["accuracy"],
                        "macro_precision":r["test"]["precision"],"macro_recall":r["test"]["recall"],
                        "test_prediction_seconds":r["test_prediction_seconds"]})
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-archive",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    a=p.parse_args()
    result=run(a.source_archive,a.output_dir)
    print("Final selection:",result["selected_model"],flush=True)


if __name__=="__main__":
    main()
