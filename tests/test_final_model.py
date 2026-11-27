"""Real final-model selection, four-way comparison, UI and dataset inference tests."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import duckdb
import ml_service
from recommendation_ml.model_registry import MODEL_IDS, AdaptedLogisticModel, load_candidate
from recommendation_ml.train_final import ROOT, LABELS, _metrics, sha

REPORT_DIR=ROOT/"reports/final_model"
ENV={k:"" for k in ("DIGIKALA_MODEL_PATH","QBC13_API_KEY","QBC13_BACKEND_URL","QBC13_PREDICT_FUNCTION",
                     "QBC13_SEARCH_FUNCTION","QBC13_REVIEW_FUNCTION")}
ENV.update(FOROUHAR_LANGFUSE_ENABLED="0",QBC13_REVIEW_GENERATION_MODE="extractive",QBC13_UI_SEARCH_MODE="bm25")
ACTIVE=ROOT/"bonus_artifacts/adapted_head.npz"
SAMPLES=("کیفیت عالی بود و خرید را پیشنهاد می‌کنم","افتضاح بود اصلا نخرید","هنوز استفاده نکرده‌ام")


class FinalModelTests(TestCase):
    def test_comparison_and_artifact_selection(self):
        report=json.loads((REPORT_DIR/"report.json").read_text())
        config=json.loads((ROOT/"m2_outputs/active_model.json").read_text())
        rows=report["models"]
        self.assertEqual(set(row["id"] for row in rows),set(MODEL_IDS))
        self.assertEqual(rows[0]["id"],config["kind"])
        self.assertEqual(rows[0]["id"],report["selected_model"])
        self.assertEqual(rows[0]["id"],"lr_adapted")
        self.assertEqual(rows[0]["validation"]["macro_f1"],max(row["validation"]["macro_f1"] for row in rows))
        self.assertFalse(report["test_used_for_selection"])
        self.assertFalse(report["new_blind_test"])
        self.assertEqual(report["audit"]["shared_products"],0)
        self.assertEqual(report["audit"]["shared_comparable_text_keys"],0)
        for r in rows:
            for name,digest in r["artifact_sha256"].items():
                self.assertEqual(sha(ROOT/name),digest)
            self.assertEqual(sum(map(sum,r["confusion_matrix"])),report["fixed_test_count"])
        with (REPORT_DIR/"test_predictions.csv").open(encoding="utf8",newline="") as f:
            predictions=list(csv.DictReader(f))
        with (ROOT/"m2_outputs/test_predictions.csv").open(encoding="utf-8-sig",newline="") as f:
            ids={r["comment_id"]:r["recommendation_status"] for r in csv.DictReader(f)}
        self.assertEqual({r["comment_id"]:r["recommendation_status"] for r in predictions},ids)
        actual=_metrics([r["recommendation_status"] for r in predictions],[r["predicted_status"] for r in predictions])
        for k,v in actual.items():self.assertAlmostEqual(v,rows[0]["test"][k],places=12)

    def test_default_and_four_actual_candidates(self):
        from ui_services import predict_recommendation
        with patch.dict(os.environ,ENV),patch.object(ml_service,"MODEL_PATH",ACTIVE),patch.object(ml_service,"_model",None), \
                patch("requests.request",side_effect=AssertionError("Local inference must not call API")):
            self.assertEqual(ml_service._find_model_path(),ACTIVE)
            self.assertIsInstance(ml_service.load_model(),AdaptedLogisticModel)
            for sample in SAMPLES:
                out=predict_recommendation(sample)["data"]
                self.assertIn("Head Fine-tuning",out["model_version"])
                for kind in MODEL_IDS:
                    ans=ml_service.predict_model(kind,sample)
                    self.assertTrue(ans["valid"])
                    self.assertIn(ans["label"],LABELS)
                    self.assertAlmostEqual(sum(ans["probabilities"].values()),1,places=3)
                self.assertEqual(out["recommendation_status"],ml_service.predict_model("lr_adapted",sample)["label"])
            for kind in MODEL_IDS:
                self.assertFalse(ml_service.predict_model(kind,"۱۲۳۴۵")["valid"])
            self.assertIsInstance(ml_service.load_model(),AdaptedLogisticModel)

    def test_real_streamlit_final_prediction(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ,ENV),patch.object(ml_service,"MODEL_PATH",ACTIVE),patch.object(ml_service,"_model",None), \
                patch("requests.request",side_effect=AssertionError("Local app")):
            app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=60)
            app.session_state["main_tabs"]="داشبورد و پیش‌بینی رضایت"
            app.run()
            self.assertFalse(app.exception)
            app.text_area(key="prediction_text").set_value(SAMPLES[0])
            app.button(key="FormSubmitter:prediction-پیش‌بینی").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertIn("Head Fine-tuning",app.session_state["prediction_result"]["data"]["model_version"])
            self.assertTrue(any(m.label=="Macro F1" and m.value=="0.6923" for m in app.metric))
            self.assertTrue(any("بهترین مدل نهایی" in e.label for e in app.expander))
            self.assertTrue(any("Validation Macro F1" in df.value.columns and len(df.value)==4 for df in app.dataframe))

    def test_real_streamlit_four_models_and_search_tables(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ,ENV),patch.object(ml_service,"MODEL_PATH",ACTIVE),patch.object(ml_service,"_model",None), \
                patch("requests.request",side_effect=AssertionError("Local app")):
            app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=60)
            app.session_state["main_tabs"]="آزمایشگاه شواهد و امتیازی"
            app.session_state["bonus_lab_tabs"]="مدل و آموزش تکمیلی"
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(any(m.label=="Test Macro F1" and m.value=="69.23%" for m in app.metric))
            self.assertTrue(any(len(df.value)==4 and "Validation Macro F1" in df.value.columns for df in app.dataframe))
            self.assertTrue(any(len(df.value)==8 and "n_estimators" in df.value.columns for df in app.dataframe))
            self.assertTrue(any(len(df.value)==8 and "class_weight" in df.value.columns for df in app.dataframe))
            app.button(key="FormSubmitter:bonus_model_trial-مقایسه چهار مدل").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            answers=app.session_state["bonus_models_final"]
            self.assertEqual({x["id"] for x in answers},set(MODEL_IDS))
            self.assertTrue(all(x["result"]["valid"] for x in answers))
            self.assertEqual(answers[0]["id"],"lr_adapted")

    def test_streamed_parquet_and_csv_inference(self):
        from recommendation_ml.predict_dataset import infer,DEFAULT_INPUT
        with TemporaryDirectory() as t,patch.object(ml_service,"MODEL_PATH",ACTIVE),patch.object(ml_service,"_model",None):
            folder=Path(t)
            output=folder/"real_reviews.csv"
            summary=infer(DEFAULT_INPUT,output,limit=100,batch_size=12)
            self.assertEqual(summary["rows"],100)
            with output.open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),100)
            self.assertIn("Head Fine-tuning",summary["model_version"])
            p=folder/"source.csv"
            with p.open("w",encoding="utf8",newline="") as f:
                w=csv.writer(f);w.writerow(["id","title","body","recommendation_status","rating"])
                w.writerows([(1,"عالی","حتما پیشنهاد میکنم","not_recommended",1),
                            (2,"عالی","حتما پیشنهاد میکنم","recommended",5),
                            (3,"","۱۲۳۴۵","recommended",5)])
            out=folder/"custom.csv";r=infer(p,out,batch_size=2)
            self.assertEqual(r["rows"],3);self.assertEqual(r["valid"],2)
            with out.open(encoding="utf-8-sig",newline="") as f:rr=list(csv.DictReader(f))
            self.assertEqual(rr[0]["predicted_status"],rr[1]["predicted_status"])
            self.assertEqual(rr[0]["p_recommended"],rr[1]["p_recommended"])
            self.assertEqual(rr[2]["predicted_status"],"")
            self.assertEqual(rr[2]["reason"],"digits_only")
            with self.assertRaises(ValueError):infer(p,out)

    def test_router_uses_final_model(self):
        from bonus.graph import Workflow
        from bonus.observability import TraceStore,ResultCache
        with TemporaryDirectory() as t,patch.object(ml_service,"MODEL_PATH",ACTIVE),patch.object(ml_service,"_model",None):
            flow=Workflow(cache=ResultCache(),store=TraceStore(Path(t)/"test.sqlite"),engine="langgraph",export=False)
            result=flow.run({"intent":"predict","query":SAMPLES[0],"allow_llm":False,"use_cache":False})
            self.assertEqual(result["trace"]["api_calls"],0)
            self.assertEqual(result["result"]["status"],"answered")
            self.assertIn("Head Fine-tuning",json.dumps(result["result"]))
