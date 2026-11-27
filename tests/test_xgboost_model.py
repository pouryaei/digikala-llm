"""XGBoost search, artifact, active prediction and real Streamlit checks."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

import ml_service
from recommendation_ml.train_final import ROOT, LABELS, _metrics, sha
from recommendation_ml.train_xgboost import run_search, DEFAULT_OUTPUT
from recommendation_ml.xgb_service import XGBoostTextModel, load_xgboost

REPORT = DEFAULT_OUTPUT / "report.json"
SAMPLES = (("کیفیت عالی بود و خرید را پیشنهاد میکنم","recommended"),
           ("افتضاح بود اصلا نخرید","not_recommended"),
           ("هنوز استفاده نکرده ام","no_idea"))
ENV = {k:"" for k in ("DIGIKALA_MODEL_PATH","QBC13_API_KEY","QBC13_BACKEND_URL",
                      "QBC13_PREDICT_FUNCTION","QBC13_SEARCH_FUNCTION","QBC13_REVIEW_FUNCTION")}
ENV.update(FOROUHAR_LANGFUSE_ENABLED="0", QBC13_REVIEW_GENERATION_MODE="extractive",
           QBC13_UI_SEARCH_MODE="bm25")


def synthetic():
    phrases=("نظر خاصی ندارم","این کالا خراب است نخرید","کیفیت عالی خرید را پیشنهاد میکنم")
    result={}
    for split in ("train","validation","test"):
        marker={"train":"","validation":"validateword validateword","test":"testtoken testtoken"}[split]
        result[split]=[(f"{split}-{i}",f"{split}-product-{i}",
                        f"{phrase} {marker} نمونه {split} {i}",label,f"{split}-{i}")
                       for i,(label,phrase) in enumerate(zip(LABELS*12,phrases*12))]
    return result


class XGBoostTests(TestCase):
    def test_real_report_matches_saved_test_and_artifacts(self):
        r=json.loads(REPORT.read_text())
        self.assertEqual(r["model"],"XGBoost")
        self.assertEqual(r["n_trials"],8)
        self.assertEqual(r["best_validation"]["macro_f1"],max(t["validation_macro_f1"] for t in r["trials"]))
        self.assertFalse(r["test_used_for_selection"])
        self.assertEqual(r["audit"]["shared_products"],0)
        self.assertEqual(r["audit"]["shared_text_keys"],0)
        for filename,digest in r["artifacts"].items():
            self.assertEqual(sha(DEFAULT_OUTPUT/filename),digest)
        for filename,digest in r["historical_artifacts_sha256"].items():
            self.assertEqual(sha(ROOT/filename),digest)
        with (DEFAULT_OUTPUT/"test_predictions.csv").open(encoding="utf8",newline="") as f:
            predicted=list(csv.DictReader(f))
        with (ROOT/"m2_outputs/test_predictions.csv").open(encoding="utf-8-sig",newline="") as f:
            recorded={r["comment_id"]:r["recommendation_status"] for r in csv.DictReader(f)}
        self.assertEqual(len(predicted),15000)
        self.assertEqual({x["comment_id"]:x["recommendation_status"] for x in predicted},recorded)
        metrics=_metrics([row["recommendation_status"] for row in predicted],
                         [row["predicted_status"] for row in predicted])
        for key,value in metrics.items():
            self.assertAlmostEqual(value,r["test"][key],places=12)

    def test_active_model_and_historical_head_are_distinct(self):
        from ui_services import predict_recommendation
        from bonus.adapt_classifier import predict_adapted
        with patch.dict(os.environ,{**ENV,"DIGIKALA_MODEL_PATH":str(REPORT)}),patch.object(ml_service,"MODEL_PATH",REPORT),patch.object(ml_service,"_model",None), \
                patch("requests.request",side_effect=AssertionError("Prediction must be local")):
            self.assertEqual(ml_service._find_model_path(),REPORT)
            self.assertIsInstance(ml_service.load_model(),XGBoostTextModel)
            for body,label in SAMPLES:
                result=predict_recommendation(body)["data"]
                self.assertEqual(result["recommendation_status"],label)
                self.assertIn("XGBoost",result["model_version"])
                self.assertEqual(set(result["probabilities"]),set(LABELS))
                self.assertAlmostEqual(sum(result["probabilities"].values()),1,places=3)
            self.assertFalse(predict_recommendation("۱۲۳۴۵")["data"]["valid"])
            self.assertIn("Logistic Regression",ml_service.predict_legacy_recommendation(SAMPLES[0][0])["model_version"])
            self.assertTrue(predict_adapted(SAMPLES[0][0])["valid"])
            self.assertIsInstance(ml_service.load_model(),XGBoostTextModel)

    def test_grid_trains_real_boosters_and_fits_transform_on_train_only(self):
        with TemporaryDirectory() as tmp:
            target=Path(tmp)/"xgb"
            rows=synthetic()
            result=run_search(rows,target,threads=2,grid={"max_depth":[2,3],"learning_rate":[0.1,0.2],"n_estimators":[2,4]})
            self.assertEqual(result["n_trials"],8)
            self.assertEqual(result["selected_on"],"validation")
            self.assertFalse(result["test_used_for_selection"])
            model=load_xgboost(target)
            self.assertNotIn("valid",model.preprocessor["tfidf"].vocabulary_)
            self.assertNotIn("testt",model.preprocessor["tfidf"].vocabulary_)
            self.assertEqual(model.booster.num_boosted_rounds(),result["best_params"]["n_estimators"])
            predictions=model.predict([r[2] for r in rows["test"]]).tolist()
            with (target/"test_predictions.csv").open(encoding="utf8",newline="") as f:
                expected=[r["predicted_status"] for r in csv.DictReader(f)]
            self.assertEqual(predictions,expected)
            self.assertTrue(np.allclose(model.predict_proba([r[2] for r in rows["test"]]).sum(axis=1),1))
            with self.assertRaises(FileExistsError):
                run_search(rows,target,threads=1)

    def test_main_streamlit_predicts_with_xgboost_and_reports_new_scores(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ,{**ENV,"DIGIKALA_MODEL_PATH":str(REPORT)}),patch.object(ml_service,"MODEL_PATH",REPORT),patch.object(ml_service,"_model",None), \
                patch("requests.request",side_effect=AssertionError("Prediction must be local")):
            app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=60)
            app.session_state["main_tabs"]="داشبورد و پیش‌بینی رضایت"
            app.run()
            self.assertFalse(app.exception)
            app.text_area(key="prediction_text").set_value(SAMPLES[0][0])
            app.button(key="FormSubmitter:prediction-پیش‌بینی").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            data=app.session_state["prediction_result"]["data"]
            self.assertEqual(data["recommendation_status"],"recommended")
            self.assertIn("XGBoost",data["model_version"])
            self.assertTrue(any(m.label=="Macro F1" and m.value=="0.6698" for m in app.metric))
            self.assertTrue(any("مدل فعال XGBoost" in e.label for e in app.expander))

    def test_xgboost_is_visible_as_candidate(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ,ENV),patch("requests.request",side_effect=AssertionError("Local dashboard")):
            app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=60)
            app.session_state["main_tabs"]="آزمایشگاه شواهد و امتیازی"
            app.session_state["bonus_lab_tabs"]="مدل و آموزش تکمیلی"
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(any("n_estimators" in df.value.columns and len(df.value)==8 for df in app.dataframe))
            self.assertTrue(any("Test Macro F1" in df.value.columns and len(df.value)==4 for df in app.dataframe))
