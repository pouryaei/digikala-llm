"""Regression checks for the three-tab UI and its service contracts."""

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

import duckdb
import requests
from streamlit.testing.v1 import AppTest

from local_catalog import ask_reviews, search_products
from ui_services import APIError, api_call, predict_recommendation


APP = str(Path(__file__).with_name("app.py"))
MODEL_FILE = Path(APP).parent / "m2_outputs" / "best_model.joblib"


def json_response(data, status=200):
    result = requests.Response()
    result.status_code = status
    result.encoding = "utf-8"
    result._content = json.dumps({"mode": "serving", "data": data}, ensure_ascii=False).encode("utf-8")
    return result


class UIAssemblyTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.environment = {
            "QBC13_ANALYTICS_DIR": str(self.folder), "QBC13_SEARCH_FUNCTION": "",
            "QBC13_REVIEW_FUNCTION": "", "QBC13_PREDICT_FUNCTION": "",
            "QBC13_BACKEND_URL": "http://127.0.0.1:8000",
        }
        env = patch.dict(os.environ, self.environment)
        env.start()
        self.addCleanup(env.stop)
        # Existing contract tests deliberately exercise a backend without any local model.
        isolated_root = patch("ui_services.ROOT", self.folder)
        isolated_root.start()
        self.addCleanup(isolated_root.stop)

        with duckdb.connect() as con:
            con.execute("""
                CREATE TABLE p(product_id VARCHAR, title_fa VARCHAR, category1 VARCHAR, brand VARCHAR,
                    price DOUBLE, rating_100 DOUBLE, rating_count BIGINT, product_text_lexical VARCHAR);
                INSERT INTO p VALUES
                    ('1','کفش مردانه آلفا','کفش مردانه','آلفا',150,80,5,'کفش مردانه آلفا'),
                    ('2','کفش مردانه بتا','کفش مردانه','بتا',270,60,3,'کفش مردانه بتا'),
                    ('3','کفش مردانه کتابی','کفش مردانه','آلفا',NULL,70,1,'کفش مردانه کتابی'),
                    ('4','کتاب ادبی','کتاب','بتا',95,100,2,'کتاب ادبی بتا');
                CREATE TABLE r(comment_id VARCHAR, product_id VARCHAR, rating_5 DOUBLE, recommendation_status VARCHAR,
                    title_raw VARCHAR, body_raw VARCHAR, advantages_raw VARCHAR, disadvantages_raw VARCHAR,
                    title_clean VARCHAR, body_clean VARCHAR, advantages_clean VARCHAR, disadvantages_clean VARCHAR);
                INSERT INTO r VALUES
                    ('101','1',1,'not_recommended','دوخت ضعیف','کیفیت دوخت پایین است.','', 'دوخت ضعیف','دوخت ضعیف','کیفیت دوخت پایین است.',NULL,'دوخت ضعیف'),
                    ('102','1',5,'recommended','کیفیت خوب','کیفیت ساخت خوب است.','سبک و راحت','ندارد','کیفیت خوب','کیفیت ساخت خوب است.','سبک و راحت','ندارد'),
                    ('103','1',3,'no_idea','امتحان نکردم','هنوز استفاده نکرده ام.',NULL,NULL,'امتحان نکردم','هنوز استفاده نکرده ام.',NULL,NULL),
                    ('104','1',NULL,NULL,'خالی',NULL,NULL,NULL,'خالی',NULL,NULL,NULL),
                    ('201','2',2,'not_recommended','سایز','سایزش کوچک است.',NULL,'سایز کوچک','سایز','سایزش کوچک است.',NULL,'سایز کوچک'),
                    ('401','4',5,'recommended','کتاب','کیفیت چاپ خوبه.','چاپ خوب',NULL,'کتاب','کیفیت چاپ خوبه.','چاپ خوب',NULL);
            """)
            con.table("p").write_parquet(str(self.folder / "products.parquet"))
            con.table("r").write_parquet(str(self.folder / "reviews.parquet"))

    def test_local_filters_counts_and_empty_results(self):
        items = search_products("کفش مردانه", {"brand": "آلفا", "min_price": 100, "max_price": 200}, 10)["items"]
        self.assertEqual([row["product_id"] for row in items], [1])
        self.assertEqual((items[0]["comment_count"], items[0]["valid_label_count"]), (4, 3))
        self.assertAlmostEqual(items[0]["recommendation_rate"], 1 / 3)
        self.assertFalse(search_products("qwertynonexistent")['items'])
        self.assertFalse(search_products("؟؟؟")['items'])

    def test_local_review_excerpts_and_no_evidence(self):
        reply = ask_reviews(1, "ایرادهای پرتکراری که خریداران نوشته‌اند چیست؟")
        self.assertEqual(reply["answer_mode"], "extractive")
        self.assertEqual([item["comment_id"] for item in reply["evidence"]], ["101"])
        self.assertTrue(any("دوخت" in item["text"] for item in reply["evidence"]))
        with self.assertRaises(LookupError):
            ask_reviews(999, "کیفیت")
        self.assertEqual(ask_reviews(1, "qwertynonexistent")["answer_mode"], "insufficient_evidence")
        self.assertEqual(ask_reviews(3, "کیفیت")["answer_mode"], "insufficient_evidence")

    def test_backend_success_and_empty_result_do_not_trigger_local_search(self):
        with patch("requests.request", return_value=json_response({"items": [], "applied_filters": {"brand": "آلفا"}})) as request, \
                patch("local_catalog.search_products", side_effect=AssertionError("Should not fall back on a valid empty result")):
            entry = api_call("POST", "/api/search", {"query": "کفش", "filters": {"brand": "آلفا"}, "limit": 4})
            self.assertEqual(entry["mode"], "serving")
            self.assertEqual(request.call_args.kwargs["json"]["filters"], {"brand": "آلفا"})
            self.assertEqual(entry["data"]["items"], [])

    def test_direct_search_and_review_functions_receive_the_team_contract(self):
        sample_module = types.ModuleType("test_search_contract")
        calls = []

        def search(*, query, filters, limit):
            calls.append((query, filters, limit))
            return search_products(query, filters, limit)

        def answer(*, product_id, query):
            calls.append((product_id, query))
            return ask_reviews(product_id, query)

        sample_module.search_products = search
        sample_module.ask_reviews = answer
        configuration = {"QBC13_SEARCH_FUNCTION": "test_search_contract:search_products",
                         "QBC13_REVIEW_FUNCTION": "test_search_contract:ask_reviews"}
        with patch.dict(sys.modules, {sample_module.__name__: sample_module}), \
                patch.dict(os.environ, configuration), \
                patch("requests.request", side_effect=AssertionError("Configured Python functions must be used")):
            result = api_call("POST", "/api/search", {"query": "کفش", "filters": {"brand": "آلفا"}, "limit": 2})
            self.assertEqual(result["mode"], "team_function")
            self.assertEqual(calls[0], ("کفش", {"brand": "آلفا"}, 2))
            result = api_call("POST", "/api/reviews/ask", {"product_id": 1, "query": "دوخت"})
            self.assertEqual(result["mode"], "team_function")
            self.assertEqual(calls[1], (1, "دوخت"))
            self.assertTrue(result["data"]["evidence"])

    def test_unavailable_and_invalid_review_response_fall_back_to_real_evidence(self):
        invalid = {"product_id": 1, "answer": "پاسخ نامعتبر", "answer_mode": "grounded", "evidence": [
            {"comment_id": 201, "product_id": 1, "text": "شاهد محصول دیگری"},
        ]}
        with patch("requests.request", return_value=json_response(invalid)):
            entry = api_call("POST", "/api/reviews/ask", {"product_id": 1, "query": "کیفیت دوخت"})
            self.assertEqual(entry["mode"], "local")
            self.assertNotEqual(entry["data"]["answer"], invalid["answer"])
            self.assertTrue(all(item["product_id"] == 1 for item in entry["data"]["evidence"]))
        with patch("requests.request", side_effect=requests.Timeout):
            entry = api_call("POST", "/api/reviews/ask", {"product_id": 1, "query": "کیفیت دوخت"})
            self.assertEqual(entry["mode"], "local")

    def test_valid_team_reply_checks_ids_and_uses_original_text(self):
        data = {"product_id": "1", "answer": "تجربهٔ مربوط به دوخت", "answer_mode": "grounded", "evidence": [
            {"comment_id": "101", "product_id": 1, "text": "متن جایگزین غیرواقعی"},
        ]}
        with patch("requests.request", return_value=json_response(data)):
            entry = api_call("POST", "/api/reviews/ask", {"product_id": 1, "query": "دوخت"})
            self.assertEqual(entry["mode"], "serving")
            self.assertIn("دوخت ضعیف", entry["data"]["evidence"][0]["text"])
            self.assertNotIn("غیرواقعی", entry["data"]["evidence"][0]["text"])

    def test_predict_calls_actual_python_entrypoint_contract(self):
        calls = []
        sample_module = types.ModuleType("test_prediction_contract")

        def predict(text):
            calls.append(text)
            return {"label": "recommended", "model_version": "contract-test"}

        sample_module.predict_recommendation = predict
        with patch.dict(sys.modules, {sample_module.__name__: sample_module}), \
                patch.dict(os.environ, {"QBC13_PREDICT_FUNCTION": "test_prediction_contract:predict_recommendation"}), \
                patch("requests.request", side_effect=AssertionError("Direct model function must be used")):
            entry = predict_recommendation("کیفیت خوب بود")
            self.assertEqual(calls, ["کیفیت خوب بود"])
            self.assertEqual(entry["data"]["recommendation_status"], "recommended")
            app = AppTest.from_file(APP, default_timeout=20)
            app.session_state["page"] = "تحلیل فروشگاه"
            app.run()
            app.text_area(key="prediction_text").set_value("جنسش خوب است")
            app.button(key="FormSubmitter:prediction-پیش‌بینی").click().run()
            self.assertEqual(calls, ["کیفیت خوب بود", "جنسش خوب است"])
            self.assertFalse(app.exception)
            self.assertEqual(len(next(t for t in app.tabs if t.label == "Analytics & Dashboard").get("plotly_chart")), 3)
            self.assertIn("recommended", [entry.value for entry in app.code])

    def test_bundled_model_predicts_all_three_classes(self):
        import ml_service
        samples = (("کیفیت عالیه، واقعاً پیشنهاد می‌کنم", "خیلی راضی‌ام", "recommended"),
                   ("افتضاح بود اصلا نخرید", None, "not_recommended"),
                   ("بد نیست، معمولیه", None, "no_idea"))
        with patch("ui_services.ROOT", Path(APP).parent), \
                patch.object(ml_service, "MODEL_PATH", MODEL_FILE), patch.object(ml_service, "_model", None), \
                patch("requests.request", side_effect=AssertionError("The bundled model must run locally")):
            for body, title, expected in samples:
                actual = ml_service.predict_recommendation(body, title=title)
                entry = predict_recommendation(body, title=title)
                self.assertEqual(entry["mode"], "team_function")
                self.assertEqual(entry["data"]["recommendation_status"], expected)
                self.assertEqual(entry["data"]["confidence"], actual["confidence"])
                self.assertEqual(entry["data"]["probabilities"], actual["probabilities"])
                self.assertEqual(set(entry["data"]["probabilities"]), {"recommended", "not_recommended", "no_idea"})
                self.assertAlmostEqual(sum(entry["data"]["probabilities"].values()), 1, places=3)

    def test_invalid_input_is_not_a_no_idea_prediction(self):
        import ml_service
        with patch("ui_services.ROOT", Path(APP).parent), \
                patch.object(ml_service, "load_model", side_effect=AssertionError("Invalid inputs cannot reach the model")), \
                patch("requests.request", side_effect=AssertionError("Rejected input cannot reach the API")):
            for body, reason in (("", "empty_text"), ("😍😍", "symbols_or_emoji_only"),
                                 ("۱۲۳۴۵", "digits_only"), ("خوب خوب خوب خوب خوب", "spam_repetition")):
                entry = predict_recommendation(body)
                self.assertFalse(entry["data"]["valid"])
                self.assertIsNone(entry["data"]["recommendation_status"])
                self.assertEqual(entry["data"]["reason"], reason)

    def test_real_model_in_streamlit_with_scores_and_report(self):
        import ml_service
        with patch("ui_services.ROOT", Path(APP).parent), \
                patch.object(ml_service, "MODEL_PATH", MODEL_FILE), patch.object(ml_service, "_model", None), \
                patch("requests.request", side_effect=AssertionError("The bundled model must run locally")):
            app = AppTest.from_file(APP, default_timeout=30)
            app.session_state["main_tabs"] = "داشبورد و پیش‌بینی رضایت"
            app.run()
            app.text_input(key="prediction_title").set_value("خیلی راضی‌ام")
            app.text_area(key="prediction_text").set_value("کیفیت عالیه واقعاً پیشنهاد می‌کنم")
            app.button(key="FormSubmitter:prediction-پیش‌بینی").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            result = app.session_state["prediction_result"]["data"]
            self.assertEqual(result["recommendation_status"], "recommended")
            self.assertEqual(len(result["probabilities"]), 3)
            self.assertTrue(any(m.label == "اطمینان برآوردشدهٔ مدل" for m in app.metric))
            self.assertTrue(any(m.label == "Macro F1" and m.value == "0.6791" for m in app.metric))
            self.assertIn("recommended", [entry.value for entry in app.code])
            self.assertTrue(any("ارزیابی مدل تحویلی" in item.label for item in app.expander))
            app.text_input(key="prediction_title").set_value("")
            app.text_area(key="prediction_text").set_value("۱۲۳۴۵")
            app.button(key="FormSubmitter:prediction-پیش‌بینی").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("فقط شامل عدد" in item.value for item in app.warning))
            self.assertNotIn("no_idea", [entry.value for entry in app.code])
            self.assertFalse(any(m.label == "اطمینان برآوردشدهٔ مدل" for m in app.metric))
            self.assertEqual(len(next(t for t in app.tabs if t.label == "Analytics & Dashboard").get("plotly_chart")), 3)

    def test_missing_local_pipeline_uses_team_backend_or_specific_error(self):
        import ml_service
        with patch("ui_services.ROOT", Path(APP).parent), \
                patch.object(ml_service, "MODEL_PATH", self.folder / "missing.joblib"), \
                patch.object(ml_service, "_model", None):
            with patch("requests.request", return_value=json_response({"recommendation_status": "no_idea", "probability": 0.7})) as request:
                entry = api_call("POST", "/api/predict", {"text": "هنوز استفاده نکرده ام", "title": "نظری ندارم"})
                self.assertEqual(entry["mode"], "serving")
                self.assertEqual(entry["data"]["confidence"], 0.7)
                self.assertEqual(request.call_args.kwargs["json"], {"text": "نظری ندارم هنوز استفاده نکرده ام"})
                self.assertTrue(entry["notice"])
            with patch("requests.request", side_effect=requests.Timeout):
                with self.assertRaisesRegex(APIError, "sentiment_model.joblib"):
                    api_call("POST", "/api/predict", {"text": "کیفیت خوب بود"})

    def test_invalid_prediction_probabilities_do_not_crash_ui(self):
        with patch("requests.request", return_value=json_response({"recommendation_status": "recommended", "confidence": 2})):
            app = AppTest.from_file(APP, default_timeout=20)
            app.session_state["main_tabs"] = "داشبورد و پیش‌بینی رضایت"
            app.run()
            app.text_area(key="prediction_text").set_value("کیفیت خوب بود")
            app.button(key="FormSubmitter:prediction-پیش‌بینی").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("احتمال نامعتبر" in item.value for item in app.error))
            self.assertNotIn("recommended", [entry.value for entry in app.code])
            self.assertEqual(len(next(t for t in app.tabs if t.label == "Analytics & Dashboard").get("plotly_chart")), 3)

    def test_saved_m2_report_reproduces_metrics_and_confusion_matrix(self):
        from verify_ml_integration import verify_recorded_test
        _, report = verify_recorded_test()
        self.assertEqual(report["test_count"], 15000)
        self.assertAlmostEqual(report["macro_f1"], 0.6790791799331871)
        self.assertAlmostEqual(report["accuracy"], 0.7831333333333333)
        self.assertEqual(report["confusion_matrix"], [[1156,384,321],[391,1376,42],[1833,282,9215]])

    def test_three_tabs_local_search_to_chat_and_history(self):
        with patch("requests.request", side_effect=requests.ConnectionError):
            app = AppTest.from_file(APP, default_timeout=20).run()
            self.assertFalse(app.exception)
            labels = [tab.label for tab in app.tabs]
            self.assertEqual([label for label in labels if label in (
                "جست‌وجوی محصول", "دستیار تحلیلی کالا (RAG)", "داشبورد و پیش‌بینی رضایت")],
                ["جست‌وجوی محصول", "دستیار تحلیلی کالا (RAG)", "داشبورد و پیش‌بینی رضایت"])
            app.text_input(key="search_query").set_value("کفش مردانه")
            app.text_input(key="search_brand").set_value("آلفا")
            app.text_input(key="search_max_price").set_value("۲۰۰")
            app.button(key="FormSubmitter:search-جست‌وجو").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("محلی" in info.value for info in app.info))
            app.button(key="ask_search_0").click().run()
            self.assertEqual(app.session_state["main_tabs"], "دستیار تحلیلی کالا (RAG)")
            self.assertEqual(app.selectbox(key="review_product_choice").value, 1)
            app.chat_input(key="review_chat_input").set_value("کیفیت دوخت").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.chat_message), 2)
            self.assertTrue(any("شواهد نظرات" in expander.label for expander in app.expander))
            app.chat_input(key="review_chat_input").set_value("qwertynonexistent").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.chat_message), 4)
            self.assertTrue(any("شواهد کافی" in info.value for info in app.info))
            app.text_input(key="review_id").set_value("2")
            app.button(key="FormSubmitter:review_product_lookup-یافتن محصول").click().run()
            self.assertEqual(app.selectbox(key="review_product_choice").value, 2)
            self.assertEqual(len(app.chat_message), 0)

    def test_search_empty_and_prediction_timeout_no_crash(self):
        with patch("requests.request", return_value=json_response({"items": []})):
            app = AppTest.from_file(APP, default_timeout=20).run()
            app.text_input(key="search_query").set_value("دادهٔ ناموجود")
            app.button(key="FormSubmitter:search-جست‌وجو").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("پیدا نشد" in info.value for info in app.info))
        with patch("requests.request", side_effect=requests.Timeout):
            app.session_state["main_tabs"] = "داشبورد و پیش‌بینی رضایت"
            app.run()
            app.text_area(key="prediction_text").set_value("کیفیت خوب بود")
            app.button(key="FormSubmitter:prediction-پیش‌بینی").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("زمان انتظار" in error.value for error in app.error))
            self.assertEqual(len(next(t for t in app.tabs if t.label == "Analytics & Dashboard").get("plotly_chart")), 3)

    def test_missing_data_and_unavailable_api_remain_an_actionable_error(self):
        with patch.dict(os.environ, {"QBC13_ANALYTICS_DIR": str(self.folder / "missing")}), \
                patch("requests.request", side_effect=requests.ConnectionError):
            with self.assertRaisesRegex(APIError, "دادهٔ محلی"):
                api_call("POST", "/api/search", {"query": "کفش"})


if __name__ == "__main__":
    unittest.main()
