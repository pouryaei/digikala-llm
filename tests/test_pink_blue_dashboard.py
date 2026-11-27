"""End-to-end checks of the original dashboard layout with local RAG."""
import hashlib
from importlib.util import find_spec
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch

from src.config import ROOT
from src.original_dashboard import load_original
from scripts.check_rag import demo_examples, read_manifest

ENV = {key: "" for key in (
    "QBC13_BACKEND_URL", "QBC13_SEARCH_FUNCTION", "QBC13_REVIEW_FUNCTION", "QBC13_ANALYTICS_DIR",
    "FOROUHAR_COMMENTS_PATH", "QBC13_REVIEW_SOURCE", "QBC13_API_KEY", "QBC13_UI_SEARCH_MODE",
)}
ENV["QBC13_REVIEW_GENERATION_MODE"] = "extractive"
EXAMPLES = demo_examples()
QUERY = EXAMPLES["primary"]["title_fa"]
PID = EXAMPLES["primary"]["product_id"]
OTHER = EXAMPLES["comparison"]["product_id"]
N_REVIEWS = EXAMPLES["primary"]["review_rows"]
PID_FA = str(PID).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))
MANIFEST = read_manifest()


@skipUnless(find_spec("streamlit"), "Install requirements-ui.txt for dashboard tests.")
class PinkBlueDashboardTests(TestCase):
    def test_search_select_chat_stays_in_same_app(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ, ENV), patch("requests.request", side_effect=AssertionError("unexpected API call")):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["main_tabs"], "جست‌وجوی محصول")
            app.text_input(key="search_query").set_value(QUERY)
            app.button(key="FormSubmitter:search-جست‌وجو").click().run()
            self.assertFalse(app.exception)
            hits = app.session_state["search_result"]["data"]["items"]
            self.assertEqual((hits[0]["product_id"], hits[0]["comment_count"]), (PID,N_REVIEWS))
            app.button(key="ask_search_0").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["main_tabs"], "دستیار تحلیلی کالا (RAG)")
            self.assertEqual(app.selectbox(key="review_product_choice").value, PID)
            app.chat_input(key="review_chat_input").set_value("ایرادهای پرتکرار این محصول چیست؟").run()
            self.assertFalse(app.exception)
            answer = app.session_state["review_chats"][PID][-1]["entry"]["data"]
            self.assertEqual((answer["stats"]["total_reviews"], len(answer["evidence"])), (N_REVIEWS,EXAMPLES["primary"]["evidence_count"]))
            self.assertEqual(answer["meta"]["api_calls"],0)
            self.assertTrue(all(row["product_id"] == PID for row in answer["evidence"]))
            self.assertTrue(any("شواهد نظرات" in box.label for box in app.expander))
            self.assertGreaterEqual(len(app.chat_message),2)

    def test_catalogue_filters_comparison_and_analytics_scope(self):
        from analytics_service import BUNDLED_ANALYTICS_DIR, get_category_review_distribution, resolve_analytics_files
        from ui_services import api_call
        with patch.dict(os.environ, ENV), patch("requests.request", side_effect=AssertionError("unexpected API call")):
            search = api_call("POST", "/api/search", {"query": PID_FA, "filters": {}})
            self.assertEqual([r["product_id"] for r in search["data"]["items"]], [PID])
            exclude = api_call("POST", "/api/search", {"query": QUERY, "filters": {"brand": "برند ناموجود"}})
            self.assertFalse(exclude["data"]["items"])
            result = api_call("POST", "/api/products/compare", {"product_ids": [PID,OTHER]})["data"]
            self.assertEqual([r["product_id"] for r in result["items"]], [PID,OTHER])
            self.assertTrue({r["product_id"] for r in result["evidence"]} <= {PID,OTHER})
            self.assertEqual(resolve_analytics_files()[0].parent,BUNDLED_ANALYTICS_DIR)
            self.assertEqual(int(get_category_review_distribution()["review_count"].sum()),MANIFEST["review_rows"])

    def test_existing_file_is_not_rewritten(self):
        from ui_services import api_call
        original = """def api_call(*args,**kwargs): raise RuntimeError()
def main(): return 'original user style'
"""
        with TemporaryDirectory() as folder:
            path = Path(folder) / "forouhar_app1.py"
            path.write_text(original, encoding="utf-8")
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            mod = load_original(path)
            self.assertIs(mod.api_call, api_call)
            self.assertEqual(mod.main(), "original user style")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
