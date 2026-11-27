"""UI wiring and an optional real Streamlit smoke test in the pinned environment."""
import ast
import importlib.util
from pathlib import Path
import os
from unittest import TestCase,skipUnless
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/"app.py"
TAB="آزمایشگاه شواهد و امتیازی"
UI_READY=all(importlib.util.find_spec(x) for x in ("streamlit","duckdb","altair"))


class BonusUITests(TestCase):
    def test_main_tab_wiring(self):
        tree=ast.parse(APP.read_text(encoding="utf8"))
        tabs=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=="MAIN_TABS" for t in node.targets))
        self.assertEqual(len(tabs),4)
        self.assertEqual(tabs[-1],TAB)
        self.assertIn("render_bonus(open_review)",APP.read_text(encoding="utf8"))

    @skipUnless(UI_READY,"Streamlit/DuckDB UI dependencies not installed")
    def test_bonus_tab_opens(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ,{"FOROUHAR_LANGFUSE_ENABLED":"0","QBC13_API_KEY":"","QBC13_BACKEND_URL":"",
                                    "QBC13_REVIEW_GENERATION_MODE":"extractive"}):
            app=AppTest.from_file(str(APP),default_timeout=45)
            app.session_state["main_tabs"]=TAB
            app.session_state["bonus_lab_tabs"]="مسیریاب و مانیتورینگ"
            app.run()
            self.assertFalse(app.exception)
            self.assertIn(TAB,[t.label for t in app.tabs])


    @skipUnless(UI_READY and importlib.util.find_spec("langgraph"), "Graph/UI dependencies not installed")
    def test_qa_demo_executes_in_streamlit(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ, {"FOROUHAR_LANGFUSE_ENABLED": "0", "QBC13_API_KEY": "",
                                     "QBC13_BACKEND_URL": "", "QBC13_REVIEW_GENERATION_MODE": "extractive"}):
            app = AppTest.from_file(str(APP), default_timeout=45)
            app.session_state["main_tabs"] = TAB
            app.session_state["bonus_lab_tabs"] = "مسیریاب و مانیتورینگ"
            app.run()
            self.assertFalse(app.exception)
            app.selectbox(key="bonus_router_example").select("آمار مستقیم از داده").run()
            next(button for button in app.button if button.label == "اجرای گراف").click().run()
            self.assertFalse(app.exception)
            out = app.session_state["bonus_router_result"]
            self.assertEqual(out["trace"]["engine"], "LangGraph")
            self.assertEqual(out["trace"]["api_attempts"], 0)
            self.assertEqual(out["result"]["answer_mode"], "extractive")
            self.assertIn("qa_policy", [event["node"] for event in out["trace"]["events"]])
