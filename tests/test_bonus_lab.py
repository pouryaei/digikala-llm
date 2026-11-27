"""Bonus regressions. Fake human/SDK records only ever live in temp fixtures."""
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase,skipUnless
from unittest.mock import patch

from bonus.graph import Workflow,classify_intent,validate_request,generation_policy,EDGE_PAIRS
from bonus.observability import ResultCache,TraceStore
from bonus.evaluate_workflow import result_key,ROUTER_CASES
from bonus.human_review import cases,display_case,submit,records,agreement,export_csv
from bonus.quality_gap import wilson_upper,audit
from bonus.retrieval import search
from bonus.adapt_classifier import predict_adapted

ROOT=Path(__file__).resolve().parents[1]


class BonusTests(TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        patcher=patch.dict(os.environ,{"QBC13_API_KEY":"","QBC13_SOURCE_PRICE_UNIT":"","QBC13_UI_SEARCH_MODE":"bm25","FOROUHAR_LANGFUSE_ENABLED":"0"})
        patcher.start();self.addCleanup(patcher.stop)
        self.store=TraceStore(Path(self.temp.name)/"traces.sqlite")
        self.runner=Workflow(cache=ResultCache(),store=self.store,engine="python",export=False)

    def test_router_intents_and_ambiguity(self):
        for q,ids,c,expected in ROUTER_CASES:
            actual,_=classify_intent(validate_request({"query":q,"product_ids":ids,"category":c or None}))
            self.assertEqual(actual,expected,q)

    def test_graph_cache_same_content_and_redacted_trace(self):
        req={"query":"هدفون","intent":"search","use_cache":True}
        first=self.runner.run(req);second=self.runner.run(req)
        self.assertFalse(first["trace"]["cached"]);self.assertTrue(second["trace"]["cached"])
        self.assertEqual(result_key(first["result"]),result_key(second["result"]))
        self.assertEqual([x["node"] for x in second["trace"]["events"]],["route_intent","cache_lookup","finalize"])
        self.assertNotIn("هدفون",json.dumps(self.store.recent(),ensure_ascii=False))

    def test_missing_evidence_never_generates(self):
        go,why=generation_policy({"query":"چرا این محصول ارزش خرید دارد","allow_llm":True},{"status":"insufficient_evidence","evidence":[]})
        self.assertFalse(go);self.assertEqual(why,"missing_or_insufficient_evidence")
        out=self.runner.run({"query":"ایرادهای این محصول چیست؟","intent":"qa","product_ids":[1696173],"allow_llm":False})
        self.assertEqual(out["trace"]["api_calls"],0)
        self.assertNotIn("qa_generate",[x["node"] for x in out["trace"]["events"]])

    def test_hybrid_recovers_typo_and_preserves_no_results(self):
        cases_=json.loads((ROOT/"reports/bonus/retrieval_stress_cases.json").read_text())["cases"]
        r=cases_[0]
        ids={p["product_id"] for p in search(r["query"],method="hybrid_char")["items"]}
        self.assertIn(int(r["target_product_id"]),ids)
        self.assertEqual(search("بلیط هواپیما از تهران به پاریس",method="hybrid_char")["items"],[])
        self.assertEqual(search("کرم مرطوب کننده",{"brand":"brand-not-in-catalog-5821"},method="hybrid_char")["items"],[])

    def test_new_audit_real_sources(self):
        r=audit(limit=5)
        self.assertGreater(r["conservative_flags"],0)
        self.assertGreater(r["naive_flags"],r["conservative_flags"])
        for p in r["items"]:
            self.assertGreaterEqual(p["valid_labels"],20)
            self.assertLessEqual(p["conservative_gap"],p["naive_gap"])
            self.assertAlmostEqual(p["wilson_upper"],wilson_upper(p["recommended"],p["valid_labels"]))
            self.assertTrue(all(e["product_id"]==p["product_id"] for e in p["evidence"]))

    def test_adapted_model_real_inference(self):
        out=predict_adapted("کیفیت عالی بود و خرید را پیشنهاد میکنم")
        self.assertTrue(out["valid"])
        self.assertIn(out["label"],["recommended","not_recommended","no_idea"])
        self.assertAlmostEqual(sum(out["probabilities"].values()),1.,places=5)

    def test_human_review_blind_and_empty_until_submission(self):
        path=Path(self.temp.name)/"human.sqlite"
        case=cases()[0]
        self.assertNotIn("assessment",display_case(case))
        self.assertEqual(agreement(records(path))["human_submissions"],0)
        # Only a unit-test fixture; not included in the real human review DB.
        submit("test_fixture",case["case_id"],2,1,"test rationale",path=path)
        rows=records(path)
        self.assertEqual(len(rows),1);self.assertIn(b"test_fixture",export_csv(rows))
        self.assertEqual(agreement(rows)["unique_cases"],1)

    @skipUnless(importlib.util.find_spec("langgraph"),"Optional LangGraph SDK not installed")
    def test_real_langgraph_matches_fallback(self):
        req={"query":"هدفون","intent":"search","use_cache":False}
        pure=self.runner.run(req)
        engine=Workflow(cache=ResultCache(),store=self.store,engine="langgraph",export=False)
        actual=engine.run(req)
        self.assertEqual(actual["trace"]["engine"],"LangGraph")
        self.assertEqual(result_key(pure["result"]),result_key(actual["result"]))
