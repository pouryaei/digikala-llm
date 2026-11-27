import json
from pathlib import Path
import tempfile
from unittest import TestCase
from src.bm25_index import BM25IndexManager
from src.task2_evidence_selector import polarity
from src.review_qa import ask_reviews
from evaluation.run_evaluation import CASE_PATH, ranked_metrics, token_usage
from evaluation.cost_from_usage import calculate

class Section4Tests(TestCase):
    def test_search_guard_rejects_accidental_word_match_and_keeps_named_item(self):
        with tempfile.TemporaryDirectory() as folder:
            index=BM25IndexManager(Path(folder)/'index.pkl')
            index.build_from_records([
                {"product_id":"1","title_fa":"هدفون مانیتورینگ","product_text_lexical":"هدفون مانیتورینگ"},
                {"product_id":"2","title_fa":"کرم پاریس","product_text_lexical":"کرم پاریس"},
                {"product_id":"3","title_fa":"کتاب سفر تهران","product_text_lexical":"کتاب سفر تهران"},
            ])
            self.assertEqual(index.search("بلیط هواپیما از تهران به پاریس"),[])
            self.assertEqual(index.search("هدفون مانیتورینگ برای میکس موسیقی")[0]['product_id'],'1')
            self.assertEqual(index.search("میخوام هدفون"),[r for r in index.search("هدفون")])
    def test_literal_mild_disagreement_not_a_complaint(self):
        self.assertNotEqual(polarity("disadvantages_raw","بد نبود"),"negative")
        self.assertEqual(polarity("body_raw","متوسط بد نبود"),"positive")
    def test_ranked_metrics(self):
        m=ranked_metrics([2,1,0],[2,2,1,0],8)
        self.assertEqual(m["strict_precision_at_k"],1/8)
        self.assertEqual(m["strict_precision_returned"],1/3)
        self.assertEqual(m["direct_recall_at_k"],1/2)
        self.assertEqual(m["hit_at_k"],1)
    def test_usage_cost_unknown_never_zeroed(self):
        self.assertEqual(calculate([{"api_calls":0,"api_attempts":0,"prompt_tokens":0,"completion_tokens":0,"usage_known":True}])["complete_cost_estimate_usd"],0)
        value=calculate([{"api_calls":1,"api_attempts":2,"prompt_tokens":1000,"completion_tokens":100,"usage_known":True}],1,2)
        self.assertAlmostEqual(value["cost_from_reported_tokens_usd"],.0012)
        self.assertIsNone(value["complete_cost_estimate_usd"])
        self.assertFalse(token_usage({"api_calls":1,"api_attempts":1})["usage_known"])
    def test_reference_split_has_no_shared_products(self):
        cases=json.loads(CASE_PATH.read_text(encoding='utf-8'))
        ids={g:{r["product_id"] for r in cases["qa"] if r["split"]==g} for g in ("dev","holdout")}
        self.assertFalse(ids["dev"]&ids["holdout"])
        self.assertEqual(sum(len(r["qrels"]) for r in cases["qa"]),275)
