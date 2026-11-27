"""Regression tests for the actual packaged corpus and API-free pipeline."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pipeline import run_rag_pipeline, find_product_candidates, compare_products
from scripts.check_rag import verify_files, read_manifest, demo_examples
from src.catalog import open_catalog
from src.rag_query import search_request
from src.search_qa import answer_product_question
from src.task2_evidence_selector import polarity, _list_segments
from src.task2_review_store import FIELDS, SQLiteReviewStore
from src.config import ROOT, Config

EXAMPLES = demo_examples()
SHOE = EXAMPLES["primary"]["product_id"]
SHOE_TITLE = EXAMPLES["primary"]["title_fa"]
SHOE_REVIEWS = EXAMPLES["primary"]["review_rows"]
OTHER = EXAMPLES["comparison"]["product_id"]
SHOE_FA = str(SHOE).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))
MANIFEST = read_manifest()


class NeverGenerate:
    model = "must-not-run"

    def generate(self, *_):
        raise AssertionError("Did not expect an LLM invocation")


class Integration(unittest.TestCase):
    def test_indices_and_catalogue_agree(self):
        a = verify_files()
        self.assertEqual(a["products"], MANIFEST["product_count"])
        self.assertEqual(a["bm25_documents"], MANIFEST["product_count"])
        self.assertEqual(a["reviews"], MANIFEST["review_rows"])
        self.assertEqual(a["products_with_reviews"], MANIFEST["product_count"])
        self.assertEqual(a["analytics_reviews"], MANIFEST["review_rows"])
        self.assertEqual(a["dense_available"], MANIFEST.get("dense_available", False))
        if a["dense_available"]:
            self.assertEqual(a["faiss_vectors"], MANIFEST["product_count"])

    def test_search_then_grounded_answer(self):
        hits = find_product_candidates(SHOE_TITLE, top_k=3)
        self.assertEqual(hits[0]["product_id"], str(SHOE))
        r = run_rag_pipeline(f"ایرادهای پرتکرار {SHOE_TITLE} چیست؟", SHOE, generation_mode="extractive")
        self.assertEqual((r["status"], r["product_id"], r["stats"]["total_reviews"]), ("answered", SHOE, SHOE_REVIEWS))
        self.assertGreater(len(r["evidence"]), 0)
        store = SQLiteReviewStore(Config.CATALOG_PATH)
        rows = {row["comment_id"]: row for row in store.get(SHOE).reviews}
        for e in r["evidence"]:
            self.assertEqual(e["product_id"], SHOE)
            self.assertIn(e["comment_id"], rows)
            self.assertTrue(any(e["text"] in (rows[e["comment_id"]][f] or "") for f in FIELDS))

    def test_persian_id_and_question_id(self):
        with_digits = run_rag_pipeline("نظرها دربارهٔ جنس این محصول چیست؟", SHOE_FA, generation_mode="extractive")
        in_query = run_rag_pipeline(f"ایرادهای محصول با شناسه محصول {SHOE_FA} چیست؟", generation_mode="extractive")
        self.assertEqual(with_digits["product_id"], SHOE)
        self.assertEqual((in_query["product_id"], in_query["product_resolution"]), (SHOE, "question_id"))

    def test_ambiguity_no_model_call(self):
        for question in ("ایرادهای این محصول چیست؟", "کیفیت کفش چطور است؟", f"ایرادهای #{SHOE} و #{OTHER} چیست؟"):
            with self.subTest(question=question):
                r = run_rag_pipeline(question, generation_mode="llm", llm=NeverGenerate())
                self.assertEqual(r["status"], "needs_product")
                self.assertIsNone(r["answer"])
                self.assertEqual(r["meta"]["api_calls"], 0)

    def test_album_with_two_full_titles_needs_selection(self):
        cat = open_catalog().get_products([SHOE, OTHER])
        titles = " و ".join(x["title_fa"] for x in cat.values())
        r = run_rag_pipeline(f"ایرادهای {titles} چیست؟", llm=NeverGenerate(), generation_mode="llm")
        self.assertEqual(r["status"], "needs_product")
        self.assertEqual({p["product_id"] for p in r["candidates"]}, {SHOE, OTHER})

    def test_no_data_separates_missing_reviews_from_out_of_scope(self):
        with open_catalog().connection() as con:
            missing = con.execute("SELECT MAX(product_id)+1 FROM products").fetchone()[0]
        a = run_rag_pipeline("کیفیت چطور است؟", missing, generation_mode="llm", llm=NeverGenerate())
        self.assertEqual(a["status"], "insufficient_evidence")
        self.assertIn("خارج", a["reason"])
        self.assertEqual(a["meta"]["api_calls"], 0)

    def test_price_not_silently_inferred(self):
        q = "کفش بین ۳۰۰ تا ۶۰۰ هزار تومان"
        req = search_request(q)
        self.assertEqual((req.text, req.min_price, req.max_price), ("کفش", 300000, 600000))
        with patch.dict(os.environ, {"QBC13_SOURCE_PRICE_UNIT":""}):
            with self.assertRaises(ValueError):
                find_product_candidates(q)
        hits = find_product_candidates(q, source_price_unit="rial")
        self.assertTrue(hits)
        for hit in hits:
            self.assertLessEqual(300000, hit["price_toman"])
            self.assertLessEqual(hit["price_toman"], 600000)
            self.assertEqual(hit["price_toman"], hit["price_recorded"]/10)

    def test_comparison_citations_never_cross_products(self):
        c = compare_products([SHOE, OTHER], "راحتی کفش چطور است؟")
        self.assertEqual(c["status"], "compared")
        self.assertEqual([item["product_id"] for item in c["products"]], [SHOE, OTHER])
        for item in c["products"]:
            self.assertTrue(item["facts"])
            self.assertTrue(item["strengths"] or item["weaknesses"])
            for e in item["strengths"] + item["weaknesses"] + item["related_reviews"]:
                self.assertEqual(e["product_id"], item["product_id"])
        self.assertEqual(c["meta"]["api_calls"], 0)

    def test_llm_missing_key_uses_local_result(self):
        with patch.dict(os.environ, {"QBC13_API_KEY":""}):
            r = run_rag_pipeline("جنس این محصول چطور است؟", SHOE, generation_mode="llm", use_cache=False)
        self.assertEqual(r["answer_mode"], "extractive")
        self.assertEqual(r["meta"]["api_calls"], 0)
        self.assertIn("fallback_reason", r["meta"])

    def test_explicit_invalid_source_does_not_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_rag_pipeline("کیفیت چطور است؟", SHOE, source=Path(d)/"missing.parquet", generation_mode="extractive")
        self.assertEqual(r["status"], "insufficient_evidence")

    def test_legacy_structured_contract(self):
        r = answer_product_question(SHOE, "کیفیت و جنس چطور است؟")
        self.assertIsInstance(r["answer"], dict)
        self.assertFalse(r["answer"]["insufficient_evidence"])
        self.assertEqual(r["product_id"], str(SHOE))

    def test_negated_praise_and_list_quotes(self):
        self.assertEqual(polarity("body_raw", "جنس مناسب نیست"), "negative")
        self.assertEqual(polarity("body_raw", "کیفیت بد نیست"), "positive")
        raw = r"['جنس مقاوم و باکیفیت\r', 'نرم و راحت']"
        parts = _list_segments(raw)
        self.assertEqual(parts, ["جنس مقاوم و باکیفیت", "نرم و راحت"])
        self.assertTrue(all(part in raw for part in parts))

    def test_can_call_from_a_different_working_directory(self):
        with tempfile.TemporaryDirectory(prefix="Forouhar test ") as d:
            cmd = [sys.executable, "-S", "-X", "utf8", str(ROOT/"script/run_rag.py"),
                   "ask", "--product-id", str(SHOE), "--query", "کیفیت چیست؟", "--json"]
            r = subprocess.run(cmd, cwd=d, capture_output=True, encoding="utf-8", check=True, timeout=20)
        ans = json.loads(r.stdout)
        self.assertEqual(ans["stats"]["total_reviews"], SHOE_REVIEWS)


if __name__ == "__main__":
    unittest.main()
