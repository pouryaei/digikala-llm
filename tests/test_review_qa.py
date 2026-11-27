import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.task2_evidence_selector import parse_question, polarity
from src.review_qa import ask_reviews
from src.task2_review_store import FIELDS, JsonlReviewStore, MemoryReviewStore, PayloadReviewStore, ParquetReviewStore, ReviewDataError

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "task2_demo" / "reviews.jsonl"
SHOE = 10133811
BOOK = 1126863
NO_REVIEWS = 10065041


class ValidFakeLLM:
    model = "offline_contract_test"
    def __init__(self): self.calls = 0
    def generate(self, system, user):
        self.calls += 1
        row = json.loads(user)["retrieved_reviews"][0]
        return {"text": json.dumps({"summary": "فقط پاسخ ساختاریافتهٔ آزمون", "observations": [{"point": "این تجربه در نظر ثبت شده است.", "comment_ids": [row["comment_id"]], "quote": row["text"][:60]}]}, ensure_ascii=False), "latency": .001, "prompt_tokens": 100, "completion_tokens": 25}


class FakeBadID:
    model = "offline_invalid_id_test"
    def generate(self, system, user):
        return {"text": json.dumps({"summary": "ادعای ساختگی آزمون", "positive_points": [{"point": "ادعای ساختگی آزمون", "comment_ids": [999999999999], "quote": "ساختگی"}]}, ensure_ascii=False), "latency": .001, "prompt_tokens": 10, "completion_tokens": 5}


class FakeBadQuote:
    model = "offline_invalid_quote_test"
    def generate(self, system, user):
        # Retry appends feedback, so split after the original JSON if necessary.
        row = json.JSONDecoder().raw_decode(user)[0]["retrieved_reviews"][0]
        return {"text": json.dumps({"observations": [{"point": "ادعای ساختگی آزمون", "comment_ids": [row["comment_id"]], "quote": "نقل قول ساختگی که در نظر وجود ندارد"}]}, ensure_ascii=False), "latency": 0, "prompt_tokens": 2, "completion_tokens": 1}


class FakeMissingQuote(FakeBadQuote):
    model = "offline_missing_quote_test"
    def generate(self, system, user):
        row = json.JSONDecoder().raw_decode(user)[0]["retrieved_reviews"][0]
        return {"text": json.dumps({"observations": [{"point": "ادعای ساختگی آزمون", "comment_ids": [row["comment_id"]]}]}, ensure_ascii=False), "latency": 0, "prompt_tokens": 2, "completion_tokens": 1}


class FakeInvalidJson:
    model = "offline_invalid_json_test"
    def generate(self, system, user): return {"text": "not JSON", "latency": 0, "prompt_tokens": 2, "completion_tokens": 1}


class NeverCall:
    model = "must_not_be_called"
    def generate(self, *args): raise AssertionError("No evidence must prevent LLM calls")


class ReviewQATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.store = JsonlReviewStore(SAMPLE)

    def assert_evidence(self, result):
        pid = result["product_id"]
        rows = {r["comment_id"]: r for r in self.store.get(pid).reviews}
        ids = []
        for evidence in result["evidence"]:
            self.assertEqual(evidence["product_id"], pid)
            self.assertIn(evidence["comment_id"], rows)
            row = rows[evidence["comment_id"]]
            self.assertTrue(any(evidence["text"] in (row.get(f) or "") for f in FIELDS))
            ids.append(evidence["comment_id"])
        self.assertEqual(len(set(ids)), len(ids))
        if result["answer"]:
            cited = {int(i) for i in re.findall(r"نظر\s+(\d+)", result["answer"])}
            self.assertTrue(cited <= set(ids))
        for t in result.get("theme_counts", []):
            self.assertTrue(0 < t["count"] <= t["denominator"])
            self.assertTrue(set(t["evidence_ids"]) <= set(ids))

    def test_four_official_questions_use_real_evidence(self):
        cases = [("مردم بیشتر از چه چیزی در این محصول راضی بودند؟", "positive"),
                 ("ایرادهای پرتکرار این محصول چیست؟", "negative"),
                 ("خریداران درباره کیفیت این محصول چه گفته‌اند؟", "aspect"),
                 ("آیا با توجه به تجربه کاربران ارزش خرید دارد؟", "value")]
        for q, intent in cases:
            with self.subTest(q=q):
                result = ask_reviews(SHOE, q, store=self.store)
                self.assertEqual(result["answer_mode"], "extractive")
                self.assertEqual(result["intent"], intent)
                self.assertTrue(result["evidence"])
                self.assert_evidence(result)
                if intent in ("positive", "negative"): self.assertTrue(result["theme_counts"])

    def test_partial_question_does_not_claim_missing_facet(self):
        r = ask_reviews(10046185, "آیا کاربران از کیفیت و کارایی کیف پول راضی هستند؟", store=self.store)
        self.assertEqual(r["answer_mode"], "extractive")
        facets={v["theme"]:v for v in r["facet_coverage"]}
        self.assertTrue(facets["کیفیت و جنس"]["evidence_ids"])
        self.assertFalse(facets["کارایی و عملکرد"]["evidence_ids"])
        self.assertIn("شاهد صریح کافی", r["answer"])
        self.assert_evidence(r)

    def test_recommendation_denominator_is_all_valid_labels(self):
        r = ask_reviews(SHOE, "ارزش خرید دارد؟", store=self.store)
        self.assertEqual(r["stats"]["total_reviews"], 65)
        self.assertEqual(r["stats"]["valid_label_count"], 57)
        self.assertEqual(r["stats"]["recommended_count"], 40)
        self.assertEqual(r["stats"]["not_recommended_count"], 4)
        self.assertEqual(r["stats"]["no_idea_count"], 13)
        self.assertAlmostEqual(r["stats"]["recommendation_rate"], 40/57)
        self.assert_evidence(r)

    def test_missing_reviews_abstain_and_never_call_llm(self):
        r = ask_reviews(NO_REVIEWS, "کیفیت چطور است؟", store=self.store, llm=NeverCall())
        self.assertEqual(r["answer_mode"], "insufficient_evidence")
        self.assertEqual(r["stats"]["total_reviews"], 0)
        self.assertEqual(r["meta"]["api_calls"], 0)
        self.assertFalse(r["evidence"])

    def test_out_of_domain_facet_abstains(self):
        r = ask_reviews(BOOK, "باتری این محصول چند ساعت شارژ نگه می‌دارد؟", store=self.store)
        self.assertEqual(r["answer_mode"], "insufficient_evidence")
        self.assertEqual(r["stats"]["total_reviews"], 68)
        self.assertFalse(r["evidence"])

    def test_unknown_text_abstains(self):
        self.assertEqual(ask_reviews(SHOE, "qwertynonexistent", store=self.store)["answer_mode"], "insufficient_evidence")

    def test_product_outside_sample_is_not_reported_as_no_reviews(self):
        r = ask_reviews(999999999, "کیفیت", store=self.store)
        self.assertFalse(r["stats"]["complete_for_product"])
        self.assertIn("محدوده", r["reason"])

    def test_no_cross_product_ids_or_duplicates(self):
        rows = [{"comment_id": 1, "product_id": 10, "body": "کیفیت خوبی دارد", "recommendation_status": "recommended"},
                {"comment_id": 1, "product_id": 10, "body": "کیفیت خوبی دارد"},
                {"comment_id": 2, "product_id": 20, "body": "کیفیت بسیار عالی"}]
        result = ask_reviews(10, "کیفیت", store=MemoryReviewStore(rows))
        self.assertEqual(result["stats"]["total_reviews"], 1)
        self.assertEqual([x["comment_id"] for x in result["evidence"]], [1])
        self.assertFalse(result["stats"]["complete_for_product"])

    def test_word_boundary_and_common_persian_suffixes(self):
        spec = parse_question("مردم بیشتر از چه چیزی راضی بودند؟")
        self.assertEqual(spec.intent, "positive")
        self.assertNotIn("مزه و بو", spec.themes)
        spec = parse_question("کیفیتش چطوره؟")
        self.assertIn("کیفیت و جنس", spec.themes)
        self.assertEqual(polarity("body_raw", "نسبت به قیمتش بد نیست"), "positive")

    def test_invalid_input(self):
        for pid, query in [(False, "کیفیت"), (-1, "کیفیت"), (SHOE, "   ")]:
            with self.assertRaises(ValueError): ask_reviews(pid, query, store=self.store)

    def test_grounded_model_contract_with_exact_quote(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"QBC13_LLM_CACHE_PATH":str(Path(temp)/"cache.sqlite")}):
            llm = ValidFakeLLM()
            r = ask_reviews(SHOE, "کیفیت و جنس", store=self.store, llm=llm)
            self.assertEqual(r["answer_mode"], "grounded")
            self.assertEqual(llm.calls, 1)
            self.assert_evidence(r)
            self.assertIn("جمع‌بندی مدل زبانی", r["answer"])
            warm = ask_reviews(SHOE, "کیفیت و جنس", store=self.store, llm=llm)
            self.assertTrue(warm["meta"]["cached"])
            self.assertEqual(warm["meta"]["api_calls"], 0)
            self.assertEqual(warm["meta"]["prompt_tokens"], 0)
            self.assertEqual(llm.calls, 1)

    def test_invalid_ids_quotes_missing_quotes_and_json_fallback(self):
        for model in (FakeBadID(), FakeBadQuote(), FakeMissingQuote(), FakeInvalidJson()):
            with self.subTest(model=model.model):
                r = ask_reviews(SHOE, "کیفیت و جنس", store=self.store, llm=model, use_cache=False)
                self.assertEqual(r["answer_mode"], "extractive")
                self.assertNotIn("ادعای ساختگی", r["answer"])
                self.assert_evidence(r)

    def test_tampered_sample_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d)/"reviews.jsonl"
            shutil.copy(SAMPLE, target)
            shutil.copy(SAMPLE.with_name("manifest.json"), target.with_name("manifest.json"))
            with target.open("a", encoding="utf-8") as fp: fp.write("\n")
            with self.assertRaises(ReviewDataError): JsonlReviewStore(target)


@unittest.skipUnless(importlib.util.find_spec("zstandard"), "zstandard not installed")
class PayloadTests(unittest.TestCase):
    def test_payload_and_cache_integrity(self):
        import zstandard
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path = root/"reviews_payload.csv.zst"
            buffer = io.StringIO(newline="")
            w=csv.writer(buffer); w.writerow(["comment_id","product_id","body_raw","recommendation_status"])
            w.writerow([1,10,"کیفیت خوب است","recommended"]);w.writerow([2,20,"متن محصول دیگر","not_recommended"])
            path.write_bytes(zstandard.ZstdCompressor().compress(buffer.getvalue().encode()))
            (root/"reviews_payload.json").write_text(json.dumps({"row_count":2,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}))
            store=PayloadReviewStore(path, cache_dir=root/"cache");r=store.get_many([10,20,30])
            self.assertEqual([len(r[x].reviews) for x in [10,20,30]],[1,1,0])
            self.assertEqual(r[10].reviews[0]["comment_id"],1)
            f=store.cache_dir/"10.json";v=json.loads(f.read_text());v["reviews"][0]["body_raw"]="tampered";f.write_text(json.dumps(v))
            fresh=PayloadReviewStore(path, cache_dir=root/"cache")
            self.assertEqual(fresh.get(10).reviews[0]["body_raw"],"کیفیت خوب است")


@unittest.skipUnless(importlib.util.find_spec("duckdb"), "DuckDB not installed in this execution environment")
class ParquetTests(unittest.TestCase):
    def test_parquet_product_filter_and_aliases(self):
        import duckdb
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"comments_clean.parquet"
            with duckdb.connect() as c:
                c.execute("CREATE TABLE r(id VARCHAR,product_id VARCHAR,body VARCHAR,recommendation_status VARCHAR)")
                c.execute("INSERT INTO r VALUES ('1','10','کیفیت عالی','recommended'),('2','20','کیفیت ضعیف','not_recommended')")
                c.table("r").write_parquet(str(p))
            r=ask_reviews(10,"کیفیت",source=p)
            self.assertEqual(r["answer_mode"],"extractive")
            self.assertEqual([e["comment_id"] for e in r["evidence"]],[1])
    def test_reject_slim_data(self):
        import duckdb
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"reviews.parquet"
            with duckdb.connect() as c:
                c.execute("CREATE TABLE r(comment_id BIGINT,product_id BIGINT,rate DOUBLE,recommendation_status VARCHAR)")
                c.execute("INSERT INTO r VALUES (1,10,5,'recommended')")
                c.table("r").write_parquet(str(p))
            with self.assertRaises(ReviewDataError): ParquetReviewStore(p)


@unittest.skipUnless(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"), "FastAPI test dependencies not installed")
class APITests(unittest.TestCase):
    def test_real_sample_api_contract(self):
        from fastapi.testclient import TestClient
        from task2_api import app
        with patch.dict(os.environ,{"QBC13_REVIEW_SOURCE":str(SAMPLE),"QBC13_REVIEW_GENERATION_MODE":"extractive"}):
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code,200)
                r=client.post("/api/reviews/ask",json={"product_id":SHOE,"query":"کیفیت"})
                self.assertEqual(r.status_code,200)
                self.assertEqual(r.json()["mode"],"serving")
                self.assertEqual(r.json()["data"]["answer_mode"],"extractive")
                self.assertTrue(r.json()["data"]["evidence"])
                self.assertEqual(client.post("/api/reviews/ask",json={"product_id":False,"query":"کیفیت"}).status_code,422)
                self.assertEqual(client.post("/api/reviews/ask",json={"product_id":NO_REVIEWS,"query":"کیفیت"}).json()["data"]["answer_mode"],"insufficient_evidence")


if __name__ == "__main__": unittest.main()
