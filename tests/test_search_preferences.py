"""Preference ranking and currency contracts on a tiny independent catalogue."""
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from scripts.build_review_store import SCHEMA
from src.bm25_index import BM25IndexManager
from src.catalog import Catalog
from src.dashboard_rag import search
from src.hybrid_search import HybridSearchEngine
from src.rag_query import search_request
from src.search_qa import search_products


class PreferenceTests(TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        db = root / "catalog.sqlite3"
        records = [
            (1, "کیف زنانه مدل الف", "اکسسوری زنانه", "کیف زنانه", 120000, 30, 10),
            (2, "کیف زنانه مدل ب", "اکسسوری زنانه", "کیف زنانه", 180000, 95, 14),
            (3, "کیف زنانه مدل ج", "اکسسوری زنانه", "کیف زنانه", 900000, 90, 20),
            (4, "کتاب راهنمای سفر", "کتاب تاریخ و جغرافیا", "کتاب", 50000, 100, 20),
            (5, "ساک سفر", "لوازم سفر", "ساک سفری", 500000, 85, 12),
            (6, "ظرف سفر", "لوازم کمپینگ", "اجاق و ظرف سفری", 100000, 90, 20),
        ]
        with sqlite3.connect(db) as con:
            con.executescript(SCHEMA)
            for pid, title, category, group, price, rating, n in records:
                con.execute("INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (pid,title,category,group,None,"برند",price,None,0,rating,n,title,title))
            for cid, pid, label in [(1,1,"not_recommended"), (2,2,"recommended"),(3,2,"recommended"),
                                    (4,3,"recommended"),(5,4,"recommended"),(6,5,"recommended"),(7,6,"recommended")]:
                con.execute("INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?)",
                            (cid,pid,None,"کیفیت محصول",None,None,4,label))
            con.execute("INSERT INTO metadata VALUES (?,?)", ("supplied_price_unit_confirmed", json.dumps(False)))
        catalog = Catalog(db)
        index = BM25IndexManager(root / "bm25.pkl")
        index.build_from_records([{"product_id": str(pid), "title_fa": title, "product_text_lexical": title}
                                  for pid,title,*_ in records])
        engine = HybridSearchEngine(None, None, index)
        for target, value in (("src.search_qa.open_catalog", catalog), ("src.dashboard_rag.open_catalog", catalog),
                              ("src.search_qa.load_search_engine", engine)):
            mock = patch(target, return_value=value)
            mock.start(); self.addCleanup(mock.stop)
        env = patch.dict(os.environ, {"QBC13_SOURCE_PRICE_UNIT": "", "QBC13_UI_SEARCH_MODE": "bm25"})
        env.start(); self.addCleanup(env.stop)

    def test_satisfaction_preference_affects_rank(self):
        before = search_products("کیف زنانه", top_k=3)
        after = search_products("کیف زنانه با رضایت کاربران", top_k=3)
        self.assertEqual(before[0]["product_id"], "1")
        self.assertNotEqual(after[0]["product_id"], "1")
        self.assertTrue(after[0]["positive_feedback_signal"])
        self.assertFalse(after[-1]["positive_feedback_signal"])

    def test_official_bag_query_combines_preferences(self):
        q = "یک کیف برای استفاده‌ی روزمره می‌خوام که خیلی گرون نباشه و خریدارها هم ازش راضی باشن."
        req = search_request(q)
        self.assertTrue(req.prefer_satisfaction and req.prefer_affordability)
        result = search(q, limit=2)
        self.assertEqual(result["items"][0]["product_id"], 2)
        self.assertTrue(any("تأیید نشده" in note for note in result["search_notes"]))

    def test_travel_scope_excludes_books(self):
        r = search("چند وسیله‌ی کاربردی برای سفر معرفی کن که نظر کاربران درباره‌شون خوب باشه.")
        self.assertEqual({p["product_id"] for p in r["items"]}, {5,6})

    def test_raw_budget_and_explicit_currency(self):
        r = search("کیف زنانه تا ۲۰۰ هزار")
        self.assertEqual({p["product_id"] for p in r["items"]}, {1,2})
        self.assertTrue(any("واحد ثبت‌شده" in x for x in r["search_notes"]))
        with self.assertRaises(ValueError):
            search_products("کیف زنانه تا ۲۰ هزار تومان")
        converted = search_products("کیف زنانه تا ۲۰ هزار تومان", source_price_unit="rial")
        self.assertEqual({p["product_id"] for p in converted}, {"1", "2"})
        self.assertTrue(all(p["price_toman"] <= 20000 for p in converted))

    def test_structured_filters_remain_hard_constraints(self):
        r = search("کیف زنانه با رضایت کاربران", {"max_price": 150000})
        self.assertEqual([p["product_id"] for p in r["items"]], [1])
        self.assertTrue(any("شواهد کافی" in x for x in r["search_notes"]))
