import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
from streamlit.testing.v1 import AppTest

from analytics_service import (
    CLEAN_FILENAMES,
    FINAL_FILENAMES,
    get_category_review_distribution,
    get_price_vs_rating_data,
    get_top_rated_brands,
)
from build_analytics_data import build_analytics_data


PRODUCTS = [
    ("p1", "کیف یک", "کیف‌سفر", "برند‌یک", 100.0),
    ("p2", "کیف دو", "کیف سفر", "برند یک", 200.0),
    ("p3", "کتاب", "کتاب", "برند دو", None),
    ("p4", "محصول بدون دسته", None, "", 1000.0),
    ("p5", "کیف بدون قیمت", "کیف سفر", "برند دو", 0.0),
]
REVIEWS = [
    ("r1", "p1", 5.0, "کیف سبک است."),
    ("r2", "p1", 3.0, "اندازه خوب است."),
    ("r3", "p2", 2.0, "کیف جا دارد."),
    ("r4", "p2", None, "هنوز استفاده نکردم."),
    ("r5", "p3", 4.5, "کتاب مفیدی است."),
    ("r6", "p4", 4.5, "خوب بود."),
    ("r7", "p5", 6.0, "امتیاز این ردیف نامعتبر است."),
]


def write_fixture(directory, products=PRODUCTS, reviews=REVIEWS):
    directory.mkdir()
    with duckdb.connect(config={"threads": 1}) as con:
        con.execute("""
            CREATE TABLE p (product_id VARCHAR, title_fa VARCHAR, category1 VARCHAR,
                            brand VARCHAR, price DOUBLE);
            CREATE TABLE r (comment_id VARCHAR, product_id VARCHAR, rating_5 DOUBLE,
                            text_clean VARCHAR);
        """)
        if products:
            con.executemany("INSERT INTO p VALUES (?, ?, ?, ?, ?)", products)
        if reviews:
            con.executemany("INSERT INTO r VALUES (?, ?, ?, ?)", reviews)
        products_query = """
            SELECT *, title_fa AS title_raw, title_fa AS product_text_norm,
                   'old' AS product_text_lexical FROM p
        """
        reviews_query = """
            SELECT *, text_clean AS body_raw, text_clean AS body_clean,
                   'old' AS text_key, 99::BIGINT AS word_count, false AS rag_eligible FROM r
        """
        for query, name in zip((products_query, reviews_query), CLEAN_FILENAMES):
            con.sql(query).write_parquet(str(directory / name), compression="zstd")


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="m1 analytics ")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "processed_repo_based"
        self.output = self.root / "analytics"

    def test_final_parquet_contains_normalized_text_and_preserves_source_fields(self):
        title = "\u200c كيف  دستى\u00a0"
        body = " \u200cبررسي  كردم؛ مى \u200c\u200c ارزد \u200c "
        write_fixture(
            self.source,
            products=[("01", title, "كيف \u200cسفر", "كالا ياس", 100.0)],
            reviews=[("02", "01", 5.0, body)],
        )
        self.assertEqual(build_analytics_data(self.source, self.output), {"products": 1, "reviews": 1})
        with duckdb.connect() as con:
            products, reviews = (con.read_parquet(str(self.output / name)) for name in FINAL_FILENAMES)
            self.assertEqual(products.project(
                "product_id, title_fa, category1, brand, title_raw, product_text_lexical"
            ).fetchall()[0], ("01", "کیف دستی", "کیف‌سفر", "کالا یاس", title, "کیف دستی"))
            self.assertEqual(reviews.project(
                "comment_id, product_id, text_clean, body_clean, body_raw, word_count, rag_eligible"
            ).fetchall()[0], ("02", "01", "بررسی کردم؛ می‌ارزد", "بررسی کردم؛ می‌ارزد", body, 3, True))
            calculated_key = hashlib.md5("بررسیکردممیارزد".encode()).hexdigest()
            self.assertEqual(reviews.project("text_key").fetchall()[0], (calculated_key,))
            for name in FINAL_FILENAMES:
                compression = con.execute(
                    "SELECT DISTINCT compression FROM parquet_metadata(?)", [str(self.output / name)]
                ).fetchall()
                self.assertEqual(compression, [("ZSTD",)])
        self.assertEqual(get_category_review_distribution(data_dir=self.output).review_count.sum(), 1)

    def test_distribution_counts_unrated_reviews_and_missing_categories(self):
        write_fixture(self.source)
        result = get_category_review_distribution(data_dir=self.source).set_index("category")
        self.assertEqual(result.review_count.to_dict(), {"کیف سفر": 5, "کتاب": 1, "نامشخص": 1})
        self.assertAlmostEqual(result.review_share_pct.sum(), 100.0)

    def test_brands_combine_half_space_variants_and_count_valid_ratings(self):
        write_fixture(self.source)
        result = get_top_rated_brands(1, data_dir=self.source)
        self.assertEqual(result.brand.tolist(), ["برند دو"])
        result = get_top_rated_brands(category="كيف سفر", data_dir=self.source)
        self.assertEqual(result.brand.tolist(), ["برند یک"])
        self.assertEqual(result.rated_review_count.tolist(), [3])
        self.assertAlmostEqual(result.mean_review_rating_5.iloc[0], 10 / 3)

    def test_price_sample_does_not_change_correlation(self):
        write_fixture(self.source)
        all_points = get_price_vs_rating_data("کیف‌سفر", data_dir=self.source)
        sample = get_price_vs_rating_data("کیف سفر", max_points=1, data_dir=self.source)
        second_sample = get_price_vs_rating_data("کیف سفر", max_points=1, data_dir=self.source)
        self.assertEqual(all_points["data"].product_id.tolist(), ["p1", "p2"])
        self.assertEqual(len(sample["data"]), 1)
        self.assertEqual(sample["data"].product_id.tolist(), second_sample["data"].product_id.tolist())
        self.assertEqual(sample["product_count"], 2)
        self.assertAlmostEqual(all_points["pearson_r"], -1.0)
        self.assertAlmostEqual(sample["pearson_r"], all_points["pearson_r"])

    def test_unknown_category_and_undefined_correlation(self):
        write_fixture(self.source)
        unknown = get_price_vs_rating_data("دستهٔ ناموجود", data_dir=self.source)
        self.assertTrue(unknown["data"].empty)
        self.assertEqual(unknown["product_count"], 0)
        self.assertIsNone(unknown["pearson_r"])
        self.assertTrue(get_top_rated_brands(category="دستهٔ ناموجود", data_dir=self.source).empty)
        one = get_price_vs_rating_data("نامشخص", data_dir=self.source)
        self.assertEqual(one["product_count"], 1)
        self.assertIsNone(one["pearson_r"])

    def test_empty_parquet_tables(self):
        write_fixture(self.source, products=[], reviews=[])
        self.assertEqual(build_analytics_data(self.source, self.output), {"products": 0, "reviews": 0})
        self.assertTrue(get_category_review_distribution(data_dir=self.output).empty)
        self.assertTrue(get_top_rated_brands(data_dir=self.output).empty)
        result = get_price_vs_rating_data(data_dir=self.output)
        self.assertTrue(result["data"].empty)
        self.assertIsNone(result["pearson_r"])

    def test_orphan_review_does_not_publish_a_partial_pair(self):
        write_fixture(self.source, reviews=[("r1", "absent", 5.0, "نمونه")])
        with self.assertRaisesRegex(ValueError, "no matching product"):
            build_analytics_data(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_streamlit_dashboard_tab_uses_finalized_parquet(self):
        write_fixture(self.source)
        build_analytics_data(self.source, self.output)
        with patch.dict("os.environ", {"QBC13_ANALYTICS_DIR": str(self.output)}):
            app = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=20)
            app.session_state["page"] = "تحلیل فروشگاه"
            app.run()
            self.assertFalse(app.exception)
            dashboard = next(tab for tab in app.tabs if tab.label == "Analytics & Dashboard")
            self.assertEqual(dashboard.label, "Analytics & Dashboard")
            self.assertEqual(len(dashboard.get("plotly_chart")), 3)

            app.selectbox(key="m1_category").set_value("کتاب").run()
            self.assertFalse(app.exception)
            dashboard = next(tab for tab in app.tabs if tab.label == "Analytics & Dashboard")
            self.assertEqual(len(dashboard.get("plotly_chart")), 2)
            self.assertTrue(any("قیمت مثبت" in item.value for item in app.info))

    def test_streamlit_dashboard_tab_reports_empty_data(self):
        write_fixture(self.source, products=[], reviews=[])
        build_analytics_data(self.source, self.output)
        with patch.dict("os.environ", {"QBC13_ANALYTICS_DIR": str(self.output)}):
            app = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=20)
            app.session_state["page"] = "تحلیل فروشگاه"
            app.run()
            self.assertFalse(app.exception)
            dashboard = next(tab for tab in app.tabs if tab.label == "Analytics & Dashboard")
            self.assertFalse(dashboard.get("plotly_chart"))
            self.assertTrue(any("ردیفی برای تحلیل" in item.value for item in dashboard.info))


if __name__ == "__main__":
    unittest.main()
