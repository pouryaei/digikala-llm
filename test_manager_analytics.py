import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import duckdb
from streamlit.testing.v1 import AppTest

from manager_analytics import (
    QUESTION_EXAMPLES,
    ManagerOptions,
    ManagerScope,
    answer_manager_question,
    manager_intent,
    parse_product_ids,
)
from manager_issue_rules import find_review_issues


QUESTIONS = tuple(QUESTION_EXAMPLES.values())


class ManagerAnalyticsTests(unittest.TestCase):
    def setUp(self):
        work = TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.directory = Path(work.name)
        with duckdb.connect() as con:
            con.execute("""
                CREATE TABLE p (product_id VARCHAR, title_fa VARCHAR, category1 VARCHAR, brand VARCHAR);
                INSERT INTO p VALUES ('1', 'کفش یک', 'کفش مردانه', 'آلفا'),
                    ('2', 'کفش دو', 'کفش مردانه', 'آلفا'), ('3', 'کفش سه', 'کفش مردانه', 'بتا'),
                    ('4', 'کفش دیگر', 'کفش مردانه', 'متفرقه'), ('5', 'سرم', 'مراقبت پوست', 'بتا'),
                    ('6', 'کفش بی‌نظر', 'کفش مردانه', 'بتا');
                CREATE TABLE r (comment_id VARCHAR, product_id VARCHAR, rating_5 DOUBLE,
                    recommendation_status VARCHAR, disadvantages_clean VARCHAR, title_clean VARCHAR, body_clean VARCHAR);
                INSERT INTO r VALUES
                    ('r1', '1', 1, 'not_recommended', 'جنس ضعیف ؛ جنس ضعیف', 'جنسش خوب نیست', 'دوخت ضعیف بود. سایزش کوچک بود.'),
                    ('r2', '1', 4, 'recommended', 'جنس', NULL, NULL),
                    ('r3', '1', 0, 'no_idea', NULL, 'بسته بندی نامناسب', NULL),
                    ('r4', '1', 5, NULL, NULL, 'صداش ضعیفه', NULL),
                    ('r5', '2', 5, 'recommended', 'گران', NULL, NULL),
                    ('r6', '2', 2, 'recommended', 'نکته منفی ندارد', 'جنسش نازکه برای تابستون خوبه', 'گران نیست. خرابی نداشت.'),
                    ('r7', '3', 5, 'invalid', 'مغایرت با عکس', NULL, NULL),
                    ('r8', '3', 2, 'not_recommended', NULL, 'بسته بندی ضعیف', NULL),
                    ('r9', '3', 4, 'recommended', NULL, NULL, NULL),
                    ('r10', '4', 4, 'recommended', 'نازک', NULL, NULL),
                    ('r11', '5', 1, 'not_recommended', NULL, 'باعث حساسیت شد', NULL);
            """)
            con.table("p").write_parquet(str(self.directory / "products.parquet"))
            con.table("r").write_parquet(str(self.directory / "reviews.parquet"))

    def answer(self, index, scope=ManagerScope("کفش مردانه"), options=ManagerOptions(), **kwargs):
        return answer_manager_question(QUESTIONS[index], scope, options, data_dir=self.directory, **kwargs)

    def test_routes_the_four_sample_questions(self):
        self.assertEqual(tuple(map(manager_intent, QUESTIONS)), ("complaints", "features", "low_recommendation", "brands"))
        with self.assertRaisesRegex(ValueError, "چهار نوع"):
            manager_intent("فردا هوا چطور است؟")

    def test_counts_complaints_once_per_review_with_traceable_sources(self):
        result = self.answer(0)
        self.assertEqual((result.product_count, result.review_count), (5, 10))
        issues = {row["topic"]: row for row in result.rows}
        top_topic = "کیفیت پایین جنس یا ساخت"
        self.assertEqual(issues[top_topic]["mentions"], 2)
        self.assertEqual(issues[top_topic]["review_share_pct"], 20)
        self.assertEqual(issues["بسته‌بندی ضعیف یا آسیب در تحویل"]["mentions"], 1)
        self.assertEqual(issues["نازکی یا ضخامت نامناسب"]["mentions"], 1)
        self.assertNotIn("کیفیت صدا یا صدای اضافی", issues)
        cited = result.evidence[top_topic][0]
        self.assertEqual((cited.comment_id, cited.product_id, cited.quote), ("r1", "1", "جنس ضعیف"))

    def test_feature_rollup_counts_a_review_once_and_omits_shipping(self):
        result = self.answer(1)
        features = {row["topic"]: row["mentions"] for row in result.rows}
        self.assertEqual(features, {"جنس و کیفیت ساخت": 3, "اندازه و تناسب": 1})
        self.assertNotIn("بسته‌بندی ضعیف یا آسیب در تحویل", result.evidence)

    def test_recommendation_rate_uses_valid_labels_and_applies_exact_threshold(self):
        result = self.answer(2, options=ManagerOptions(3, 2, 50))
        self.assertEqual([row["product_id"] for row in result.rows], ["1", "3"])
        first, second = result.rows
        self.assertEqual((first["review_count"], first["valid_label_count"], first["no_idea_count"]), (4, 3, 1))
        self.assertAlmostEqual(first["recommendation_pct"], 100 / 3)
        self.assertEqual(second["recommendation_pct"], 50.0)
        strict = answer_manager_question(
            "کدام محصولات دستهٔ كفش مردانه حداقل ۳ نظر دارند و نرخ پیشنهاد خریدشان کمتر از ۵۰٪ است؟",
            options=ManagerOptions(min_labeled=2), data_dir=self.directory,
        )
        self.assertEqual(strict.scope.category, "کفش مردانه")
        self.assertEqual([row["product_id"] for row in strict.rows], ["1"])

    def test_compares_named_brands_on_the_same_scope(self):
        result = self.answer(3)
        self.assertEqual([row["brand"] for row in result.rows], ["آلفا", "بتا"])
        first, second = result.rows
        self.assertEqual((first["review_count"], first["valid_label_count"], first["recommendation_pct"]), (6, 5, 60))
        self.assertEqual(second["not_recommended_pct"], 50)
        self.assertTrue(second["main_issue"])
        self.assertIn("10.0 واحد درصد", result.summary)
        selected = self.answer(3, options=ManagerOptions(compare_brands=("بتا", "ناموجود")))
        self.assertEqual([row["brand"] for row in selected.rows], ["بتا"])
        self.assertTrue(any("ناموجود" in note for note in selected.notes))

    def test_resolves_product_sets_missing_ids_and_empty_scopes(self):
        self.assertEqual(parse_product_ids("۱، ۲ ۱"), ("1", "2"))
        result = self.answer(0, ManagerScope(product_ids=("1", "9999")))
        self.assertEqual((result.product_count, result.review_count, result.missing_product_ids), (1, 4, ("9999",)))
        self.assertFalse(self.answer(0, ManagerScope("کفش مردانه", ("6",))).rows)
        self.assertFalse(self.answer(3, ManagerScope("دستهٔ ناموجود")).rows)
        with self.assertRaisesRegex(ValueError, "شناسهٔ محصولات"):
            self.answer(0, ManagerScope())

    def test_avoids_denials_and_requires_critical_context(self):
        result = find_review_issues((("متن نظر", "گران نیست. خرابی نداشت. جنسش نازکه برای تابستون خوبه."),))
        self.assertFalse(result)
        result = find_review_issues((("عنوان نظر", "کیفیت خوب نیست. شارژ نمی شه."),))
        self.assertEqual({rule.aspect for rule in result}, {"جنس و کیفیت ساخت", "باتری و شارژ"})

    def test_streamlit_question_tab_runs_a_local_query(self):
        with patch.dict(os.environ, {"QBC13_ANALYTICS_DIR": str(self.directory)}):
            app = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=15)
            app.session_state["page"] = "تحلیل فروشگاه"
            app.session_state["analytics_tabs"] = "پرسش تحلیلی"
            app.run()
            self.assertFalse(app.exception)
            app.button(key="manager_example_شکایت‌های پرتکرار").click().run()
            self.assertEqual(app.text_area(key="manager_question").value, QUESTIONS[0])
            app.selectbox(key="manager_category").set_value("کفش مردانه")
            app.button(key="FormSubmitter:manager_questions-اجرای تحلیل").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(app.metric[1].value, "10")
            self.assertEqual(app.dataframe[0].value.iloc[0]["mentions"], 2)


if __name__ == "__main__":
    unittest.main()
