"""Test the real sklearn recipe and leakage guard without full-data retraining."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from ml_service import _prepare, compose_model_text
from recommendation_ml.train_final import Example, LABELS, SPLITS, audit_splits, character_model, fit


def example_splits():
    result = {}
    phrases = ("نظری ندارم معمولی", "بسیار ضعیف پیشنهاد نمی کنم", "عالی خرید را پیشنهاد می کنم")
    i = 0
    for split in SPLITS:
        rows = []
        for repeat in range(3):
            for label, phrase in zip(LABELS, phrases):
                i += 1
                rows.append(Example(str(i), f"{split}-{i}", f"{phrase} نمونه {i}", label, f"full{i}", f"core{i}"))
        result[split] = rows
    return result


class TrainingTests(TestCase):
    def test_feature_matches_training_separator_and_dedup(self):
        self.assertEqual(compose_model_text("كيفيت عالی", "راضی"), "راضی . کیفیت عالی")
        self.assertEqual(compose_model_text("عالی", "عالی"), "عالی")
        self.assertEqual(_prepare("عالی", "عالی"), ("عالی", None))
        self.assertEqual(_prepare("کیفیت خوب", "راضی"), ("راضی . کیفیت خوب", None))
        self.assertEqual(_prepare("۱۲۳۴۵")[1], "digits_only")

    def test_detects_product_leakage(self):
        rows = example_splits()
        audit_splits(rows)
        old = rows["test"][0]
        rows["test"][0] = Example(old.comment_id, rows["train"][0].product_id, old.text, old.label, old.text_key, old.core_text_key)
        with self.assertRaisesRegex(ValueError, "Product shared"):
            audit_splits(rows)

    def test_detects_text_leakage(self):
        rows = example_splits()
        old = rows["test"][0]
        rows["test"][0] = Example(old.comment_id, old.product_id, rows["train"][0].text, old.label, old.text_key, old.core_text_key)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit_splits(rows)

    def test_observed_pipeline_and_small_training_run(self):
        p = character_model()
        self.assertEqual(p["tfidf"].analyzer, "char")
        self.assertEqual(p["tfidf"].ngram_range, (2,5))
        self.assertEqual(p["tfidf"].max_features, 200000)
        self.assertTrue(p["tfidf"].sublinear_tf)
        self.assertEqual(p["model"].class_weight, "balanced")
        with TemporaryDirectory() as t:
            output = Path(t) / "fresh"
            manifest = fit(example_splits(), output)
            self.assertEqual(manifest["audit"]["shared_products"], 0)
            self.assertTrue((output / "best_model.joblib").is_file())
            self.assertEqual(len(json.loads((output / "confusion_matrix.json").read_text())["matrix"]), 3)
            with self.assertRaises(FileExistsError):
                fit(example_splits(), output)
