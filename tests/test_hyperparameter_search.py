"""Offline regression checks for the grid, artifact isolation and UI wiring."""
from __future__ import annotations

import ast
import csv
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import MagicMock

import joblib
import pandas as pd

from recommendation_ml.hyperparameter_search import (
    DEFAULT_OUTPUT, LABELS, PARAM_GRID, PROTECTED_ARTIFACTS, ROOT,
    SPLITS, _audit_rows, run_search, sha,
)


def examples():
    phrases = ("نظری ندارم", "اصلا ارزش خرید ندارد", "عالی و با کیفیت")
    rows = {}
    for split in SPLITS:
        marker = {"train": "", "validation": "validateword validateword", "test": "testtoken testtoken"}[split]
        rows[split] = []
        for i, (label, phrase) in enumerate(zip(LABELS * 4, phrases * 4)):
            key = f"{split}-{i}"
            rows[split].append((key, key, f"{phrase} {marker} نمونه {key}", label, key))
    return rows


class HyperparameterSearchTests(TestCase):
    def test_product_and_text_leakage_rejected(self):
        for location, expected in ((1, "Product shared"), (4, "Duplicate")):
            rows = examples()
            old = list(rows["validation"][0])
            old[location] = rows["train"][0][location]
            rows["validation"][0] = tuple(old)
            with self.assertRaisesRegex(ValueError, expected):
                _audit_rows(rows)

    def test_grid_runs_on_fixed_fold_and_persists_candidate(self):
        before = {p: sha(p) for p in PROTECTED_ARTIFACTS if p.is_file()}
        with TemporaryDirectory() as temp:
            out = Path(temp) / "candidate"
            rows = examples()
            result = run_search(rows, out, threads=1)
            self.assertEqual(result["n_trials"], len(PARAM_GRID["C"]) * len(PARAM_GRID["class_weight"]))
            self.assertEqual(result["best_validation"]["macro_f1"],
                             max(t["validation_macro_f1"] for t in result["trials"]))
            self.assertFalse(result["test_used_for_selection"])
            self.assertEqual(result["audit"]["shared_products"], 0)
            self.assertEqual(result["vectorizer_fitted_on"], "train")
            self.assertFalse(result["production_model_replaced"])
            path = out / "tuned_candidate.joblib"
            self.assertEqual(sha(path), result["candidate_sha256"])
            candidate = joblib.load(path)
            vocab = candidate["tfidf"].vocabulary_
            self.assertNotIn("valid", vocab)
            self.assertNotIn("testt", vocab)
            self.assertEqual(candidate["tfidf"].ngram_range, (2, 5))
            for key, value in result["best_params"].items():
                self.assertEqual(candidate["model"].get_params()[key], value)
            with (out / "test_predictions.csv").open(encoding="utf-8", newline="") as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual([r["predicted_status"] for r in saved],
                             candidate.predict([row[2] for row in rows["test"]]).tolist())
            self.assertIsNone(result["recorded_test_reference"])
            self.assertEqual(json.loads((out / "report.json").read_text(encoding="utf-8"))["best_params"],
                             result["best_params"])
            with self.assertRaises(FileExistsError):
                run_search(rows, out)
        self.assertEqual(before, {p: sha(p) for p in before})

    def test_deployed_model_directory_cannot_be_used_as_output(self):
        for path in (ROOT / "m2_outputs" / "not_this", ROOT / "bonus_artifacts" / "not_this"):
            with self.assertRaises(FileExistsError):
                run_search(examples(), path)

    @skipUnless((DEFAULT_OUTPUT / "report.json").is_file(), "No precomputed grid report")
    def test_precomputed_result_and_panel_rendering(self):
        # Execute only the panel against a recording UI stand-in: actual
        # Streamlit rendering is tested separately when the dependencies exist.
        result = json.loads((DEFAULT_OUTPUT / "report.json").read_text(encoding="utf-8"))
        tree = ast.parse((ROOT / "bonus" / "dashboard.py").read_text(encoding="utf-8"))
        panel = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "hyperparameter_panel")
        model_panel = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                           and node.name == "model_panel")
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                             and n.func.id == "hyperparameter_panel" for n in ast.walk(model_panel)))
        self.assertEqual(result["n_trials"], 8)
        self.assertEqual(result["best_params"], {name: result["trials"][0][name] for name in PARAM_GRID})
        self.assertEqual(result["recorded_test_reference"]["matched_recorded_test_ids_and_labels"], 15000)
        self.assertEqual(sha(DEFAULT_OUTPUT / result["candidate_file"]), result["candidate_sha256"])
        for name, digest in result["protected_artifacts_sha256"].items():
            self.assertEqual(sha(ROOT / name), digest)
        ui = MagicMock()
        cols = [MagicMock() for _ in range(4)]
        ui.columns.return_value = cols
        namespace = {"st": ui, "pd": pd, "report": lambda _: result,
                     "number": lambda val: f"{val:,}", "REPORT": DEFAULT_OUTPUT.parent}
        exec(compile(ast.Module(body=[panel], type_ignores=[]), "bonus/dashboard.py", "exec"), namespace)
        namespace["hyperparameter_panel"]()
        self.assertTrue(all(col.metric.called for col in cols))
        self.assertEqual(len(ui.dataframe.call_args_list[0].args[0]), 8)
        self.assertTrue(ui.bar_chart.called)
        self.assertEqual(ui.download_button.call_count, 2)

    @skipUnless(all(importlib.util.find_spec(name) for name in ("streamlit", "duckdb", "altair")),
                "Streamlit/DuckDB UI dependencies not installed")
    @skipUnless((DEFAULT_OUTPUT / "report.json").is_file(), "No precomputed grid report")
    def test_streamlit_hyperparameter_panel(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string("from bonus.dashboard import hyperparameter_panel\nhyperparameter_panel()",
                                  default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertIn("Validation Macro F1", [m.label for m in app.metric])
        self.assertTrue(app.dataframe)
