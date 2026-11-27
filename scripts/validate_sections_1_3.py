"""Offline regression of the repaired sections using the actual bundled 100K data."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation.run_evaluation import result_fingerprint
from scripts.restore_100k_assets import REQUIRED, REVISION
from scripts.validate_live_qa import scoped_quotes
from src.catalog import open_catalog
from src.dashboard_rag import search
from src.rag import compare_products, run_rag_pipeline
from src.search_qa import load_mapping, load_search_engine


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate():
    start = perf_counter()
    manifest = json.loads((ROOT / "storage/catalog_manifest.json").read_text(encoding="utf-8"))
    require(manifest["dataset_revision"] == REVISION, "Wrong dataset revision")
    for rel, key in REQUIRED.items():
        with (ROOT / rel).open("rb") as f:
            require(hashlib.file_digest(f, "sha256").hexdigest() == manifest[key], f"Hash mismatch: {rel}")
    catalogue = open_catalog()
    with catalogue.connection() as con:
        products = con.execute("SELECT count(*) FROM products").fetchone()[0]
        reviews = con.execute("SELECT count(*) FROM reviews").fetchone()[0]
        require((products, reviews) == (100000, 1874457), "Catalogue counts changed")
        require(con.execute("PRAGMA quick_check").fetchone()[0] == "ok", "Invalid SQLite")
        require(con.execute("SELECT count(*) FROM reviews r LEFT JOIN products p USING(product_id) WHERE p.product_id IS NULL").fetchone()[0] == 0, "Orphan reviews")
    idx = load_search_engine("bm25").bm25_manager
    require(len(load_mapping()) == len(idx.id_mapping) == products, "Index count changed")
    demos = []
    bag_query = "یک کیف برای استفاده‌ی روزمره می‌خوام که خیلی گرون نباشه و خریدارها هم ازش راضی باشن."
    travel_query = "چند وسیله‌ی کاربردی برای سفر معرفی کن که نظر کاربران درباره‌شون خوب باشه."
    for query, scope in ((bag_query, "everyday_bag"), (travel_query, "travel")):
        result = search(query)
        allowed = catalogue.preference_scope_ids(scope)
        require(bool(result["items"]) and all(str(p["product_id"]) in allowed for p in result["items"]), f"Official search example failed: {scope}")
        demos.append({"query": query, "scope": scope, "items": result["items"], "notes": result["search_notes"]})
    before, after = search("هدفون"), search("هدفون با رضایت کاربران")
    require([x["product_id"] for x in before["items"]] != [x["product_id"] for x in after["items"]], "Satisfaction ranking had no effect")
    demos.append({"query": "هدفون با رضایت کاربران", "before": before["items"], "after": after["items"], "notes": after["search_notes"]})
    raw = search("ساعت ضد آب تا ۵ میلیون")
    require(bool(raw["items"]) and all(0 < p["price"] <= 5000000 for p in raw["items"]), "Raw budget outside constraints")
    demos.append({"query": "ساعت ضد آب تا ۵ میلیون", "items": raw["items"], "notes": raw["search_notes"]})
    try:
        search("کفش بین ۳۰۰ تا ۶۰۰ هزار تومان")
    except ValueError:
        unknown_currency_refuses_conversion = True
    else:
        unknown_currency_refuses_conversion = bool(manifest.get("supplied_price_unit_confirmed"))
    require(unknown_currency_refuses_conversion, "Unverified currency conversion")
    with patch.dict(os.environ, {"QBC13_SOURCE_PRICE_UNIT": "rial"}):
        currency_test = search("کفش بین ۳۰۰ تا ۶۰۰ هزار تومان")
    require(bool(currency_test["items"]) and all(3000000 <= p["price"] <= 6000000 for p in currency_test["items"]), "Configured rial conversion failed")
    # The configured conversion above is a controlled test, not a source-unit claim.
    old = json.loads((ROOT / "evaluation/results/final/evaluation.json").read_text(encoding="utf-8"))
    changed_search = []
    for row in old["search"]:
        result = search(row["query"], row["filters"], limit=5)
        original = [int(p["product_id"]) for p in row["results"]]
        current = [int(p["product_id"]) for p in result["items"]]
        if original != current:
            changed_search.append({"case_id": row["case_id"], "before": original, "after": current})
    changed_qa = []
    qa_evidence = 0
    source = str(ROOT / "storage/catalog.sqlite3")
    for row in old["qa"]:
        result = run_rag_pipeline(row["question"], row["product_id"], source=source, generation_mode="extractive", use_cache=False)
        if result_fingerprint(result) != row["fingerprint"]:
            changed_qa.append(row["case_id"])
        if result["status"] == "answered":
            require(scoped_quotes(result, source), "Invalid real source quotation")
            qa_evidence += len(result["evidence"])
    comparisons = []
    for row in old["comparison"]:
        ids = row["ids"]
        result = compare_products(ids, source=source)
        require(result["status"] == "compared" and [x["product_id"] for x in result["products"]] == [int(i) for i in ids], "Comparison scope changed")
        count = 0
        for item in result["products"]:
            evidence = item["strengths"] + item["weaknesses"] + item["related_reviews"]
            if evidence:
                require(scoped_quotes({"product_id": item["product_id"], "evidence": evidence}, source), "Comparison cross-product evidence")
            count += len(evidence)
        comparisons.append({"ids": ids, "verified_evidence": count, "pass": True})
    model_replay_file = ROOT / "reports/model_replay_complete.json"
    return {
        "scope": "Actual 100K service-level offline regression; not a Streamlit browser test",
        "dataset": {"revision": REVISION, "products": products, "reviews": reviews,
                    "runtime_asset_checksums_verified": True, "sqlite_integrity": True,
                    "orphan_reviews": 0, "index_mapping_count": products},
        "search": {"official_and_preference_demos": demos, "currency_guard": True,
                   "configured_currency_test_pass": True,
                   "original_cases": len(old["search"]), "unchanged_ranked_results": len(old["search"])-len(changed_search),
                   "changed_cases": changed_search},
        "review_qa": {"original_cases": len(old["qa"]), "unchanged_fingerprints": len(old["qa"])-len(changed_qa),
                      "changed_fingerprints": changed_qa, "exact_scoped_evidence": qa_evidence},
        "comparison": comparisons,
        "prediction_full_replay_report": model_replay_file.name if model_replay_file.is_file() else None,
        "manager": {"implementation_changed": False, "prior_validation_cases": 8,
                    "rerun_here": False, "note": "Run the original full evaluator for DuckDB analytics and Streamlit tests in the pinned environment."},
        "api_calls": 0, "live_llm_verified": False,
        "elapsed_seconds": round(perf_counter()-start, 3),
        "optional_dependencies_here": {n: bool(importlib.util.find_spec(n)) for n in ("duckdb", "streamlit", "rank_bm25", "openai")}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/sections_1_3_validation.json")
    args = parser.parse_args()
    with patch.dict(os.environ, {"QBC13_SOURCE_PRICE_UNIT": "", "QBC13_UI_SEARCH_MODE": "bm25",
                                 "QBC13_REVIEW_GENERATION_MODE": "extractive", "QBC13_API_KEY": ""}):
        report = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "search"}, ensure_ascii=False, indent=2))
    print("Search:", {k:v for k,v in report["search"].items() if k != "official_and_preference_demos"})
