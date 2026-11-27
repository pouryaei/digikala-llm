"""Measure real retrieval and grounding for the current (not historical) catalogue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_rag import verify_files, demo, demo_examples
from scripts.evaluate_task2 import validate
from src.bm25_index import mapping_digest
from src.config import Config
from src.rag import run_rag_pipeline
from src.search_debug_tools import evaluate_known_items, load_query_set, make_queries
from src.search_qa import load_mapping, search_products
from src.task2_review_store import SQLiteReviewStore


def evaluate(modes=("bm25",), *, k=5, deep=False):
    assets = verify_files(deep=deep)
    prep = demo()
    mapping = load_mapping()
    query_path = ROOT / "queries.json"
    current = json.loads(query_path.read_text(encoding="utf-8")) if query_path.is_file() else None
    if current is None or current.get("mapping_sha256") != mapping_digest(mapping):
        current = make_queries(mapping, seed=42)
        queries = [q for q in current["queries"] if q.get("target_product_id") is not None]
        untargeted = [q for q in current["queries"] if q.get("target_product_id") is None]
    else:
        _, queries, untargeted = load_query_set(query_path, mapping)
    retrieval = {}
    for mode in modes:
        if mode != "bm25" and not assets["dense_available"]:
            raise ValueError("Dense/Hybrid require an aligned index for the current catalogue.")
        t0 = perf_counter()
        search_products("کفش پیاده روی", top_k=k, mode=mode)
        load_seconds = perf_counter() - t0
        result = evaluate_known_items(lambda q, top_k: search_products(q, top_k=top_k, mode=mode),
                                      queries, k=k, mode=mode)
        retrieval[mode] = {**result, "warmup_seconds": round(load_seconds, 6)}
        print("RETRIEVAL", mode, {f"Hit@{k}": result[f"Hit@{k}"], f"MRR@{k}": result[f"MRR@{k}"]}, flush=True)

    # Select actual product IDs in the current corpus; the old 15k fixture is
    # retained only for historical unit tests, not used for this result.
    examples = demo_examples()
    pids = [examples["primary"]["product_id"], examples["comparison"]["product_id"]]
    pids.extend(int(q["target_product_id"]) for q in queries if q.get("query_type") == "review_qa")
    pids = list(dict.fromkeys(pids))
    store = SQLiteReviewStore(Config.CATALOG_PATH)
    qa = []
    for i, pid in enumerate(pids, 1):
        for kind, question in (("complaints", "ایرادهای پرتکرار این محصول چیست؟"),
                               ("satisfaction", "خریداران چقدر از کیفیت این محصول راضی بوده‌اند؟")):
            result = run_rag_pipeline(question, pid, generation_mode="extractive", source=Config.CATALOG_PATH)
            batch = store.get(pid)
            checks = validate(result, batch)
            passed = result["status"] in ("answered", "insufficient_evidence") and all(checks[key] for key in (
                "all_evidence_exact_and_scoped", "no_duplicate_ids", "answer_references_in_evidence",
                "stats_match_source", "theme_references_valid"))
            if result["status"] == "answered" and not result["evidence"]:
                passed = False
            qa.append({"case_id": f"{i:02d}_{kind}", "product_id": pid, "question": question,
                       "pass": passed, "checks": checks, "result": result})
            print("QA", qa[-1]["case_id"], result["status"], "PASS" if passed else "FAIL", flush=True)
    timings = sorted(x["result"]["meta"]["rag_total_seconds"] for x in qa)
    return {
        "assets": assets, "smoke": prep, "retrieval": retrieval,
        "retrieval_scope": "20 deterministic title-derived known-item queries from the current catalogue; not a representative or human-labelled benchmark.",
        "non_target_query_ids": [q["query_id"] for q in untargeted],
        "qa_summary": {
            "cases": len(qa), "passed": sum(row["pass"] for row in qa),
            "answered": sum(row["result"]["status"] == "answered" for row in qa),
            "abstained": sum(row["result"]["status"] == "insufficient_evidence" for row in qa),
            "evidence": sum(row["checks"]["evidence_count"] for row in qa),
            "exact_evidence": sum(row["checks"]["exact_evidence_count"] for row in qa),
            "median_seconds": median(timings),
            "p95_seconds": timings[max(0, (95 * len(timings) + 99) // 100 - 1)],
            "api_calls": 0, "live_gateway_tested": False, "human_semantic_judgments": False,
        },
        "qa_cases": qa,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=("bm25", "dense", "hybrid"), default=["bm25"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--deep", action="store_true", help="Verify all review-ID pairs across the two stores as well.")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/rag_evaluation_100k.json")
    args = parser.parse_args()
    if not 2 <= args.top_k <= 50:
        parser.error("--top-k must be between 2 and 50")
    report = evaluate(list(dict.fromkeys(args.modes)), k=args.top_k, deep=args.deep)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("QA summary:", json.dumps(report["qa_summary"], ensure_ascii=False))
    raise SystemExit(0 if report["qa_summary"]["passed"] == report["qa_summary"]["cases"] else 1)


if __name__ == "__main__":
    main()
