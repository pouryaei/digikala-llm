"""Explicit, bounded opt-in test of the real product-review LLM path.

Without --live this only runs the same query locally. No key or network needed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.cost_from_usage import calculate
from phase4_llm.llm_client import LLMClient, LLMUnavailable, is_configured_key
from src.config import Config
from src.rag import run_rag_pipeline
from src.task2_review_store import FIELDS, SQLiteReviewStore


def scoped_quotes(result, source):
    reviews = {r["comment_id"]: r for r in SQLiteReviewStore(source).get(result["product_id"]).reviews}
    return bool(result.get("evidence")) and all(
        e["product_id"] == result["product_id"] and e["comment_id"] in reviews
        and any(e["text"] in (reviews[e["comment_id"]].get(field) or "") for field in FIELDS)
        for e in result["evidence"]
    )


def smoke(product_id, question, *, live=False, input_rate=None, output_rate=None):
    source = Config.CATALOG_PATH
    offline = run_rag_pipeline(question, product_id, source=source, generation_mode="extractive",
                               use_cache=False)
    checked = offline["status"] == "answered" and scoped_quotes(offline, source)
    report = {"product_id": product_id, "question": question, "offline_pass": checked,
              "offline": offline, "live_requested": live, "live_verified": False,
              "live_result": None, "real_gateway_calls": 0}
    if not checked:
        report["status"] = "NO_EVIDENCE"
        return report
    if not live:
        report["status"] = "OFFLINE_PASS"
        report["cost"] = calculate([offline["meta"]])
        return report
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass
    if not is_configured_key(os.getenv("QBC13_API_KEY")):
        report["status"] = "MISSING_GROUP_KEY"
        return report
    # Two service attempts at most; each client generation makes no
    # automatic retry. Keep this smoke test within a small token cap.
    try:
        llm = LLMClient(max_retries=1, max_tokens=350)
    except LLMUnavailable:
        report["status"] = "SDK_OR_PROVIDER_UNAVAILABLE"
        return report
    generated = run_rag_pipeline(question, product_id, source=source, generation_mode="llm",
                                 llm=llm, use_cache=False)
    meta = generated["meta"]
    passed = (generated["answer_mode"] == "grounded" and meta.get("generation_mode") == "llm"
              and meta.get("api_calls", 0) > 0 and scoped_quotes(generated, source))
    report.update(live_result=generated, live_verified=passed, real_gateway_calls=meta.get("api_calls", 0),
                  status="LIVE_PASS_STRUCTURAL" if passed else "LIVE_FALLBACK_OR_INVALID",
                  cost=calculate([meta], input_rate, output_rate))
    report["limit_note"] = "Structural/exact-quote check only; semantic answer quality must be reviewed against the real reviews."
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--product-id", type=int, default=1696173)
    p.add_argument("--question", default="ایرادهای پرتکرار این محصول چیست؟")
    p.add_argument("--live", action="store_true", help="Explicitly permit a small paid gateway smoke test.")
    p.add_argument("--input-usd-per-million", type=float)
    p.add_argument("--output-usd-per-million", type=float)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    # Reject invalid prices before any call.
    calculate([], args.input_usd_per_million, args.output_usd_per_million)
    result = smoke(args.product_id, args.question, live=args.live,
                   input_rate=args.input_usd_per_million, output_rate=args.output_usd_per_million)
    output = args.output or ROOT / "reports" / ("llm_live_validation.json" if args.live else "llm_smoke_offline.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("offline","live_result")}, ensure_ascii=False, indent=2))
    if result["status"] not in {"OFFLINE_PASS", "LIVE_PASS_STRUCTURAL"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
