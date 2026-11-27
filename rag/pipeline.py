"""Run the portable Forouhar RAG system on real Digikala data."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys

from src.config import DenseUnavailable
from src.rag import compare_products, find_product_candidates, run_rag_pipeline
from src.task2_review_store import ReviewDataError

__all__ = ["run_rag_pipeline", "find_product_candidates", "compare_products", "main"]


def _json(data):
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _print_candidates(candidates):
    if any(item.get("budget_unit") == "recorded" for item in candidates):
        print("بودجه به واحد ثبت‌شدهٔ منبع اعمال شد؛ واحد تومان تأیید نشده است.")
    if any(item.get("purpose_match_relaxed") for item in candidates):
        print("گزینه‌های هم‌نوع؛ ویژگی‌های دقیق کاربری تأیید نشده‌اند.")
    for rank, candidate in enumerate(candidates, 1):
        details = []
        if (price := candidate.get("price_toman")) is not None:
            details.append(f"{price:,.0f} تومان در داده" + (" (مقادیر متفاوت در منبع)" if candidate.get("price_conflict") else ""))
        elif (price := candidate.get("price_recorded")) is not None:
            details.append(f"{price:,.0f} در منبع (واحد نامشخص)")
        if "review_count" in candidate:
            details.append(f"{candidate['review_count']:,} نظر")
        suffix = f" | {'، '.join(details)}" if details else ""
        print(f"{rank}. [{candidate['product_id']}] {candidate.get('title_fa') or ''}{suffix}")


def _print_answer(result):
    if result["status"] == "needs_product":
        print(result["reason"])
        _print_candidates(result["candidates"])
        return
    product = result.get("product") or {}
    print(f"محصول: [{result['product_id']}] {product.get('title_fa', '')}")
    print(result["answer"] or result["reason"])
    if result["evidence"]:
        print("\nEvidence IDs:", ", ".join(str(x["comment_id"]) for x in result["evidence"]))
    meta = result["meta"]
    print(f"Mode: {result['answer_mode']} | API calls: {meta.get('api_calls', 0)} | seconds: {meta.get('rag_total_seconds', 0):.3f}")
    if meta.get("fallback_reason"):
        print("مدل زبانی نتیجهٔ قابل نمایش نداد؛ پاسخ محلی و مستند نمایش داده شد.")


def _print_comparison(result):
    for item in result["products"]:
        facts = item["facts"] or {}
        print(f"\n[{item['product_id']}] {facts.get('title_fa', '')}")
        if facts.get("price_toman") is not None:
            print(f"قیمت ثبت‌شده: {facts['price_toman']:,.0f} تومان")
        elif facts.get("price_recorded") is not None:
            print(f"عدد قیمت ثبت‌شده: {facts['price_recorded']:,.0f} (واحد منبع تأیید نشده)")
        if facts.get("rating_100") is not None:
            print(f"امتیاز ثبت‌شده: {facts['rating_100']:g}/100 ({facts['rating_count']:,} امتیاز)")
        stats = item["review_stats"]
        print(f"تعداد نظرات در منبع: {stats['total_reviews']:,}")
        if stats["valid_label_count"]:
            print(f"پیشنهاد خرید: {stats['recommended_count']:,} از {stats['valid_label_count']:,} نظر دارای برچسب")
        for name, key in (("مزایا", "strengths"), ("معایب", "weaknesses"), ("نظرهای مرتبط با سؤال", "related_reviews")):
            if item[key]:
                print(name + ":")
                for e in item[key]:
                    print(f"• «{e['text']}» (نظر {e['comment_id']})")
        if item.get("related_review_note"):
            print(item["related_review_note"])
    print("\n" + "\n".join(result["comparison_notes"]))


def main(argv=None):
    # Legacy Windows console encodings should not garble Persian data on files/CI.
    if getattr(sys.stdout, "encoding", None) and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="Find products by a Persian search query.")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="bm25")
    search.add_argument("--min-price", type=float, help="Minimum recorded price in toman.")
    search.add_argument("--max-price", type=float, help="Maximum recorded price in toman.")
    search.add_argument("--source-price-unit", choices=["rial", "toman"], help="Set only if the original price unit has been confirmed.")
    search.add_argument("--min-rating", type=float, help="Minimum catalogue rating, from 0 to 100.")
    search.add_argument("--min-reviews", type=int, default=0)
    search.add_argument("--json", action="store_true")

    ask = sub.add_parser("ask", help="Answer a question with the selected product's real reviews.")
    ask.add_argument("--query", required=True)
    ask.add_argument("--product-id")
    ask.add_argument("--source", help="Optional alternate SQLite/Parquet/JSONL/M1 ZIP review source.")
    ask.add_argument("--mode", choices=["extractive", "llm"], default="extractive")
    ask.add_argument("--search-mode", choices=["bm25", "dense", "hybrid"], default="bm25")
    ask.add_argument("--top-k", type=int, default=8)
    ask.add_argument("--no-cache", action="store_true")
    ask.add_argument("--json", action="store_true")

    compare = sub.add_parser("compare", help="Compare real catalogue facts and reviews across selected products.")
    compare.add_argument("--product-id", nargs="+", required=True)
    compare.add_argument("--query")
    compare.add_argument("--source")
    compare.add_argument("--source-price-unit", choices=["rial", "toman"])
    compare.add_argument("--top-k", type=int, default=3)
    compare.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "search":
            results = find_product_candidates(
                args.query, args.top_k, args.mode, min_price=args.min_price,
                max_price=args.max_price, min_rating=args.min_rating, min_reviews=args.min_reviews,
                source_price_unit=args.source_price_unit,
            )
            if args.json:
                _json(results)
            elif results:
                _print_candidates(results)
            else:
                print("برای این عبارت و فیلترها محصولی در کاتالوگ بسته پیدا نشد.")
        elif args.command == "ask":
            result = run_rag_pipeline(args.query, args.product_id, source=args.source, search_mode=args.search_mode,
                                      generation_mode=args.mode, top_k=args.top_k, use_cache=not args.no_cache)
            _json(result) if args.json else _print_answer(result)
        else:
            result = compare_products(args.product_id, args.query, source=args.source, top_k=args.top_k,
                                      source_price_unit=args.source_price_unit)
            _json(result) if args.json else _print_comparison(result)
    except (ValueError, FileNotFoundError, ReviewDataError, DenseUnavailable, sqlite3.Error) as exc:
        parser.exit(2, f"خطا: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
