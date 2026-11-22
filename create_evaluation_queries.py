#!/usr/bin/env python3
"""Create the QBC13 evaluation query set.

This script uses only the Python standard library. Copy it to the repository
root and run: python create_evaluation_queries.py
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_NOTES = [
    "Do not fabricate relevant product IDs, review IDs, SQL answers, or benchmark scores; populate those only from verified project data and runs.",
    "Finglish requires explicit transliteration and multilingual handling; Yeh/Kaf/ZWNJ normalization alone does not solve Finglish.",
    "Manual Top-10 Yes/No relevance labels permit Precision@10 only.",
    "Title-seeded Method A, using real product titles and real target IDs, is required for Hit Rate@1/@5 and MRR.",
    "Analytics are separate deterministic SQL tasks, not retrieval scoring.",
    "Before exact label analytics are frozen, check the schema for brand, sales, and timestamp fields and reconcile the known 39-row label-count discrepancy.",
]


def record(query_id: str, task_type: str, query: str, features: list[str], note: str) -> dict:
    return {
        "id": query_id,
        "task_type": task_type,
        "query": query,
        "language": "fa",
        "features": features,
        "note": note,
    }


CORE_QUERIES = [
    record("PD-01", "product_discovery", "گوشی سامسونگ سری S با قیمت مناسب", ["brand", "series", "price_intent"], "confirm relevant catalog products exist; broad intent needs relevance judgments"),
    record("PD-02", "product_discovery", "یه گوشی شیائومی با باتری قوی و شارژدهی طولانی", ["colloquial", "brand", "battery"], "colloquial wording; verify catalog coverage"),
    record("PD-03", "product_discovery", "لپ‌تاپ ایسوس با رم ۱۶ گیگ برای برنامه‌نویسی", ["brand", "ram", "use_case"], "confirm RAM/spec fields or relevant product descriptions"),
    record("PD-04", "product_discovery", "مانیتور ۲۷ اینچ خمیده برای بازی و کار", ["size", "shape", "use_case"], "mixed constraints"),
    record("PD-05", "product_discovery", "هندزفری بلوتوسی شیائومی با حذف نویز", ["intentional_typo", "brand", "feature"], "intentional typo in بلوتوسی; confirm noise-cancellation evidence"),
    record("PD-06", "product_discovery", "پاوربانک ۲۰۰۰۰ میلی‌آمپرساعت با شارژ سریع ۲۰ وات", ["capacity", "fast_charging"], "numeric/spec matching"),
    record("PD-07", "product_discovery", "کفش پیاده‌روی مردانه سبک و راحت", ["category", "gender", "comfort"], "attribute intent"),
    record("PD-08", "product_discovery", "ماشین لباسشویی کم‌مصرف با ظرفیت حدود ۸ کیلو", ["category", "energy", "capacity"], "confirm metadata availability"),
    record("RQ-09", "review_qa", "در نظرهای خریداران، بدنه کتری برقی موقع جوشیدن آب چقدر داغ می‌شود؟", ["review_evidence", "thermal_safety"], "label supporting and contradicting review evidence manually"),
    record("RQ-10", "review_qa", "آیا کابل شارژ تایپ‌سی با گوشی‌های قدیمی سامسونگ سازگار بوده؟", ["review_evidence", "compatibility"], "need product/model resolution; don't assume compatibility from category alone"),
    record("RQ-11", "review_qa", "کاربران درباره دوام باتری گوشی‌های شیائومی پس از چند ماه چه گفته‌اند؟", ["review_evidence", "durability", "time_phrase"], "temporal phrase does not prove review timestamp; retrieve actual evidence"),
    record("RQ-12", "review_qa", "صدای فن لپ‌تاپ‌های گیمینگ هنگام کار معمولی آزاردهنده است؟", ["review_evidence", "noise", "usage_context"], "need relevant product/review evidence"),
    record("RQ-13", "review_qa", "خریداران از کیفیت تصویر تلویزیون‌های ایکس‌ویژن راضی بوده‌اند؟", ["review_evidence", "brand", "quality"], "separate individual anecdotes from aggregate claims"),
    record("RQ-14", "review_qa", "آیا در نظرات درباره خرابی زودهنگام جاروبرقی پارس‌خزر شکایتی هست؟", ["review_evidence", "negative_evidence", "durability"], "avoid treating retrieved examples as population-wide rate"),
    record("RQ-15", "review_qa", "نظر خریداران درباره کیفیت میکروفون هدفون‌های بی‌سیم چیه؟", ["colloquial", "review_evidence", "audio_quality"], "conversational Persian"),
    record("AN-16", "analytics", "کدام ۱۰ محصول بیشترین تعداد نظر ثبت‌شده را دارند؟", ["deterministic_count", "top_k"], "aggregate comments by product_id; check duplicate product metadata does not multiply counts"),
    record("AN-17", "analytics", "از بین نظرهای دارای برچسب معتبر، سهم هر یک از سه برچسب توصیه چقدر است؟", ["label_distribution", "denominator"], "use valid labels only and reconcile 39-row discrepancy before freezing exact expected values"),
    record("AN-18", "analytics", "کدام ۱۰ محصول بیشترین تعداد نظر «توصیه نمی‌کنم» را دارند؟", ["label_filter", "top_k"], "count only valid not_recommended labels; report denominators/coverage"),
    record("AN-19", "analytics", "در هر برند موبایل، نرخ «توصیه می‌کنم» چقدر است؟", ["brand_aggregation", "recommendation_rate"], "conditional on verified brand column/product mapping; define denominator and minimum sample size"),
    record("AN-20", "analytics", "کدام برند ساعت هوشمند امسال بیشترین فروش را داشته؟", ["sales_analytics", "temporal_scope"], "negative-capability test; do not answer unless verified sales data and year coverage exist; otherwise state limitation"),
]

STRESS_VARIANTS = [
    {"id": "ST-01", "variant_of": "PD-01", "query": "gooshi samsung seri S ba gheymate monaseb", "language": "finglish", "features": ["finglish", "transliteration"]},
    {"id": "ST-02", "variant_of": "PD-03", "query": "laptop asus ba ram 16 baraye barnamerizi", "language": "finglish", "features": ["finglish", "transliteration", "numeric"]},
    {"id": "ST-03", "variant_of": "PD-06", "query": "powerbank 20000 ba charge sari 20 watt", "language": "finglish", "features": ["finglish", "transliteration", "numeric"]},
    {"id": "ST-04", "variant_of": "RQ-12", "query": "sedaye fan laptop gaming vaghte kar-e mamooli aziyat konande ast?", "language": "finglish", "features": ["finglish", "review_qa"]},
    {"id": "ST-05", "variant_of": "RQ-13", "query": "nazare kharidaran dar morede keifiat tasvire TV haye X-Vision chie?", "language": "finglish", "features": ["finglish", "review_qa", "brand"]},
]


def build_payload() -> dict:
    queries = CORE_QUERIES + STRESS_VARIANTS
    return {
        "project": "QBC13 — Digikala AI Shopping & Review Intelligence",
        "dataset": "evaluation_queries",
        "version": "1.0",
        "notes": PROJECT_NOTES,
        "queries": queries,
    }


def validate(payload: dict) -> None:
    queries = payload["queries"]
    ids = [item["id"] for item in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("query IDs must be unique")

    core = [item for item in queries if "variant_of" not in item]
    variants = [item for item in queries if "variant_of" in item]
    if len(core) != 20:
        raise ValueError(f"expected exactly 20 core queries, found {len(core)}")
    if len(variants) != 5:
        raise ValueError(f"expected exactly 5 stress variants, found {len(variants)}")
    if any(item.get("language") != "fa" for item in core):
        raise ValueError("every core query must have language='fa'")
    if any(item.get("language") != "finglish" for item in variants):
        raise ValueError("every stress variant must have language='finglish'")

    core_ids = {item["id"] for item in core}
    missing = [item["id"] for item in variants if item["variant_of"] not in core_ids]
    if missing:
        raise ValueError(f"variants point to missing core IDs: {missing}")
    if any(not item.get("query") or not item.get("features") for item in queries):
        raise ValueError("every query must have non-empty query text and features")


def main() -> None:
    payload = build_payload()
    validate(payload)
    output_dir = Path(__file__).resolve().parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation_queries.json"
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote {len(payload['queries'])} queries to {output_path}")


if __name__ == "__main__":
    main()
