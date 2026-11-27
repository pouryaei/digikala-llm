"""Adapt the local RAG to the original Forouhar dashboard service contract."""
from __future__ import annotations

import os
from pathlib import Path
import unicodedata
from functools import lru_cache

from .bm25_index import tokenize_query
from .catalog import open_catalog, resolve_price_unit
from .config import ROOT
from .rag import compare_products, run_rag_pipeline
from .review_data import BUNDLED_STORE, is_bundled_sample
from .search_qa import search_products
from .rag_query import search_request
from .product_preferences import recorded_budget
from .task2_review_store import ReviewDataError, identifier, ParquetReviewStore

DATA_SCOPE = "indexed_catalogue"


@lru_cache(maxsize=8)
def _has_review_text(path, mtime_ns, size):
    if is_bundled_sample(path):
        return False
    try:
        ParquetReviewStore(path)
    except ReviewDataError:
        return False
    return True


def review_source():
    explicit = (os.getenv("QBC13_REVIEW_SOURCE") or os.getenv("FOROUHAR_COMMENTS_PATH") or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        path = path if path.is_absolute() else ROOT / path
        if is_bundled_sample(path):
            raise ReviewDataError("این فایل، نمونهٔ چهارردیفی قدیمی است و برای پاسخ واقعی استفاده نمی‌شود.")
        if not path.is_file():
            raise ReviewDataError("فایل تعیین‌شده برای نظرها پیدا نشد. مسیر QBC13_REVIEW_SOURCE را بررسی کنید.")
        return path

    data_dir = os.getenv("QBC13_ANALYTICS_DIR", "").strip()
    folders = [Path(data_dir)] if data_dir else []
    folders.extend(ROOT / p for p in ("processed_repo_based", "data/processed_repo_based", "data/analytics"))
    for folder in folders:
        for name in ("comments_clean.parquet", "reviews.parquet"):
            path = (folder / name).resolve()
            if path.is_file():
                st = path.stat()
                if _has_review_text(path, st.st_mtime_ns, st.st_size):
                    return path
        recoverable = folder / "reviews_payload.csv.zst"
        if recoverable.is_file():
            return recoverable
    return BUNDLED_STORE


def _field_key(text):
    raw = str(text or "").strip().casefold()
    if raw == "نامشخص":
        return raw
    return " ".join(tokenize_query(unicodedata.normalize("NFKC", raw)))


def _dash_products(product_ids):
    cat = open_catalog()
    rows = cat.get_products(product_ids)
    if not rows:
        return {}
    stats = cat.review_stats(rows)
    unit = resolve_price_unit()
    raw_unit = {"rial": "ریال", "toman": "تومان"}.get(unit, "واحد ثبت‌شده")
    output = {}
    for pid, r in rows.items():
        n, positive, valid = stats.get(pid, (0,0,0))
        output[pid] = {
            "product_id": pid, "title_fa": r["title_fa"], "category": r["category1"],
            "brand": r["brand"], "price": r["price_recorded"],
            "price_unit": raw_unit if r["price_recorded"] is not None else None,
            "rating": r["rating_100"], "rating_scale": 100, "rating_count": r["rating_count"],
            "comment_count": n, "recommended_count": positive, "valid_label_count": valid,
            "recommendation_rate": positive/valid if valid else None,
        }
    return output


def _matching_ids(filters):
    if not filters:
        return None
    category, brand = (_field_key(filters.get(k)) for k in ("category", "brand"))
    min_raw, max_raw = filters.get("min_price"), filters.get("max_price")
    if min_raw is not None and max_raw is not None and min_raw > max_raw:
        raise ValueError("حداقل قیمت از حداکثر بیشتر است.")
    criteria, params = [], []
    for op, value in ((">=", min_raw), ("<=", max_raw)):
        if value is not None:
            criteria.append(f"price_recorded {op} ?")
            params.append(value)
    sql = "SELECT product_id, category1, brand FROM products"
    if criteria:
        sql += " WHERE " + " AND ".join(criteria)
    with open_catalog().connection() as con:
        return {str(row["product_id"]) for row in con.execute(sql, params)
                if (not category or _field_key(row["category1"] or "نامشخص") == category)
                and (not brand or _field_key(row["brand"]) == brand)}


def search(query, filters=None, limit=5):
    filters = dict(filters or {})
    if filters.keys() - {"category", "brand", "min_price", "max_price"}:
        raise ValueError("فیلتر جست‌وجو شناخته نشد.")
    allowed = _matching_ids(filters)
    try:
        exact = identifier(query)
    except ValueError:
        exact = None
    notes = []
    if exact is not None:
        selected = [exact] if allowed is None or str(exact) in allowed else []
    else:
        mode = os.environ.get("QBC13_UI_SEARCH_MODE", "bm25").strip().lower() or "bm25"
        req = search_request(query)
        if req.prefer_satisfaction:
            notes.append("اولویت رضایت: ابتدا گزینه‌های مرتبط با دست‌کم ۵۰٪ پیشنهاد خرید، یا در نبود برچسب امتیاز دست‌کم ۷۰ از ۱۰۰ با حداقل دو رأی؛ سپس رتبه‌بندی با توجه به تعداد شواهد.")
        if req.prefer_affordability:
            notes.append("اولویت قیمت: قیمت ثبت‌شده میان گزینه‌های مرتبط مقایسه شد؛ ارزان‌بودن نسبی است.")
        if req.scope_hint == "travel":
            notes.append("دامنهٔ پیشنهادها: دسته‌های واقعی لوازم سفر و کمپینگ.")
        elif req.scope_hint == "everyday_bag":
            notes.append("دامنهٔ پیشنهادها: دسته‌های کیف زنانه و مردانه؛ کاربری از مشخصات ثبت‌شده فراتر است.")
        if recorded_budget(req, open_catalog()):
            notes.append("درخواست واحد پول مشخص نکرده است؛ بازه به واحد ثبت‌شدهٔ داده اعمال شد، نه لزوماً تومان.")
        elif req.min_price is not None or req.max_price is not None:
            notes.append("بازهٔ بودجهٔ نوشته‌شده در سؤال با واحد تومان و واحد منبعِ تأییدشده اعمال شد.")
        results = search_products(query, top_k=limit, mode=mode, allowed_product_ids=allowed)
        if req.prefer_satisfaction and any(not r.get("positive_feedback_signal") for r in results):
            notes.append("برخی گزینه‌های مرتبط شواهد کافی از رضایت ندارند؛ آمار هر محصول را جدا بررسی کنید.")
        if any(r.get("purpose_match_relaxed") for r in results):
            notes.append("ویژگی‌های دقیق درخواست در عنوان‌ها یافت نشد؛ گزینه‌های هم‌نوع پیشنهاد شدند. کاربری، وزن یا سازگاری خاص تأیید نشده است.")
        selected = [identifier(r["product_id"]) for r in results]
    metadata = _dash_products(selected)
    return {"items": [metadata[pid] for pid in selected if pid in metadata],
            "applied_filters": filters, "scope": DATA_SCOPE, "search_notes": notes}


def get_product(product_id):
    product = _dash_products([product_id]).get(identifier(product_id))
    return {"product": product} if product else None


def categories():
    with open_catalog().connection() as c:
        labels = {r[0] for r in c.execute("SELECT DISTINCT category1 FROM products")
                  if isinstance(r[0], str) and r[0].strip()}
    return {"items": sorted(labels), "scope": DATA_SCOPE}


def ask(product_id, query, *, source=None):
    # Local answering is independent of the dashboard web backend address.
    # Direct product tab stays local; paid requests use ProfessionalWorkflow.
    return run_rag_pipeline(query, product_id, source=source if source is not None else review_source(),
                            generation_mode="extractive")


def compare(product_ids, query=None):
    ids = [identifier(pid) for pid in product_ids]
    raw = compare_products(ids, query=query, source=review_source())
    items = _dash_products(ids)
    evidence = {}
    for item in raw["products"]:
        pid = item["product_id"]
        stats = item["review_stats"]
        if pid in items:
            items[pid].update(
                comment_count=stats["total_reviews"], recommendation_rate=stats["recommendation_rate"],
                recommended_count=stats["recommended_count"], valid_label_count=stats["valid_label_count"],
            )
        for source in item["strengths"] + item["weaknesses"] + item["related_reviews"]:
            if source["product_id"] == pid:
                evidence.setdefault((pid, source["comment_id"]), source)
    return {"items": [items[pid] for pid in ids if pid in items], "evidence": list(evidence.values()),
            "notes": raw["comparison_notes"], "scope": DATA_SCOPE, "meta": raw["meta"]}


def dispatch(method, path, payload):
    if method == "POST" and path == "/api/search":
        return search(payload["query"], payload.get("filters"), payload.get("limit", 5))
    if method == "POST" and path == "/api/reviews/ask":
        return ask(payload["product_id"], payload["query"])
    if method == "POST" and path == "/api/products/compare":
        return compare(payload["product_ids"], payload.get("query"))
    if method == "GET" and path == "/api/categories":
        return categories()
    if method == "GET" and path.startswith("/api/products/"):
        return get_product(path.rsplit("/", 1)[-1])
    if method == "GET" and path == "/health":
        return {"status": "ok", "service": "local_rag", "scope": DATA_SCOPE}
    return None
