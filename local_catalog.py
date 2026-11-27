"""Local lexical search and extractive review fallback over Forouhar's real data."""

from __future__ import annotations

import re
from pathlib import Path

from analytics_service import CATEGORY_LABEL, open_analytics_tables, resolve_analytics_files
from manager_issue_rules import normalize_persian


STOP_WORDS = set("""
یک یه این آن اون محصول کالا محصولات برای که با از در به را رو و یا چه چی کدام کدوم هست است بود
دارم دارد دارن داشته باشه باشد می میخوام بخوام خواهم خرید بخرم دنبال میخواهم لطفا درباره درمورد
مورد کاربران کاربر خریداران خریدارها خریدار نظر نظرات نظرها گفته گفتند گفتن بودند بودن بودنش
بیشتر چه چیزی چیزهایی ویژگی ویژگیها نکات نکته نقاط قوت ضعف مثبت منفی ایراد ایرادها مشکل مشکلات شکایت
خوب بهتر بهترین عالی مناسب راضی رضایت ناراضی شده نیست هم خیلی چند روزمره
چیست چگونه چطور مردم بیشترش بوده بودند اند نوشته نوشتن نوشتهاند هایی های ها همه یهچیزی همش
ایرادهای پرتکرار پرتکراری پرتکرارترین مشکلاتی معایب مزایا کاربرانش خریدارانش مثبتش منفیش نظرش
""".split())
LABELS = "'recommended', 'not_recommended', 'no_idea'"
EMPTY_REVIEW = set("ندارد نداره نداشت هیچی هیچ نداردش ندیدم هیچینداره نداردن خوب عالی هیچکدام نامشخص".split()) | {
    "هیچ چیز", "نکته منفی ندارد", "نکته منفی نداره", "موردی ندیدم", "چیزی ندیدم", "هیچ مشکلی ندارد",
}


def keywords(text: str) -> list[str]:
    return list(dict.fromkeys(word for word in re.findall(r"[^\W_]+", normalize_persian(text))
                              if word not in STOP_WORDS and len(word) > 1))


def _records(con, sql, parameters):
    cursor = con.execute(sql, parameters)
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _selected_product_records(con):
    # Rating count describes submitted product ratings, while comment count comes from reviews.
    return _records(con, f"""
        WITH review_stats AS (
            SELECT r.product_id, count(*) AS comment_count,
                   count(*) FILTER (WHERE recommendation_status = 'recommended') AS recommended_count,
                   count(*) FILTER (WHERE recommendation_status IN ({LABELS})) AS valid_label_count
            FROM reviews r JOIN selected_catalog p USING (product_id) GROUP BY r.product_id
        )
        SELECT try_cast(p.product_id AS BIGINT) AS product_id, p.title_fa, p.category, p.brand,
               p.price, 'واحد ثبت‌شده' AS price_unit, p.rating, 100 AS rating_scale, p.rating_count,
               coalesce(s.comment_count, 0) AS comment_count,
               coalesce(s.recommended_count, 0) AS recommended_count,
               coalesce(s.valid_label_count, 0) AS valid_label_count,
               s.recommended_count::DOUBLE / nullif(s.valid_label_count, 0) AS recommendation_rate
        FROM selected_catalog p LEFT JOIN review_stats s USING (product_id)
        ORDER BY p.rank_order
    """, [])


PRODUCT_FIELDS_SQL = """
    p.product_id, p.title_fa, p.category1 AS category, p.brand,
    CASE WHEN p.price > 0 AND isfinite(p.price) THEN p.price END AS price,
    CASE WHEN p.rating_100 BETWEEN 0 AND 100 AND p.rating_count > 0 THEN p.rating_100 END AS rating,
    p.rating_count
"""


def search_products(query: str, filters: dict | None = None, limit: int = 5,
                    *, data_dir: str | Path | None = None) -> dict:
    filters = dict(filters or {})
    terms = keywords(query)
    if not terms:
        return {"items": [], "applied_filters": filters, "scope": "processed_products"}
    if not 1 <= limit <= 20:
        raise ValueError("تعداد نتایج باید بین ۱ تا ۲۰ باشد.")
    allowed = {"category", "brand", "min_price", "max_price"}
    if filters.keys() - allowed:
        raise ValueError("فیلتر محلی شناخته نشد.")

    predicates = ["try_cast(p.product_id AS BIGINT) > 0"]
    parameters = [terms]
    for name, column in (("category", CATEGORY_LABEL), ("brand", "p.brand")):
        if name in filters:
            predicates.append(f"lower(fa_group_key({column})) = lower(fa_group_key(?))")
            parameters.append(filters[name])
    for name, operator in (("min_price", ">="), ("max_price", "<=")):
        if name in filters:
            predicates.extend(["p.price > 0 AND isfinite(p.price)", f"p.price {operator} ?"])
            parameters.append(filters[name])

    with open_analytics_tables(resolve_analytics_files(data_dir)) as con:
        con.execute(f"""
            CREATE TEMP TABLE selected_catalog AS
            WITH scored AS (
                SELECT {PRODUCT_FIELDS_SQL}, len(list_intersect(
                    string_split(lower(coalesce(p.product_text_lexical, '')), ' '), ?::VARCHAR[]
                )) AS matches
                FROM products p WHERE {' AND '.join(predicates)}
            )
            SELECT *, row_number() OVER (ORDER BY matches DESC, rating_count DESC NULLS LAST,
                       rating DESC NULLS LAST, product_id) AS rank_order
            FROM scored WHERE matches > 0 ORDER BY rank_order LIMIT ?
        """, [*parameters, limit])
        products = _selected_product_records(con)
    return {"items": products, "applied_filters": filters, "scope": "processed_products"}


def get_product(product_id: int, *, data_dir: str | Path | None = None) -> dict:
    with open_analytics_tables(resolve_analytics_files(data_dir)) as con:
        con.execute(f"""
            CREATE TEMP TABLE selected_catalog AS
            SELECT {PRODUCT_FIELDS_SQL}, 1 AS rank_order FROM products p
            WHERE try_cast(p.product_id AS BIGINT) = ?
        """, [product_id])
        products = _selected_product_records(con)
    if len(products) != 1:
        raise LookupError("محصولی با این شناسه در دادهٔ محلی پیدا نشد.")
    return {"product": products[0]}


def get_categories(*, data_dir: str | Path | None = None) -> dict:
    with open_analytics_tables(resolve_analytics_files(data_dir)) as con:
        rows = con.execute(f"""
            SELECT min({CATEGORY_LABEL}) FROM products p GROUP BY fa_group_key({CATEGORY_LABEL}) ORDER BY 1
        """).fetchall()
    return {"items": [row[0] for row in rows]}


def _feedback_intent(question: str) -> str:
    text = normalize_persian(question)
    if re.search(r"ارزش خرید|پیشنهاد|توصیه|بخرم|می ارزه|میارزه", text):
        return "recommendation"
    negative = bool(re.search(r"ناراض|شکایت|ایراد|معایب|نکات منفی|نقاط ضعف|مشکل", text))
    positive = bool(re.search(r"نکات مثبت|مزایا|نقاط قوت|راضی|رضایت", text))
    if negative and not positive:
        return "negative"
    if positive and not negative:
        return "positive"
    return "neutral"


def ask_reviews(product_id: int, query: str, *, data_dir: str | Path | None = None) -> dict:
    """Retrieve genuine source excerpts, avoiding generated reviews or unsupported opinions."""
    terms = keywords(query)
    intent = _feedback_intent(query)
    files = resolve_analytics_files(data_dir)
    with open_analytics_tables(files) as con:
        if con.execute("SELECT count(*) FROM products WHERE try_cast(product_id AS BIGINT)=?", [product_id]).fetchone()[0] != 1:
            raise LookupError("محصولی با این شناسه در دادهٔ محلی پیدا نشد.")
        con.execute("""
            CREATE TEMP TABLE product_reviews AS
            SELECT comment_id, recommendation_status, rating_5,
                   title_raw, body_raw, advantages_raw, disadvantages_raw,
                   title_clean, body_clean, advantages_clean, disadvantages_clean
            FROM reviews WHERE try_cast(product_id AS BIGINT) = ?
        """, [product_id])
        total, valid, recommended = con.execute(f"""
            SELECT count(*), count(*) FILTER (WHERE recommendation_status IN ({LABELS})),
                   count(*) FILTER (WHERE recommendation_status='recommended') FROM product_reviews
        """).fetchone()

        candidates = _records(con, """
            WITH fields AS (
                SELECT comment_id, recommendation_status, rating_5, 'body' AS source, body_raw AS text, body_clean AS normalized FROM product_reviews
                UNION ALL SELECT comment_id, recommendation_status, rating_5, 'title', title_raw, title_clean FROM product_reviews
                UNION ALL SELECT comment_id, recommendation_status, rating_5, 'positive', advantages_raw, advantages_clean FROM product_reviews
                UNION ALL SELECT comment_id, recommendation_status, rating_5, 'negative', disadvantages_raw, disadvantages_clean FROM product_reviews
            )
            SELECT comment_id, recommendation_status, rating_5, source, text, normalized,
                   len(list_intersect(regexp_extract_all(lower(fa_group_key(normalized)), '[\\p{L}\\p{N}]+'), ?::VARCHAR[])) AS overlap
            FROM fields WHERE nullif(trim(text), '') IS NOT NULL AND nullif(trim(normalized), '') IS NOT NULL
        """, [terms])

    def relevant(row):
        normalized = " ".join(re.findall(r"[^\W_]+", normalize_persian(row["normalized"])))
        if normalized in EMPTY_REVIEW or not any(character.isalnum() for character in normalized):
            return False
        if intent == "recommendation":
            return row["recommendation_status"] in ("recommended", "not_recommended")
        if terms and row["overlap"] == 0:
            return False
        if intent == "negative":
            return row["source"] == "negative" or (row["source"] in ("body", "title") and row["recommendation_status"] == "not_recommended")
        if intent == "positive":
            return row["source"] == "positive" or (row["source"] in ("body", "title") and row["recommendation_status"] == "recommended")
        return row["overlap"] > 0

    candidates = [row for row in candidates if relevant(row)]
    preferred = {"negative": "negative", "positive": "positive"}.get(intent, "body")
    candidates.sort(key=lambda row: (-row["overlap"], row["source"] != preferred,
                                     -min(len(row["text"]), 400), str(row["comment_id"])))
    selected = []
    seen = set()
    if intent == "recommendation":
        # Include both kinds of experience whenever both are available.
        for label in ("recommended", "not_recommended"):
            row = next((item for item in candidates if item["recommendation_status"] == label), None)
            if row is not None and row["comment_id"] not in seen:
                selected.append(row)
                seen.add(row["comment_id"])
    for row in candidates:
        if row["comment_id"] not in seen:
            selected.append(row)
            seen.add(row["comment_id"])
        if len(selected) >= 5:
            break

    evidence = [{"comment_id": row["comment_id"], "product_id": product_id, "text": row["text"][:1500]}
                for row in selected]
    if not evidence:
        return {"product_id": product_id, "answer_mode": "insufficient_evidence", "answer": None, "evidence": []}

    if intent == "recommendation" and valid:
        answer = (f"از {valid:,} نظر دارای برچسب معتبر، {recommended:,} نفر خرید را پیشنهاد کرده‌اند "
                  f"({recommended / valid:.1%}). کل نظرهای ثبت‌شده برای محصول {total:,} است. "
                  "بخش شواهد نمونه‌هایی واقعی از تجربهٔ کاربران را نشان می‌دهد.")
    else:
        source_count = len({row["comment_id"] for row in candidates})
        excerpts = "\n".join(f"• {row['text'][:280]} (نظر {row['comment_id']})" for row in selected[:3])
        answer = (f"{source_count:,} نظر با معیارهای بازیابی محلی مرتبط بود. بخش‌هایی از متن واقعی آن‌ها:\n{excerpts}\n"
                  f"کل نظرات محصول {total:,} است. این شواهد نمونه‌های بازیابی‌شده برای پرسش هستند.")
    return {"product_id": product_id, "answer_mode": "extractive", "answer": answer, "evidence": evidence}
