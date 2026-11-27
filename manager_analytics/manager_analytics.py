"""Answer Persian store-manager questions with local data and traceable evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Literal

import duckdb

from analytics_service import CATEGORY_LABEL, open_analytics_tables, resolve_analytics_files
from manager_issue_rules import find_review_issues, normalize_persian


ManagerIntent = Literal["complaints", "features", "low_recommendation", "brands"]
QUESTION_EXAMPLES = {
    "شکایت‌های پرتکرار": "پرتکرارترین شکایت کاربران در این دسته چیست؟",
    "ویژگی‌های ناراضی‌کننده": "کاربران بیشتر دربارهٔ چه ویژگی‌هایی از محصولات این دسته ناراضی هستند؟",
    "محصولات پرنظر با پیشنهاد کم": "کدام محصولات نظر زیادی دارند اما درصد پیشنهاد خرید آن‌ها پایین است؟",
    "مقایسهٔ بازخورد برندها": "بازخورد کاربران دربارهٔ چند برند اصلی این دسته چه تفاوتی دارد؟",
}

VALID_LABELS = "'recommended', 'not_recommended', 'no_idea'"
RATING = "CASE WHEN r.rating_5 BETWEEN 0 AND 5 THEN r.rating_5 END"
NEGATIVE_REVIEW = "(r.recommendation_status = 'not_recommended' OR r.rating_5 > 0 AND r.rating_5 <= 2)"
UNNAMED_BRANDS = "'', 'متفرقه', 'بدون برند', 'فاقد برند', 'نامشخص', 'other', 'generic', 'no brand', 'unknown'"
RESULT_LIMIT = 20
TOP_BRANDS = 5
EVIDENCE_LIMIT = 3
ISSUE_LIMIT = 12


@dataclass(frozen=True)
class ManagerScope:
    category: str | None = None
    product_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ManagerOptions:
    min_reviews: int = 100
    min_labeled: int = 30
    max_recommendation_pct: float = 50.0
    compare_brands: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReviewEvidence:
    comment_id: str
    product_id: str
    source: str
    quote: str
    specific: bool


@dataclass
class ManagerAnswer:
    intent: ManagerIntent
    scope: ManagerScope
    product_count: int
    review_count: int
    summary: str = ""
    rows: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    evidence: dict[str, list[ReviewEvidence]] = field(default_factory=dict)
    missing_product_ids: tuple[str, ...] = ()


@dataclass
class FeedbackCounts:
    candidates: int = 0
    matched: int = 0
    complaints: Counter = field(default_factory=Counter)
    features: Counter = field(default_factory=Counter)
    complaint_evidence: dict[str, list[ReviewEvidence]] = field(default_factory=dict)
    feature_evidence: dict[str, list[ReviewEvidence]] = field(default_factory=dict)


def parse_product_ids(text: str) -> tuple[str, ...]:
    ids = tuple(dict.fromkeys(filter(None, re.split(r"[،,;؛\s]+", normalize_persian(text)))))
    if any(not re.fullmatch(r"[0-9]+", value) for value in ids):
        raise ValueError("شناسه‌های عددی محصول را با ویرگول یا فاصله جدا کن.")
    return ids


def parse_brand_names(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(filter(None, (item.strip() for item in re.split(r"[،,;؛\n]+", text)))))


def manager_intent(question: str) -> ManagerIntent:
    text = normalize_persian(question)
    if re.search(r"\bبرند(?: ?ها(?:ی|یشان|ست)?|ی)?\b", text) and re.search(
        r"مقایسه|تفاوت|اختلاف|بازخورد|امتیاز|اصلی|کدام|نرخ|کنار هم|رضایت|محبوب", text
    ):
        return "brands"
    if re.search(r"پیشنهاد|توصیه", text) and re.search(
        r"پایین|کمترین|کمتر|کم(?: است| بود| دارند| دارن|ه\b)|ضعیف|زیر |پایین تر|پر ?نظر|کم پیشنهاد", text
    ):
        return "low_recommendation"
    if re.search(r"ناراض|نارضایت|شکایت|ایراد|اشکال|مشکل|ضعف|نقص|گله|نکات منفی", text):
        if re.search(r"ویژگی|خصوصیت|خصوصیات|مشخصه|مشخصات|جنبه|چه چیز|کدام قسمت", text):
            return "features"
        return "complaints"
    raise ValueError("یکی از چهار نوع پرسش نمونه را انتخاب کن یا پرسشت را به همان موضوع‌ها محدود کن.")


def get_manager_categories(data_dir: str | Path | None = None) -> tuple[str, ...]:
    with open_analytics_tables(resolve_analytics_files(data_dir)) as con:
        values = con.execute(f"""
            SELECT min({CATEGORY_LABEL}) AS category FROM products p
            GROUP BY fa_group_key({CATEGORY_LABEL}) ORDER BY category
        """).fetchall()
    return tuple(category for category, in values)


def _get_scope(question: str, scope: ManagerScope, categories: tuple[str, ...]) -> ManagerScope:
    if scope.category:
        return scope
    text = normalize_persian(question)
    matches = []
    for name in sorted(categories, key=len, reverse=True):
        if name == "نامشخص":
            continue
        for match in re.finditer(rf"(?<!\w){re.escape(normalize_persian(name))}(?!\w)", text):
            if not any(start <= match.start() and match.end() <= end for _, start, end in matches):
                matches.append((name, match.start(), match.end()))
    names = {name for name, _, _ in matches}
    if len(names) > 1:
        raise ValueError("چند نام دسته در پرسش پیدا شد. دستهٔ موردنظر را از فهرست انتخاب کن.")
    if names:
        return ManagerScope(names.pop(), scope.product_ids)
    if not scope.product_ids:
        raise ValueError("دسته را از فهرست انتخاب کن یا شناسهٔ محصولات را در تنظیمات وارد کن.")
    return scope


def _create_scope(con: duckdb.DuckDBPyConnection, scope: ManagerScope) -> tuple[int, int, tuple[str, ...]]:
    con.execute(f"""
        CREATE TEMP TABLE scoped_products AS
        SELECT p.product_id, p.title_fa, nullif(trim(p.brand), '') AS brand,
               CASE WHEN lower(fa_group_key(p.brand)) NOT IN ({UNNAMED_BRANDS})
                    THEN lower(fa_group_key(p.brand)) END AS brand_key
        FROM products p
        WHERE (? IS NULL OR fa_group_key({CATEGORY_LABEL}) = fa_group_key(?))
          AND (NOT ? OR CAST(p.product_id AS VARCHAR) IN (SELECT unnest(?::VARCHAR[])))
    """, [scope.category, scope.category, bool(scope.product_ids), list(scope.product_ids)])
    product_count = con.execute("SELECT count(*) FROM scoped_products").fetchone()[0]
    missing_ids = ()
    if scope.product_ids:
        found = {str(row[0]) for row in con.execute("SELECT product_id FROM scoped_products").fetchall()}
        missing_ids = tuple(value for value in scope.product_ids if value not in found)
    review_count = con.execute("SELECT count(*) FROM scoped_products p JOIN reviews r USING (product_id)").fetchone()[0]
    return product_count, review_count, missing_ids


def _keep_evidence(samples_by_topic: dict[str, list[ReviewEvidence]], label: str, evidence: ReviewEvidence) -> None:
    samples = samples_by_topic.setdefault(label, [])
    if len(samples) < EVIDENCE_LIMIT:
        samples.append(evidence)
    elif evidence.specific and not samples[-1].specific:
        samples[-1] = evidence
    samples.sort(key=lambda item: not item.specific)


def _collect_feedback(con: duckdb.DuckDBPyConnection, brand_keys: tuple[str, ...] = ()) -> dict[str | None, FeedbackCounts]:
    counts = defaultdict(FeedbackCounts)
    cursor = con.execute(f"""
        SELECT r.comment_id, CAST(p.product_id AS VARCHAR),
               CASE WHEN ? THEN p.brand_key END AS group_key,
               nullif(trim(r.disadvantages_clean), ''),
               CASE WHEN {NEGATIVE_REVIEW} THEN nullif(trim(r.title_clean), '') END,
               CASE WHEN {NEGATIVE_REVIEW} THEN nullif(trim(r.body_clean), '') END
        FROM reviews r JOIN scoped_products p USING (product_id)
        WHERE (NOT ? OR p.brand_key IN (SELECT unnest(?::VARCHAR[])))
          AND (nullif(trim(r.disadvantages_clean), '') IS NOT NULL
               OR {NEGATIVE_REVIEW} AND (nullif(trim(r.title_clean), '') IS NOT NULL
                                        OR nullif(trim(r.body_clean), '') IS NOT NULL))
        ORDER BY r.comment_id
    """, [bool(brand_keys), bool(brand_keys), list(brand_keys)])
    while rows := cursor.fetchmany(2048):
        for comment_id, product_id, brand_key, disadvantage, title, body in rows:
            group = counts[brand_key]
            group.candidates += 1
            hits = find_review_issues((("نکات منفی", disadvantage), ("عنوان نظر", title), ("متن نظر", body)))
            if not hits:
                continue
            group.matched += 1
            seen_aspects = set()
            for item, (source, quote, specific) in hits.items():
                evidence = ReviewEvidence(str(comment_id), product_id, source, quote, specific)
                group.complaints[item.label] += 1
                _keep_evidence(group.complaint_evidence, item.label, evidence)
                if item.aspect is None or item.aspect in seen_aspects:
                    continue
                seen_aspects.add(item.aspect)
                group.features[item.aspect] += 1
                _keep_evidence(group.feature_evidence, item.aspect, evidence)
    return counts


def _rank_counts(counts: Counter) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


FEEDBACK_METHOD = (
    "موضوع‌ها از نکات منفی و از عنوان یا متن نظرهای «پیشنهاد نمی‌کنم» یا دارای امتیاز مثبتِ حداکثر ۲ "
    "شناسایی شده‌اند. قواعد فارسی ممکن است بعضی تعبیرها را از دست بدهند."
)
LABEL_DENOMINATOR = (
    "مخرج نرخ پیشنهاد خرید، همهٔ نظرهای دارای یکی از سه برچسب معتبر است. «نظر مشخصی ندارد» در مخرج می‌آید. "
    "برچسب خالی یا نامعتبر کنار گذاشته می‌شود."
)


def _answer_feedback(con: duckdb.DuckDBPyConnection, answer: ManagerAnswer) -> None:
    feedback = _collect_feedback(con).get(None, FeedbackCounts())
    features = answer.intent == "features"
    counts = feedback.features if features else feedback.complaints
    samples = feedback.feature_evidence if features else feedback.complaint_evidence
    ranking = _rank_counts(counts)
    answer.notes = [
        f"{feedback.candidates:,} نظرِ دارای متن یا نکتهٔ منفی واجد بررسی بود و در {feedback.matched:,} نظر دست‌کم یک شکایت شناخته‌شده پیدا شد.",
        FEEDBACK_METHOD,
        "هر نظر برای هر موضوع یک‌بار شمرده می‌شود. سهم هر موضوع از کل نظرهای محدوده است و یک نظر می‌تواند چند موضوع داشته باشد.",
    ]
    if features:
        answer.notes.append("این جدول ویژگی‌های محصول مانند جنس، تناسب و عملکرد را نشان می‌دهد. قیمت و موارد ارسال در گزارش شکایت‌ها می‌آیند.")
    if not ranking:
        answer.summary = "در نظرهای این محدوده موضوعی مطابق قواعد تحلیل شناسایی نشد."
        return
    label, count = ranking[0]
    kind = "ویژگی با بیشترین نارضایتیِ شناسایی‌شده" if features else "پرتکرارترین شکایت شناسایی‌شده"
    answer.summary = f"{kind} «{label}» است که در {count:,} نظر آمده ({100 * count / answer.review_count:.1f}٪ از نظرهای محدوده)."
    for label, count in ranking[:ISSUE_LIMIT]:
        answer.rows.append({"topic": label, "mentions": count, "review_share_pct": 100 * count / answer.review_count})
        answer.evidence[label] = samples[label]


REVIEW_METRICS = f"""
    count(*) AS review_count,
    count(*) FILTER (WHERE r.recommendation_status IN ({VALID_LABELS})) AS valid_label_count,
    count(*) FILTER (WHERE r.recommendation_status = 'recommended') AS recommended_count,
    count(*) FILTER (WHERE r.recommendation_status = 'not_recommended') AS not_recommended_count,
    count(*) FILTER (WHERE r.recommendation_status = 'no_idea') AS no_idea_count,
    count({RATING}) AS rated_count,
    avg({RATING}) AS mean_rating
"""
RECOMMENDATION_PCT = "100.0 * recommended_count / nullif(valid_label_count, 0)"


def _validate_options(options: ManagerOptions) -> None:
    if any(type(value) is not int or not 1 <= value < 2**63 for value in (options.min_reviews, options.min_labeled)):
        raise ValueError("حداقل تعداد نظر و برچسب معتبر باید عدد صحیح مثبت باشد.")
    if not 0 <= options.max_recommendation_pct <= 100:
        raise ValueError("سقف نرخ پیشنهاد خرید باید بین صفر تا صد درصد باشد.")


def _low_thresholds(question: str, options: ManagerOptions) -> tuple[int, float, bool]:
    text = re.sub(r"(?<=[0-9])[,٬](?=[0-9])", "", normalize_persian(question))
    min_reviews = options.min_reviews
    ceiling = options.max_recommendation_pct
    inclusive = True
    if count_match := re.search(r"\b(حداقل|بیش از|بیشتر از|بالای) ([0-9]+) (?:تا )?(?:نظر|دیدگاه)\b", text):
        min_reviews = int(count_match[2]) + (count_match[1] != "حداقل")
    if percentage := re.search(r"\b(کمتر از|زیر|حداکثر|تا) ([0-9]+(?:\.[0-9]+)?)\s*(?:درصد|[%٪])", text):
        ceiling = float(percentage[2])
        inclusive = percentage[1] in ("حداکثر", "تا")
    _validate_options(ManagerOptions(min_reviews, options.min_labeled, ceiling))
    return min_reviews, ceiling, inclusive


def _records(con: duckdb.DuckDBPyConnection, query: str, parameters: list) -> list[dict]:
    cursor = con.execute(query, parameters)
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _answer_low_recommendation(con: duckdb.DuckDBPyConnection, answer: ManagerAnswer, question: str, options: ManagerOptions) -> None:
    min_reviews, ceiling, inclusive = _low_thresholds(question, options)
    comparison = "<=" if inclusive else "<"
    answer.rows = _records(con, f"""
        WITH metrics AS (
            SELECT CAST(p.product_id AS VARCHAR) AS product_id, p.title_fa, p.brand, {REVIEW_METRICS}
            FROM scoped_products p JOIN reviews r USING (product_id)
            GROUP BY p.product_id, p.title_fa, p.brand
        ), rated AS (
            SELECT *, {RECOMMENDATION_PCT} AS recommendation_pct FROM metrics
            WHERE review_count >= ? AND valid_label_count >= ?
        )
        SELECT *, count(*) OVER () AS matched_products FROM rated
        WHERE recommendation_pct {comparison} ?
        ORDER BY recommendation_pct, review_count DESC, product_id
        LIMIT ?
    """, [min_reviews, options.min_labeled, ceiling, RESULT_LIMIT])
    wording = "حداکثر" if inclusive else "کمتر از"
    total = answer.rows[0]["matched_products"] if answer.rows else 0
    answer.summary = (
        f"{total:,} محصول با دست‌کم {min_reviews:,} نظر و {options.min_labeled:,} برچسب معتبر پیدا شد که نرخ پیشنهاد خریدشان "
        f"{wording} {ceiling:g}٪ است."
    )
    answer.notes = [LABEL_DENOMINATOR]
    if total > RESULT_LIMIT:
        answer.notes.append(f"{RESULT_LIMIT} محصول با پایین‌ترین نرخ نمایش داده شده است.")


def _answer_brands(con: duckdb.DuckDBPyConnection, answer: ManagerAnswer, options: ManagerOptions) -> None:
    names = options.compare_brands
    answer.rows = _records(con, f"""
        WITH summary AS (
            SELECT p.brand_key, min(p.brand) AS brand, count(DISTINCT p.product_id) AS products_reviewed,
                   {REVIEW_METRICS}
            FROM scoped_products p JOIN reviews r USING (product_id)
            WHERE p.brand_key IS NOT NULL AND (NOT ? OR p.brand_key IN (
                SELECT lower(fa_group_key(brand)) FROM unnest(?::VARCHAR[]) AS selected(brand)
            ))
            GROUP BY p.brand_key
        )
        SELECT *, {RECOMMENDATION_PCT} AS recommendation_pct,
               100.0 * not_recommended_count / nullif(valid_label_count, 0) AS not_recommended_pct
        FROM summary ORDER BY review_count DESC, brand LIMIT ?
    """, [bool(names), list(names), len(names) if names else TOP_BRANDS])
    answer.notes = [LABEL_DENOMINATOR, FEEDBACK_METHOD, "سهم هر شکایت برند از کل نظرهای همان برند در محدوده محاسبه می‌شود."]
    if names:
        matched = {normalize_persian(row["brand_key"]) for row in answer.rows}
        missing = [name for name in names if normalize_persian(name) not in matched]
        if missing:
            answer.notes.append(f"برای این نام‌ها در محدوده نظری پیدا نشد. {'، '.join(missing)}")
    else:
        answer.notes.append(f"حداکثر {TOP_BRANDS} برند مشخص با بیشترین تعداد نظر در محدوده انتخاب شده‌اند.")
    if not answer.rows:
        answer.summary = "در این محدوده برای برندهای موردنظر نظرات قابل مقایسه پیدا نشد."
        return

    groups = _collect_feedback(con, tuple(row["brand_key"] for row in answer.rows))
    for row in answer.rows:
        brand = row["brand"]
        feedback = groups.get(row["brand_key"], FeedbackCounts())
        ranking = _rank_counts(feedback.complaints)
        topic, count = ranking[0] if ranking else (None, 0)
        row["main_issue"] = topic
        row["issue_mentions"] = count if topic else None
        row["issue_share_pct"] = 100 * count / row["review_count"] if topic else None
        if topic:
            answer.evidence[f"{brand} · {topic}"] = feedback.complaint_evidence[topic]

    by_rate = [row for row in answer.rows if row["recommendation_pct"] is not None]
    answer.summary = f"بازخورد {len(answer.rows)} برند مقایسه شد."
    if len(by_rate) >= 2:
        low = min(by_rate, key=lambda row: row["recommendation_pct"])
        high = max(by_rate, key=lambda row: row["recommendation_pct"])
        gap = high["recommendation_pct"] - low["recommendation_pct"]
        if gap > 0:
            answer.summary += (
                f" نرخ پیشنهاد خرید «{high['brand']}» ({high['recommendation_pct']:.1f}٪) و «{low['brand']}» "
                f"({low['recommendation_pct']:.1f}٪) به‌ترتیب بالاترین و پایین‌ترین مقدار این جدول‌اند. "
                f"فاصلهٔ آن‌ها {gap:.1f} واحد درصد است."
            )
        else:
            answer.summary += " نرخ پیشنهاد خرید برندهای دارای برچسب معتبر برابر است."
    elif len(by_rate) == 1:
        answer.summary += " نرخ پیشنهاد خرید برای یک برند قابل محاسبه بود."
    else:
        answer.summary += " برای محاسبهٔ نرخ پیشنهاد، برچسب معتبر در دسترس نیست."


def answer_manager_question(
    question: str, scope: ManagerScope = ManagerScope(), options: ManagerOptions = ManagerOptions(),
    *, data_dir: str | Path | None = None, categories: tuple[str, ...] | None = None,
) -> ManagerAnswer:
    if not question.strip():
        raise ValueError("پرسش مدیر فروشگاه را وارد کن.")
    intent = manager_intent(question)
    _validate_options(options)
    if not scope.category and categories is None:
        categories = get_manager_categories(data_dir)
    scope = _get_scope(question, scope, categories or ())
    with open_analytics_tables(resolve_analytics_files(data_dir)) as con:
        products, reviews, missing_ids = _create_scope(con, scope)
        answer = ManagerAnswer(intent, scope, products, reviews, missing_product_ids=missing_ids)
        if not reviews:
            answer.summary = "در محدودهٔ انتخاب‌شده نظر ثبت‌شده‌ای پیدا نشد."
        elif intent == "low_recommendation":
            _answer_low_recommendation(con, answer, question, options)
        elif intent == "brands":
            _answer_brands(con, answer, options)
        else:
            _answer_feedback(con, answer)
        return answer
