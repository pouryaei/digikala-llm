"""Streamlit form and results for the four store-manager question types."""

from __future__ import annotations

from html import escape
from pathlib import Path

import duckdb
import streamlit as st

from analytics_service import resolve_analytics_files
from manager_analytics import (
    QUESTION_EXAMPLES,
    ManagerAnswer,
    ManagerOptions,
    ManagerScope,
    answer_manager_question,
    get_manager_categories,
    parse_brand_names,
    parse_product_ids,
)


TABLE_COLUMNS = {
    "complaints": ("topic", "mentions", "review_share_pct"),
    "features": ("topic", "mentions", "review_share_pct"),
    "low_recommendation": (
        "title_fa", "product_id", "brand", "review_count", "valid_label_count", "recommended_count",
        "not_recommended_count", "no_idea_count", "recommendation_pct",
    ),
    "brands": (
        "brand", "products_reviewed", "review_count", "mean_rating", "rated_count", "valid_label_count",
        "recommended_count", "recommendation_pct", "not_recommended_pct", "no_idea_count",
        "main_issue", "issue_mentions", "issue_share_pct",
    ),
}


@st.cache_data(show_spinner=False, max_entries=8)
def _categories(directory, file_versions):
    return get_manager_categories(directory)


@st.cache_data(show_spinner=False, max_entries=24)
def _answer(directory, file_versions, question, scope, options, categories):
    return answer_manager_question(question, scope, options, data_dir=directory, categories=categories)


def _use_example(question):
    st.session_state["manager_question"] = question
    st.session_state.pop("manager_request", None)


def _render_evidence(answer):
    if not answer.evidence:
        return
    with st.expander("نمونه شواهد از نظرهای ثبت‌شده"):
        topic = st.selectbox("موضوع یا برند", list(answer.evidence), key="manager_evidence_topic")
        for sample in answer.evidence[topic]:
            st.caption(f"{sample.source} · شناسهٔ نظر {sample.comment_id} · شناسهٔ محصول {sample.product_id}")
            st.html(f'<blockquote lang="fa" dir="auto" style="white-space:pre-wrap">{escape(sample.quote)}</blockquote>')


def _render_answer(answer: ManagerAnswer):
    scope = [answer.scope.category] if answer.scope.category else []
    if answer.scope.product_ids:
        scope.append(f"شناسه‌های درخواستی {'، '.join(answer.scope.product_ids)}")
    st.caption("محدودهٔ تحلیل · " + " / ".join(scope))
    if answer.missing_product_ids:
        st.warning(f"این شناسه‌ها در محدوده پیدا نشدند. {'، '.join(answer.missing_product_ids)}")

    products, reviews = st.columns(2)
    products.metric("محصولات محدوده", f"{answer.product_count:,}", border=True)
    reviews.metric("کل نظرات محدوده", f"{answer.review_count:,}", border=True)
    st.html(f'<p lang="fa" dir="rtl">{escape(answer.summary)}</p>')

    if answer.rows:
        number = st.column_config.NumberColumn
        st.dataframe(
            answer.rows, hide_index=True, width="stretch", column_order=TABLE_COLUMNS[answer.intent],
            column_config={
                "topic": "موضوع", "mentions": number("تعداد نظرهای مرتبط", format="localized"),
                "review_share_pct": number("سهم از کل نظرات محدوده", format="%.1f%%"),
                "title_fa": "محصول", "product_id": "شناسهٔ محصول", "brand": "برند",
                "review_count": number("تعداد نظرات", format="localized"),
                "products_reviewed": number("محصولات دارای نظر", format="localized"),
                "valid_label_count": number("برچسب‌های معتبر", format="localized"),
                "recommended_count": number("پیشنهاد می‌کنند", format="localized"),
                "not_recommended_count": number("پیشنهاد نمی‌کنند", format="localized"),
                "no_idea_count": number("بدون نظر مشخص", format="localized"),
                "recommendation_pct": number("نرخ پیشنهاد خرید", format="%.1f%%"),
                "not_recommended_pct": number("نرخ عدم پیشنهاد", format="%.1f%%"),
                "mean_rating": number("میانگین امتیاز از ۵", format="%.2f"),
                "rated_count": number("نظرهای امتیازدار", format="localized"),
                "main_issue": "شکایت اول شناسایی‌شده",
                "issue_mentions": number("تعداد نظرهای مرتبط با شکایت اول", format="localized"),
                "issue_share_pct": number("سهم شکایت اول از نظرات برند", format="%.1f%%"),
            },
        )
    for note in answer.notes:
        st.caption(note)
    _render_evidence(answer)


def render_manager_questions(data_dir: str | Path | None = None) -> None:
    try:
        files = resolve_analytics_files(data_dir)
        directory = str(files[0].parent)
        stats = [path.stat() for path in files]
        versions = tuple((str(path), stat.st_mtime_ns, stat.st_size) for path, stat in zip(files, stats))
        categories = _categories(directory, versions)
    except FileNotFoundError:
        st.info("برای پرسش‌های مدیر فروشگاه، ابتدا فایل‌های `products.parquet` و `reviews.parquet` را آماده کن.")
        return
    except (OSError, duckdb.Error):
        st.error("خواندن دسته‌بندی‌ها از دادهٔ فروشگاه با خطا روبه‌رو شد.")
        return

    st.subheader("پرسش مدیر فروشگاه", anchor=False)
    st.caption("یک دسته انتخاب کن یا در تنظیمات، شناسهٔ محصولات را وارد کن. می‌توانی نام دقیق دسته را در پرسش هم بنویسی.")
    for column, (label, question) in zip(st.columns(4), QUESTION_EXAMPLES.items()):
        column.button(label, key=f"manager_example_{label}", on_click=_use_example, args=(question,), width="stretch")

    with st.form("manager_questions"):
        category = st.selectbox("دستهٔ محصول", categories, index=None, placeholder="انتخاب دسته…", key="manager_category")
        question = st.text_area("پرسش تحلیلی", max_chars=500, key="manager_question", height=105)
        with st.expander("تنظیمات تحلیل"):
            product_ids = st.text_input("شناسهٔ محصولات، اختیاری", placeholder="مثلاً ۱۲۳۴۵، ۶۷۸۹۰")
            brands = st.text_input("برندها برای مقایسه، اختیاری", placeholder="نام برندها را با ویرگول جدا کن")
            st.caption("آستانه‌های گزارش محصولات پرنظر با نرخ پیشنهاد پایین")
            left, middle, right = st.columns(3)
            min_reviews = left.number_input("حداقل تعداد نظر", min_value=1, value=100, step=10)
            min_labeled = middle.number_input("حداقل برچسب معتبر", min_value=1, value=30, step=5)
            max_rate = right.number_input("حداکثر نرخ پیشنهاد (٪)", min_value=0.0, max_value=100.0, value=50.0, step=5.0)
        submitted = st.form_submit_button("اجرای تحلیل", type="primary", icon=":material/bar_chart:")

    if submitted:
        st.session_state.pop("manager_request", None)
        try:
            scope = ManagerScope(category, parse_product_ids(product_ids))
            options = ManagerOptions(min_reviews, min_labeled, max_rate, parse_brand_names(brands))
        except ValueError as error:
            st.error(str(error))
            return
        st.session_state["manager_request"] = (question, scope, options)

    request = st.session_state.get("manager_request")
    if request is None:
        return
    try:
        with st.spinner("در حال بررسی نظرهای محدودهٔ انتخاب‌شده…"):
            answer = _answer(directory, versions, *request, categories)
    except ValueError as error:
        st.error(str(error))
        return
    except (OSError, duckdb.Error):
        st.error("تحلیل داده اجرا نشد. بررسی کن هر دو فایل نهایی در مسیر داده موجود باشند.")
        return
    _render_answer(answer)
