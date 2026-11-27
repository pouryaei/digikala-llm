"""Persian Streamlit client for the Forouhar project.

Install: python -m pip install -r requirements.txt
Run:     python -m streamlit run app.py --server.port 8503

The bundled 100k catalogue and local review pipeline work without a separate server.
QBC13_BACKEND_URL is an optional explicit team-service override.

The catalog-to-comparison flow is inspired by the Mesbah team dashboard:
https://github.com/mmz1380/QBC_AI12_G2_P3/blob/ffacf9cb79f5902a54ec88059b1772c184f3a0c5/dashboard/app.py
"""

import os
import re
import argparse
from pathlib import Path
from html import escape
from math import isfinite

import altair as alt
import streamlit as st

from ui_services import APIError, api_call, backend_url, product_id_value


PAGES = (
    "جست‌وجوی محصول",
    "پرسش از نظرات",
    "مقایسهٔ محصولات",
    "تحلیل فروشگاه",
    "پیش‌بینی پیشنهاد خرید",
)
MAIN_TABS = (
    "جست‌وجوی محصول",
    "دستیار تحلیلی کالا (RAG)",
    "داشبورد و پیش‌بینی رضایت",
    "آزمایشگاه شواهد و امتیازی",
)
CATALOG_TABS = ("جست‌وجو و کشف محصول", "مقایسهٔ محصولات")
SEARCH_EXAMPLES = (
    ("کولهٔ لپ‌تاپ", "کولهٔ سبک برای لپ‌تاپ که خریدارها از فضای آن راضی‌اند"),
    ("هدفون بی‌سیم", "هدفون بی‌سیم با کیفیت صدای خوب برای استفادهٔ روزمره"),
    ("کفش پیاده‌روی", "کفش پیاده‌روی راحت با نظرهای مثبت خریداران"),
)
REVIEW_EXAMPLES = (
    ("کیفیت ساخت", "خریداران دربارهٔ کیفیت ساخت این محصول چه گفته‌اند؟"),
    ("ایرادهای پرتکرار", "ایرادهای پرتکراری که خریداران نوشته‌اند چیست؟"),
    ("نکات مثبت", "خریداران بیشتر از کدام ویژگی‌ها راضی بوده‌اند؟"),
)
PREDICTION_EXAMPLES = (
    ("نظر مثبت", "بعد از یک هفته استفاده از کیفیتش راضی‌ام و خریدش را پیشنهاد می‌کنم."),
    ("نظر منفی", "محصول خیلی زود خراب شد و خریدش را پیشنهاد نمی‌کنم."),
    ("نظر نامشخص", "بسته رسید. هنوز از محصول استفاده نکرده‌ام."),
)
LABELS = {
    "recommended": "پیشنهاد می‌کند",
    "not_recommended": "پیشنهاد نمی‌کند",
    "no_idea": "نظر مشخصی ندارد",
}
METRICS = {
    "product_count": "تعداد محصولات",
    "review_count": "تعداد نظرات",
    "recommendation_rate": "نرخ پیشنهاد خرید",
}
CHART_GROUPS = {
    "category": "دسته",
    "brand": "برند",
    "title_fa": "محصول",
    "product_title": "محصول",
    "term": "موضوع",
    "label": "گروه",
}
CHART_METRICS = {
    **METRICS,
    "comment_count": "تعداد نظرات",
    "count": "تعداد",
    "recommended_count": "تعداد پیشنهادها",
    "valid_label_count": "تعداد نظرات با برچسب معتبر",
}
SCOPES = {
    "processed_reviews": "نظرات پردازش‌شده",
    "processed_products": "محصولات پردازش‌شده",
    "fixture": "دادهٔ نمونه",
    "indexed_catalogue": "دادهٔ واقعی در محدودهٔ شاخص",
}
PRODUCT_FIELDS = {
    "category": "دسته",
    "brand": "برند",
    "price": "قیمت",
    "rating": "امتیاز",
    "rating_count": "تعداد امتیازها",
    "comment_count": "تعداد نظرات",
    "recommendation_rate": "نرخ پیشنهاد خرید",
    "recommended_count": "تعداد پیشنهادها",
    "valid_label_count": "تعداد نظرات با برچسب معتبر",
}
FILTER_LABELS = {
    "category": "دسته",
    "brand": "برند",
    "min_price": "حداقل قیمت",
    "max_price": "حداکثر قیمت",
}
DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
MAX_ID = 2**63 - 1
MAX_COMPARE_PRODUCTS = 5
MISSING = "ثبت نشده"
CSS = """
<style>
@font-face {
    font-family: "Vazirmatn";
    src: url(data:font/woff2;base64,{font}) format("woff2");
    font-weight: 100 900;
    font-style: normal;
    font-display: swap;
}

:root {
    --deep-violet: #152f6a;
    --purple: #315bb7;
    --magenta: #b03986;
    --coral-pink: #f482b8;
    --coral: #f295c2;
    --orange: #eaa3d5;
    --warm-yellow: #f9c2df;
    --canvas: #f5f5ff;
    --paper: #fffafe;
    --ink: #19294f;
    --muted: #485979;
    --line: #dce2f3;
    --accent: var(--purple);
    --accent-deep: var(--deep-violet);
    --accent-soft: #e9efff;
    --font-fa: "Vazirmatn", Tahoma, "Segoe UI", sans-serif;
}

[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background: var(--canvas);
    color: var(--ink);
}

[data-testid="stAppViewContainer"] {
    background-image: radial-gradient(ellipse at 0% 60%, #f9c2df25, transparent 53%),
                      radial-gradient(ellipse at 100% 0%, #315bb716, transparent 54%);
}

[data-testid="stAppDeployButton"] { display: none; }

[data-testid="stMainBlockContainer"],
[data-testid="stPopoverBody"] {
    direction: rtl;
    text-align: start;
    color: var(--ink);
    color-scheme: light;
    font-family: var(--font-fa);
}

[data-testid="stMainBlockContainer"] {
    max-width: 77rem;
    padding: 4.45rem clamp(1rem, 3.5vw, 2.8rem) 3rem;
}

[data-testid="stPopoverBody"] {
    background: var(--paper);
}

[data-testid="stMarkdownContainer"] :is(p, li),
[data-testid="stHeading"] :is(h1, h2, h3),
[data-testid="stWidgetLabel"] p,
[data-testid="stMetric"],
[data-testid="stText"],
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    font-family: var(--font-fa);
}

[data-testid="stMetricValue"] { direction: auto; }
[data-testid="stHeading"] :is(h1, h2, h3) { color: var(--ink); }
[data-testid="stCaptionContainer"] p,
[data-testid="stWidgetLabel"] p { color: var(--muted); }
[data-testid="stText"] { white-space: pre-wrap; overflow-wrap: anywhere; }

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    text-align: start;
    color: var(--ink);
    background: var(--paper);
}

[data-testid="stCodeBlock"] {
    direction: ltr;
    text-align: left;
}

[data-testid="stForm"],
[class*="st-key-product_card_"],
.st-key-search_panel,
.st-key-review_form_panel,
.st-key-prediction_form_panel,
.st-key-comparison_selection_panel,
.st-key-analytics_rows_panel,
[data-testid="stMetric"][data-border="true"] {
    background: var(--paper);
    border-color: var(--line);
    border-radius: 1rem;
}

[data-testid="stExpander"] details {
    background: var(--paper);
    border-color: var(--line);
    border-radius: .85rem;
}

[data-testid="stMainBlockContainer"] :is(button, a, input, textarea, [tabindex]):focus-visible,
[data-testid="stPopoverBody"] :is(button, a, input, textarea, [tabindex]):focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 3px;
}

:is([data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"]) {
    background: linear-gradient(105deg, var(--accent-deep), var(--accent));
    border-color: var(--accent);
    color: #fff;
}

:is([data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"]):hover {
    background: var(--accent-deep);
    border-color: var(--accent-deep);
    color: #fff;
}

[data-testid="stMainBlockContainer"] button p,
[data-testid="stPopoverBody"] button p { font-size: .9rem; }

.st-key-page :is([role="group"], [role="radiogroup"]) {
    flex-wrap: wrap;
    gap: .38rem;
}
.st-key-page button {
    min-height: 2.65rem;
    border-radius: .65rem !important;
    margin: 0 !important;
    border: 1px solid var(--line);
    background: var(--paper);
    color: var(--ink);
}
:is(.st-key-page, .st-key-analytics_view) button[aria-checked="true"] {
    background: var(--accent-soft) !important;
    color: var(--accent-deep) !important;
    border-color: var(--accent) !important;
    font-weight: 700;
    border-width: 1px;
}

.st-key-search_panel { box-shadow: 0 10px 34px #152f6a12; }
[data-testid="stChatMessage"] {
    flex-direction: row;
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 1rem;
    padding: 1rem 1.15rem;
}
[data-testid="stChatInput"] { direction: rtl; }
[data-testid="stChatInput"] textarea { text-align: start; font-family: var(--font-fa); }
.st-key-examples_search,
.st-key-examples_review,
.st-key-examples_prediction { gap: .55rem !important; }

.st-key-examples_search button,
.st-key-examples_review button,
.st-key-examples_prediction button {
    min-height: 2.4rem;
    background: var(--accent-soft);
    color: var(--accent-deep);
    border: 1px solid var(--line);
    border-radius: 2rem;
}
.st-key-examples_search button:hover,
.st-key-examples_review button:hover,
.st-key-examples_prediction button:hover { border-color: var(--accent); }

.brand {
    display: flex;
    align-items: center;
    gap: .68rem;
    line-height: 1.45;
}

.brand-mark {
    display: grid;
    place-items: center;
    width: 2.7rem;
    height: 2.7rem;
    border-radius: .9rem;
    background: linear-gradient(145deg, var(--deep-violet) 18%, var(--purple) 72%, var(--magenta));
    color: #fff;
    font-size: 1.55rem;
    font-weight: 750;
    flex: 0 0 auto;
}

.brand-name {
    color: var(--ink);
    font-size: 1.12rem;
    font-weight: 800;
}

.brand-detail {
    color: var(--muted);
    font-size: .76rem;
}

.forouhar-hero {
    position: relative;
    isolation: isolate;
    display: flex;
    align-items: center;
    min-height: 16.25rem;
    padding: clamp(1.45rem, 4vw, 2.9rem);
    overflow: hidden;
    background: radial-gradient(ellipse at 0% 100%, var(--orange), transparent 54%),
                linear-gradient(112deg, var(--coral-pink), var(--magenta) 25%, var(--purple) 50%, var(--deep-violet) 72%);
    border-radius: 1.25rem;
    color: #fff9ff;
}

.forouhar-hero::after {
    content: "";
    position: absolute;
    inset: auto 0 0;
    height: 3px;
    background: linear-gradient(90deg, var(--warm-yellow), var(--orange), var(--coral),
                                var(--coral-pink), var(--magenta), var(--purple), var(--deep-violet));
    pointer-events: none;
}

.hero-photos {
    position: absolute;
    z-index: -1;
    inset: 0 auto 0 0;
    width: 60%;
    mask-image: linear-gradient(to right, #000 57%, transparent 100%);
}

.hero-photo {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center 58%;
    opacity: 0;
    transition: opacity .65s ease;
}

.hero-photo.is-current { opacity: 1; }
.hero-copy { max-width: 52%; }

.hero-controls {
    position: absolute;
    inset-inline-end: .9rem;
    bottom: .75rem;
    display: flex;
    align-items: center;
    gap: .15rem;
    padding: .15rem .28rem;
    color: #fff9ff;
    background: #14285de8;
    border: 1px solid #efdffc42;
    border-radius: 2rem;
    visibility: hidden;
}

[data-carousel-ready="true"] .hero-controls { visibility: visible; }

.hero-controls button {
    color: inherit;
    font: inherit;
    min-height: 1.8rem;
    background: transparent;
    border: 0;
    border-radius: 1rem;
    cursor: pointer;
}

.hero-controls button:hover { background: #fff9ff20; }
[data-testid="stMainBlockContainer"] .hero-controls button:focus-visible {
    outline: 2px solid var(--warm-yellow);
    outline-offset: 1px;
}

.hero-controls .hero-playback { padding: 0 .45rem; font-size: .76rem; }
.hero-dot { display: grid; place-items: center; width: 1.65rem; }
.hero-dot::before {
    content: "";
    width: .44rem;
    height: .44rem;
    border-radius: 50%;
    background: #fff9ff91;
}
.hero-dot[aria-current="true"]::before { width: .78rem; border-radius: .35rem; background: var(--warm-yellow); }

.forouhar-hero h1,
.page-heading h1 {
    margin: 0;
    padding: 0;
    font-size: clamp(1.7rem, 3.5vw, 2.7rem);
    font-weight: 800;
    line-height: 1.4;
}

.forouhar-hero h1 { color: #fff9ff; }
.forouhar-hero p,
.page-heading p {
    margin: .48rem 0 0;
    padding: 0;
    font-size: .94rem;
    line-height: 1.8;
}
.forouhar-hero p { color: #f9e8f4; }

.page-heading {
    padding: .65rem 0 .2rem;
}
.page-heading h1 {
    color: var(--ink);
    font-size: clamp(1.5rem, 3vw, 2.15rem);
}
.page-heading p { color: var(--muted); }

[class*="st-key-product_card_"] {
    transition: box-shadow .15s ease, border-color .15s ease;
}
[class*="st-key-product_card_"]:hover {
    border-color: #d489bd;
    box-shadow: 0 9px 26px #152f6a14;
}

.product-title {
    margin: .55rem 0 .8rem;
    padding: 0;
    color: var(--ink);
    font-size: 1.12rem;
    font-weight: 780;
    line-height: 1.65;
    overflow-wrap: anywhere;
}

.meta-list {
    display: flex;
    flex-wrap: wrap;
    gap: .35rem;
    align-items: center;
}

.meta-chip {
    display: inline-block;
    padding: .19rem .55rem;
    max-width: 100%;
    border-radius: 2rem;
    background: var(--accent-soft);
    color: var(--accent-deep);
    font-size: .75rem;
    overflow-wrap: anywhere;
}

.meta-chip.id-chip {
    background: #fceaf4;
    color: #873b6e;
}

.product-stats {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: .48rem 1rem;
    padding-top: .7rem;
    border-top: 1px solid var(--line);
    margin: 0;
}
.product-stats dt {
    color: var(--muted);
    font-size: .77rem;
}
.product-stats dd {
    color: var(--ink);
    margin: .13rem 0 0;
    font-size: .93rem;
    font-weight: 720;
    overflow-wrap: anywhere;
}
.product-stats dd.missing { font-weight: 400; color: var(--muted); }
.product-stats dd.price { color: var(--accent-deep); }
[class*="st-key-product_actions_"] button { min-height: 2.55rem; }

.answer-card {
    margin: 0;
    padding: 1.15rem 1.35rem;
    background: var(--accent-soft);
    border: 1px solid var(--line);
    border-inline-start: 4px solid var(--accent);
    border-radius: .85rem;
    color: var(--ink);
}
.answer-card p,
.review-card blockquote {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    margin: 0;
    font-size: .94rem;
    line-height: 1.95;
    color: var(--ink);
}

.review-card {
    padding: .72rem .8rem;
    margin: 0 0 .5rem;
    border: 1px solid var(--line);
    border-radius: .72rem;
    background: var(--paper);
}
.review-card:last-child { margin-bottom: 0; }
.review-card blockquote { margin-top: .45rem; }

.compare-table {
    position: relative;
    width: 100%;
    overflow: auto;
    max-height: 35rem;
    border: 1px solid var(--line);
    border-radius: .9rem;
    background: var(--paper);
}
.compare-table table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    text-align: start;
    font-size: .88rem;
}
.compare-table :is(th, td) {
    text-align: start;
    vertical-align: top;
    padding: .62rem .85rem;
    border-block-end: 1px solid var(--line);
    color: var(--ink);
    overflow-wrap: anywhere;
}
.compare-table thead th {
    position: sticky;
    top: 0;
    min-width: 10rem;
    background: #e2eaff;
    z-index: 1;
}
.compare-table th small {
    display: block;
    color: var(--muted);
    font-weight: 450;
    line-height: 1.6;
}
.compare-table tbody tr:nth-child(odd) { background: #f9f4fc; }
.compare-table tbody th { background: #eff2ff; }
.compare-table th:first-child {
    position: sticky;
    inset-inline-start: 0;
    min-width: 7rem;
    z-index: 1;
}
.compare-table thead th:first-child { z-index: 2; }

@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding: 3.9rem .85rem 2rem; }
    .forouhar-hero {
        min-height: 0;
        padding: 1.2rem 1rem 10.5rem;
        border-radius: 1rem;
        background: linear-gradient(165deg, var(--deep-violet) 13%, var(--purple) 58%, var(--coral-pink));
    }
    .hero-copy { max-width: none; }
    .hero-photos {
        inset: auto 0 0;
        width: 100%;
        height: 11rem;
        mask-image: linear-gradient(to bottom, transparent, #000 30%);
    }
    .hero-controls { bottom: .58rem; inset-inline-end: .6rem; }
    .st-key-page button { min-height: 2.75rem; }
    .compare-table :is(th, td) { padding: .55rem; }
    .compare-table th:first-child { min-width: 5.75rem; }
    .compare-table thead th { min-width: 8rem; }
}

@media (prefers-reduced-motion: reduce) {
    [class*="st-key-product_card_"], .hero-photo { transition: none; }
}
</style>
"""


def run_request(key, path, payload=None, method="POST"):
    st.session_state.pop(key, None)
    try:
        with st.spinner("در انتظار پاسخ سرویس…"):
            st.session_state[key] = api_call(method, path, payload)
    except APIError as exc:
        st.session_state[key] = {"error": str(exc)}


def result(key, with_details=True):
    entry = st.session_state.get(key)
    if not entry:
        return None
    if "error" in entry:
        st.error(entry["error"])
        return None
    if with_details:
        if entry["mode"] == "mock" or entry["data"].get("scope") == "fixture":
            st.warning("این پاسخ از دادهٔ نمونه است.")
        if entry.get("notice"):
            st.info(entry["notice"])
        st.caption(f"زمان پاسخ · {entry['elapsed']:.2f} ثانیه")
    return entry["data"]


def number(raw, minimum=1):
    value = raw.strip().translate(DIGITS)
    if not re.fullmatch(r"[0-9]{1,19}", value):
        raise ValueError("یک عدد صحیح وارد کنید.")
    parsed = int(value)
    if not minimum <= parsed <= MAX_ID:
        raise ValueError(f"عدد باید در بازهٔ {minimum:,} تا {MAX_ID:,} باشد.")
    return parsed


def require_text(value, message):
    if not value.strip():
        raise ValueError(message)
    return value.strip()


def display(value):
    if value is None or value == "":
        return MISSING
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value)


def percent(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1:
        return f"{value:.1%}"
    return MISSING


def product_value(item, key):
    value = item.get(key)
    if key == "recommendation_rate":
        return percent(value)
    if key == "price" and value is not None and item.get("price_unit"):
        return f"{display(value)} {item['price_unit']}"
    if key == "rating" and value is not None and item.get("rating_scale"):
        return f"{display(value)} / {display(item['rating_scale'])}"
    return display(value)


def records(data, field):
    rows = data.get(field)
    if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
        return rows
    st.error(f"فیلد `{field}` در پاسخ سرویس ساختار مورد انتظار را ندارد.")
    return None


def loaded_categories():
    categories = st.session_state.get("categories", {}).get("data", {}).get("items", [])
    if not isinstance(categories, list):
        return []
    return sorted({item.strip() for item in categories if isinstance(item, str) and item.strip()})


def filter_inputs(prefix, show_prices=False):
    fields = st.columns(2)
    with fields[0]:
        options = loaded_categories()
        if options:
            category = st.selectbox(
                "دسته", options=options, index=None, placeholder="همهٔ دسته‌ها",
                key=f"{prefix}_category_select",
            )
        else:
            category = st.text_input("دسته", key=f"{prefix}_category")
    with fields[1]:
        brand = st.text_input("برند", key=f"{prefix}_brand")

    minimum = maximum = ""
    if show_prices:
        low, high = st.columns(2)
        minimum = low.text_input("حداقل قیمت", key=f"{prefix}_min_price")
        maximum = high.text_input("حداکثر قیمت", key=f"{prefix}_max_price")
        st.caption("بازهٔ قیمت را با واحد ثبت‌شده در داده وارد کنید.")
    return category, brand, minimum, maximum


def build_filters(category, brand, minimum, maximum):
    filters = {}
    if category and category.strip():
        filters["category"] = category.strip()
    if brand.strip():
        filters["brand"] = brand.strip()
    if minimum.strip():
        filters["min_price"] = number(minimum, minimum=0)
    if maximum.strip():
        filters["max_price"] = number(maximum, minimum=0)
    if "min_price" in filters and "max_price" in filters:
        if filters["min_price"] > filters["max_price"]:
            raise ValueError("حداقل قیمت باید کمتر از یا برابر با حداکثر قیمت باشد.")
    return filters


def show_filters(data, requested=False):
    filters = data.get("applied_filters")
    if isinstance(filters, dict) and filters:
        values = [f"{FILTER_LABELS.get(key, key)} · {display(value)}" for key, value in filters.items()]
        st.caption("فیلترهای اعمال‌شده")
        st.text("  |  ".join(values))
    elif requested:
        st.warning("سرویس اعمال فیلترهای انتخاب‌شده را در پاسخ تأیید نکرد.")
    notes = data.get("search_notes")
    if isinstance(notes, list):
        for note in notes:
            if isinstance(note, str) and note.strip():
                st.caption(note)


def page_heading(title, detail=None):
    description = f"<p>{escape(detail)}</p>" if detail else ""
    st.html(f'<section class="page-heading" lang="fa" dir="rtl"><h1>{escape(title)}</h1>{description}</section>')


HERO_ROTATION_SCRIPT = """
<script>
(() => {
    const carousel = document.getElementById("forouhar-search-hero");
    if (!carousel || carousel.dataset.carouselReady === "true") return;

    const photographs = Array.from(carousel.querySelectorAll(".hero-photo"));
    const dots = Array.from(carousel.querySelectorAll(".hero-dot"));
    const playback = carousel.querySelector(".hero-playback");
    let position = 0;
    let playing = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let timer;

    function selectPhoto(index) {
        photographs[position].classList.remove("is-current");
        dots[position].removeAttribute("aria-current");
        position = index;
        photographs[position].classList.add("is-current");
        dots[position].setAttribute("aria-current", "true");
    }

    function schedule() {
        window.clearTimeout(timer);
        if (!playing) return;
        timer = window.setTimeout(() => {
            if (!carousel.isConnected) return;
            selectPhoto((position + 1) % photographs.length);
            schedule();
        }, 5000);
    }

    function updatePlayback() {
        playback.textContent = playing ? "توقف پخش" : "ادامهٔ پخش";
        schedule();
    }

    playback.addEventListener("click", () => {
        playing = !playing;
        updatePlayback();
    });
    dots.forEach((dot, index) => dot.addEventListener("click", () => {
        selectPhoto(index);
        schedule();
    }));
    carousel.dataset.carouselReady = "true";
    updatePlayback();
})();
</script>
"""


def search_hero():
    photographs = []
    dots = []
    for index, (label, content) in enumerate(HERO_PHOTOGRAPHS):
        active_class = " is-current" if index == 0 else ""
        current = ' aria-current="true"' if index == 0 else ""
        label = escape(f"نمایش {label}")
        photographs.append(
            f'<img class="hero-photo{active_class}" src="data:image/webp;base64,{content}" alt="">'
        )
        dots.append(
            f'<button type="button" class="hero-dot" aria-label="{label}" title="{label}"{current}></button>'
        )
    # The markup and script here only use bundled content, never search or API input.
    st.html(
        '<section id="forouhar-search-hero" class="forouhar-hero" lang="fa" dir="rtl">'
        f'<div class="hero-photos" aria-hidden="true">{"".join(photographs)}</div>'
        '<div class="hero-copy"><h1>محصول مناسب را پیدا کنید</h1>'
        '<p>نام محصول، بودجه یا ویژگی‌های مدنظرتان را بنویسید.</p></div>'
        '<div class="hero-controls" role="group" aria-label="کنترل تصاویر سربرگ">'
        '<button type="button" class="hero-playback">توقف پخش</button>'
        f'{"".join(dots)}</div>'
        '</section>'
        f'{HERO_ROTATION_SCRIPT}',
        unsafe_allow_javascript=True,
    )


def set_example(input_key, value):
    st.session_state[input_key] = value
    if input_key == "prediction_text":
        st.session_state["prediction_title"] = ""


def example_buttons(prefix, input_key, examples):
    with st.container(key=f"examples_{prefix}", horizontal=True, horizontal_alignment="right",
                      vertical_alignment="center"):
        st.caption("نمونه برای شروع")
        for index, (label, value) in enumerate(examples):
            st.button(
                label,
                key=f"{prefix}_example_{index}",
                on_click=set_example,
                args=(input_key, value),
            )


def switch_page(page):
    st.session_state["page"] = page
    if page in (PAGES[0], PAGES[2]):
        st.session_state["main_tabs"] = MAIN_TABS[0]
        st.session_state["catalog_tabs"] = CATALOG_TABS[1] if page == PAGES[2] else CATALOG_TABS[0]
    elif page == PAGES[1]:
        st.session_state["main_tabs"] = MAIN_TABS[1]
    else:
        st.session_state["main_tabs"] = MAIN_TABS[2]


def remember_products(items):
    known = st.session_state.setdefault("review_product_catalog", {})
    for item in items:
        identifier = product_id_value(item.get("product_id"))
        if identifier is not None:
            known[identifier] = str(item.get("title_fa") or f"محصول {identifier}")
    return known


def open_review(product_id, title=None):
    remember_products([{"product_id": product_id, "title_fa": title}])
    switch_page(PAGES[1])
    st.session_state["review_id"] = str(product_id)
    st.session_state["review_product_choice"] = product_id
    st.session_state.pop("review_result", None)


def comparison_selection():
    return st.session_state.setdefault("comparison_selection", {})


def add_to_comparison(product_id, title):
    selection = comparison_selection()
    if product_id not in selection and len(selection) < MAX_COMPARE_PRODUCTS:
        selection[product_id] = title
        st.session_state.pop("comparison_result", None)


def remove_from_comparison(product_id):
    comparison_selection().pop(product_id, None)
    st.session_state.pop("comparison_result", None)


def comparison_add_button(product_id, title, key):
    selection = comparison_selection()
    added = product_id in selection
    full = len(selection) >= MAX_COMPARE_PRODUCTS
    st.button(
        "در فهرست مقایسه" if added else "افزودن به مقایسه",
        key=key,
        disabled=added or full,
        help="برای افزودن محصول جدید، یکی از انتخاب‌ها را در بخش مقایسه حذف کنید."
        if full and not added else None,
        icon=":material/check:" if added else ":material/add:",
        on_click=add_to_comparison,
        args=(product_id, title),
    )


def product_card(item, origin):
    identifier = item.get("product_id")
    action_id = product_id_value(identifier)
    title = str(item.get("title_fa") or "محصول بدون عنوان")
    chips = [
        f'<span class="meta-chip">{escape(str(item[key]))}</span>'
        for key in ("category", "brand") if item.get(key)
    ]
    if identifier is not None:
        chips.append(f'<span class="meta-chip id-chip">شناسهٔ <bdi dir="ltr">{escape(str(identifier))}</bdi></span>')
    stats = []
    for key in ("price", "rating", "comment_count", "recommendation_rate"):
        value = product_value(item, key)
        style = "missing" if value == MISSING else key
        stats.append(
            f'<div><dt>{PRODUCT_FIELDS[key]}</dt>'
            f'<dd class="{style}"><bdi>{escape(value)}</bdi></dd></div>'
        )
    with st.container(border=True, key=f"product_card_{origin}", height="stretch"):
        st.html(
            f'<div lang="fa" dir="rtl"><div class="meta-list">{"".join(chips)}</div>'
            f'<h3 class="product-title">{escape(title)}</h3>'
            f'<dl class="product-stats">{"".join(stats)}</dl></div>'
        )
        if action_id is not None:
            with st.container(horizontal=True, key=f"product_actions_{origin}"):
                st.button(
                    "پرسش از نظرات این محصول",
                    key=f"ask_{origin}",
                    icon=":material/forum:",
                    on_click=open_review,
                    args=(action_id, title),
                )
                comparison_add_button(action_id, title, f"compare_add_{origin}")


def discovery():
    search_hero()
    with st.container(border=True, key="search_panel"):
        with st.form("search", border=False):
            search_input, search_action = st.columns([4, 1], vertical_alignment="bottom")
            query = search_input.text_input(
                "دنبال چه محصولی هستید؟", max_chars=500, key="search_query",
                placeholder="مثلاً کولهٔ سبک برای لپ‌تاپ که خریدارها از جیب‌هایش راضی‌اند",
            )
            submitted = search_action.form_submit_button(
                "جست‌وجو", type="primary", icon=":material/search:", width="stretch",
            )
            with st.expander("فیلترهای اختیاری", icon=":material/tune:"):
                filter_values = filter_inputs("search", show_prices=True)
                limit = st.slider("حداکثر تعداد نتایج", min_value=1, max_value=20, value=5)
        example_buttons("search", "search_query", SEARCH_EXAMPLES)
    if submitted:
        st.session_state.pop("search_result", None)
        try:
            payload = {"query": require_text(query, "نیاز خود را کوتاه توضیح دهید."), "limit": limit}
            filters = build_filters(*filter_values)
            if filters:
                payload["filters"] = filters
            st.session_state["search_used_filters"] = bool(filters)
            st.session_state["submitted_search_query"] = payload["query"]
            run_request("search_result", "/api/search", payload)
        except ValueError as exc:
            st.error(str(exc))

    selected_count = len(comparison_selection())
    if selected_count:
        st.button(
            f"رفتن به مقایسه ({selected_count})", type="secondary",
            icon=":material/compare_arrows:",
            on_click=switch_page, args=(PAGES[2],),
        )

    data = result("search_result")
    if data is not None:
        if submitted_query := st.session_state.get("submitted_search_query"):
            st.text(f"درخواست جست‌وجو · {submitted_query}")
        show_filters(data, requested=st.session_state.get("search_used_filters", False))
        items = records(data, "items")
        if items == []:
            st.info("محصولی برای این درخواست پیدا نشد. نام محصول یا فیلتر برند و قیمت را تغییر دهید.")
        if items:
            remember_products(items)
            st.subheader(f"نتایج جست‌وجو · {len(items)} محصول", anchor=False)
            for index in range(0, len(items), 2):
                group = items[index:index + 2]
                columns = st.columns(2) if len(group) == 2 else (st.container(),)
                for offset, (column, item) in enumerate(zip(columns, group)):
                    with column:
                        product_card(item, f"search_{index + offset}")

    with st.expander("یافتن محصول با شناسه", icon=":material/tag:"):
        with st.form("lookup"):
            raw_id = st.text_input("شناسهٔ محصول", key="lookup_id")
            lookup = st.form_submit_button("نمایش محصول")
        if lookup:
            st.session_state.pop("lookup_result", None)
            try:
                identifier = number(raw_id)
                run_request("lookup_result", f"/api/products/{identifier}", method="GET")
            except ValueError as exc:
                st.error(str(exc))
        data = result("lookup_result")
        if data is not None:
            product = data.get("product")
            if isinstance(product, dict):
                remember_products([product])
                product_card(product, "lookup")
            else:
                st.error("پاسخ سرویس شامل اطلاعات محصول نبود.")


def verified_evidence(items, product_ids):
    evidence = []
    allowed = {str(identifier) for identifier in product_ids}
    for item in items:
        identifier = item.get("comment_id") or item.get("review_id")
        if isinstance(identifier, bool) or not isinstance(identifier, (int, str)):
            continue
        if not str(identifier).strip():
            continue
        if item.get("product_id") is not None and str(item["product_id"]) not in allowed:
            continue
        evidence.append((identifier, item))
    return evidence


def show_evidence(evidence, label="شواهد نظرات"):
    if not evidence:
        return
    cards = []
    for identifier, item in evidence:
        metadata = f'<span class="meta-chip id-chip">نظر <bdi dir="ltr">{escape(str(identifier))}</bdi></span>'
        if item.get("product_id") is not None:
            product_id = escape(str(item["product_id"]))
            metadata += f'<span class="meta-chip">محصول <bdi dir="ltr">{product_id}</bdi></span>'
        quote = f"<blockquote>{escape(str(item['text']))}</blockquote>" if item.get("text") else ""
        cards.append(
            f'<figure class="review-card"><figcaption class="meta-list">{metadata}</figcaption>{quote}</figure>'
        )
    with st.expander(f"{label} · {len(evidence)} مورد", expanded=True, icon=":material/format_quote:"):
        st.html(f'<div lang="fa" dir="rtl">{"".join(cards)}</div>')


def show_answer(text):
    st.html(f'<div class="answer-card" lang="fa" dir="rtl"><p>{escape(text)}</p></div>')


def select_review_product():
    known = st.session_state.setdefault("review_product_catalog", {})
    with st.container(border=True, key="review_form_panel"):
        with st.expander("یافتن محصول با نام یا شناسه", expanded=not bool(known), icon=":material/search:"):
            with st.form("review_product_lookup", border=False):
                field, action = st.columns([4, 1], vertical_alignment="bottom")
                lookup = field.text_input("نام یا شناسهٔ محصول", key="review_id", max_chars=500)
                submitted = action.form_submit_button("یافتن محصول", width="stretch")
            if submitted:
                st.session_state.pop("review_lookup_result", None)
                try:
                    value = require_text(lookup, "نام یا شناسهٔ محصول را وارد کنید.")
                    if re.fullmatch(r"[0-9]+", value.translate(DIGITS)):
                        identifier = number(value)
                        run_request("review_lookup_result", f"/api/products/{identifier}", method="GET")
                    else:
                        run_request("review_lookup_result", "/api/search", {"query": value, "limit": 8})
                except ValueError as error:
                    st.error(str(error))
            data = result("review_lookup_result")
            if data is not None:
                product = data.get("product")
                items = [product] if isinstance(product, dict) else records(data, "items")
                if items == []:
                    st.info("محصولی برای انتخاب پیدا نشد. نام دقیق یا شناسهٔ دیگری وارد کنید.")
                if items:
                    choices = [product_id_value(item.get("product_id")) for item in items]
                    choices = [identifier for identifier in choices if identifier is not None]
                    remember_products(items)
                    if submitted and len(choices) == 1:
                        st.session_state["review_product_choice"] = choices[0]

        if st.session_state.get("review_product_choice") not in known:
            st.session_state["review_product_choice"] = None
        return st.selectbox(
            "محصول موردنظر برای گفتگو", list(known),
            key="review_product_choice", index=None,
            format_func=lambda identifier: f"{known[identifier]} · {identifier}",
            placeholder="محصولی از نتایج جست‌وجو انتخاب کنید", disabled=not known,
        )


def queue_review_question(identifier, question):
    st.session_state["queued_review_question"] = (identifier, question)


def render_chat_reply(entry, identifier):
    if "error" in entry:
        st.warning(entry["error"])
        return
    data = entry["data"]
    if entry.get("notice"):
        st.info(entry["notice"])
    if entry.get("mode") == "mock" or data.get("scope") == "fixture":
        st.warning("این پاسخ از دادهٔ نمونه است.")
    if data.get("answer_mode") == "insufficient_evidence":
        st.info(data.get("reason") or "برای این سؤال شواهد کافی پیدا نشد. دربارهٔ ویژگی دیگری از همین محصول بپرسید.")
    else:
        show_answer(data["answer"])
        show_evidence(verified_evidence(data["evidence"], [identifier]))
    if data.get("meta", {}).get("fallback_reason"):
        st.caption("جمع‌بندی مدل زبانی معتبر نبود؛ پاسخ مستند محلی نمایش داده شد.")
    st.caption(f"زمان پاسخ · {entry['elapsed']:.2f} ثانیه")


def review_qa():
    page_heading(MAIN_TABS[1], "محصول را انتخاب کنید و دربارهٔ تجربهٔ خریداران گفتگو کنید.")
    identifier = select_review_product()
    histories = st.session_state.setdefault("review_chats", {})
    if identifier is None:
        st.info("ابتدا محصولی از جست‌وجو انتخاب کنید یا نام و شناسهٔ آن را وارد کنید.")
        st.chat_input("سؤال دربارهٔ محصول", disabled=True, key="review_chat_inactive")
        return

    title = st.session_state["review_product_catalog"][identifier]
    history = histories.setdefault(identifier, [])
    left, right = st.columns([4, 1], vertical_alignment="center")
    left.text(title)
    if right.button("پاک کردن گفتگو", key=f"reset_chat_{identifier}", disabled=not history):
        history.clear()

    for column, (label, question) in zip(st.columns(3), REVIEW_EXAMPLES):
        column.button(label, key=f"review_example_{label}", on_click=queue_review_question,
                      args=(identifier, question), width="stretch")

    for message in history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.text(message["text"])
            else:
                render_chat_reply(message["entry"], identifier)

    prompt = st.chat_input("دربارهٔ این محصول بپرسید…", key="review_chat_input", max_chars=500)
    queued = st.session_state.pop("queued_review_question", None)
    question = prompt or (queued[1] if queued and queued[0] == identifier else None)
    if not question or not question.strip():
        return
    question = question.strip()
    history.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.text(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("در حال پیدا کردن نظرها و آماده کردن پاسخ…"):
                entry = api_call("POST", "/api/reviews/ask", {"product_id": identifier, "query": question})
        except APIError as error:
            entry = {"error": str(error)}
        history.append({"role": "assistant", "entry": entry})
        render_chat_reply(entry, identifier)


def find_comparison_products():
    with st.expander("جست‌وجوی محصول برای مقایسه", expanded=True, icon=":material/search:"):
        with st.form("compare_search", border=False):
            field, action = st.columns([4, 1], vertical_alignment="bottom")
            query = field.text_input(
                "نام، برند یا دستهٔ محصول", max_chars=500,
                placeholder="مثلاً کولهٔ لپ‌تاپ",
            )
            submitted = action.form_submit_button("یافتن محصول", width="stretch")
        if submitted:
            st.session_state.pop("compare_search_result", None)
            try:
                query = require_text(query, "نام، برند یا دستهٔ محصول را وارد کنید.")
                run_request("compare_search_result", "/api/search", {"query": query, "limit": 5})
            except ValueError as exc:
                st.error(str(exc))

        data = result("compare_search_result")
        if data is None:
            return
        items = records(data, "items")
        if items == []:
            st.info("محصولی برای این جست‌وجو پیدا نشد.")
        for index, item in enumerate(items or []):
            identifier = product_id_value(item.get("product_id"))
            if identifier is None:
                continue
            title = str(item.get("title_fa") or "محصول بدون عنوان")
            name, action = st.columns([3, 2], vertical_alignment="center")
            name.text(f"{title}  |  شناسهٔ {identifier}")
            with action:
                comparison_add_button(identifier, title, f"compare_add_match_{index}")


def comparison_matrix(items):
    st.subheader("اطلاعات ثبت‌شدهٔ محصولات", anchor=False)
    only_differences = st.toggle(
        "فقط تفاوت‌ها", key="only_differences",
        help="ردیف‌هایی را که مقدار نمایش‌داده‌شدهٔ آن‌ها برای همهٔ محصول‌ها یکسان است پنهان می‌کند.",
    )
    headers = ['<th scope="col">ویژگی</th>']
    for item in items:
        title = escape(str(item.get("title_fa") or "محصول بدون عنوان"))
        identifier = item.get("product_id")
        suffix = f'<small>شناسهٔ <bdi dir="ltr">{escape(str(identifier))}</bdi></small>' if identifier is not None else ""
        headers.append(f'<th scope="col">{title}{suffix}</th>')

    rows = []
    for key, label in PRODUCT_FIELDS.items():
        values = [product_value(item, key) for item in items]
        if only_differences and len(set(values)) == 1:
            continue
        cells = "".join(f"<td><bdi>{escape(value)}</bdi></td>" for value in values)
        rows.append(f'<tr><th scope="row">{label}</th>{cells}</tr>')
    if not rows:
        st.info("در اطلاعات ثبت‌شده تفاوتی برای نمایش پیدا نشد.")
        return
    st.html(
        '<div class="compare-table" lang="fa" dir="rtl" tabindex="0" role="region" aria-label="جدول مقایسه">'
        f'<table aria-label="مقایسهٔ مشخصات"><thead><tr>{"".join(headers)}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def comparison():
    page_heading(PAGES[2], "۲ تا ۵ محصول را از جست‌وجو یا با شناسه انتخاب کنید.")
    find_comparison_products()
    selection = comparison_selection()
    if selection:
        with st.container(border=True, key="comparison_selection_panel"):
            st.subheader(f"محصولات انتخاب‌شده · {len(selection)} از {MAX_COMPARE_PRODUCTS}", anchor=False)
            for identifier, title in selection.items():
                name, action = st.columns([3, 2], vertical_alignment="center")
                name.text(f"{title}  |  شناسهٔ {identifier}")
                action.button(
                    "حذف از مقایسه", key=f"remove_comparison_{identifier}",
                    type="tertiary", icon=":material/remove_circle:",
                    on_click=remove_from_comparison, args=(identifier,),
                )

    with st.form("compare"):
        raw_ids = st.text_input(
            "ورود مستقیم شناسهٔ محصول", max_chars=140, placeholder="مثلاً 12345، 67890",
            help="می‌توانید به انتخاب‌های بالا شناسه اضافه کنید. شناسه‌ها را با ویرگول یا فاصله جدا کنید.",
        )
        submitted = st.form_submit_button(
            "مقایسهٔ محصولات", type="primary", icon=":material/compare_arrows:",
        )
    if submitted:
        st.session_state.pop("comparison_result", None)
        try:
            parts = re.split(r"[,،\s]+", raw_ids.strip())
            manual_ids = [number(part) for part in parts if part]
            identifiers = list(dict.fromkeys([*selection, *manual_ids]))
            if len(set(manual_ids)) != len(manual_ids):
                raise ValueError("شناسه‌های واردشده باید متمایز باشند.")
            if not 2 <= len(identifiers) <= MAX_COMPARE_PRODUCTS:
                raise ValueError("۲ تا ۵ شناسهٔ متمایز وارد کنید.")
            st.session_state["compared_products"] = identifiers
            run_request("comparison_result", "/api/products/compare", {"product_ids": identifiers})
        except ValueError as exc:
            st.error(str(exc))

    data = result("comparison_result")
    if data is None:
        return
    items = records(data, "items")
    if items is None:
        return
    if not items:
        st.info("سرویس اطلاعاتی برای مقایسه برنگرداند.")
        return

    comparison_matrix(items)

    identifiers = st.session_state["compared_products"]
    evidence = data.get("evidence", [])
    if isinstance(evidence, list):
        groups = {}
        for reference in verified_evidence([item for item in evidence if isinstance(item, dict)], identifiers):
            product_id = reference[1].get("product_id")
            groups.setdefault(str(product_id) if product_id is not None else None, []).append(reference)
        for identifier in identifiers:
            show_evidence(groups.get(str(identifier), []), f"شواهد نظرات محصول {identifier}")
        show_evidence(groups.get(None, []), "شواهد بدون شناسهٔ محصول")
    notes = data.get("notes")
    if isinstance(notes, str):
        notes = [notes]
    summary = "\n".join(note for note in notes if isinstance(note, str) and note.strip()) if isinstance(notes, list) else ""
    if summary:
        st.subheader("جمع‌بندی سرویس", anchor=False)
        show_answer(summary)


def chart_number(value):
    if type(value) is int:
        return 0 <= value < 2**53
    return type(value) is float and isfinite(value) and value >= 0


def analytics_rows(rows):
    group_key = None
    for key in CHART_GROUPS:
        names = [row.get(key) for row in rows]
        if all(isinstance(name, str) and name.strip() for name in names):
            if len(set(name.strip() for name in names)) == len(rows):
                group_key = key
                break

    metrics = []
    if len(rows) > 1 and group_key:
        for key in CHART_METRICS:
            values = [row.get(key) for row in rows]
            if all(chart_number(value) for value in values):
                if key != "recommendation_rate" or max(values) <= 1:
                    metrics.append(key)
    if not metrics:
        st.dataframe(rows, hide_index=True, width="stretch")
        return

    with st.container(border=True, key="analytics_rows_panel"):
        view = st.segmented_control(
            "نمای نتیجه", ("نمودار", "جدول"), default="نمودار",
            required=True, key="analytics_view",
        )
        if view == "جدول":
            st.dataframe(rows, hide_index=True, width="stretch")
            return

        metric = metrics[0]
        if len(metrics) > 1:
            metric = st.selectbox("معیار نمودار", metrics, format_func=CHART_METRICS.get, key="chart_metric")
        group_title = CHART_GROUPS[group_key]
        metric_title = CHART_METRICS[metric]
        chart_rows = [
            {"group": row[group_key].strip(), "value": row[metric]}
            for row in sorted(rows, key=lambda row: row[metric], reverse=True)[:12]
        ]
        is_rate = metric == "recommendation_rate"
        value_format = ".1%" if is_rate else ",.4~g"
        scale = alt.Scale(domain=[0, 1]) if is_rate else alt.Scale(zero=True)
        chart = (
            alt.Chart(alt.Data(values=chart_rows), height=max(130, 33 * len(chart_rows)))
            .mark_bar(color="#b03986", cornerRadiusEnd=5)
            .encode(
                x=alt.X("value:Q", title=metric_title, scale=scale, axis=alt.Axis(format=value_format)),
                y=alt.Y("group:N", sort=[row["group"] for row in chart_rows],
                        title=None, axis=alt.Axis(labelLimit=150)),
                tooltip=[
                    alt.Tooltip("group:N", title=group_title),
                    alt.Tooltip("value:Q", title=metric_title, format=value_format),
                ],
            )
            .configure(background="transparent", font="Vazirmatn")
            .configure_axis(labelColor="#485979", titleColor="#152f6a", gridColor="#dce2f3")
            .configure_view(stroke=None)
        )
        st.altair_chart(chart, width="stretch", theme=None, alt=f"{metric_title} به تفکیک {group_title}")
        if len(rows) > len(chart_rows):
            st.caption("نمودار ۱۲ گروه با بالاترین مقدار را نشان می‌دهد. همهٔ گروه‌ها در جدول هستند.")


def analytics():
    from analytics_service import BUNDLED_ANALYTICS_DIR, bundled_catalog_scope, resolve_analytics_files
    page_heading(MAIN_TABS[2])
    try:
        using_bundled = resolve_analytics_files()[0].parent == BUNDLED_ANALYTICS_DIR
    except FileNotFoundError:
        using_bundled = False
    if using_bundled:
        scope = bundled_catalog_scope()
        if scope:
            st.caption(f"تحلیل روی مجموعهٔ همراه بسته است: {scope['product_count']:,} محصول و {scope['review_rows']:,} نظر واقعی، نه کل فروشگاه.")
    dashboard, queries = st.tabs(
        ("Analytics & Dashboard", "پرسش تحلیلی"),
        key="analytics_tabs", on_change="rerun",
    )
    if dashboard.open:
        from analytics_dashboard import render_analytics_dashboard

        with dashboard:
            prediction(show_heading=False)
            st.divider()
            render_analytics_dashboard()
    if queries.open:
        from manager_dashboard import render_manager_questions

        with queries:
            render_manager_questions()
            if os.environ.get("QBC13_BACKEND_URL", "").strip():
                with st.expander("پرسش از سرویس تیم"):
                    analytics_query()


def analytics_query():
    with st.form("analytics"):
        query = st.text_area(
            "پرسش تحلیلی", max_chars=500,
            placeholder="مثلاً نرخ پیشنهاد خرید محصولات این دسته چقدر است؟",
        )
        metric = st.selectbox(
            "معیار اختیاری", options=[None, *METRICS],
            format_func=lambda key: METRICS.get(key, "بر اساس پرسش"),
        )
        with st.expander("محدودهٔ تحلیل", icon=":material/filter_alt:"):
            filter_values = filter_inputs("analytics")
        submitted = st.form_submit_button("اجرای تحلیل", type="primary", icon=":material/bar_chart:")
    if submitted:
        st.session_state.pop("analytics_result", None)
        try:
            payload = {"query": require_text(query, "پرسش تحلیلی خود را بنویسید.")}
            filters = build_filters(*filter_values)
            if metric:
                payload["metric"] = metric
            if filters:
                payload["filters"] = filters
            st.session_state["analytics_used_filters"] = bool(filters)
            run_request("analytics_result", "/api/analytics", payload)
        except ValueError as exc:
            st.error(str(exc))

    data = result("analytics_result")
    if data is None:
        return
    show_filters(data, requested=st.session_state.get("analytics_used_filters", False))
    scope = data.get("scope")
    if isinstance(scope, str) and scope:
        st.caption("دامنهٔ محاسبه")
        st.text(SCOPES.get(scope, scope))
    name = data.get("metric")
    value = data.get("value")
    has_metric = isinstance(name, str) and "value" in data and not isinstance(value, (dict, list))
    if has_metric:
        if name == "recommendation_rate":
            rate, numerator, denominator = st.columns(3)
            rate_label = percent(value)
            rate.metric(METRICS[name], rate_label, border=True)
            numerator.metric("تعداد پیشنهادها", display(data.get("recommended_count")), border=True)
            denominator.metric("تعداد نظرات با برچسب معتبر", display(data.get("valid_label_count")), border=True)
            if data.get("valid_label_count") == 0:
                st.info("در این محدوده نظر دارای برچسب معتبر برای محاسبهٔ نرخ وجود ندارد.")
            elif rate_label != MISSING:
                st.progress(float(value), text=f"سهم نظرهای پیشنهاددهنده · {rate_label}")
        else:
            st.metric(METRICS.get(name, name), display(value), border=True)
    rows = data.get("rows", data.get("items", data.get("value")))
    has_rows = isinstance(rows, list) and all(isinstance(row, dict) for row in rows)
    if has_rows:
        if rows:
            analytics_rows(rows)
        else:
            st.info("نتیجه‌ای برای این محدوده پیدا نشد.")
    explanation = data.get("explanation")
    if isinstance(explanation, str) and explanation.strip():
        st.subheader("توضیح نتیجه", anchor=False)
        st.text(explanation)
    elif not has_rows and not has_metric:
        st.error("پاسخ سرویس نتیجهٔ تحلیلی قابل نمایش ندارد.")


def prediction(show_heading=True):
    if show_heading:
        page_heading(PAGES[4])
    else:
        st.subheader(PAGES[4], anchor=False)
    with st.container(border=True, key="prediction_form_panel"):
        with st.form("prediction", border=False):
            title = st.text_input("عنوان نظر (اختیاری)", key="prediction_title")
            text = st.text_area(
                "متن نظر", max_chars=5000, height=145, key="prediction_text",
                placeholder="مثلاً بعد از یک هفته استفاده، کیفیتش خوب بود و از خریدم راضی‌ام.",
            )
            submitted = st.form_submit_button("پیش‌بینی", type="primary", icon=":material/rate_review:")
        example_buttons("prediction", "prediction_text", PREDICTION_EXAMPLES)
    if submitted:
        st.session_state.pop("prediction_result", None)
        if text.strip() or title.strip():
            payload = {"text": text.strip()}
            if title.strip():
                payload["title"] = title.strip()
            run_request("prediction_result", "/api/predict", payload)
        else:
            st.error("متن یا عنوان نظر را وارد کنید.")

    data = result("prediction_result")
    if data is not None:
        render_prediction_result(data)
    from prediction_report import render_prediction_report
    render_prediction_report()


def render_prediction_result(data):
    if data.get("valid") is False:
        reasons = {
            "empty_text": "متن قابل پردازش نیست؛ یک نظر واقعی و غیرخالی وارد کنید.",
            "symbols_or_emoji_only": "نظر فقط شامل نشانه یا ایموجی است؛ توضیح متنی اضافه کنید.",
            "digits_only": "نظر فقط شامل عدد است؛ توضیح متنی اضافه کنید.",
            "spam_repetition": "نظر شامل تکرار غیرعادی حروف یا واژه‌هاست؛ متن کامل‌تری وارد کنید.",
        }
        reason = data.get("reason")
        st.warning(reasons.get(reason if isinstance(reason, str) else "",
                               "مدل ورودی را معتبر تشخیص نداد. متن نظر را بررسی کنید."))
        st.caption("برای این ورودی هیچ کلاس پیش‌بینی نشده است.")
        return
    label = data.get("recommendation_status")
    if not isinstance(label, str) or label not in LABELS:
        st.error("سرویس برچسب معتبر پیش‌بینی را برنگرداند.")
        return
    color, icon = {
        "recommended": ("green", ":material/thumb_up:"),
        "not_recommended": ("red", ":material/thumb_down:"),
        "no_idea": ("gray", ":material/remove:"),
    }[label]
    with st.container(border=True):
        st.subheader("نتیجهٔ پیش‌بینی", anchor=False)
        st.badge(LABELS[label], color=color, icon=icon)
        st.code(label, language=None)
        confidence = percent(data.get("confidence"))
        if confidence != MISSING:
            st.metric("اطمینان برآوردشدهٔ مدل", f"{float(data['confidence']):.2%}")
        probabilities = data.get("probabilities")
        if isinstance(probabilities, dict):
            available = [(key, value) for key, value in LABELS.items()
                         if percent(probabilities.get(key)) != MISSING]
            if available:
                st.caption("احتمال کلاس‌ها در این پیش‌بینی")
                for key, name in available:
                    probability = float(probabilities[key])
                    st.progress(probability, text=f"{name} · {probability:.2%}")
        if confidence != MISSING or isinstance(probabilities, dict):
            st.caption("این احتمال‌ها برآورد خود مدل برای نظر شما هستند، نه دقت آن در مجموعهٔ آزمون.")
        if data.get("low_confidence") is True:
            st.warning("اطمینان مدل پایین است. نتیجه را قطعی در نظر نگیرید.")
        if data.get("model_version"):
            st.caption("نسخهٔ مدل")
            st.text(str(data["model_version"]))


def service_popover():
    from analytics_service import bundled_catalog_scope
    with st.popover("سرویس", help="وضعیت اتصال و بارگذاری دسته‌بندی‌ها", icon=":material/dns:"):
        scope = bundled_catalog_scope()
        if scope and not os.environ.get("QBC13_BACKEND_URL", "").strip():
            st.success("موتور محلی فروهر در همین برنامه استفاده می‌شود.")
            st.caption(f"{scope['product_count']:,} محصول · {scope['review_rows']:,} نظر · BM25 آماده")
        else:
            url = backend_url()
            if url is None:
                st.warning("آدرس QBC13_BACKEND_URL را بررسی کنید.")
            else:
                st.code(url, language=None)
        if st.button("بررسی اتصال", width="stretch", icon=":material/sync:"):
            run_request("health", "/health", method="GET")
        data = result("health", with_details=False)
        if data is not None:
            if data.get("status") == "ok":
                flag = st.session_state["health"]["mode"]
                st.success("سرویس نمونه در دسترس است." if flag == "mock" else "سرویس در دسترس است.")
            else:
                st.warning("سرویس وضعیت آماده را اعلام نکرد.")
        st.divider()
        if st.button("بارگذاری دسته‌بندی‌ها", width="stretch", icon=":material/category:"):
            run_request("categories", "/api/categories", method="GET")
        categories = result("categories", with_details=False)
        if categories is not None:
            count = len(loaded_categories())
            if count:
                st.success(f"{count} دسته برای فرم‌ها بارگذاری شد.")
            else:
                st.info("سرویس دسته‌ای برای انتخاب برنگرداند.")


def _load_local_settings():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


def main():
    from analytics_service import bundled_catalog_scope
    _load_local_settings()
    st.set_page_config(page_title="فروهر | دستیار خرید و تحلیل", layout="wide")
    st.html(CSS.replace("{font}", FONT_WOFF2))
    with st.container(key="app_bar", horizontal=True, horizontal_alignment="distribute",
                      vertical_alignment="center"):
        st.html(
            '<div class="brand" lang="fa" dir="rtl">'
            '<span class="brand-mark" aria-hidden="true">ف</span>'
            '<div><div class="brand-name">فروهر</div>'
            '<div class="brand-detail">دستیار خرید و تحلیل دیجی‌کالا</div></div></div>',
            width="content",
        )
        service_popover()
    scope = bundled_catalog_scope()
    if scope:
        st.caption(f"دادهٔ همراه نسخه: {scope['product_count']:,} محصول و {scope['review_rows']:,} نظر واقعی")
    if "main_tabs" not in st.session_state:
        switch_page(st.session_state.get("page", PAGES[0]))
    search_tab, review_tab, dashboard_tab, bonus_tab = st.tabs(MAIN_TABS, key="main_tabs", on_change="rerun")
    if search_tab.open:
        with search_tab:
            discovery_tab, comparison_tab = st.tabs(CATALOG_TABS, key="catalog_tabs", on_change="rerun")
            if discovery_tab.open:
                with discovery_tab:
                    discovery()
            if comparison_tab.open:
                with comparison_tab:
                    comparison()
    if review_tab.open:
        with review_tab:
            review_qa()
    if dashboard_tab.open:
        with dashboard_tab:
            analytics()
    if bonus_tab.open:
        with bonus_tab:
            from bonus.dashboard import render_bonus
            render_bonus(open_review)


# Embedded WebP and WOFF2 assets keep the app runnable as a single file.
# The studio photographs were created for Forouhar.
HERO_PHOTOGRAPHS = (
    (
        "هدفون و هندزفری",
        "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",
    ),
    (
        "کوله و لوازم سفر",
        "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"
        "tP7dpOmAAAA=",
    ),
    (
        "لوازم آشپزخانه",
        "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",
    ),
    (
        "کفش ورزشی",
        "UklGRjxTAABXRUJQVlA4IDBTAACwoQGdASpIAzACPpVKn0wlpC2wI5fJEgASiWVu3O26jfi54vNo38heh7ysqHKJzS+Yp2HM"
        "HfMb4bU5sCwXIFJ3/f9D/kXx4khZvXnmfH8R3x/Rr+rvYP803qX/v/oU/cv1wPTV/k9+N9E3zsfWe/wvSAf//29ulvjhsffP"
        "+wkvd/UeCP9u/kOdb/M74/oNqF4l/8/ucty/5voTe6Ogng9/SazLQA8o//m8m37X6jPSz9JoW/b99zCOAkfVHFREcCdkBdt2"
        "Mk5OyAnZAgjeDju6a2TaGgZGRKs33zv4FJCWxOjA7wGMunK5j6Ld3J4FRpFZ5AfNgCBMOBgzujbK83/4dq6gBJzRTnavI8DJ"
        "g/ZQKmjXNLZ9Nf6jqTFXptPHedcooxDqd3TY7nbQS2OJ8H9FZ7vq9m0U+42FwlujQvaYd7ECtXpwEmLkGIZiCjiNKriXryhx"
        "879f45ivhwR26Pflq+Vu3eiNpyPZvET4xDB1UF5zLuBmpd05ToKkhhp7/1SJL+syFvLU8rmawI82ybrzmty1KcEhdifXA2M6"
        "2gdNUug0NEQovrZN6LvzufusNI4kaLURwqvQjVqpwWvXglG91GgaxEjQy8N/D7PYFy9LMXTDG/e7f7Y+uMgr4SvW5FvupgUO"
        "/Qb/tCK1oElN8SAwk8Ezq2p1qvdnLJEr6ZzVVdZgPrXL+FiZGlShHbqhcmccVSRyuOVQocjV+9ie8oco0Xp4AaPfcKJV9r0b"
        "T4H1OhT4+UYAL7MjOmfVUYLKHsJ6mwxrjGu66bDm71DV1cDne9Ue/C6XLzayZ7cCcKxN8ddRJj2FGmo4J4fktt4CbAV+5i56"
        "NF+PvAFg48Qfsj6hxlaFKFit6vYAK+NApRoSoN0lZ9qOcW6Mmo0TXVzVXeQ4LRrrwVZppbC6zJ/VWYxeWc/Een+nkhl5sLLl"
        "7QYHyo6o3Ikp/KKN8vRa31opC5MNJwMc4pBMdyBYTuXa9POtg3KgeyE9V8hCKXib/a4rQuu5did7nrkgJU7fOHDr4TYrHZXd"
        "bbbBCbzAU/6JJpn3oZbYD0AHTLk/iruISZ6Xgj87Ba+l9TuQ4Wv9lh/WMN+XwW5q9x7KkszMiy3vAfJDt9+IiB7JdwthKYPK"
        "91ehOdQsV9h/K4XqvBOk16NRolqDndWyOdLzkQ8rIIoO0jxgFyOQYU2twGKzsNK73QGMAFiABynfrQQZ6IT3ABxkja4M+TCD"
        "LyeW8tteWrTvv4UnFH7/Z57oIMDyyseyqDH9sMiqNBYBebpqYAEM5UT9yX0dlfoOj0UQTF2Hm4edLmsoHwJWLUXxnHLGTHD2"
        "3GLEDh2dbW/bqb161jHPHnAp3WLhSbcWFRGz+DNT6RpE5EbH88aKPdpR4qU5izR9IoT3+o0tsypiIFe1fqTatHF2PoGuU3t2"
        "xlqZ5UchzR3po5I13JtEUPSk7XAEayXNhlrahHaR8P4l4eW4Ajf/5tAvFBfJrynJ8dHGlbxa9IGzorb/hcxHTulKIUmat8mP"
        "2Mf98ZRTr805ddKJ05TDrAlns9YgREGSG5WkoeowjwYMXAFDvsQmBqH8BC5If6OwTeORhtxVZGeAOuwKXoVeUFo1RplHXwBz"
        "7zf6p8QO83JmRBPQL/yvDlhk30QujhAxUbor06HVDPSjTolEOBeuOIh8cTwK2VmOEL1CK+Nz2NxFQPI5naOSn6V3adK/a+au"
        "6bmEKktF/bGFg7JfagoNMxoMpKbww4fThSpl+0qD4Pc4wM3aANXmTeoUzLgYW5TOJ6iCzcdmmW/v3l0lnx12lNNWGqf+8ebI"
        "m4dDrlRktdLFIeNeCUeYr1B3V6uEQty4bkkqNHuJjV7miDS438b+WbCzm0UqHwyQHnqPC72SQRijVVT1XnBD5S4v2MKmb/tk"
        "O2p9lFqIdRZ5QzHrtpcDGxUOe0P5b54bY5PdWzA5Q4tSqMk1Hhe6V3YT4oOrTFutK1g/nL0d+3LBshoJbNgZfJ5uTTnjW6wn"
        "o4xC0eg6+lQBVbsh9MfWI2i1VzPJLtSBWNh5txbbMxwZW+Dyr4NvGsyLNbSC2Ji9p2s/LUgE8vrM3R6eJ6j5tc7olxmHbCXe"
        "ug4oQnNm40s//+8B/itAksn2aP7vR46rp6uV+QCIBL7zwenhAYilexILL9+r9nIGb1kzZQ1dV9oqdAcE/HbiZ/489+4Cs5ON"
        "DqNPbsMiAQjsCGkShmGkHedz5qDYmH3n4X2A0+f/ovOctYyRBrftCof+A3yRWUcZQgU4pHLnjYsIVIwPctSr904ZjIXGMkNi"
        "tm0paJ6Ulv9AWipb3O9IDKy6K8Sy2ikNhH/l9nT+wYrGykGWWDaW5f9Ib01hKe/ntvpv/QP3VKFkt/rnTSHAe37ChibgiFdD"
        "ju6eozzRPw69lKrkioOW9d/4BJBiMNoG4+gcLzaXtcv7ObbY2vmMuZKUbfpQamHP8Lk62wTN1kyfQJwLJjzadV2L0KSn4Y03"
        "FHQIkDpVkPUVoJpcBhQUfVcn80jFihGusC/474drqObAm7XTYKlVPv2Xn7O3XQPVj5F8z6BZrd77SNvJ7xOHx3k5HpKBsPgJ"
        "hwOBuljnRsXH3ZJ9S5DxgY+HZ3cGqbg0FDP2a2xLaUGYoH0urV5+6xhavdwlNI1JldgQ0jxYOCfXzn/Z8LQ4vlLg/UrPJIRJ"
        "ojMxZuoxNGBck+f9PrtZ2H/wuvyB1O39450PmfdKhhtY0fqsQN5OUsZDgLkbUyQklLmL+4+tyP5JZOqn5Q6pClZ6siNdePjS"
        "NSWdAk39Q5gGAvcd9dLXCVTz6lm02GdQQoIiFgRUq18svULRPeh5+8wKbPBYyHtqvs+7iuwNCFHgVrET61NR8Jp6hFgxgLt1"
        "3PAeRPuLUXcXE7ojVF+QpwRNmcGi0uC3ydr/7sEKkwbIZcWK2KJVnM+YG/tx9cLaXMRFqZkWPZpMKRh3MuXa9GRwhzujTRqv"
        "Ju+I51h2A5qgU9TWavX/JYVEzV4UUvpIwwhkYSTGKEUF3/02nxSlCeE/KRbola90F34ViZsIdCU1mMaMit/92NqKsoW58aPR"
        "Yr/q291H5v+zDu0/QyVkCT/y/4Na2DfxgWyp+VPWe597papQnUxh2YQ3cJiQpBJ3ffDtF/w8n0971riQen5/J4Q3twtUYjxC"
        "hZvHFnj/vBPg72vY+8+3RWIXXw1IngjHzQEO8KfFx2lp/3mTB/1V3wfAjKzcZ9Qd8Lpbo3VG/E4XMNPeDOC+G+oyhotrURHR"
        "l+82uQ/2oHlldKAPwBv8XvG7Vqnj/J7tFVMu4PDHFzCxA4KOCWScfdEiSmU/FtI+nhG4i7gRjq8/V7b+Bf+1F0wk6wNbyhpx"
        "rE4wh9UUFyYN1D+LbOAXGNe4D4/jf6Hiqt+SKp3Z/PbQUib9PrWf/Z/VgdLrHBgyd8lIObXFVpxLhxlEV9Kty7/Llt9jYKA2"
        "tWoMKmrMDCLd0J5utnKlxrvsMobKfgyTaSUS67SP4Ut6vHn3CQacx+MfDy5tKvq3C4CeRy1jzILOITpiwlfWqq4blxCa1J+U"
        "U/cghi4cnPVPpFAgejf2Ie9uTEIQGT4UIsQJ5GMDDAtpZl0bkZ5Ypyc66Ex9xQMfCIiujeTOzozG9xS3f8xq46nB4Cb+NB9S"
        "4kXI69YGYwiuNEh2h66DfWY9aamUkTebinLmnYioiccj+Rcou6zBRUtHoe7N987zGW3cJ6RyT5myLpIcXvAYDV4fS14M1FWB"
        "AzWal41yop9wXquFnbPAXxKJ1uEVARnbS9mRHiD8G08nV0OPIGNzq9Zn5AaGrnUIqT6O34lYRq1XjSmTK3zXOFr6PkoYdHoD"
        "F2k7jtDQj23lUvv8/RuoJZ4YJ7twitA8n2+Ym+AszyRIS87EcoDKuUB81wmO4XwkrhNGPBmWsK7H19i5YUUoUPg6a6REnzZe"
        "HLYgo0SKj/OCewZk8kocaQ+/64v6XyUjXgY3/ImxijBtbk2/+YnASexevWWhoc6RYcGTRjxOdKg/PjKDGnvaSfIo5Ag6MpnH"
        "Ub2cmkjQnhAsJIq3fNYJ/6s4kBbUkCy7LG2NhoG0GvUi+p3Xk2NNX/XpqeXX4gzylBn3eCuEoWI2lFQFiJYSxqRivnq2Wy2b"
        "yK5JomLdi9+E41Ou7h60nq3gJFGbOxsStipZl7FXZtyvuKyWXLSSuu5GCfFc51fWN3k3lKl/FcXZenfxTqHQUGOWcZT6s9Ub"
        "M3BDYc1IjQqEHnjIAPEk/1RPiLYIXmZRlxeYuKsp8pU00+F61rmo2LvcrSbL6td4giIKlafg4AysgIO+ngfYAQ4amEgB4gPG"
        "Neg81x3HU9ZKVF8S3p0e7IDmky9xTnSwlmi711ncPHTM+Vxm+4TuQMgYwalxffo6Z5vz8ieWnZ6x+8OEwFTagzDRCfTW+PWm"
        "mHqSrNkv/IDrxp0350PqpNv4J/9p6+PJcMPUN2Nmatn2HZCfnQ973CEE1eR/NuN944yg8/l0rnPEQAD+vdbfotTldL//0HH/"
        "6Dj/9Bx9ceNrlqJTQIcWXGxsGexFMtvl+WLCxAlqij5tFCai028umedF47BWZoqzmYY9bMQr643fJg9ZyEzzp1FlJEwjyeS6"
        "3BCLgfQcYC1EXPJ6qmkGiSEvcxpyXjbPjOIpAjNTal9wca//+oKV4AW8lxUKmO99M57sdVoI/cycELvOUP0DCjZUd1rBfb9k"
        "HNmJVYzMg1pPgGk3wR6LLeGKTvRpMc4upvmv2PyacX79Yz1h3MQ11HOqb8cd6jQUAlBl/W8ggLKQZUcynqUpkGG6tSq3vnUA"
        "YVEG9P8hT+K4mHr8WS+EnjfAtUYKGleOS3EToNvlPzGDvroTRUa3FTYABDpi8G+kRk7nRrxFuFKwYdRq3RdIkhMuLz15gAwi"
        "Z0SkgrnKDoocftN1oZMy3DgXBQWvY6BumZDLpiAHc02m5PjCew3F8NeebadOrCRLo3QUUOmwHVvYcxbgs5+SebeM64CtJ1J1"
        "7u3jdUR8YGfQCgiMEdvIRZhtMrbeC9SIEMCbKaLR5Ch2VD7RXJxggmrDhtuU3vusrYNreGNF92uFxJ6vktXibzEsvcdOzYoa"
        "1o+r0jpC+eWY2znEXFqvNavnUcNkjtTNKezD29XBy4Q7BlWNkSqpMi8wMXQQp2YBvweCwxuNTF/cHoCLqvz0RJJVxGgBi5FO"
        "czATHX91yTzQ5LjB0mTHb+WMA3bifIwu5i7i+5bBTkOBiw9JgQus2E8YmEZRx8pBN/wxrgWSPOX1naA/iFiwYYqtRaXy71zX"
        "JVTFe4j7bHRCo80iPUaKIS3l0xEJk/sLhX5pnNPERv386HtehxB3Djw0YdulyDIrALbLCgiupuFASH8MYo53f6+tgSZTVoDj"
        "9h8nfhWC8nvo4UAlvhtYSHFLvxJdxkAGUCYCKAYjDU7skwAFhXCmkwTHqJBXGUzaZzHbBH2V+wg9R1MSdcVqniPy8JETRDEh"
        "yF5L1bpH5ggMy19IsvW6DMelZyQBHp5woMVp/6RN1h9+kpjj0+m6sPBCoBEz0y22Uon9LFjENalJg9g896SoD0NMo8hxmlpd"
        "o68Qg11vQVp51RC9toSHMXKNat4k6mMvpmfHssF/49oxN9jG6tSdnuz0CIr7JZuTmh/m5RyqGfJ7ptHmvwtNqoLq7ukx5MFO"
        "5ZvX04cETJlnxvQ62Ycv2nSu0XXNX6pff2QjWSn8LxLNXdxES599hCpmQcvLMfX1joYzAgfjky2HXFWNW9Eyw3k3kOAlgrWd"
        "apqdWSwOCqW30fMqI9++PC+4LwiYs3MSMCH5RkyBvnKsaPhy1X8JSDRyXJFCohqtBApjPgs66paFJkRLZGG8AiZd5mTCrqx8"
        "0jRej4FsJtyITwOZx6BXzgJVQ7gWko4pR8p260Qt0EicEQ2D8e/SdvJAK7DGu4vsrv0Z6PLGO7WVhPPqa+KGQikde+TkdmNF"
        "7J2vvNkL6N5D7R5BWtFqUhr8x4xBWQ7PwgWvb7R+Vt+sPgYc5LNay6T63sT1LyvquHzHdpoWa+6s8ToPQ3wB4gb8wlenVQDy"
        "q8nKgSof9Dl+8kFp3lV6VJkqo4mVZF5mW0GOQgLD0So4KOPv0XYSKYU8JMaFwU7qSTsAEuJEgiCUA7AvDpvNsuoGuHDcq14+"
        "dfCt0MpR4gFyPs5aoZpwfu8u75IzMI66JaRKAX0Ip+7VPga2w1liUvZKHrXV1AxRVGwpaLlNnQz6SpN7rrjAYt33UUsA897d"
        "T7Bs6TrP3IvtkIrSfMU+Av60GymDa75OZGj2Y27BL+HKOyD//erdwQXhc1b3aWEfWpROwpM1hWmewwC7TFYIHv9ysWtl38G5"
        "cDhxx61s/2ezqRRTVuHxQ15bSExmUuwsNMqbBMYJ/LkAMfqauWrnAe42/RIpLSui3dRuEkCQBrVNDIflsNCmla5IL2ss7vgU"
        "9REPbtV/QDolRBqtcj7yp0yWUazsKebB1QDs6hwUBTVyMRI6ianTG+Rj6sSEXbmYXfAlxi9MQfdgjKBKNcciCO6Kjv1aBh+n"
        "tx8/hxVphX1q+kOy+weXWYL2I+EMaqWv+6l4CWVBDrNQR+kEhlcOh9ZNM1I1070FJQUkucrYi6B6sVl0pAjzGbwEevuYKvp+"
        "dkRAs2o+3l1gYF2UYyeE20RG+r+B5/ua6ugMaDOy6vviZS3S4LW1uHfQALOAjOKN5Zx7/nCN5aWDTX3kfX+Ke0W6l8YEaAVD"
        "G2cNVDDC74WrBIqDNtooRhWYQYsvxPrzo/VlH2Q1nLbVDJPH/MOt+HPqwvXum/Mm0KXf2ivD1RemR4LU5wnSiEwo0l9qMgKn"
        "EFU2eXFwKWTKkUucq5qnnOmWEFBszHOmET0B9jh/OojyIUuZN2/7xm46aCFfZGs2jiFS6GaThrxjp8VoZigO2qVvwH0KWUYj"
        "nGYDaiDIe/dv3I9PgB1Y0H/L44VvFkOkqxxKKpo+IXC+9P6S+g36GHTqetzittaGMXDugaVoGIVJ+R6DDpu4cngSSic3nc1O"
        "OwxI5dNFK4L5uvrbWOudgXLnGRHVw9lY/oXLcIhAAJrt0RKbvL8ThTe9MrhqNGzxlzeJE9nJcDIjz5QfribZVF03T1myYhsF"
        "Rwf48mgPRcEg/cdY2WvHQcRcT+WBIHi4hEIX8qDLMcnW/GkI1Zzg9yHNExuZT+q0DW0LlQGrm25lBEQcpVNVb5Yx8HmlRvaf"
        "sO8Y9L25lY7q716rhMaql74e5IqDk3TCKPVmR/ARs2/kYIcXnMvTCW+OE2W54SiOwivjdC5GbK4/gu7Bj1kFO7mv+IUsdzNS"
        "Ehi0/WwbUqQqvBp9DNqZBScI23mkGYkc/xTKSUIJscIXb30PY/ihnSQj6wGxoFCn6R4ld25H0XZdHaTuynmB68GkjqAl0ykW"
        "Er7z5lZYToGWQ3x9U0YxaUWdTJW+YpU6ZSHUrGS4qTpG4/iTZdHi6e3ELdhShT9hYKtvJlvV5zoSHlmGXQ9yJtVDOAL3xH/b"
        "hxG+RjqP1vlFNM4e5qopj4IAFPHzmg7nhrhx4waVOxHRkBPNjzVm3qHzoaCTrI+hMy0qkFAV9v69DiJbFXCRZ/X2hGzBn8QG"
        "zCYG6TQ5kV416+IIYT1h2UC7k/U9A0fWyHeT7oLpcFgEWAIgmP2OPdgupTf8uqM6/HIRFAoto/7MUq/Ylraa6dBxWZlUKWIL"
        "fqiF0HcLmSme9ZecYMeV6Rd4N+LB+hb7d/hAAgeWqW4eKGgjX9DkwP+HGIBqCDnwQBh+77WRDaDsGvlVo+w20xNXjFeHDxxc"
        "GaZtCoUaq8gN4iHKgQIf5JEnWoBmc59XPH74Z/Yu7Yb7wdz6wFH6VaBirkZTAV+a6VZ5VOt5035qN05FXtbiDqIoMldIvyNG"
        "1ZQrUFl8x0KSpHK7jkppDX99xkhzQ3D4tOD4aGGsePf8hXbpq6PPEZOCFDWWpdG0a/BOV/J3m33lk78fcpbKgnRQ8JJ1NYPM"
        "abA+4c2e/2HMkcXtx6CUydt5nvAKskhEFckedMzGzhrZ/UP3U4fEOToFm0w87gAfscrW2KH0xNQmEQBW62Bu0EdcCul0ZOkg"
        "fdbJKiHB5Y759CxI4aWYpKA5zkyIKqGXCDaR5oywjVCHaMhkwWCgt9NdfcWmyiCG7azyi/Lrb6j/FG5HrFHunBijzOoYJteA"
        "O5CcFJaP8oODnOqasfed/fxtqo0llz+bIuGmLtjfotuNe7rCgFxWAw5mc5niAxuOFKa7fQ/eJNRjlWY0aK9fgsds/AoP6MzY"
        "07ik1MGbSqCXjYJtS9gseMvxZnjyHxvWm1PNcRWnaHJoxW9RexNFDzh2BkGz07ogWg9/MgJ0VXc5wQDPeqbniFQIybREtbfQ"
        "+sMc6EhKObKOIt59yHuPXIc6L7/rudO+W6djIBnz48LTq5JcqAsLe4oji0Mcl4ZnB25cqwMUEVtri2cVbgSJfGSJsR1P8CRy"
        "7YlPnJcpkULuPLyoO+JaVmsmy85hfX5boMDdteClve8hrt8sZbhs5ZgXg1QdHiR4g3SzPYfRwQLpDmeEh1BR6DYzcYEkcQTu"
        "2ST7RW9J0Mwj0suKZ8UQECICCgSbS094JK9polcTQx49CW90Rp18Es7DfYufm7vFJpK6My6KtVW/buyjpKyatjYA2FXIgi9v"
        "Aoau6s6AJ1/3o+bA48bcbx2zDXyBXqPVcbaTxk/aZQdDVSgQv3rXweNIF9XyFxMuQVw5AnlD8J5VO6f90orTz3pjvieJnmeP"
        "5vlaa2fbtvbTc6G0y3utsWMi7FOMMX/yYjyYMfjD0zWL6js7CWDHp1WfijbcAzO6Vl8AP9By+xXlvhirGlUZGB/tJdh4QvnJ"
        "LRHBwHb0ehUvCR3HqWj9Qu1qQtd2nlOhsvEyX+5s6NNsGMS9qr0PYKksNhURiA62d3f8Gf1moy6tIYdUJ5HB8on9N8QF+RmN"
        "dJtxcceQ3lpVlNnI9ZRlsj2fabe8EyiafVBvxvGqv1W5pJ7ux8IqXXfXWNfUEt4f9TJFPy7GcbZkfPszzPxBrucm2tRtKKaV"
        "sqnYpP48YjucNzEGsSTygF4tXmcWu5hROmBs/aSLKzr6Y85hJGLWUp0m43MRBL9X9DBrViQD0lb6YQ6oXUkv09p7bC0FltMV"
        "CJk78gM8kgJ5vZo1NCag82eBum+eZgp/5d9d4Wkqh12StRJydFi4CRLlukqLZTXe+SQ2XZqbwt8lk3alk58s1Os7Cmms6/06"
        "k1ARREEpSm5RNiUTqL1GBmhhEBMD8PFBufgDNGk0eaBngwXvffIHdUbRN7o/j+tmMMfEztQ4RSNM+gPCeY+kIKvP5M9GH6Hv"
        "f/V7GokLVZbbXNrmkPJaPAYCZAyIse3RpT9pt+HYp0mSTxukych2bk7xe1aLCStcjJtVkWyaQ6vbSww2iprUlg6GqssohGiP"
        "Rsu4e/zsnoS0NYZ5M1p5nR1R6jZSPB8ZFucMqZdcL+D6G+c1Qy87wgWR2UlE+1x1n+Hahl47ZEBuyqKlJyvLGVjNp3qqkpgS"
        "fVNb1DY5MxBB6NWIjLEjF+2oYMA4LS4Pj3j9JyscgiRp/7zbPPjPiD1b3Dv2SHkFelbMFdeuc+bu9tmiLcnzI9uUOWhBzQAD"
        "aRc/0zMKGza0+KMK7PTtiLuwLpPhtB1cR5pQWKjKtTkaOjpT7AKF8Y9uooAjnOqEDcMzMnX6lD0fz+7e1bSxNWn5MxOdYOSb"
        "TeOpmavROyywXtpq0iOfh6H8WfaC9YcdzHKSGbB5NkNmIH1Zpr4cfwthqyiEEnA3NwqkhQAGFABcyOTC15NTvy7pWWBOIAFS"
        "GW+bMvcraeKs+I+9XHcCeM94kEz2iiwuSpZdJzBHd3RVvhJkZOvMqAEf7uz+T00GRtEquQECJTUwB4f4prtpRX+ffhb5DKV5"
        "Q5NTbVuU7XOSlVAuy/y068GxmfP1NHvAe/mCLVtn9Wu6dBMZdyNxvF85SfApMCZS4vhf7y966z1WjCYWJk/xfxt8NaeKJ7W3"
        "YAGZUjXeEGpkSi3PsMEC7pQ1wiM6BQxe3A3nKXm41P0r59potNjj3UDWnL0ilF4k0yjCLiJCRYlxYyoJk4AzYXP5vh5WXvAA"
        "+JNh8Ab78CLSeUL8mLdsmhY188XgN4KdlR0Ii1uFpKdtwRxHSpcKalG3kU3JvtN8ujFuqIiAfX75oWYFLGlt42Ip6Qm2lgtL"
        "jjk+UyTuhQHCBUb7/7J+2AgxG+KMEAxpZ0vVz0HHvOwHgUL2mFCWt+c3GSp2ewhbrwWAMnpU60filOvCvloaq1h3FnVml/Iq"
        "TBW53e/HmAGfJdIthrOnI1CayWYZBZBC1w/7qXqLnDsC3nUEPxFSgAsPe39liZMaL3FzfMiMAS0fvl7WzTLcSIEpsWEVcDZH"
        "HPrCiDD/FPHfMcWNIVrhW8Ar8/jPT67iX7rheQln0N1oUEOfoHljMLaTaiMwnW8isM6LiPkrnmjvmCkU+tvyKZQC1KjjyB2X"
        "IwkHcnEZJUYYWUG9AIChKB0M/9k6/XL6/QUzs4Dik2cKekylR4EyJhOyVc5o0JqBUK1/UxtAJEQPL5R19XinhuzhzRb0pXjJ"
        "Ky40fK7INX8hzWGY+rqwLy6sQVYVK3LnTRne7I83Sm2CHjegv4uw3bW1Czi7WfjoMftXs1vKbiL9El3MAYMSvzKoLrhY9akJ"
        "nBf0BkrpCb+lsFT2Wd94XXDrJpd/HvxgxYOMHveDhpGdmj0+y9r4ZecbQNZ1f1FV9f0D9vUBqakJEOygj0jObAZKvdDIRfXV"
        "b2fB6+ZMmiGmCNz5NGVMh+mQc/Yd5iHfRA/frTKjYE3cbZ7FJG1E46Q3pluzNtyIOMPtVPpMNkdRmNdXZiyhNqtYXzfNp9qx"
        "5nxP3Nacz13aO2n+NiDIhjUQv2zjPN/ZNP9pTHoopFcRTXYh4/H/NDOgUikSuO9DMYaDY+S3BjYBpNgqGHF1rtIppt0KNnh2"
        "Cg/ikcWInmmt5rEtg6qO77rdCQ7kwQnO43J5B3Wc3CsinIFYItmqzfK0wIcMrcJgD03K2gjyn6eBJiOlK5ORsBVWu9Cjo87h"
        "9z5TBYCP/y1agPOJJlu8CCdXDccPBI5W7FdJesEPxRy46PDizbqNRBAQsWHymu6zemS77msTMYmwPqmQ1h8ALM9DnH96d+l4"
        "sTJ3XWvXymTBdMbYP6D7f0gGy7T1tfXoCCcR0xfVmL+h5JYAs8OMOybFcMfhgFe9u7RN6STJc2bJneeEZEbPbjatCTxdEtTc"
        "6l0r/a8PNW05x0xhza/zZpaN2DnF6boRqFYewM20vDPYfbIAzSvNfZaMqYJVc5RIYuqH9JZKLvbGwtaw9OS/mqL72s3BFv7q"
        "RAH0sjzdYKQFIAhFIvqYwgpSjc6vPnHPCdlEXEserxLLBWBpNBdRVl7tfNtNx1C86vzyHidFONvdVQUSBS7tCRq+zkQUPEXs"
        "xBnxrFiCQ86iGEPGgsWtfRjLTUDFWqddOlhr8qWAuXPKLwjRMwAgDjQPyWb5V1Cz2zeZIBne55RhsPZNQcTDWlMC1MzmfLcx"
        "JJyjCJa7BUqWppKHSDL4zNVvt7Kej0xVqzuUDUxK/zmm/gei1+/y39n0UbfcVqoVwjZ7Yf3C1enS2DWA0sDaS2xfW7PsbmNv"
        "ai5dhxFkhTP0uv521pn70HltZbF3ml6r6V8QmCNoR+nPj2wDddFIubvMcqcV/xKUxNNdfRbZMzrN3hrEZGWdT/gT3IC9o5Es"
        "yvCyVnlJh03+a5YzvMowO/DmU5e80SFvTyfUTEpqGCnhG09KZBasI53u0GlLEoLMoP1QOIh25ub8fc34R3JxQryFAXtjBjOZ"
        "SB+NEhLIDW5qmkB+GFPB/9BO5BD9WsmwsthDpJw+mDXIEmz1ePHFJyYBRYcnmGZx/2w1sfuEc5n5mnCvueHkML+F3jPkWy9i"
        "eYPjSpdPgiBDtJUPFI+dMHoL4l6OqXJcP+3IVEpl9o6ZVhw2AKqvTa+YW/rdVLY6wKzlYYb+a5lY8uOjULia8bXYZMAWg5gH"
        "095qkRIVgBA6ghN2eY+GHChDU2mzLtlvM+JdskdwCO9VQHN+8DcXtByhBQd1ZOYHErOKhnaW25d8GBj0SUZvjDrJGrr4MYpu"
        "PJ24+t+qFd4KTfAV0a+EafIEkFr2CHkUDDzVvlmitLTF2BwcPugWqUMch/3IvbC//tvyolHL7HhBVy9HFXNG1ew0ux385tcR"
        "9pOhvbNrDNvrXbsQedMKHrvPSNdqPsI+BBk3RbfMlLZ580GRsixMcRdBKBsCCLqisy0fzvho8Rt1efvhtCizyOn2LNwmz93t"
        "3HsL5OwznBsQYbVqeYthsC9BKR6tq5AmmuU5A9cRBz5vqUQxEupzYGRVWt/Z7RcsWYMwOyD04mlK4/7y5YaPk6MFKd8JxzVj"
        "i0dNLBGwsrE7cTtB4ZaMcLFHas1RqzYrJ8WBrKUeOXb8S+Hz89+TbHXYc8w5WQcN9vGZO4o9UetlfQ/2G1DCMc1rjfBicSdu"
        "oFKS5ZFspKnKWgVfJYfJkftV8L/zaseTSYNgKWjntnZP6WTEHqSrChHT4LFl3HlaKTZyIBRFteErM5fYDFKy55t2mmp6Etqj"
        "TZWAICS4AetWRl8YnBgd0soXMiLYDMfTLWQQHl8bL1M5kScvQflOSeXSh06NQb1XI5L+PG9dgfVbwV/JgUDm48uke2WFhYpK"
        "QqouVLImQmkfl0LvItY8ykezSRXVCWtJhG8RYLli9sizQsKOfZQec4ae+XFYOyw79jGz+YAeL83AKsGeIfx4xi9NSOODDEV7"
        "C39HHTam5cZUvc1kpYmOf7GWO/YDFEGjIcUEjtY+qr6ASk2/4xQmp5bLVHahfsqt3vTHoxRiogWfzeCjaN7gXNrD8OZ9wpvW"
        "3MTpQJibnkwAaMxYDDBCebgiqSFB6etb+poJNXQVdST0UCEMLDnpSp5Dr1ZDSng2SjCnMWD09AnyH+/NdrOvVoj0IYzIdoV1"
        "pR9OPk7yzmZOOmQU2wH7icwDWy9+aos1EtTbnQEpMnXttPW5iK//EdfWTUZEnp165JYMce+AMFxElVdpu8GNVbVrdtTYTUvo"
        "L1NWtqptgB0yetkclbU0ktRjVcpz/J8/Q58Z8PyExu0UogpcFWY+Kdr4UD9+Y/M9dwi5P6Rj/Idjj1z+S59nkQphksbBrNe6"
        "oy6XjgjtDAdICLtHlcGhj+6q7MYzbOpATitjfvqwYNqW9Vsv9ckucv1+rclar92YNzYsinw8Eblmc7SfrPraMQg5MRSQU3B0"
        "F2KRjTLY/BZreCcJFP2T0anAYWAowE18usNYrBcrXXuNb0lwpfx0o0CMT38hrr1Nae/F2vR5br+2v3HIpmTBhI6BH32nGtHQ"
        "tO9Pax2MtxL1835sngu78Bl8aLJTVLgRnuBaALNxnHjAI4z/rfSyQOy+ne7uIYeU56JJ1wCUnHcxwN0KGZQ5XIiQ1z+XhOqj"
        "ArEjQKn4IYb7ULS+GToP4Ix28RcPT89lQlc0R7n1p80Lr/hPbCAOwja3agycZrpBJhEdk9KT/mXnpggzSsBZ0b0qvWDGZMtT"
        "KIyDzdQ6b+udVJ6aAkarwbL/4gtqZB6DYRafaOn5JTk/I3Uirak6ei4r/rZ4VA9ouDdJuvaBv9YEqERy1augR4+dEviT7/03"
        "Z9b+/Xpa6YWOdmoajxfJrYoALroppRubvlZ02X2cfpK/UEHXMIaFim8m4wB91qjoa/7r4FICUBgCSLWMegd8X0WhmTGIQXLI"
        "WeO5RrEc+/icuoaQ9BJ0r7zQw3HIY61nzQ2RfDgvLpMHp6FkWh+9/jaOdNi2m4bV0FP+ccozFk0cyJKlfP+hQZYq9sQLAGL3"
        "sFrk8/bRCwiiFspVKcn9X75ivdwCcZpcbNmD4FzIwo1QGoTGooXcr+Nl/K3pLL0BEOUaV1wIRxRqeDklWD9AE8ALmdnK1hg+"
        "pNzDbVzGJBkOQdoS9zsCj0es8hDMOBFXKifV820ukD3t4iK/ZyYriwcTGdbGeDcEPaVGDRh0gJweXrX3tfB80lr3NITgnSlT"
        "yyAETYPBfk+EQupPCCX+ifHqRa9zob0eHZoOVT1mB7ipGIJKiSicdDw6rn4t0FcVwlprzxREkrHqQhTXlvWsRSuadiyAOsbT"
        "hqr2Yat7dIezBe4iZ2OUI+1QfbteUU6bwtmaoI4duKss2iUKcY7eHUojFPrNj7zdJlfMofGMjlog5D8gQXOgBAYkxawJuBi/"
        "AyJM7DTlzsr7jlSKIjt2XSgszlOcRSPmJFZ6fVSvTBrdn/UZrCLMZRXCYdsGHhlRc1eE7x1AGFUlzRxhyfdEzqf8QTnXGLsy"
        "XWDkzNTI4wM1FDSMNIKY4slj+z9LOc6gMfSPV+69OQgmERYPGi8/4s1ElyXLfuXCemELVjBMHHWnhL9KDpYZcPkWwMaz8zp3"
        "myAVUuGu2n48R3LPOvfXq3fOQ7C/iASbBqtHJpLYrqDtucZbSfmINiC4aPO85Ra010ouTzIAO5fD4xEbxNW26RIxHIG99KDl"
        "zyedY5nuj6X/HZMr5gfcXfkvIrIaYuAoYtgmJSUPIpjtH9RnNoZl4AgmTD2uepcL1umrwC/Cr7puVnQf4qkxbMDbwjcJb0Xe"
        "zlovhonGyJ0VPQslcx+kewj7vjgAJQ0yAQHtQdTBWOV+QsSMSL41M3sYEcvALEUwQO7eAf93XVBvXyV1P/JVmuABVIuVMXxB"
        "5ZdHPhELc7ku5Z0LZuT4oGc+sVaBUcQzYkHR5IZEzWIQmBAp+MtJawc4N8OcjklM9tqN5g1g7kYM1juRUNgrzFGv06cNr42t"
        "/+WA6I6IcabDeF2TSZFImrM3DuRaZXikJGunsVAuSoEnrRlZ70FGHoe3IxjhYI8+pI2mHunCs2QbgSjNdZa023aWw00e+xaN"
        "J7zMT+nnWqEpR2dDew80tzk77O1wl4bVAF+bTfI1GS4EUMWQph34Fj9sT88w4iAHAjFT20Id7v8dq4uPQTM/AqCCS9Z5ZGaS"
        "tDW1Hxe+CNO8SZas2bY9yn91BbwYpZfIi+hnlx+Ta0/gUDJlBABWeVHJ8mJmZG/NMoyDbqoDlmv1k/K/JeCTKmOGGLPEuYAB"
        "K5IK84VUCYe/B+hs0RPf3ECMsK/PkjV6/ww8X5+uMZpNF/IFiNhrdWtBOcZk/wh/PGc0D87zF58YfktHRmT9lETTx6uSZYOL"
        "3xZyK3rvTVtESatE/VTXPmETQy2hC5kRGa7XAPIKgFVfmWReHghUy6xLzee2ADefUmOrxMGiyDwFQVQP/uLhmoOLw9fXvHKs"
        "TRjJDzreNoTOEGQ5Ujz1O2FlUqZ76hfo+o4/QTTCTH+Y5TBdIHcgNLn4W5B+VzwxzrAZTmdvVnDWKKldiIx0XzLjRAhDGK3S"
        "QDEFBUHdJg1kH8nKaCj1MjEMuX6FSyzoyPxmZVV9Oi6Vt7xmdjPmHzFUcgTr6Ubw08tTFIibTzykGh3eOQhX0xXYtlLjG+rB"
        "Qabdl2BGfKjgJ8r+HAPfMkgX/u3T/ZEtnucVn2FHhORucJmRe7xuwjsnXSC0On38htPwaXSGYoDhlOD3O1P3xKw8fzqoIG+t"
        "wxPTv/xs/IxnR20LvBVqj3+ixDedVJkKJA00B7JQHITZwDDYwgBty85HueRXngj6DH43voJa9DlLw/0qEtkYPlrwNgM/YO+F"
        "UJakkJB0OI/iPhI2gjp3z3jlY3iS+VCC9gw6JqceP2w7gWG8fjGjoJa3nxqkvj/krJdDcEFovppJv9rXeciVIsqNZW/X/BbG"
        "vkUpGVj7qmYuh/lUUf4Q8KMKsQeH9EL8ezlSQ8ONU28aX9XBWc6FaCSM/yH8VOenLpSG86GYerkO9JiD6J9f1fTS+2dUJBmK"
        "D3NW0gin7leQWNUgqLoUDWZWceNq8MVcAI8ROyZ7l7BNI4PVvFGAV6baed9iZAAcNegES2OXq4aczDd4zxMu/IknhyMzya3r"
        "dal7VA3v6XEtDRjuLM6e41MMdeGz/ZMpfmx56Iutcrnj8RwQbjgAGPgiruhbjeqTzWxde60sHhIoUxh8dT66UBRE3/5qCLfw"
        "hVIKaNchRi51freKdNcr9TVhR/654NJLTZHf3awNUhTAlQrh8H+JvXprYQCySxXIVuKh0vzGhUD4yoChG421HfLKQqAGQuxX"
        "aCwVm7iyAUG4ErbbqS9zjDfXEtkbJ6Uhs8AYuwirYJ01sgIpRWSwNjuIHgtHJWwmW8MtuIoHlM9yZceeG9a0KaJVYN1KifpK"
        "z6pcieW8WSltGj/eDtXG+Nf20NNFcZt30kJHKOOGrY+FOvqpoiKszoak+f/2AgvEaJD5+ojkpzFmrQkhWQRQzqimn8QLvHEi"
        "05zq2qm4YiimIWGdKmQP+V3Qg+u/mXRl1HyNbpEc/8ut68ym3KVCFdmCZBxG09zOMTPzLDCf2+Cx/cuFITsXumfUinM+2AyE"
        "YTvwjyd8knGmji1DCqFk8wx1ABPSBEvMSNULv8qFjbIJzPRoZW8Zh8/9nK9lkE0LjCuYbSBzgCL8pkAW6X5c3hQX0kX5nmSI"
        "6wmB8yfNwytRh44rORnOheR3xsNmLvJrg9M9a13iX+V1QkFndcBqoGRyVKu0OVYTlqwjrn19z6mEc2JHX0HoLbo1jwmknFh+"
        "E/XX0R7IYd2zsQTbHhItIGvY8AhCZg/MkVX+f6LgZBK6ohxWrzdPGwHKUwvupzGYQWokB6j87nY4+bMsO+xviKXf+Zl3Tw48"
        "1tmsd2r2FUT0ezAb/RKvuYaAKqMWYCGhCb4IFRds8ZxGjFcl0/wgqxZcU1Mfn0iB/OYjFel060trjfmo1V5aWPH8yv37+kLI"
        "plyVd/BL0+i8QWWORW9Y8gDLfjfaGVLDA6JKXGimsk3bHZUY74mJOy4b2TMn6cy6fNRfwkLrudIg6esSGCSf4w4VcKuU42Vb"
        "xLNgblu8pVkgdsfqqbn0R1Z6UHDk9q/IYwStzXrtnLu4O7tDfTR9+V1WNOYWBecn+waAlF5zQbmAZkmxErCfwU6ObEZNYuyk"
        "CFQ7cxU2YJRh5ZI0vBA/3dyXPAqYWDExt5VU5GIYwBxia8L0WFP2fleVFbBZ8mkmxmhZqML0OK2bhe0P0IDVdaD2IUjhDq2n"
        "w7UaSEhjL5SfT7/dgtq/IaI0Oq4bhJwruda/ei1hCEFwmY5xS/CZRgRwTdTQvpDsMkQA2ietHUzJjLB4nSS1Kpnc+hb2ZL4Z"
        "2f07mKmv5MsabQJmBekMOziJ0DMj5NCAIYj7r8fwTy5Tx1e5isRG6pbX0waBBEpFK1SUCRN+T+40ntbNJ6rhH9q+UsZF+fcD"
        "P+bO+/LArVY1KeEVAtTwIUCaQOIXLB2U9AZ57skU/NgKMTQrByS7HYdIZ2WqRwmHk6/6WvA7tunx2vZJm38QnSUTYGfpJbyM"
        "0HzbxvLQJer+d6zW+NozdFG9Id0JT2OaphYtYnI5TYcuj1O8rjlH/7simG78vkRnQdLl7barv2nfT9pKPwsb8a2+WrnxVaP8"
        "ZcBCJE4y7oAn+1z5gwC5I3b4db17OXWg2J706F3iQ2TCR9Avlu4Cfj2tM8JCFE+5YTphEUBc3vyvsoUtgEYCfbqIYfdEt7/v"
        "V+ILMbugQfFIjCgAqwwtl/P9EasgwDgMjIJM9Za3TcpW4dOk7kegWKAyehP2QYrcwB9Ky+Ob4yHZPKNHTgVFbJA+ZM76P8LR"
        "vchkCy7A/2LGmW40Mf1nGuJuzaQJHx5i2jYKhnggyVIqa6jfBpUrRUnHy62FhiPgyIWQ2gnC+wdSyLhTn5TZz8+ctic5Si0N"
        "7BNfwQQSTLP4wCvVFUx27ukUatyfQ5o3qkPxQ+0QV3QEkx5ONu0md7aC2paptc4qj5jaq9hx44iA0i/FDdixzZts8N7vjrEy"
        "wHMzMG1KE+WFltB7bmnvalWtLonIElJu9AkkD4EnZAaWBpqHM2R2+idYDeT5EVgpqXRlc5eQCzCt9KrZ4GfzNxuCuU094L56"
        "9QTVDRi6CUoOp/KlFz9X8WXxOezQRPqVam3j1I34P6pL+VdHviZPWRTffTY/9Dp1XABzj3retQvlTmLdzfA6QHvyH4htDe7W"
        "9g3vkOsL88XLHDnslz1NPF6QopH58YWnIOVDF/Zt9tZOYRy9OApEQIil0NKZ0RxuTmAO6YpVOAPmB+FmvCo0pwn0EfCAo08a"
        "os/+eFhHJAhEEfRhyQY+KbJ2eHSC5za2hbxxIxDAqyk2L3rOWT/pI1chTQc8zrSBf5ZPs1i92quLaPU1YDAf6kLd5KPFmsdc"
        "H0ODxCetYPGhOIwgC1T2zaC3daRl2Bg8y2HAzSoKXaZfIt7LJyyj7v8HkbadTICE+3R+x8BttRbHMRKY4xvGWxZk5c6LU2hI"
        "2IXgiitiQYr5jjtveDu1+moyc6dTMfcpd2DIhP6iEeRYfQNvanojwx9eqdUxqss2w+/qy1sadhIVWRyzjmBUe2nLNIjzq01A"
        "1efg1qlzXAO5S6RBW0v7uvFaCo0mhWczuoaCUTVkJygy2EJb+fldil7o11SYczXE1Nr/un3zSsji1g+9x6bwUXhx7nhbvaFK"
        "RAtadWb4QfWriSf8A/QehCvq1pHt4V93jl8OkPXahXmEd9x1gVDASOSri7O1GAogR16A56rprEZFUVrHTBJyOjQqZyC2ktYl"
        "pxnlE89ce74MHFJ8XHwKKJKj+hUZzjcVZ1qAbBYnDRz1SjVp+8TQoAmp79edAnqQr5PpUh7JQknTJJe7shO49pylYVLnyBOe"
        "iXcqQmXrzayZAJ1IedImMKg9gEWc+zqYfPSAYXY20N98dvN7iLu+EBKJ3z9lA9in06tSsZILxW+OhWmgftEgytDKd41XwzIA"
        "3wdMdRSWogwgvSCz7peDidPaYszozrQpQO5/UDlihU1tDHkfn6yx2JZBaPhsa5nJzumdKWNwIENGh2z6UskY9x5BtDFI+m9Y"
        "kxZiDwqc4AyCy/uCllgmmPRr7/1m83bnZBgLK3566Z/vNZtZnEzV1wxaUizV0z8z8JiDFpTe3QEEkznuz4gIsuNodEpblGzM"
        "h7HDZWqeEF1Gtg8ftyPMYnOpAGFhgmljkXFwXQTtWEMMTj6s8g9qR0PJmW6lKebppxBnT9hJZfGoocawh1mRXuKYVQMBwU9f"
        "7b4jjwfnSwS1Uc8/Q/5F/bClhDhAz9ndUMDxnwuvk++hgwIZE4MnGCveiFUeqOzwNElefiCV6oTyvvjYvejvuAiXcwif8Ik1"
        "xneuNuWRq0tOxdn8jlv+yP44QJNqYRz4YpN1zRGrhVoOAB/vZPiKSppY8zTchoyB3RW3uHrHxlTsXBSsdC4Ie2duhXmgwjji"
        "m70F6/C14FMbzR3dPhrb8utPIrFe3NXShrVUNH+CiH+idIwAYgXeFxHAqjpBLSqCjDah2mIMgIfqpCXMPCX/xwEOLC9VTIXd"
        "zk90O/UnItOgQRFxkSan93jY1E0DQQ67XCSP0hQWV9GTADGTjnIeu4gKaWSu3jq18t5/9WKJQ25FOg4v+I4BKONxrzk7AXz6"
        "ODsdca69n5pGecPBxgg+8zJ5YPYFa5l+CT39GF9jE4nodzIPQELk0lucndYpL+iytCoaPVaUYlpDaiT08DFr3w1RK22P9yrN"
        "LPZfwgYWF2b4PtCheRpv6wRjv7KzDva8h4L9+gUqXk1QwFaaXvBW6zPTc4dYHtOIWovGFL0SQqSMC1MCHKb7BldvZ3r5rIDg"
        "iNLX3+PsB3miETiGtfw4PzKLWoZSfIjz+s5gIVSZXpc44hdaXUFb/zembvQna1OSSyOqai9ocI/G0mZjmR5SQuKgXEqOWNr1"
        "IiJN52w46+ZU3NTgnbx5Zz+BhMvwf1fsfUeiMM0Ycssk/R90aKNWuibAufFiXAQDozllA/HEjTfMuVxbcgKNdWDXRxHfN4CE"
        "W8srA9cSn6lpyoX6M6MvsM8+ZdV8kSqXbCMBzb98+EMFhJw4FKuJ+1Yz0YfNXEoyHmTJ2f6mBNoW9idMe23ySvE0xG+VShg6"
        "9yxLuqxlrwmTBAmypShS8PLzRHEShktkKUVb+4ftn8xOHyb+zgpdpS8RygqlgqCFlMvdMf/zgj2MEEjwzd3Y0hwv/Q1iNJoz"
        "ooJ3kpcdZySZh8CIFgjb4AZx3/gZ4yQzOWr1FSo6uf2Bpd/4xZUHiXoezGcEvGQODhiad5vHVncMcvG6g1oHg7s1hrQ/zNXr"
        "UW3J5/URIHGZo9daWpltxufw6rAyWeEMC8eSW+4o+Wf+v2CoThga2oiCCSUx82NBu8JoEGPp8lBhaFBRywGZmLMub9GnyFgI"
        "uljD7nEaws4lHCxPzn0TboUQmgu7MpdPDpRYM7x3SimLEDYoyiyAGmVQYNLlPVnU1ueN88yj+jRVkLWfwT3dAIsP2jsLZVdt"
        "og1KThpznKhOBgQDqozMHLbgPpXcplwG+HYpHdvO+/+IfpVsJD6Fkr5rEAsDe5Wbab3Bh2/utmdc6H9QUfxZpyHAz8U7//Fj"
        "cr53R0LslSFKFhFZSNnS7IZj4eMtLsTle+njHvu9ZXZgIzTqK/MlZ7Ns6wTAkhsBGcxmjzMIwr61aNip0WA/NQUTCLg1kkuK"
        "LcCXvY4/hyxGywBV2Fu6kr4iihQmyeM+8LCwZQ0Yxs53gUOSHe6r8bS9G3OimDlsnxXLvS+WhRZEVdTgzAebDXvHsa6C2i+W"
        "K4lEKKL1QOwmwtqEg6cu448jgfT5Gcn4WH05MkR3SgF5rvHAAGxwsJ/2wmuEJdjuBrAMJK22rSxOnhXJb+KmYwJC6JsWUs9+"
        "Jz6ZkWLlDKLWfd61FdcxPfqWVrww2IqV+6z35183C1p4qoSmYKpVIBvRF9aTxU/TCu7Z67JPdJjqqyxfM8Zwi18c1755Leji"
        "HRfjcfDsJ13Ocntv+NRACGIv9uofrpjLPZRrybCalb7a4QffYU41oAaX+tsC8toVzlJXrD/IkL5UZ2b7/t2Ucj8AWHYhIsYZ"
        "3LWV3c++lSY5ZPueR02wQGpBvEFvXiB2YnAdGhZ3So3G00cALt+3bvommrS2Hiv4834mLVbmZRflg9PfjVTGyKXmGtu7tRon"
        "B+6AIljptFn+Ql23EVI/bqCWZawwwK0clmtgEV5IsJ2fWRo7EbI3hxom1cvEyCBcJyIdmKSF8RoXUA/uHG1lbgAWKagLFRFQ"
        "bNOoU0whcp3CRQG4R8vO+uMKnET8cWAa6+3S+StA0e8o60gDvrnf9RuMLW/YR0WcByCGtnBZfi+5+Pc+c/rL6ap61mDoY9lh"
        "l+Gx6xW5UopvCuMT98C7YjRU04nCdbWRYdKTajRwZfi5oiSbDNC76ErzeWfNnsOEFwHZzhN5cSTO1toHsZsN7zQrZbF1hiwC"
        "oeJl4s2Ue+7P0NxkR0DyAOZ11HTMCVJZiGHLbirpDer8rfqjX6aDDkPNudtUCOi9XWB+QxytAY36lFODKuRIfYzyMwccypUx"
        "b2XOW+gMKzpc7MLlpQlKsigNVW7JM646BQRIKI23J/McwhyanCaCLJRcIkvePGWU6ywrb6IXgv8W5uwAhVlKJNbYq9cvLPQ5"
        "719RXTpJY6QEarhQnMqw2D/It+5efbNYXZpw4HPOTysJzmRRFviZTBU1YCwXoM1uscVs7ET6KdmOt7Wutv7ngfdE1tz+spxK"
        "iPoTDBIWR7QV5TZA/K9q0owvjQMkkPVQUolCMcJsJOL2/djQlEXhk+huqgsA/sIUGae+oO/2lDmeq/Q5HCCZK8bqsMJGXpfD"
        "EoGWHPDlkX90SEP+IfGMLYo2+PB86IjYlOQkCv+ILPdCjn1USoIe80S2/mdnenrAb/x7X4OMYz0YGzyLjZBKJGIshKFFRmuN"
        "hx/Wx9n+dYNch65a9cqnLSCR2/eT+DLPUXcd2rqUS5t3uXfkj3cpcFU4KV/SsLRwxI2bmdG/NLNIOGht31F+bWnzmYX2a8QR"
        "P5m5mAxmLckpGmWM4Evaln4WPgr6yxxof+ltyzrwFeJlKbeFG5yyl6c4rCCIQv7KyQGm/kHS3dKN2vqVhQ+tmYAzShWJlqVs"
        "9UcnNvTczmUnFel56tt2qx6GqLiRDE3jzIQcd2ka3uV7Sqqseop6ZiP8QcTHYukBV1oKet1BXatV2ACCleMgXA31lESKUQIQ"
        "oQa0goT8ozYvhwF8dwD5h9BjYuyN9uOk8eMafWoB8iMV+R0LGjn1kNN1W2GKFdh6hoRhye8NPSI1LD4XaMq5S1G9YvGzXCUR"
        "uKnmbRl3nuGZufDDaoTC4gIDIgtTtfMpx7RVGTr8NBjIhHox9C5IOZ2oNuJX6gGLu99y52UVGLxuf6N8xnmr4MQpKBunUl8C"
        "2naE5YSu36jAaPXeBpQjxjwbbJhW9hmLF0CMISL8lo3SRAgLSAakkacxvHOqUfEsDnaZ0Z5YfF+Izh+rDJBVkwL6MLXGm2ji"
        "14Zk0C92YFMwxPXVb4H0li9iIIhe+xupG2hZ6REfrClthpSSRcqT7uckl5k8LktiVNU1T40hW27JbpR/dQz8BZ5ThPsaTz9J"
        "XQQupoSEfXQhdCeGC69RX48Qbkq0axe4mTqq1CKK6vZFZo5oXWCN2ocx78jwyADsaBedXeCxnqSgx//7dMM1t9RoK993QGYb"
        "Jaslv6svdJ9XgnSYW3uHsJ7Bnbqo8dj52TexlZbPebbdFCZFzxUFHVT0v2TvvkTfZ+KXuhhn5EnnLM9A+/9q9qfh0/TMY33J"
        "8SmKldyAB/kelgHaa0oAmNqDnhGMRM/E/n6x6VEXlMjAY8Krzez5MGWUXglYhwopg6Gl2GKEFTxGkZQ+xh5dhRz/wH4VxmZj"
        "3OX4KXpUhq9SJVZU8j1KW0zIxJojrfxu5cLa8UetzChGwFdeEf7vueUMdIqmxihCbJkcHgkPkVnBNlHj+865Zgc6kKilMx80"
        "Rp7QnvYCCV+u3UQms7O3HjQNr/5jaTyo7cEhiFv9gSd5UkIZIyMbyIGIlg0lvGMUt5GTdyo8DqkbdGJY0RW2Jsy3deAgqmcM"
        "Ee0KOMGbXT9voRqqfZuRyB3nywApr+qFvydK5OoMMCpYlVEWel5Spq4hNJDfmktkQQ6/sL1u56Tb2u5KVI73bCWzcqdd3/ml"
        "YDkRle6Ket3T9rJKRuiE6gMlfJBCOHiCxc2tfs508M62xidfcuVl16G03FPWdy2w0Zt9FOIxN5L17q414yUPYh6oX4R6kYWG"
        "JxPdMx7EK34f3ZlsNVKbNVmKh5GgbZCEobLi6G8mZb1UKZ18lQWNvZl6sCjtYmt6CBWSJ+cdJQtFr/hESfOa4nk/3gM27qPi"
        "8Fn1MQpmhhnnQJGDbyg+fa+8pfU0pN+50j0JW7IMC38H5ncCoqgzh+mreCx6cf0X25cYyVuecn7pPA5Fw7feEgobTKzyihyJ"
        "2HHAliBaAlpuinGLimjC//9U453HV2JL611b1gwTnXkg+3Pd8SOVpP1hFpSHqvKMlZCViIFPtPt7DY3xKJd3yXTW9IBUPfJd"
        "mQ8OqQAbszL1sQMaLxSvHcEDTKiC0UyKzeh9yyNCd8JdQIgIJko8FAY0Bd6ch/G/fY2wzlPWlsA/zj6Pe2uyl5lPsZzbG0wj"
        "+EkU/MOdfT6RLCZZSos9Rob9iTG75vSbzKj+4eK4ZbnpWMJpXEbZcHA+z9GY3hi6QtN2KQgD0v8cn+NcmmzEf4BXTeaogbTQ"
        "GG64ID+aewRHLlW0w2GfGLeQM+3jwt3nY3IKMyq+FJ/+tWXOig6bRZXEwzFsN09mRPfUkzdWws2zfeHJ14q6Y94P98IxUEmU"
        "NlYu77HCQcaIAWJ/j0AUJ0S7q1GJmqSSsgwKApKZrcAi8bJJ9DCsh5IopE+V1D3EAwpQA+/zT5elce029mig1alPxsFt6KdB"
        "HOEUbqG2x28bhYT8+V/8kd871f/Bu0beWyAzZmpx1m0qsO+05oYzVr6HETh1qTPoNYgF9VZGLRPHWwX4M3CIW5fUuA1EXd7E"
        "tB6WVRZS4VVmfyQOHsN2rsIre9ZgjWpkDguGB8IsF4bTn4D9XyobqjKimlDJYAPAmreJbiwtIRM6zQNSlQ/RS0Oxqs+YUISi"
        "7BZlu6nqA02q4or/MgXdcUR2+jtUmxy/J/4TlnE7DZDmnmbJGF+aas0En+dudR7j1LPHkIAIN7F3/kcvCcgyB/Mg+drIK6pT"
        "32ww7Vf8TTYzpaZXMHdDCRT8fozAEoyI+J9p0+mSRl4QBCsPQI21HnnbQd6+bbDDp9iPRazegEAFVG58jVnuL6BK5WDWWSam"
        "9oOKco1JtrcRMVvZcgPJg6aYiUmb1D2hguvRC3HzjLJFIPyG1yQ6Ngbn51GmNBgpdtGCPNmP1bxKhVAtuZXC4CWRaU/DM+z0"
        "635XtNlWLtYp8datbe37JwJ5zeAbjMVvO2jvfoAhQnVhVXh0b71AiQlOWouOen2UrLaUo6efVO3SpTAofu1c/JYn4Z9ZhqMD"
        "mVIBQ7ungdTIQL0g5aF/rBViotH8iLhpX+OETsKWfN+/T2u2d3FeJKSVzREgeSSs6hEXusAdNTGd4eEdOdLg/p7cG8pQmTgK"
        "7Sxq60g4wfxsmn8t8zfQwUiXGhffugNP10tIbCb7eHiAtWh2AdKwFFa06i5J7hKoq44OqiVHSAS1wNdg7AzguKfIwmXsutW+"
        "Tz4r8rh/Y6Nf3Qyl438+iEmCMyjaJIiBBxDLAQcnxKRf0/qEwFzo+B+2mzVpRw+ycCdJgpP1/B0cepI5zGh1tfVDfT2+P9BO"
        "MDKX0PWHd5lBcHrpr6cbVcPNIPv2PxBL/n5yfEk4idyTpvVgjjbb3OVWnJBuoyWghL4UqTBNGT9Tpg84XJ9VSovVPadz3TXL"
        "HCCx0smXlWLYVXAYnYkIfGkj+oLIsop3XRPco6ckI/MuS+NemiXjLwNVRFHY5D1Q/YmPv/nWAMzhsKlrQUEXZ8tCHMl9S0XW"
        "j3RYE6B417Jdt3lat9U4Rq6Dt0girmYbVX2r0o0GG+x5vR4SizM9DNLJSZn9hYFq6r5LF9NQdH04smuKGcvSmn11NN2e4f+m"
        "WVUaUWsA/O/uRpJjtTYXY4131EM9bSkLthc8NjmY+EIK13j0blDan4dqVUaPLJQfmTHvpY1DOcBTJ88AoB4udIjuNg08jBrE"
        "qK+Z/MbDcKItD+roAZ5EXOz9r8MR2+0MzwgHcjXDeoy065dte1gjosHKjE+1r+ztpdnguvZ+Vgx515EbNv8WM6hCJmivJD5d"
        "Lkk3grgddDmRYQZIeausTYNVixIWqdkILQhnN/Kr6+x8mJUJarQATQRnJcT/1RmVJKsbTWn5Pqgx5R7H8Dw4sY/ZMHFOyhgs"
        "nUKCU39FPrHRfEXEEGiENLNoZ7LAnGtlXNFILQ9tzYCSEnOc42h3+eJzKnfJWWS/Oobp+jcs/aIyuhoWlLZjFPLsdmCQSSYy"
        "Er6XqnqQqHmatiaw0YsI6pg+oUUmL8C+DubdNGNIp8cu1KhqdK7ZM8cy3rUqSVzySkFVB3c52yXbXlaQfQeqjgf30u2zw2uN"
        "oIJj1oGLL1zOHA5n8nnC0TO11/fq+oaWP4Ax2WtN6JxVnldxA+tkadImb1FqoTvF4fNLa6Tnj8GvI+cMeUQuzZdYxGPiRwHc"
        "T1S+i/p+3/UOZZKvhTeuPXeA1NWPV6kIesuoiTIQPXDDatT2ZZ7xfOSa2JOReg26kS1Dk2e4WI9Q0Sz97MgmGbcs1auNDCed"
        "lZ/gW+F0mmpt1mN1yYVISiqJ6TJtvYf+pn1sqiTBhUG41Oam5sXyHxfM9j/Pq83OchWudMJbjAVqtHRQLGVZjREAugiaHAKR"
        "X7YuCrwGn7nQHtUKiN1P/5Dg81n9uQTxlPGKaCvS/rYyDYdpTcnc63ZStMu9/Q7pczpmI4ysGvNv2ESV1US/zllLhNQ5dMYU"
        "R88cE1bdEsqzzNljoZ1cDQ2xfNxlIjRyE6MJbFyBZ/K1QwyZOI2/fAbZOK9XJzgRttUX2QKgjRZTouxGR9Wt74+tdYAkiv8q"
        "kKpUk6Frrfutrgp1gueh3zh0UhpNk5zUmZRWTaTUh7b/w47M9TmaTk4gzYSh3YdkslhnUOk6fyjbhXY2ZTCO0+C9IyO6fY36"
        "i2pCafaCC3PIhZRIkcml+p7YjLCqulzlHTIwb+vRfcHnGsnQl2mdeWu+uybaGl8TAgeZ6mn0aBTKXIe4GJozzj4cw5YcJLTY"
        "q1bzGev9G5jT19X4pBvnk+Va7cl98x1Gh8VXDYS/EpMLLmvw0n29Eb87vg/MzKrYrZ9BAOyznv7xaigz5abakL10oZxbzSxE"
        "9oetuSCmF9c8h/IIrtPJI7iI4Wnu2bAjczujuflmSpI9VzwypL2oGGlDRK7SBXgd82B+yL28wZN/5FZCnXgvMqmdxIAH9xGM"
        "tr3g06V2gxayM9KBRrHW2J/0Lzjk5nAmsYIfeUeSaaYun8Ojd2WqozTyktRP8cViw2pW9qtOcXPyvL4YtBrEz12d/bekPwk1"
        "0mFx42iOQS+Sbv5ntjcp4jUktK+pwRaPdvWcgoPncf3SeFCcmkJeNWfh3Xu/4tYwCRrra1nEfENIgTEFBv5JsaRxVqBfBIep"
        "Ola1Mor8ChBH++RNdrExNOZA29ZiI9ICDOiCogpwLVaWqiUs8EBQfMP4TANpwV5fD51uKbhTRfjgtKi0Mwado4iYWvgEJfCz"
        "aeJWyJny1g8RDL8BjaWc4Q3NK/2C0grepZsaSvtPYjD89OzWWyNJ6vyZAsKFLtGpIkhlv1T81IAu1SZgSUZTmyaP+OkfbLCS"
        "Cqb0DoJTDnMAikT+anNYhD30yqzPznI5Kkb5rNyuexWFl+qxrHdneMOvV1oFF92PLSPsw4Z+2PS/PTIASABoGLH9yG+tKJay"
        "v2HWVrRgQuiyVC2wRCxdaguATRw8zh0z/cCE//DMhH0B+wDm4XEYqVxImB6QIus79xjyl+C/OKlxT6xpi3TMpxoUahh9MIsb"
        "+i5z2OwSLxrX4NJ7HS5YW2sFB/qLbpVL+boyofWbDHovst4FHhSm0t6sW2xIxeU5WmAZ4v5hCecSw6WgLrnKvErjf13EDFPF"
        "rjgWFtX2fg/l0ftmL9Yd91eedE/g4BaH7QhHkj2/nek1rEKHgCi3UOKkQxYadZ2AZpvs29f1nH9Bjsn088AlOLMyPZCpXXtd"
        "4TvgbnpLS3N7Oj8/zgMkbyC3JCGrI1eS/ivjWNyrEaESHP2B856B7K2BlJc6/2Q+/CcTKQHF+SvDWRu1bkBL5nzHAUd4JaLm"
        "FCIpxYVswW7UThABclvggyvGnwoHg1mnoJcFaqwCOSalIqrcxeLc8xiTfrNfDd7zOUTV85VjkQA9GofPmASIlKZG4tUv8hoO"
        "FRmwSWAdMZCIlWXhwlVYnfXWLacXubXELVnHJtujJg7U1DgfV9weJ48TLZcHYxMZ8IJUFvgZt8IDXTjh2/SzfBHrZziEC3DG"
        "iep8AHt1yVoRiRu58zJMwHoJepAYF4xrVgYK6/KOoJuW33+Vy/8ZjUlQnZ+eVAKCRWtxbnR72RLG/JVwCFQXTOGiHiXFTCCf"
        "JQJKvq433pawqu73ot34n6+SKKHOA56aJsxotn+W1dJlt5B0R8KzL/T6ZBXlwhdjZnKvDdqNmmjxuxLFUYIWT3Kktf++A9y7"
        "Lsulspt+Vo/Que2NGX/h4k8t1w1UW/X+QxVWwISYogaf+cPeMmz4rJmUa9F8I+ngrbgR/L9sSSjnMd5L6pG/goU8Y7LEnN/0"
        "9sZYdITonHVW61Uq74/e7t3Uq+OXOrWyoc277as5X3rSB3sh62msCtMzJRj6lrfKdHkEe0GnKQ7U9bsD6AvJz5tRXWCy0H8q"
        "TiyNntQOpkjByb6/M/aXA4WpYDRdlQN0T0n0kYtqOnVts95j2Tbdsuqi8rTcdec52nHb+tXYJeXYfs10rKFTPqP13NsUG4hz"
        "+Xz3ZAeayE3x0UpyKf3asxa903nAMVuxRouXb1T1yZUqJuO0+YwhPNWg8W7rcd8jKuMwstZi1HKhZkDO6Ow+XzqYpNT2ROYr"
        "2l94aEes20L6PTbxA9tsY7KB8in+DO0kv+DVWCmuKqc7vWfAYEFBabA8LfdcRnSYZ273wFlkFCUWP3PIsbk9ryj/Vmb012d1"
        "gBlYyjwmoSm1VT4AzdXfxI+cTlhyfV+njBjmKv1UDw0lRbUc3lQ2Pjoy9sjQ0sIVMBHEJO+afE+TGKFGfBoy/mRDU0wHkCpm"
        "Ar6JGE50u9gSPk5y2VcK3RlsKgxlxz+SqfG2EDHPXanbKMGxjJWMdU+dJf247nB3X1pyFarE+QSodq5qll1c+2vjZc3ZTbzC"
        "3iepwRdq4x9mmx1gVJe7cBQAfscLm0wBEqLnQd+SXdxFuDDzhtf4gMvSYRBWVt7+MRb4+syDAaRJhUzOy8xP22vbKvu2/xeW"
        "6JbkXWXRNp2EbYJYAYb6QWHP/6965snfIeokzTSoUt0Y+eSjZ72bhORTg/npx9UQt5mvVXTcPEURxjO7e40TjUeFG3f4OmAC"
        "eIAAAA==",
    ),
)

# Vazirmatn, subset to Latin and Persian/Arabic script blocks.
# Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn)
#
# This Font Software is licensed under the SIL Open Font License, Version 1.1.
# This license is copied below, and is also available with a FAQ at:
# http://scripts.sil.org/OFL
#
#
# -----------------------------------------------------------
# SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007
# -----------------------------------------------------------
#
# PREAMBLE
# The goals of the Open Font License (OFL) are to stimulate worldwide
# development of collaborative font projects, to support the font creation
# efforts of academic and linguistic communities, and to provide a free and
# open framework in which fonts may be shared and improved in partnership
# with others.
#
# The OFL allows the licensed fonts to be used, studied, modified and
# redistributed freely as long as they are not sold by themselves. The
# fonts, including any derivative works, can be bundled, embedded,
# redistributed and/or sold with any software provided that any reserved
# names are not used by derivative works. The fonts and derivatives,
# however, cannot be released under any other type of license. The
# requirement for fonts to remain under this license does not apply
# to any document created using the fonts or their derivatives.
#
# DEFINITIONS
# "Font Software" refers to the set of files released by the Copyright
# Holder(s) under this license and clearly marked as such. This may
# include source files, build scripts and documentation.
#
# "Reserved Font Name" refers to any names specified as such after the
# copyright statement(s).
#
# "Original Version" refers to the collection of Font Software components as
# distributed by the Copyright Holder(s).
#
# "Modified Version" refers to any derivative made by adding to, deleting,
# or substituting -- in part or in whole -- any of the components of the
# Original Version, by changing formats or by porting the Font Software to a
# new environment.
#
# "Author" refers to any designer, engineer, programmer, technical
# writer or other person who contributed to the Font Software.
#
# PERMISSION & CONDITIONS
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of the Font Software, to use, study, copy, merge, embed, modify,
# redistribute, and sell modified and unmodified copies of the Font
# Software, subject to the following conditions:
#
# 1) Neither the Font Software nor any of its individual components,
# in Original or Modified Versions, may be sold by itself.
#
# 2) Original or Modified Versions of the Font Software may be bundled,
# redistributed and/or sold with any software, provided that each copy
# contains the above copyright notice and this license. These can be
# included either as stand-alone text files, human-readable headers or
# in the appropriate machine-readable metadata fields within text or
# binary files as long as those fields can be easily viewed by the user.
#
# 3) No Modified Version of the Font Software may use the Reserved Font
# Name(s) unless explicit written permission is granted by the corresponding
# Copyright Holder. This restriction only applies to the primary font name as
# presented to the users.
#
# 4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font
# Software shall not be used to promote, endorse or advertise any
# Modified Version, except to acknowledge the contribution(s) of the
# Copyright Holder(s) and the Author(s) or with their explicit written
# permission.
#
# 5) The Font Software, modified or unmodified, in part or in whole,
# must be distributed entirely under this license, and must not be
# distributed under any other license. The requirement for fonts to
# remain under this license does not apply to any document created
# using the Font Software.
#
# TERMINATION
# This license becomes null and void if any of the above conditions are
# not met.
#
# DISCLAIMER
# THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
# OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
# DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM
# OTHER DEALINGS IN THE FONT SOFTWARE.
FONT_WOFF2 = (
    "d09GMgABAAAAAVA0ABQAAAACtGgAAU+/ACEAxQAAAAAAAAAAAAAAAAAAAAAAAAAAGopXG4GWBhyiFj9IVkFSk1gGYD9TVEFU"
    "gRonLgCKai9sEQgKg8BUg4MuMIWODgE2AiQDnRgLjk4ABCAFhVAHIAwHWxWNcgfVsW6fhESu3M7RtlUS+By91RKZ3Ih+nbeV"
    "i7xA9prpNg+Ru1WVaAdzbrL/////HckeGdvfDbv33jYYMEgVtaryf8jkQoRliKyoqI5Ws00QPAytmYnHsJhHR1tm1UCkmifX"
    "sEzZekFF2/KUsW/tPRah1zEHR8Y5R6iy57CVYr2c91rMUrudvU8Oh7BNxXxD3jGvMxS6ruQiM2s3p+i2QEZHKqfP0ID+Ruw+"
    "43jww4WrUead70bG1XKnRv6mnkP91DdNI4/Rx3K/2lJKmW7lo7S65c+vWi917ttWa68YWyll3BZrpRS7XZAdMfQ7+aOklNga"
    "7r/0F9dakzphPm3JX/A9qdO9Tpuz0ZbUsNxSihd1IcKw6Zc8klJSSkpDS1/eoZ/2aZijwouRi/4udBX+9UZN8NeSlGhhEypE"
    "L3z/r0bupkn036Vwkr6U9cAf54ONLBkD7/Vy3OYzjqwiCfICgx7hLioiEzmqJTq8Oou/DRstI8ZZE0qwDjTnHC2C3aBwmPxq"
    "IXJeLEOUEY6J6sdNggY0C6xm6L3B6YVqB3bDdH/QU7RyEz8nevr2EhiDywOGiIpp2eK4In3aL0TAX35VdY+TpbIAlY+TvRQv"
    "86YH+Ln17729ZZILQhiISKUOUEAkyiJ6ZKgggv13YXzFOjkwufI8zAQRsFEBSRlRY8AGgwXbEE21RrJwxQuonQWtpBXgCiyy"
    "JJtjiB2CQpq0OWZLci8HhOm1f99DKKVMSa5JCqg4iUEGGv7dLHohaN1YUUlnzsQ/7xsIFKtDDZME0yAeghdqNkplrKsO/0/7"
    "9Z/rbKjq/jATQOEJFRAKE0kgbGR8hI9QYWXiydLrWwM0tw4kVCJysMHGNsYY21gny2AbLBgxeoQSgoT4tApGEirqO8WKj/Qj"
    "/fLb7+Sf7+/5rX0uUGARBZE1H4L6M4G++eBNhLnUGR622T9dlAtr08V3FUZtc2aiooINKKUIKCggKShiEKUI2Ju5UnTqhDlz"
    "UX9ut3ZrF+miL3+en9ufe8+9720PlowQR42onoWJqHTZOSMZWICCmfEz7NjPv5v9/+oxHdPqSGcqFKlD2UmAIIEkSIDggQQL"
    "ZsXuTOeJye8Vsx4RGT0imuPfuf4nOTmxpqVGabeUIous+NPVFPaZyqflkBazCl5BUgVfKLKift1WfHjctH8SfQmBQKDQUqrr"
    "dlM92/bVLC/dV/ET3+nEbmtHaYt4QkSHp7n9EWkhFbeRTiXSBJ1BitAGH760UaSBwd6KwTZyI+dGLgRlxIgSccMCmzDqiBoi"
    "BgYgP7r1/52chBgy8ua5lb8yK+pf2m7TvmpnKMptKzG7ztwBBiJAyPA0Xf1L2rSlghaxAmuw/PsrfkphT83SYc98zVCpKG2a"
    "ZKI+8Zk3pjGfyMu2SuUUJ6VwTimUKnNTpHBWmZ2cUuecOrVSOUflBLykHOAJp75Ppx3HTvLXmB0LA5/UTWgLi2hLWAn+FMR+"
    "/O4r6tMpwT2LJ0s/FCwTiqg0QoPQrUIo9lZnUlYMn8GGVZQoV6U6KL57Nl80Ok+gK5GkF/ncbT/jUCNZnueZY/8QkxIIkU7M"
    "drdBKITGoZl8q2pqIpv0cmUWSir4zJQgd3b8a/u9PmkiLeDmgXELOZm+OLTVMuDJqxqeZF4cfIHj3/cGbSkq0ZqOXsRWThBv"
    "zsvNzDwfO08TjSCeYTBAnQ7kdT/ItZUUxkA5/YDBn+Z+sf+UwWOD+tX09uS2IROEwm30lVp8OWRitOkyrkEV5Ch8odzWzZba"
    "hCvKeN1vqqPlV3SM3CVI978UCRaBtCiHcI4EYR/T/bXnDx5x7W1yKRyOgBwqP+En2096Qqq9/z83v/cIem8SoB2z9axmLeJP"
    "gNSRlKZfqSS5m+TMrpy/UzVkBJ5ov9RmQ/ifC5ZByJCwBSUP9+Y2RNMTJoD8oQjKR7iqWt1KV59/M7UvfbcbDRYoaqZBuYJs"
    "Qf9zz6uxhTU5RUjzjfHZ341doRpQV3cDYKFBUgVAIIsNmUZTlF43SKm6Qf4B0AQFgmYkknIUxxEcR63RGMdxr5si5wGEZqqb"
    "Mg2MAzXOSess9Z130fpoNpvNfrzRhhuk3sYbpRuGG6Qu8kGyYbLZEhWrabfh1wjDSRx4R6EYlSDzPJ884WItLsgzOlDWMot3"
    "6UMWE+MDDx58RBySTk5pCfIbGrCE2AEG/X+mpu0OFysiUSIUSWVKDneuUtGQCql05dLPlV+fdv/MYHZ3domwAAhwARIMuDuA"
    "FxiUwgacvAB5icpUCKF05J1CTq93SLl1pd5F41Kl+sZFW7pu7P+nam87713O+QChPcZwhGPo/OWxKCdA+wup6LxFJ8eqphBM"
    "kcNRIJWh0QYtNmEznYNKuQu529Kl2x9i5V85hdJF5eOictu79NHUku5I9p3+P+ks6MICA+DVBAXwECiPtB5bkst9q05pXavZ"
    "XcuyTt+aP63z3liFPB0FguTQwRRCAv9uCeX7YcxlCiMiUiS4QaSn3ZlFrmw26TWQIYQwBJMKY4xqVCGECCYN6eze/9Z3U+jv"
    "xfnG4hQOEZFSShEJ4RGC/GH6t2GlD+hdn//ulyUNIoNIECthyInYYZm1/X2x6ZeebEPSxVeISYwBtzAze6rH7ac9ZGt/hI2F"
    "UR/0xlhR/UajPaylFVJo5rzTztQoIioqAQQEFKK2+12wIJ1JuzhO+oRSz9DD63WHrClBsK+27vucZZ2zECACRcCYnMlxBOZm"
    "NtRK85deA7gRzun8WcFI2mF+JYKAZrrALYBwoCOYi0OEgggTDWRbHYOIcXE93UAUiEckeUaRVnq1jx9nhhhmlElvWlSPaM0B"
    "IDEgS6pv/sBcJoEKCVejsWqNOE2kaCfTOrJtX7vbbhBW2iutduCKQ97Q/2zKaLZZX9si3jjGuOPeRKztCbs2B33Yh5jZkaEF"
    "jDjmgIj4uE+mhdiBNyHPLBAzFmMRRSyjDKWn0gaz2uRorst6M2hy2dgaV0E4E/U/2QMRKQKpO9IY+mcCeHDHb9gpCPQqBEBK"
    "AjRAzbEs/b1WqLNZyBZsJ2I4osEkmP9TFvzFlWUoFevr6QVhlBDkeKKBArlKG1uuNNvd3mC+ONZIKPgXN/guwpThTSyoJzL5"
    "a2DBGvSLkyXxufwiWhQXM4vzP6QtLdaVkSVGy5VXfscxWhsIFiQrVlBImCivpKxDi2wH9O9aV/f7O71LTsTFEwGURDLBUgBr"
    "4l/afyb9LKhsEMnGhcZgeXiFhEVWY+fOnhu673znX79OM5haukr9WWJi8yd+udkzIxP/E8/1xcTwkhWqNBL/JnrQEBwFSkrT"
    "lFUMYpHQsqnj1PkSl97oH5PuxV0TvGdgISChYOCBCIjkqdDDEMJN2cR56szAg4tIbBNKeRVkRYoSlCBVIYVVVkVTd+7u+/+C"
    "pBKm5LaHpQPThhYsWrVWrkqKK7E+VlXNaLZhhz/56Q9l8sIv8cc75ePr30zIK1BQGByBQmNwvHiQQCTz8VOFRMQkpKRpyirq"
    "Xt7ruTLhDBNE0MGGGL6IRCZ1GWYku8tUsqIKVa7NnbTYvlY71fl+7/Tcpj7KxKewPZs5+IEndFJ/0a1IMUmSqu3B9nh7e3tf"
    "e6J9oH2wfVv7SPvu9j3t+9sPtB9sP9r+yq9v/n7wEl3YfuXR+TcuvqkDq7YNbWJrnYnOgc6dnSOduzsPdB7tfM8FgyDAARs8"
    "Lf7SI2RWj3BCNXOpv8NEjMKUDXuNIcsL9eEOQYQQDbkWeTTh2qZO5oGt8+NO9+ycNXNeKMtn1PuqBD5g+qjZh9LddRODY5B0"
    "QADYzLvc5sSjR/OidlR84HOL8QMlaLCyuW1FVirRGUrSMCXHrkoTq7S7UjpGKZ2oVHRUqiqVqsFK1VD49ew4sxyg6YAAAZjx"
    "adDvI/w+y99n5fs4v8/ML1n8Jeifmfkz6N/RXAZ6Hdeu44/ruHEd9+9g/10cuouf7uPuMxz9ABwcAJdmA2AgIAAsGxrlCwer"
    "AuqqCs4CATAgxg0YW21DeYH3Vk0I4HDrVGXwfa+KefD9TNSL4AsGgCoCQAKD3N3GQENzk6OAOvK0WhksPsRqEWwWv0PUrecr"
    "ILjnC7HgohgAfhDothgHt8CCAw9h/9YgQoIMzwNzvWXMjw/uOWvPTGut9S3BkuYmFCJD9O/dqz6C0X+rhwOJuWEgQSCQoMFB"
    "hAJDYnAgkcxPFRGXosnKKyrrUeNrA3+BNgsTKUa8RMlSCWXIkiNPgSIlyuxSYa9qBxAhRoqcGeZYQE+B94gR27+DJbZjMgcn"
    "cEInfE6o2DdcCkUGRWoPhXAWBkDh4Icxm8X1cdQ3f069HoqoY16oTjXqGfLHY+uW2Pq2P4QX+VSv7FU//u15x+Lj/zUDg2wn"
    "I2+APCX5vpHgdLvvs+wpNW6mne+Qu7vKIw0oaggCEg8CARqMWSGbDkfhgcGRKK23lEakDXX7g8VLqJEkIikeSXOkx7LHSL6j"
    "cLxSbR+cXvyUaaOdt7yjF6VIkACSQZoZZpltjrnmCRNhtURJkqVIlS5Drmo1GnIuz+ZveSFroXOk3udRTYlUs6trELgl30DQ"
    "WAu4VQva1xjoSZcS1SKI156jNzTd0VNc6WvriXVTL6AnDBw8IhIKKjoGFr39/v+/oQTDN2tATXfaWXUa/NPcrWn3Xh4Xi9tI"
    "K9q0e+tdvuR1eRcbHFw8JtZsHU/TwcZ7hqZVRo2VX1MVKFRUxcdzTD88VxPnN1yo2hsIjL3ttmPnbL2EDOQn8+Tr5EhFJydX"
    "ocncx30gkMqdzAOtdr0JkCCCGMFUeCRUDHqHS/abwzXifyXxS/U7WhzvgqJ0Jx2paaKa59cadaw4EeS9xszc+uRT/xOZeYuk"
    "pGv9HwZkhcC5OsZXo4N8XfyB1iU31l1jqvucrTdAjtAOTCO9/kh3SPAYJZaayUSHNq7ZNCj5Clm5GH6+d65eqr+++Ks+EvzV"
    "48/1RW9NWawbfH1+yKZekJ50l5BtdDVjuj0Khf9K7Ooq7FXtwAYlJSUlXePAEavNy19qwj0816pNu7fe1dxdZIDSOtjFK0Qh"
    "IaOgohd3BcLdkrjnuVZt2r31ruYmuXcE6luNMDhMnxP0+CIDvlx8ddGOEl6hwpqs2icHF4+JNfNNdcLYA3mWVhD0QjYBh+es"
    "m27fBXRH8dBRst1QsksSR7h4TGrNEhSP6xA7Ux15l98kjt3d/hUQvRCBeuHAAP2NxnSYEE1Vc7q5vwCJ6lm0oGHia4oWI48i"
    "2Gm+IFPZXvJcIEa2vCQyD/upfqdcZlcpdPOnMdSK0uy+WOWXZXC2kHRq0P1zZsc/fZhn0A55GaSmhvFfXOSv1R2eu4VoVlme"
    "UyGyZkZRqPUvIUGLpbbFYembz2v9yWbCsVi1CoPAo3si+RUsrKZzPY6ZnMLG7xw5JaJG8krdaVUrPo1g7QnqsQW+U3OxiiNe"
    "r9RDxUNJ66GwHzL2ojP/HX9q5DcnJhpofmq1pn0KDWVmxuZpyPksCx2GzNDzFg2RRKYFW7QTc2R8+V8mkB2PdVnW6PiYCO0j"
    "CuYWhONVjlleybXLHESHKOa07pTiCQl1JiBa7+qiX3k0yHdLsLnO1Hr1mHg7uz+Adv776CSWfj89tp3zhiJ7ltBnHygfRko9"
    "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"
    "XK8vj3D6rpjnnNXlNA075AWCvfsYLla6414fR5zCRj0q1VsVAs7pEmQvk1qZnMRmu6hPk77//p47Xspct+Ui2ji4KxXrKUx0"
    "jD85xj7xObcmwcoosNB23O5uAxIrs8bIuy9ygE8jDEQgDLzJt2rn5CzHD2fuNqgfC0xV4/QB2beRnzyq0oyT2r3v+prf5V9X"
    "Lk3DqVgl6mn7/lI/X+ym9LsocBoC1XufAGfytUyXM9NMuGPQUFa3HTcbAjlSq4dRzW4pOjhuogjq1DUQDLaLlqQiaATqQwZf"
    "h/SPI0WeJTbFrhPIRoPDk0+rv26z7CDuw5Uf2VN4JbmNqLFvslSB2tqr9YU96lDjp4CF0+jEk2Ugu3dKp9s9z+8Xwywq8nGo"
    "szMI/RmmmunvtRrkgRYY2enKUCj3S0xMdgbStRYaLEJfDMfdvqx5DfHFvVtqgaZwM17x4AopB9l7XGM5SJERk3Zox3dhMZto"
    "afUPyO0qRq2lyJ2rj1g/hOu6WiY5zWSWWj+B1Jbs4PwHbH44chIWpliBm+MneaOiMY7IrnvGTN4CGKvyMa73RrtSuViXjrua"
    "gDjK2tqL0LPl8fLLHNnVEfIcy9EUdZ/A03Tf79BGi8Gp8vDV760uC19gV0q/RmPTSLxugrstx6kDP5KsI6J29qGdxlQdUo+s"
    "VgvbzJ30UovWRbO9nlVV3ce3MRmkuyuQRxWXxKpU7CVqMCRW3ZXOzpa70yqM96qBISo3qMUdpMDFFB0lhhNTBUpSS1e/Jyz+"
    "91fAex0GJDct9+3jVJVBxn1B37tiBv1IpeHpZ9VXRVd8y2+/tbI6V5nIqdVE9zU0BETltvOuclJe+mNE14qSkg7dpm6DYFmu"
    "+ZYqCkag0ZmOnED6VM2WdXy/pwIbCcf/Pf0h2fPqB/8BBex6/+lxqg48VrLunSDt/E1Iz6AQwilhYNyDvnCC3rO2lta1yYWw"
    "9xE4kDleqo5XqmZlsM7osOncW0s5DaVvJpg8UGzOzZMcuaqRuwozrv+CYR08VuiOOVmKuuJ/48iap98+BY0xJmQWV9q2vRxy"
    "6JU62xi9egKK0AASsqrESienRFIcQpdy1ZFfR4DrNEOgBP9CfClviz2CEguRBRc001szvYEz04EEJ29BtKGhwF5Ke/N6tLHC"
    "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"
    "oCK0iChaFN0TPRT9gK4Wp4lxYpb4oHhdUiRBSU5KLki3SZOkGqlDekJ6XvpD4lJSdE1TzWbNSzWfvynR7CO4a94PZkLl/Yfv"
    "lVDDylOBOv9QfYDjgWJOma/8K1VZlaoCqypVFJVEZVZ5VV3qbeo0dbEarWaoa9RmtVfdo55QL6lPq2+qX1E/Vc+p/01fq8nU"
    "FGt+XG9koDMmMsMzkzMvI1IRfyKHIH+hZCg7I9iINjJQcBQGlY/ioqSoBtQR1EnURNbKrHWmQlOliZ2VmnU56252cjY+m4F2"
    "sCAs1RY+OhOdh6ajhegadBv6BHoKfQM9i/6KQcxKTBt2bW1JLa6Wi03H5mGZ2CpsLfYg9jjWiD2DvYl9hv2EU+LW2+pwg3gL"
    "PAZ/H/8O/3eOLMfGcTj3PiGFcIRwn/CK8DUPnKvz7J0gZ7nzABG41hDXuiaIeuJvkkCychPdPFIKCUMqIHFIVSQdqZs0RJog"
    "XSc9JX0ifT18RrKMvOZvRxBPG/l8vn2DPL8p/1X+9wJeYO3N8zYWTFJsGm2UIeqqJjf1QuGW5p5CWWFTkV3LoSJD0XjRtaKH"
    "RW9oUpqOdpDWR5ugXac9pX2mE/pyupbeRj9GH6FfoT+gv6F/Z/RioBk9jP+KUcXE4r+ZMqZlR1ZHL/MW8wWLx5KxRtmL2LZd"
    "oK6KLjIbys5iU9gCtpJ/fdLKNrDH2dfYD9gv2J/Z3zhyDpxziDPH9fJTuFruB14wT1di1TvO7+aP8M/y3/G/CQwEhQKYgCeA"
    "CUyCKcFTwdtSaGlaqbq0ufRI6XDphdK7pa9Lf5Up4q3LCspQZaeEO4UVQo0wV8gQDgi/lTuW95ZfEFmJ8kRIEUkkErWIPlbE"
    "VHRXnKw4X7mpsrPyUVVQVXwVqopYRasSVC9Lz6u+X/2u+i+xSoaNOFfsEwvFNeIW8UPxB/E/EhXqoPSA9K70hnRW+lm6JFOm"
    "j8r3yW/Jr8mfyOfkvxUKWWWKZ8pkJV7JUJYrFcoGVaiqTb1ZXa6mqqHqDDVRzVFXq2vV7ep+9bj6svpOjbTmQa1fLae2tfas"
    "Zh5zJVOr+V0XUXdDG6tFaMlanlasrdP+1GXrbfTL+iP1FsV29T310vqG+hf1n+v35laPhux9h2jiqUI9p9ap3eqkuqak2Hhb"
    "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"
    "c74cFqTKgWDqbZZJ9oX4f1ifeeoO/7l1FAAAAA=="
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, help="Use your own existing Forouhar design file unmodified.")
    cli, _ = parser.parse_known_args()
    if cli.original is None:
        main()
    else:
        from src.original_dashboard import load_original
        _load_local_settings()
        try:
            original = load_original(cli.original)
        except (OSError, ValueError, ImportError) as error:
            st.error(str(error))
        else:
            original.main()
