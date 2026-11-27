"""Interactive manager charts for the Forouhar Streamlit app."""

from __future__ import annotations

from pathlib import Path

import duckdb
import plotly.graph_objects as go
import streamlit as st

from analytics_service import (
    get_category_review_distribution,
    get_price_vs_rating_data,
    get_top_rated_brands,
    resolve_analytics_files,
)


ACCENT = "#315bb7"
PLOT_LIMIT = 4_000


@st.cache_data(show_spinner=False, max_entries=8)
def _distribution(directory, file_versions):
    return get_category_review_distribution(data_dir=directory)


@st.cache_data(show_spinner=False, max_entries=24)
def _ratings(directory, file_versions, category):
    return (
        get_top_rated_brands(category=category, data_dir=directory),
        get_price_vs_rating_data(category, max_points=PLOT_LIMIT, data_dir=directory),
    )


def _plot(fig, key, alt):
    fig.update_layout(
        font_family="Vazirmatn, Tahoma, sans-serif",
        margin=dict(l=6, r=12, t=12, b=8),
    )
    st.plotly_chart(fig, key=key, alt=alt, width="stretch", config={"displaylogo": False})


def _render_distribution(distribution):
    review_count, category_count = st.columns(2)
    review_count.metric("نظرات پاک‌سازی‌شده", f"{distribution.review_count.sum():,}", border=True)
    category_count.metric("دسته‌های دارای نظر", f"{len(distribution):,}", border=True)

    st.subheader("توزیع نظرات به تفکیک دسته", anchor=False)
    shown = distribution.head(12)
    fig = go.Figure(go.Bar(
        x=shown.review_share_pct.tolist(), y=shown.category.tolist(), orientation="h",
        customdata=shown.review_count.tolist(), marker_color=ACCENT,
        hovertemplate="%{y}<br>سهم از کل نظرات %{x:.2f}٪<br>تعداد نظرات %{customdata:,}<extra></extra>",
    ))
    fig.update_layout(height=max(240, 28 * len(shown) + 72), xaxis_title="سهم از کل نظرات")
    fig.update_xaxes(ticksuffix="٪")
    fig.update_yaxes(autorange="reversed", title=None, automargin=True)
    _plot(fig, "m1_distribution_plot", "سهم دسته‌ها از کل نظرات پاک‌سازی‌شده")
    if len(distribution) > len(shown):
        st.caption("نمودار ۱۲ دستهٔ پرنظر را نشان می‌دهد.")
    with st.expander("جدول دسته‌ها"):
        st.dataframe(
            distribution.rename(columns={"category": "دسته", "review_count": "تعداد نظرات",
                                         "review_share_pct": "سهم از کل نظرات (%)"}),
            column_config={"سهم از کل نظرات (%)": st.column_config.NumberColumn(format="%.2f")},
            hide_index=True, width="stretch",
        )


def _render_brands(brands):
    st.subheader("برندهای با بالاترین امتیاز", anchor=False)
    if brands.empty:
        st.info("در این دسته نظری با امتیاز معتبر برای برندها وجود ندارد.")
        return
    fig = go.Figure(go.Bar(
        x=brands.mean_review_rating_5.tolist(), y=brands.brand.tolist(), orientation="h",
        customdata=brands.rated_review_count.tolist(), marker_color=ACCENT,
        hovertemplate="%{y}<br>میانگین امتیاز از ۵ %{x:.2f}<br>نظرات امتیازدار %{customdata:,}<extra></extra>",
    ))
    fig.update_layout(height=max(240, 30 * len(brands) + 72), xaxis_title="میانگین امتیاز از ۵")
    fig.update_xaxes(range=[0, 5.12], dtick=1)
    fig.update_yaxes(autorange="reversed", title=None, automargin=True)
    _plot(fig, "m1_brands_plot", "میانگین امتیاز نظرات و تعداد نظرات امتیازدار هر برند")


def _render_price_rating(result):
    st.subheader("قیمت و امتیاز محصول", anchor=False)
    points = result["data"]
    if points.empty:
        st.info("در این دسته محصولی با قیمت مثبت و نظر امتیازدار پیدا نشد.")
        return

    correlation = result["pearson_r"]
    if correlation is None:
        st.caption("برای محاسبهٔ ضریب `Pearson` به چند محصول با قیمت و امتیاز متغیر نیاز است.")
    else:
        st.caption(f"ضریب `Pearson` برابر با **{correlation:+.3f}** روی {result['product_count']:,} محصول است.")

    fig = go.Figure(go.Scatter(
        x=points.price.tolist(), y=points.mean_review_rating_5.tolist(), mode="markers",
        text=points.title_fa.tolist(),
        customdata=list(zip(points.rated_review_count.tolist(), points.product_id.astype(str).tolist())),
        marker=dict(color=ACCENT, opacity=0.52, size=6),
        hovertemplate=("%{text}<br>قیمت ثبت‌شده %{x:,}<br>میانگین امتیاز %{y:.2f}"
                       "<br>نظرات امتیازدار %{customdata[0]:,}<br>شناسهٔ محصول %{customdata[1]}<extra></extra>"),
    ))
    fig.update_layout(height=360, xaxis_title="قیمت ثبت‌شده", yaxis_title="میانگین امتیاز از ۵")
    fig.update_xaxes(type="log")
    fig.update_yaxes(range=[-0.12, 5.12], dtick=1)
    _plot(fig, "m1_price_rating_plot", "قیمت و میانگین امتیاز نظرات به تفکیک محصول")
    if len(points) < result["product_count"]:
        st.caption(f"نمودار نمونه‌ای بازتولیدپذیر از {len(points):,} محصول را نمایش می‌دهد.")
    st.caption("محور قیمت در مقیاس `log` و با واحد ثبت‌شده در داده نمایش داده می‌شود.")


def render_analytics_dashboard(data_dir: str | Path | None = None) -> None:
    """Render the M1 charts inside an existing page or Streamlit tab."""
    try:
        files = resolve_analytics_files(data_dir)
        directory = str(files[0].parent)
        stats = [path.stat() for path in files]
        versions = tuple((str(path), stat.st_mtime_ns, stat.st_size) for path, stat in zip(files, stats))
        with st.spinner("در حال شمارش نظرات…"):
            distribution = _distribution(directory, versions)
        if distribution.empty:
            st.info("در فایل نظرات، ردیفی برای تحلیل وجود ندارد.")
            return

        _render_distribution(distribution)
        category = st.selectbox(
            "دستهٔ محصول برای نمودار برند و قیمت",
            [None, *distribution.category.tolist()],
            format_func=lambda value: "همهٔ دسته‌ها" if value is None else value,
            key="m1_category",
        )
        with st.spinner("در حال محاسبهٔ شاخص‌های رضایت…"):
            brands, price_rating = _ratings(directory, versions, category)
        left, right = st.columns(2)
        with left:
            _render_brands(brands)
        with right:
            _render_price_rating(price_rating)
    except FileNotFoundError:
        st.info("مسیر پوشهٔ دو فایل `Parquet` پروژه را در `QBC13_ANALYTICS_DIR` تنظیم کنید.")
    except (OSError, duckdb.Error) as exc:
        st.error("خواندن فایل‌های تحلیلی پروژه با خطا روبه‌رو شد.")
        with st.expander("جزئیات خطا"):
            st.code(str(exc))
    except (TypeError, ValueError, KeyError, AttributeError):
        st.error("رسم نمودارها ممکن نشد. ساختار دادهٔ تحلیلی و نسخهٔ وابستگی‌ها را بررسی کنید.")


if __name__ == "__main__":
    st.set_page_config(page_title="فروهر | تحلیل فروشگاه", layout="wide")
    st.title("تحلیل فروشگاه")
    render_analytics_dashboard()
