"""Store analytics computed from the cleaned Forouhar Parquet tables."""

from __future__ import annotations

import os
import json
from contextlib import contextmanager
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, TypedDict

import duckdb

if TYPE_CHECKING:
    import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "processed_repo_based"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "analytics"
def bundled_catalog_scope() -> dict:
    """Read the actual packaged scope; never keep a hard-coded 15k caption."""
    try:
        info = json.loads((ROOT / "storage" / "catalog_manifest.json").read_text(encoding="utf-8"))
        count = info.get("product_count")
        reviews = info.get("review_rows")
        if not isinstance(count, int) or count < 1 or not isinstance(reviews, int) or reviews < 0:
            return {}
        return info
    except (OSError, ValueError, TypeError):
        return {}


def _bundled_analytics_dir() -> Path:
    scope = bundled_catalog_scope()
    if scope:
        candidate = ROOT / "data" / f"analytics_{scope['product_count']}"
        if candidate.is_dir():
            return candidate
    return ROOT / "data" / "analytics_subset"


BUNDLED_ANALYTICS_DIR = _bundled_analytics_dir()
CLEAN_FILENAMES = ("products_clean.parquet", "comments_clean.parquet")
FINAL_FILENAMES = ("products.parquet", "reviews.parquet")

CATEGORY_LABEL = "coalesce(nullif(trim(p.category1), ''), 'نامشخص')"
CATEGORY_FILTER = f"(? IS NULL OR fa_group_key({CATEGORY_LABEL}) = fa_group_key(?))"
VALID_RATING = "r.rating_5 BETWEEN 0 AND 5"


class PriceRatingData(TypedDict):
    data: pd.DataFrame
    product_count: int
    pearson_r: float | None


def resolve_analytics_files(data_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Find a complete pair in one folder, preferring the finalized filenames."""
    configured_dir = os.environ.get("QBC13_ANALYTICS_DIR")
    if data_dir is not None:
        directories = (Path(data_dir),)
    elif configured_dir:
        directories = (Path(configured_dir),)
    else:
        directories = (BUNDLED_ANALYTICS_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_DATA_DIR, ROOT / "processed_repo_based")

    for directory in directories:
        directory = directory.expanduser().resolve()
        for product_name, review_name in (FINAL_FILENAMES, CLEAN_FILENAMES):
            products, reviews = directory / product_name, directory / review_name
            if products.is_file() and reviews.is_file():
                return products, reviews

    checked = ", ".join(str(directory) for directory in directories)
    raise FileNotFoundError(f"Could not find a complete products/reviews Parquet pair in {checked}")


@contextmanager
def open_analytics_tables(files: tuple[Path, Path]) -> Iterator[duckdb.DuckDBPyConnection]:
    with duckdb.connect(config={"threads": 4, "memory_limit": "2GB"}) as con:
        for table, path in zip(("products", "reviews"), files):
            con.read_parquet(str(path), hive_partitioning=False).create_view(table)
        con.execute(r"""
            CREATE MACRO fa_group_key(label) AS trim(regexp_replace(
                translate(nfc_normalize(label),
                          'يىك٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹', 'ییک01234567890123456789'),
                '[[:space:]\p{Z}\x{200B}\x{200C}]+', ' ', 'g'
            ))
        """)
        yield con


def get_top_rated_brands(
    limit: int = 10, *, category: str | None = None, data_dir: str | Path | None = None
) -> pd.DataFrame:
    """Rank named brands by the mean of their valid review ratings."""
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")

    with open_analytics_tables(resolve_analytics_files(data_dir)) as con:
        return con.execute(f"""
            WITH brand_stats AS (
                SELECT trim(p.brand) AS brand, sum(r.rating_5) AS rating_sum,
                       count(*) AS rating_count
                  FROM reviews r JOIN products p USING (product_id)
                 WHERE p.brand IS NOT NULL AND {VALID_RATING} AND {CATEGORY_FILTER}
                 GROUP BY p.brand
            )
            SELECT min(brand) AS brand,
                   sum(rating_sum) / sum(rating_count) AS mean_review_rating_5,
                   sum(rating_count)::BIGINT AS rated_review_count
              FROM brand_stats
             WHERE fa_group_key(brand) <> ''
             GROUP BY fa_group_key(brand)
             ORDER BY mean_review_rating_5 DESC, rated_review_count DESC, brand
             LIMIT ?
        """, [category, category, limit]).df()


def get_category_review_distribution(*, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Count all cleaned reviews and compute each category's share on a 0–100 scale."""
    with open_analytics_tables(resolve_analytics_files(data_dir)) as con:
        return con.execute(f"""
            WITH category_stats AS (
                SELECT {CATEGORY_LABEL} AS category, count(*) AS review_count
                  FROM reviews r LEFT JOIN products p USING (product_id)
                 GROUP BY p.category1
            ), counts AS (
                SELECT min(category) AS category, sum(review_count)::BIGINT AS review_count
                  FROM category_stats
                 GROUP BY fa_group_key(category)
            )
            SELECT category, review_count,
                   100.0 * review_count / sum(review_count) OVER () AS review_share_pct
              FROM counts
             ORDER BY review_count DESC, category
        """).df()


def get_price_vs_rating_data(
    category: str | None = None, *, max_points: int | None = None,
    data_dir: str | Path | None = None,
) -> PriceRatingData:
    """Calculate Pearson r over eligible products and optionally sample the plot points."""
    if max_points is not None and (type(max_points) is not int or max_points < 1):
        raise ValueError("max_points must be a positive integer or None")

    with open_analytics_tables(resolve_analytics_files(data_dir)) as con:
        con.execute(f"""
            CREATE TEMP TABLE price_rating AS
            SELECT p.product_id, p.title_fa, {CATEGORY_LABEL} AS category, p.price,
                   avg(r.rating_5) AS mean_review_rating_5,
                   count(*) AS rated_review_count
              FROM reviews r JOIN products p USING (product_id)
             WHERE p.price > 0 AND isfinite(p.price) AND {VALID_RATING}
               AND {CATEGORY_FILTER}
             GROUP BY ALL
        """, [category, category])
        count, correlation = con.execute("""
            SELECT count(*), corr(price, mean_review_rating_5) FROM price_rating
        """).fetchone()
        if max_points is not None and count > max_points:
            data = con.execute("""
                SELECT * FROM price_rating
                 ORDER BY md5_number_lower(CAST(product_id AS VARCHAR)), product_id
                 LIMIT ?
            """, [max_points]).df()
        else:
            data = con.execute("SELECT * FROM price_rating ORDER BY product_id").df()

    return {
        "data": data,
        "product_count": count,
        "pearson_r": correlation if correlation is not None and isfinite(correlation) else None,
    }
