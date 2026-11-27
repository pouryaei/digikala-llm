"""Read the packaged Digikala catalogue and review subset using stdlib SQLite."""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import json
import os
from pathlib import Path
import sqlite3

from .config import Config
from .task2_review_store import identifier

CATALOG_METADATA = ("product_id", "title_fa", "category1", "category2", "sub_category", "brand",
                    "price_recorded", "price_toman", "price_conflict", "rating_100", "rating_count")


def resolve_price_unit(value=None):
    unit = value if value is not None else os.environ.get("QBC13_SOURCE_PRICE_UNIT")
    if unit not in (None, "", "rial", "toman"):
        raise ValueError("واحد قیمت منبع باید rial یا toman باشد.")
    return unit or None


class Catalog:
    def __init__(self, path=Config.CATALOG_PATH):
        self.path = Path(path).expanduser().resolve(strict=True)
        with self.connection() as con:
            if con.execute("PRAGMA user_version").fetchone()[0] != 2:
                raise ValueError("Unsupported RAG catalogue schema.")
            self.metadata = {r["key"]: json.loads(r["value"]) for r in con.execute("SELECT key,value FROM metadata")}

    @contextmanager
    def connection(self):
        con = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=5)
        con.row_factory = sqlite3.Row
        try:
            yield con
        finally:
            con.close()

    def get_products(self, product_ids, *, source_price_unit=None):
        ids = list(dict.fromkeys(identifier(pid) for pid in product_ids))
        if not ids:
            return {}
        out = {}
        unit = resolve_price_unit(source_price_unit)
        with self.connection() as con:
            for i in range(0, len(ids), 250):
                batch = ids[i:i+250]
                marks = ",".join("?" for _ in batch)
                rows = con.execute(
                    "SELECT " + ",".join("p."+k for k in CATALOG_METADATA)
                    + ",(SELECT COUNT(*) FROM reviews r WHERE r.product_id=p.product_id) AS review_count "
                    f"FROM products p WHERE p.product_id IN ({marks})", batch
                ).fetchall()
                for row in rows:
                    r = dict(row)
                    price = r["price_recorded"]
                    if r["price_toman"] is None and unit and price is not None:
                        r["price_toman"] = price / 10 if unit == "rial" else price
                    r["price_unit"] = "تومان" if r["price_toman"] is not None else ("واحد ثبت‌شده (نامشخص)" if price is not None else None)
                    out[r["product_id"]] = r
        return out

    def get_review_rows(self, product_id):
        pid = identifier(product_id)
        with self.connection() as con:
            product = con.execute("SELECT title_fa FROM products WHERE product_id=?", (pid,)).fetchone()
            if product is None:
                return None, []
            rows = con.execute(
                "SELECT * FROM reviews WHERE product_id=? ORDER BY comment_id", (pid,)
            ).fetchall()
        return product["title_fa"], [dict(r) for r in rows]

    def review_stats(self, product_ids):
        """Real recommendation counts for a bounded candidate set, never guessed."""
        ids = list(dict.fromkeys(identifier(pid) for pid in product_ids))
        out = {}
        with self.connection() as con:
            for i in range(0, len(ids), 250):
                batch = ids[i:i+250]
                marks = ",".join("?" for _ in batch)
                rows = con.execute(f"""
                    SELECT product_id, count(*) AS n,
                           sum(recommendation_status='recommended') AS positive,
                           sum(recommendation_status IN ('recommended','not_recommended','no_idea')) AS valid
                      FROM reviews WHERE product_id IN ({marks}) GROUP BY product_id
                """, batch)
                out.update({row[0]: (row[1] or 0, row[2] or 0, row[3] or 0) for row in rows})
        return {pid: out.get(pid, (0,0,0)) for pid in ids}

    def preference_scope_ids(self, scope):
        # Restrict only explicit, broad use-case queries. All conditions use
        # actual catalogue categories; do not infer physical specifications.
        conditions = {
            "travel": "(category1 IN ('لوازم سفر','لوازم کمپینگ') OR category2 IN ('کیف و کوله سفری','ساک سفری','اجاق و ظرف سفری','چمدان','تجهیزات کمپینگ','تجهیزات کوهنوردی و صخره نوردی'))",
            "everyday_bag": "(title_fa LIKE 'کیف%' AND category2 IN ('کیف زنانه','کیف مردانه','کیف زنانه و مردانه','کیف دست دوز پارچه ای','کیف چرمی دست دوز'))",
            "backpack": "(category2 IN ('کوله پشتی زنانه و مردانه','کوله پشتی مردانه','کوله پشتی زنانه','کیف و کوله مدرسه') OR title_fa LIKE 'کوله%')",
        }
        if scope not in conditions:
            raise ValueError("Unknown product preference scope.")
        with self.connection() as con:
            return {str(row[0]) for row in con.execute("SELECT product_id FROM products WHERE " + conditions[scope])}

    def filtered_ids(self, *, min_price=None, max_price=None, min_rating=None, min_reviews=0, source_price_unit=None,
                     price_in_recorded_units=False):
        unit = resolve_price_unit(source_price_unit)
        if (min_price is not None or max_price is not None) and not price_in_recorded_units and not unit and not self.metadata.get("supplied_price_unit_confirmed"):
            raise ValueError("واحد قیمت در دادهٔ اصلی مشخص نشده؛ بودجهٔ تومانی را نمی‌توان اعمال کرد. فقط اگر واحد اصلی را می‌دانی، --source-price-unit rial یا toman را تعیین کن.")
        price_column = "price_recorded" if price_in_recorded_units else {
            "rial": "COALESCE(price_toman, price_recorded/10.0)", "toman": "COALESCE(price_toman, price_recorded)"
        }.get(unit, "price_toman")
        conditions, params = [], []
        for value, relation, column in ((min_price, ">=", price_column),
                                        (max_price, "<=", price_column),
                                        (min_rating, ">=", "rating_100")):
            if value is not None:
                conditions.append(f"{column} {relation} ?")
                params.append(value)
        if min_reviews:
            conditions.append("(SELECT COUNT(*) FROM reviews WHERE reviews.product_id=products.product_id) >= ?")
            params.append(min_reviews)
        sql = "SELECT product_id FROM products" + (" WHERE " + " AND ".join(conditions) if conditions else "")
        with self.connection() as con:
            return {str(row[0]) for row in con.execute(sql, params)}


@lru_cache(maxsize=3)
def open_catalog(path=Config.CATALOG_PATH):
    return Catalog(path)
