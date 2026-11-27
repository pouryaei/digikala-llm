"""Expand the Forouhar_Pink_Blue catalog from locally cleaned Parquet files.

Place this file beside app.py; run with the Streamlit server stopped.
No LLM calls. Default: BM25. Optional --dense adds the aligned FAISS index.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def q(name):
    return '"' + name.replace('"', '""') + '"'


def col(columns, *names, required=False):
    for name in names:
        if name in columns:
            return q(name)
    if required:
        raise ValueError(f"Missing required column: {' / '.join(names)}")
    return "NULL"


def text_expr(columns, *names):
    candidates = [
        f"CASE WHEN lower(trim(CAST({q(n)} AS VARCHAR))) NOT IN "
        f"('', 'null', 'none', 'nan', 'n/a', '-', '[]') THEN CAST({q(n)} AS VARCHAR) END"
        for n in names if n in columns
    ]
    if not candidates:
        return "CAST(NULL AS VARCHAR)"
    return candidates[0] if len(candidates) == 1 else "coalesce(" + ", ".join(candidates) + ")"


def finite(expr, lower, upper=None):
    v = f"try_cast({expr} AS DOUBLE)"
    bounds = f"{v} >= {lower}" + (f" AND {v} <= {upper}" if upper is not None else "")
    return f"CASE WHEN isfinite({v}) AND {bounds} THEN {v} END"


def product_projection(cols):
    pid = col(cols, "product_id", "id", required=True)
    title = text_expr(cols, "title_fa")
    lexical = text_expr(cols, "product_text_lexical")
    norm = text_expr(cols, "product_text_norm")
    price = finite(col(cols, "price", "price_recorded"), 1)
    toman = finite(col(cols, "price_toman"), 1)
    if "price_valid" in cols:
        price = f"CASE WHEN try_cast(price_valid AS BOOLEAN) THEN {price} END"
        toman = f"CASE WHEN try_cast(price_valid AS BOOLEAN) THEN {toman} END"
    count = f"try_cast({col(cols, 'rating_count')} AS BIGINT)"
    rating = finite(col(cols, "rating_100"), 0, 100)
    return f"""
        try_cast({pid} AS BIGINT) AS product_id,
        {title} AS title_fa,
        {text_expr(cols, 'category1', 'Category1')} AS category1,
        {text_expr(cols, 'category2', 'Category2')} AS category2,
        {text_expr(cols, 'sub_category')} AS sub_category,
        {text_expr(cols, 'brand', 'Brand')} AS brand,
        {price} AS price_recorded,
        {toman} AS price_toman,
        coalesce(try_cast({col(cols, 'price_conflict')} AS BOOLEAN), false)::INTEGER AS price_conflict,
        CASE WHEN {count} > 0 THEN {rating} END AS rating_100,
        CASE WHEN {count} >= 0 THEN {count} END AS rating_count,
        coalesce({lexical}, {title}) AS product_text_lexical,
        coalesce({norm}, {title}) AS product_text_norm
    """


def review_projection(cols):
    from src.task2_review_store import TEXT_ALIASES

    if not any(n in cols for aliases in TEXT_ALIASES.values() for n in aliases):
        raise ValueError("The review file has no review text columns.")
    pid = col(cols, "product_id", required=True)
    cid = col(cols, "comment_id", "id", required=True)
    label = col(cols, "recommendation_status")
    rating = finite(col(cols, "rating_5", "rate", "rating"), 0, 5)
    text = {f: text_expr(cols, *aliases) for f, aliases in TEXT_ALIASES.items()}
    return ", ".join([
        f"try_cast({cid} AS BIGINT) AS comment_id",
        f"try_cast({pid} AS BIGINT) AS product_id",
        *(f"{v} AS {k}" for k, v in text.items()),
        f"{rating} AS rating_5",
        f"CASE WHEN {label} IN ('recommended','not_recommended','no_idea') THEN CAST({label} AS VARCHAR) END AS recommendation_status",
        *(f"{text_expr(cols, f'{field}_clean', *TEXT_ALIASES[f'{field}_raw'])} AS {field}_clean"
          for field in ("title", "body", "advantages", "disadvantages")),
    ])


def populate_catalog(con, products, review_cursor, db_path, info):
    from scripts.build_review_store import PRODUCT_FIELDS, SCHEMA, REVIEW_FIELDS
    with closing(sqlite3.connect(db_path)) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript(SCHEMA)
        with db:
            db.executemany(
                "INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (tuple(p[k] for k in PRODUCT_FIELDS) for p in products),
            )
            total = 0
            while batch := review_cursor.fetchmany(10000):
                if len(batch[0]) != len(REVIEW_FIELDS):
                    raise RuntimeError("Unexpected review columns.")
                db.executemany("INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?)", batch)
                total += len(batch)
                if total % 100000 == 0:
                    print(f"  Stored {total:,} reviews", flush=True)
            info["review_rows"] = total
            info["products_with_reviews"] = db.execute(
                "SELECT count(DISTINCT product_id) FROM reviews").fetchone()[0]
            condition = ("coalesce(title_raw,body_raw,advantages_raw,disadvantages_raw) "
                         "IS NOT NULL")
            info["rows_with_review_text"] = db.execute(
                "SELECT count(*) FROM reviews WHERE " + condition).fetchone()[0]
            info["products_with_text_reviews"] = db.execute(
                "SELECT count(DISTINCT product_id) FROM reviews WHERE " + condition).fetchone()[0]
            if info["products_with_text_reviews"] != len(products):
                raise RuntimeError("Selected products must each have at least one textual review.")
            db.executemany("INSERT INTO metadata VALUES (?,?)",
                           [(k, json.dumps(v, ensure_ascii=False)) for k, v in info.items()])
        if (db.execute("PRAGMA quick_check").fetchone()[0] != "ok"
                or db.execute("PRAGMA foreign_key_check").fetchone() is not None):
            raise RuntimeError("SQLite integrity check failed.")


def validate(storage, mapping, info, dense):
    from src.bm25_index import BM25IndexManager, mapping_ids, mapping_digest
    from src.search_debug_tools import inspect_flat_faiss
    ids = mapping_ids(mapping)
    if mapping_digest(mapping) != info["mapping_sha256"]:
        raise RuntimeError("Mapping checksum mismatch.")
    manager = BM25IndexManager(storage / "bm25_index.pkl")
    if not manager.load_index() or mapping_ids(manager.id_mapping) != ids:
        raise RuntimeError("BM25 and mapping are misaligned.")
    with closing(sqlite3.connect(storage / "catalog.sqlite3")) as db:
        db_ids = {str(r[0]) for r in db.execute("SELECT product_id FROM products")}
        n = db.execute("SELECT count(*) FROM reviews").fetchone()[0]
    if db_ids != set(ids) or n != info["review_rows"]:
        raise RuntimeError("Catalog or review count mismatch.")
    if dense:
        header = inspect_flat_faiss(storage / "faiss_index.bin")
        if header["vector_count"] != len(ids) or header["dimension"] != 384:
            raise RuntimeError("FAISS dimension or count mismatch.")


def publish(staging, analytics, limit):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destinations = [(staging, ROOT / "storage"),
                    (analytics, ROOT / "data" / f"analytics_{limit}")]
    saved = []
    installed = []
    try:
        for src, dst in destinations:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                backup = dst.with_name(dst.name + "_backup_" + stamp)
                dst.rename(backup)
                saved.append((backup, dst))
            src.rename(dst)
            installed.append(dst)
    except Exception:
        for dst in reversed(installed):
            shutil.rmtree(dst)
        for backup, dst in reversed(saved):
            backup.rename(dst)
        raise
    for backup, _ in saved:
        print(f"Backup retained: {backup}", flush=True)


def build(data_dir, limit, dense):
    import duckdb
    from src.bm25_index import BM25IndexManager, mapping_digest
    from scripts.build_review_store import PRODUCT_FIELDS, REVIEW_FIELDS
    from src.bm25_index import tokenize_query

    paths = [data_dir / n for n in ("products_clean.parquet", "comments_clean.parquet")]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    print("Reading local cleaned files...", flush=True)
    with TemporaryDirectory(prefix="forouhar_build_", dir=ROOT) as tmp:
        work = Path(tmp)
        storage, analytics = work / "storage", work / "analytics"
        storage.mkdir()
        analytics.mkdir()
        with duckdb.connect(config={"threads": 4, "memory_limit": "2GB"}) as con:
            con.execute("SET temp_directory = ?", [str(work / "spill")])
            con.execute("SET preserve_insertion_order = false")
            psrc, rsrc = [con.read_parquet(str(p), hive_partitioning=False) for p in paths]
            pcols, rcols = set(psrc.columns), set(rsrc.columns)
            psrc.create_view("psrc")
            rsrc.create_view("rsrc")
            print("Selecting products with textual reviews...", flush=True)
            proj_p = product_projection(pcols)
            proj_r = review_projection(rcols)
            # Prefer only textual-review products to ensure Q&A coverage.
            # Hash ordering is deterministic and avoids choosing only the earliest IDs.
            # Project IDs/text separately from unrelated wide source columns.
            con.execute(f"CREATE TEMP VIEW clean_reviews AS SELECT {proj_r} FROM rsrc")
            con.execute("""
                CREATE TEMP TABLE reviewed_ids AS SELECT DISTINCT product_id
                FROM clean_reviews
                WHERE product_id > 0 AND comment_id > 0 AND
                      coalesce(title_raw,body_raw,advantages_raw,disadvantages_raw) IS NOT NULL
            """)
            con.execute(f"""
                CREATE TEMP TABLE chosen AS
                SELECT p.* FROM (SELECT {proj_p} FROM psrc) p
                SEMI JOIN reviewed_ids USING (product_id)
                WHERE p.product_id > 0 AND p.title_fa IS NOT NULL
                  AND p.product_text_lexical IS NOT NULL
                ORDER BY md5(CAST(p.product_id AS VARCHAR)), p.product_id
                LIMIT {limit}
            """)
            result = con.execute("SELECT * FROM chosen ORDER BY product_id")
            cols = [c[0] for c in result.description]
            products = [dict(zip(cols, values)) for values in result.fetchall()]
            if len(products) != limit or len({r["product_id"] for r in products}) != limit:
                raise ValueError(f"Expected {limit:,} distinct products with text reviews; found {len(products):,} rows. Use the complete clean files.")
            # All source-provided reviews for selected IDs, including non-text ratings.
            # Reject duplicate/invalid review IDs instead of dropping them silently.
            print("Preparing matched reviews and analytics tables...", flush=True)
            con.execute("""
                CREATE TEMP TABLE chosen_reviews AS
                SELECT r.* FROM clean_reviews r SEMI JOIN chosen p USING (product_id)
            """)
            review_rows, valid_ids = con.execute(
                "SELECT count(*), count(DISTINCT CASE WHEN comment_id > 0 THEN comment_id END) FROM chosen_reviews"
            ).fetchone()
            if review_rows != valid_ids:
                raise ValueError("Duplicate or invalid review IDs in chosen data. Use the deduplicated clean files.")
            mapping = [{"product_id": str(r["product_id"]), "title_fa": r["title_fa"]} for r in products]
            old = ROOT / "storage" / "catalog_manifest.json"
            previous = json.loads(old.read_text(encoding="utf-8")) if old.exists() else {}
            info = {
                "dataset": previous.get("dataset", "RadeAI/Digikala_comments_products"),
                "dataset_revision": previous.get("dataset_revision", "unknown"),
                "source_products": str(paths[0]),
                "source_reviews": str(paths[1]),
                "selection": "deterministic product-id hash sample of products with at least one textual review",
                "coverage": "All supplied clean reviews for selected product IDs.",
                "product_count": limit,
                "analytics_dir": f"data/analytics_{limit}",
                "mapping_sha256": mapping_digest(mapping),
                "supplied_price_unit_confirmed": any(r["price_toman"] is not None for r in products),
                "dense_available": bool(dense),
            }
            print(f"Building SQLite: {limit:,} products and {review_rows:,} reviews...", flush=True)
            review_cursor = con.execute("SELECT " + ", ".join(REVIEW_FIELDS) + " FROM chosen_reviews")
            populate_catalog(con, products, review_cursor, storage / "catalog.sqlite3", info)
            if info["review_rows"] != review_rows:
                raise RuntimeError("Imported review count is incorrect.")
            # Retain exactly the same selected product/review universe for the charts.
            con.execute("""
                COPY (
                    SELECT CAST(product_id AS VARCHAR) AS product_id, title_fa, category1,
                           category2, sub_category, brand, price_recorded AS price, rating_100,
                           rating_count, product_text_lexical, product_text_norm FROM chosen
                ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
            """, [str(analytics / "products.parquet")])
            con.execute("""
                COPY (
                    SELECT * REPLACE (CAST(comment_id AS VARCHAR) AS comment_id,
                                      CAST(product_id AS VARCHAR) AS product_id)
                      FROM chosen_reviews
                ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
            """, [str(analytics / "reviews.parquet")])
            analytics_counts = tuple(con.execute(
                "SELECT (SELECT count(*) FROM read_parquet(?)), (SELECT count(*) FROM read_parquet(?))",
                [str(analytics / "products.parquet"), str(analytics / "reviews.parquet")]
            ).fetchone())
            if analytics_counts != (limit, review_rows):
                raise RuntimeError("Analytics data counts mismatch.")

        print("Building lexical BM25 index...", flush=True)
        builder = BM25IndexManager(storage / "bm25_index.pkl")
        builder.build_from_records(products, source_metadata={"dataset_revision": info["dataset_revision"]})
        del builder
        dump(storage / "id_mapping.json", mapping)
        if dense:
            import numpy as np
            from src.embedding import DenseEmbedder
            from src.faiss_index import FAISSIndexManager
            print("Building optional FAISS index with local encoder...", flush=True)
            embedder = DenseEmbedder()
            parts = []
            for start in range(0, limit, 2000):
                part = products[start:start + 2000]
                parts.append(embedder.generate_embeddings([r["title_fa"] for r in part], batch_size=32))
                print(f"  Embedded {min(start + len(part), limit):,}/{limit:,}", flush=True)
            vectors = np.concatenate(parts)
            del parts
            FAISSIndexManager(storage / "faiss_index.bin", storage / "id_mapping.json").build_and_save_index(vectors, mapping)
            del vectors
            info["faiss_sha256"] = sha(storage / "faiss_index.bin")
        info.update({
            "catalog_sha256": sha(storage / "catalog.sqlite3"),
            "bm25_sha256": sha(storage / "bm25_index.pkl"),
            "mapping_file_sha256": sha(storage / "id_mapping.json"),
            "analytics_products_sha256": sha(analytics / "products.parquet"),
            "analytics_reviews_sha256": sha(analytics / "reviews.parquet"),
        })
        dump(storage / "catalog_manifest.json", info)
        print("Validating indexes and data alignment...", flush=True)
        validate(storage, mapping, info, dense)
        publish(storage, analytics, limit)
    print(json.dumps({
        "status": "PASS", "indexed_products": limit, "review_rows": info["review_rows"],
        "products_with_text_reviews": info["products_with_text_reviews"],
        "dense_available": dense, "api_calls": 0,
        "analytics_dir": str(ROOT / "data" / f"analytics_{limit}"),
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "processed_repo_based")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--dense", action="store_true", help="Also build aligned FAISS with the local encoder.")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if not (ROOT / "app.py").is_file() or not (ROOT / "scripts" / "build_review_store.py").is_file():
        parser.error("Put this script beside app.py in the Forouhar_Pink_Blue project.")
    build(args.data_dir.expanduser().resolve(), args.limit, args.dense)


if __name__ == "__main__":
    main()
