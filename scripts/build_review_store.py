"""Build the portable RAG catalogue and its complete in-scope review store."""
from __future__ import annotations

import argparse
from contextlib import closing
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from time import perf_counter
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bm25_index import BM25IndexManager, mapping_digest, mapping_ids
from src.task2_review_store import FIELDS, canonical_review, identifier

REVISION = "89c3133b169c8d3793db8834f56f32fee33d9db0"
PRODUCT_MEMBER = "data/analytics/products.parquet"
REVIEW_MEMBER = "data/analytics/reviews_payload.csv.zst"
MANIFEST_MEMBER = "data/analytics/reviews_payload.json"
PRODUCT_FIELDS = (
    "product_id", "title_fa", "category1", "category2", "sub_category", "brand",
    "price_recorded", "price_toman", "price_conflict", "rating_100", "rating_count",
    "product_text_lexical", "product_text_norm",
)
REVIEW_FIELDS = ("comment_id", "product_id", *FIELDS, "rating_5", "recommendation_status")
SCHEMA = """
PRAGMA user_version=2;
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY, title_fa TEXT NOT NULL,
    category1 TEXT, category2 TEXT, sub_category TEXT, brand TEXT,
    price_recorded REAL, price_toman REAL, price_conflict INTEGER NOT NULL,
    rating_100 REAL, rating_count INTEGER,
    product_text_lexical TEXT NOT NULL, product_text_norm TEXT NOT NULL
);
CREATE TABLE reviews (
    comment_id INTEGER PRIMARY KEY, product_id INTEGER NOT NULL REFERENCES products,
    title_raw TEXT, body_raw TEXT, advantages_raw TEXT, disadvantages_raw TEXT,
    rating_5 REAL, recommendation_status TEXT
);
CREATE INDEX reviews_product ON reviews(product_id);
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _finite(value, low, high=None):
    return value if value is not None and math.isfinite(value) and value >= low and (high is None or value <= high) else None


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _catalogue(path, mapping):
    import duckdb
    ids = mapping_ids(mapping)
    columns = ", ".join("price AS price_recorded" if k == "price_recorded" else k for k in PRODUCT_FIELDS)
    with duckdb.connect() as con:
        rows = con.execute(
            "SELECT " + columns + ", price_valid FROM read_parquet(?) "
            "WHERE CAST(product_id AS VARCHAR) IN (SELECT UNNEST(?))", [str(path), ids]
        ).fetchall()
    by_id = {str(row[0]): dict(zip((*PRODUCT_FIELDS, "price_valid"), row)) for row in rows}
    if len(rows) != len(ids) or set(by_id) != set(ids):
        raise ValueError("Clean product file must supply exactly one row per indexed product.")
    for m in mapping:
        pid = str(m["product_id"])
        data = by_id[pid]
        data["product_id"] = identifier(pid)
        data["title_fa"] = m["title_fa"]
        data["price_recorded"] = _finite(data["price_recorded"], 1) if data["price_valid"] else None
        data["price_toman"] = _finite(data["price_toman"], 1) if data["price_valid"] else None
        data["rating_100"] = _finite(data["rating_100"], 0, 100) if data["rating_count"] else None
        data["price_conflict"] = int(bool(data["price_conflict"]))
        if not data["product_text_lexical"] or not data["product_text_norm"]:
            raise ValueError(f"Missing clean product text: {pid}")
        yield data


class _HashedStream:
    def __init__(self, raw):
        self.raw = raw
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        data = self.raw.read(size)
        self.digest.update(data)
        return data


def _import_reviews(archive, con, product_ids):
    import zstandard
    expected = json.loads(archive.read(MANIFEST_MEMBER))
    wanted = set(REVIEW_FIELDS)
    count = saved = with_text = 0
    batch = []
    csv.field_size_limit(10_000_000)
    insert = "INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?)"
    with archive.open(REVIEW_MEMBER) as raw:
        hashed = _HashedStream(raw)
        with zstandard.ZstdDecompressor().stream_reader(hashed, read_size=1024 * 1024) as dec:
            with io.TextIOWrapper(dec, encoding="utf-8", newline="") as text:
                rows = csv.reader(text)
                names = next(rows)
                if not wanted <= set(names):
                    raise ValueError("Review payload is missing the required original text columns.")
                fields = {f: names.index(f) for f in REVIEW_FIELDS}
                pid_index = fields["product_id"]
                for row in rows:
                    count += 1
                    if row[pid_index] not in product_ids:
                        continue
                    review = canonical_review({field: row[idx] for field, idx in fields.items()})
                    batch.append(tuple(review[f] for f in REVIEW_FIELDS))
                    saved += 1
                    with_text += any(review[f] is not None for f in FIELDS)
                    if len(batch) >= 5000:
                        con.executemany(insert, batch)
                        batch.clear()
                if batch:
                    con.executemany(insert, batch)
    actual_hash = hashed.digest.hexdigest()
    if actual_hash != expected["sha256"] or count != expected["row_count"]:
        raise ValueError("Review payload checksum or source row count does not match its manifest.")
    return {
        "source_review_rows": count, "source_review_sha256": actual_hash,
        "review_rows": saved, "rows_with_review_text": with_text,
    }


def build(archive_path, storage, *, overwrite=False):
    path = Path(archive_path).resolve(strict=True)
    storage = Path(storage).resolve()
    mapping_path = storage / "id_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8-sig"))
    ids = mapping_ids(mapping)
    db, lexical, manifest = (storage / n for n in ("catalog.sqlite3", "bm25_index.pkl", "catalog_manifest.json"))
    if not overwrite and any(p.exists() for p in (db, lexical, manifest)):
        raise FileExistsError("Output already exists; use --overwrite only to intentionally rebuild these files.")
    t0 = perf_counter()
    with tempfile.TemporaryDirectory(prefix="rag-build-", dir=storage) as work, ZipFile(path) as archive:
        folder = Path(work)
        parquet = folder / "products.parquet"
        with archive.open(PRODUCT_MEMBER) as src, parquet.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        products = list(_catalogue(parquet, mapping))
        info = {
            "dataset": "RadeAI/Digikala_comments_products", "dataset_revision": REVISION,
            "source_archive": path.name, "source_products_sha256": _sha(parquet),
            "product_count": len(ids), "mapping_sha256": mapping_digest(mapping),
            "coverage": "All clean reviews of only the product IDs in the bundled original index.",
            "mapping_file_sha256": _sha(mapping_path), "faiss_sha256": _sha(storage / "faiss_index.bin"),
        }
        info["supplied_price_unit_confirmed"] = any(p["price_toman"] is not None for p in products)
        db_out, lexical_out = folder / db.name, folder / lexical.name
        with closing(sqlite3.connect(db_out)) as con:
            con.execute("PRAGMA foreign_keys=ON")
            con.executescript(SCHEMA)
            with con:
                con.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                [tuple(r[k] for k in PRODUCT_FIELDS) for r in products])
                info.update(_import_reviews(archive, con, set(ids)))
                info["products_with_reviews"] = con.execute("SELECT COUNT(DISTINCT product_id) FROM reviews").fetchone()[0]
                info["products_with_text_reviews"] = con.execute(
                    "SELECT COUNT(DISTINCT product_id) FROM reviews WHERE "
                    "COALESCE(title_raw,body_raw,advantages_raw,disadvantages_raw) IS NOT NULL"
                ).fetchone()[0]
                con.executemany("INSERT INTO metadata VALUES (?,?)",
                                [(k, json.dumps(v, ensure_ascii=False)) for k, v in info.items()])
            if con.execute("PRAGMA integrity_check").fetchone() != ("ok",) or con.execute("PRAGMA foreign_key_check").fetchone():
                raise RuntimeError("Catalogue SQLite integrity check failed.")
            con.execute("VACUUM")
        BM25IndexManager(lexical_out).build_from_records(products, source_metadata={"dataset_revision": REVISION})
        info.update({"catalog_sha256": _sha(db_out), "bm25_sha256": _sha(lexical_out)})
        db_out.replace(db)
        lexical_out.replace(lexical)
        manifest.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**info, "build_seconds": round(perf_counter()-t0, 3)}, ensure_ascii=False, indent=2))
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Path to Forouhar_M1_Complete.zip")
    parser.add_argument("--storage", type=Path, default=ROOT / "storage")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build(args.archive, args.storage, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
