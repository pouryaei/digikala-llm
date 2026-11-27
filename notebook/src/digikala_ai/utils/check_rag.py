"""Validate the bundled real corpus and run an API-free search-to-review smoke check."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.catalog import open_catalog
from src.rag import compare_products, find_product_candidates, run_rag_pipeline
from src.search_debug_tools import validate_assets
from src.task2_review_store import FIELDS, SQLiteReviewStore


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_manifest(root=ROOT):
    return json.loads((Path(root) / "storage/catalog_manifest.json").read_text(encoding="utf-8"))


def analytics_dir(root, manifest):
    root = Path(root)
    return root / "data" / f"analytics_{manifest['product_count']}"


def _pair_digest(cursor):
    """Hash sorted (comment_id, product_id) pairs without loading rows into RAM."""
    pack = struct.Struct("<QQ").pack
    digest = hashlib.sha256()
    n = 0
    while batch := cursor.fetchmany(20000):
        for comment_id, product_id in batch:
            digest.update(pack(int(comment_id), int(product_id)))
        n += len(batch)
    return n, digest.hexdigest()


def verify_files(root=ROOT, *, deep=False):
    import duckdb

    root = Path(root)
    storage = root / "storage"
    manifest = read_manifest(root)
    count = manifest["product_count"]
    expected_reviews = manifest["review_rows"]
    dense_path = storage / "faiss_index.bin"
    dense_available = bool(manifest.get("dense_available", dense_path.is_file()))
    for name, key in (("catalog.sqlite3", "catalog_sha256"), ("bm25_index.pkl", "bm25_sha256"),
                      ("id_mapping.json", "mapping_file_sha256")):
        if sha256(storage / name) != manifest[key]:
            raise ValueError(f"Checksum mismatch: {name}")
    if dense_available:
        if not dense_path.is_file() or sha256(dense_path) != manifest.get("faiss_sha256"):
            raise ValueError("Missing or misaligned dense index")
    elif dense_path.exists():
        raise ValueError("An unregistered vector index must not be mixed with the current catalogue.")
    manager, mapping, assets = validate_assets(root)
    del manager
    if assets["mapping_sha256"] != manifest["mapping_sha256"] or len(mapping) != count:
        raise ValueError("Catalogue and search asset metadata disagree.")
    dense_info = assets["faiss"]
    if dense_available and (not dense_info["header_verified"] or dense_info["vector_count"] != count
                            or dense_info["dimension"] != 384):
        raise ValueError("Dense index is not aligned with the current catalogue.")
    catalog = open_catalog(str(storage / "catalog.sqlite3"))
    with catalog.connection() as con:
        if con.execute("PRAGMA quick_check").fetchone()[0] != "ok" or con.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Catalogue integrity check failed.")
        product_ids = {str(row[0]) for row in con.execute("SELECT product_id FROM products")}
        reviews, reviewed_products = con.execute("SELECT COUNT(*), COUNT(DISTINCT product_id) FROM reviews").fetchone()
        text = "coalesce(title_raw,body_raw,advantages_raw,disadvantages_raw) IS NOT NULL"
        rows_with_text, products_with_text = con.execute(
            f"SELECT COUNT(*),COUNT(DISTINCT product_id) FROM reviews WHERE {text}"
        ).fetchone()
        sqlite_pairs = _pair_digest(con.execute("SELECT comment_id,product_id FROM reviews ORDER BY comment_id")) if deep else None
    if (product_ids != {str(row["product_id"]) for row in mapping} or reviews != expected_reviews
            or reviewed_products != manifest["products_with_reviews"]
            or products_with_text != manifest["products_with_text_reviews"]
            or rows_with_text != manifest["rows_with_review_text"]):
        raise ValueError("Catalogue IDs or review counts do not match the build manifest.")
    folder = analytics_dir(root, manifest)
    paths = [folder / f"{name}.parquet" for name in ("products", "reviews")]
    for name, path in zip(("products", "reviews"), paths):
        expected = manifest.get(f"analytics_{name}_sha256")
        if not path.is_file() or expected and sha256(path) != expected:
            raise ValueError(f"Missing or changed analytics file: {path.name}")
    with duckdb.connect(config={"threads": 4, "memory_limit": "2GB"}) as con:
        con.read_parquet(str(paths[0]), hive_partitioning=False).create_view("p")
        con.read_parquet(str(paths[1]), hive_partitioning=False).create_view("r")
        pc, pn = con.execute("SELECT COUNT(*),COUNT(DISTINCT product_id) FROM p").fetchone()
        rc, rn, rc_products = con.execute("SELECT COUNT(*),COUNT(DISTINCT comment_id),COUNT(DISTINCT product_id) FROM r").fetchone()
        orphan = con.execute("SELECT COUNT(*) FROM r LEFT JOIN p USING (product_id) WHERE p.product_id IS NULL").fetchone()[0]
        pids = {str(row[0]) for row in con.execute("SELECT product_id FROM p").fetchall()}
        if (pc, pn, rc, rn, rc_products, orphan) != (count, count, reviews, reviews, reviewed_products, 0) or pids != product_ids:
            raise ValueError("Analytics tables are not aligned to the search catalogue.")
        parquet_pairs = _pair_digest(con.execute(
            "SELECT CAST(comment_id AS BIGINT),CAST(product_id AS BIGINT) FROM r ORDER BY CAST(comment_id AS BIGINT)")) if deep else None
    if deep and sqlite_pairs != parquet_pairs:
        raise ValueError("SQLite and Parquet contain different (comment_id, product_id) pairs.")
    return {"products": len(mapping), "bm25_documents": assets["bm25_count"],
            "reviews": reviews, "products_with_reviews": reviewed_products,
            "rows_with_review_text": rows_with_text, "products_with_text_reviews": products_with_text,
            "analytics_products": pc, "analytics_reviews": rc, "analytics_orphan_reviews": orphan,
            "same_ids_and_order": True, "exact_review_pairs_verified": deep,
            "dense_available": dense_available, "faiss_vectors": dense_info["vector_count"] if dense_available else None,
            "faiss_dimensions": dense_info["dimension"] if dense_available else None}


def demo_examples():
    """Use the recorded real examples, or discover valid ones after a rebuild."""
    info = read_manifest()
    if info.get("demo"):
        return info["demo"]
    with open_catalog().connection() as con:
        rows = con.execute("""SELECT p.product_id,p.title_fa,p.category1,count(*) AS n
            FROM products p JOIN reviews r USING (product_id)
            WHERE length(p.title_fa) BETWEEN 15 AND 150
            GROUP BY p.product_id,p.title_fa,p.category1 HAVING n BETWEEN 10 AND 250
            ORDER BY p.product_id LIMIT 80""").fetchall()
    picks = []
    question = "ایرادهای پرتکرار این محصول چیست؟"
    for pid, title, category, n in rows:
        hits = find_product_candidates(title, top_k=3)
        if not hits or int(hits[0]["product_id"]) != pid:
            continue
        answer = run_rag_pipeline(question, pid, generation_mode="extractive", source=ROOT / "storage/catalog.sqlite3")
        if answer["status"] == "answered" and answer["evidence"]:
            picks.append({"product_id": pid, "title_fa": title, "category": category,
                          "review_rows": n, "evidence_count": len(answer["evidence"])})
        if len(picks) == 2:
            return {"primary": picks[0], "comparison": picks[1], "question": question}
    raise ValueError("No suitable real demonstration products were found in the rebuilt set.")


def demo():
    examples = demo_examples()
    primary, secondary = examples["primary"], examples["comparison"]
    pid, title = primary["product_id"], primary["title_fa"]
    other = secondary["product_id"]
    path = ROOT / "storage/catalog.sqlite3"
    started = perf_counter()
    results = find_product_candidates(title, top_k=3)
    if not results or int(results[0]["product_id"]) != pid:
        raise ValueError("Known item search did not return the demonstration product first.")
    search_seconds = perf_counter() - started
    started = perf_counter()
    answer = run_rag_pipeline(examples["question"], pid, generation_mode="extractive", source=path)
    qa_seconds = perf_counter() - started
    if answer["product_id"] != pid or answer["status"] != "answered" or answer["stats"]["total_reviews"] != primary["review_rows"]:
        raise ValueError("Search-to-QA result or exact source review count failed.")
    store = SQLiteReviewStore(path)
    source = {row["comment_id"]: row for row in store.get(pid).reviews}
    if not answer["evidence"] or any(
        e["product_id"] != pid or e["comment_id"] not in source
        or not any(e["text"] in (source[e["comment_id"]].get(field) or "") for field in FIELDS)
        for e in answer["evidence"]
    ):
        raise ValueError("Review answer contains missing, cross-product, or non-verbatim evidence.")
    with open_catalog().connection() as con:
        missing = con.execute("SELECT MAX(product_id)+1 FROM products").fetchone()[0]
    empty = run_rag_pipeline("کیفیت محصول چگونه است؟", missing, generation_mode="extractive", source=path)
    ambiguous = run_rag_pipeline("ایرادهای این محصول چیست؟", generation_mode="extractive", source=path)
    comparison = compare_products([pid, other], top_k=2, source=path)
    if empty["status"] != "insufficient_evidence" or ambiguous["status"] != "needs_product" or comparison["status"] != "compared":
        raise ValueError("Out-of-scope, ambiguous product, or comparison handling failed.")
    for item in comparison["products"]:
        rows = {r["comment_id"]: r for r in store.get(item["product_id"]).reviews}
        for evidence in item["strengths"] + item["weaknesses"] + item["related_reviews"]:
            if evidence["product_id"] != item["product_id"] or evidence["comment_id"] not in rows:
                raise ValueError("Comparison has a cross-product citation.")
    return {"product_id": pid, "title_fa": title, "verified_reviews": len(source), "evidence": len(answer["evidence"]),
            "comparison_product_id": other, "out_of_scope_abstention": True, "ambiguous_product_abstention": True,
            "comparison_is_product_scoped": True, "search_seconds": round(search_seconds, 6),
            "qa_seconds": round(qa_seconds, 6), "real_api_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep", action="store_true", help="Compare every review ID and product ID across storage and analytics.")
    args = parser.parse_args()
    report = {"assets": verify_files(deep=args.deep), "demo": demo()}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("PASS: packaged search, review QA, comparison and analytics scope; no API calls.")


if __name__ == "__main__":
    main()
