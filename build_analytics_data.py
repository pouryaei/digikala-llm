"""Normalize the EDA text and publish products.parquet and reviews.parquet."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

from analytics_service import (
    CLEAN_FILENAMES,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    FINAL_FILENAMES,
    open_analytics_tables,
)


PRODUCT_TEXT = (
    "title_fa", "category1", "category2", "sub_category", "brand", "seller", "product_text_norm",
)
REVIEW_TEXT = (
    "title_clean", "body_clean", "advantages_clean", "disadvantages_clean", "seller_title", "text_clean",
)

TEXT_NORMALIZATION = r"""
    CREATE MACRO m1_collapse_spaces(value) AS
        regexp_replace(value, '[[:space:]\p{Z}\x{200B}]+', ' ', 'g');
    CREATE MACRO m1_collapse_joiners(value) AS
        regexp_replace(value, '( *\x{200C})+ *', chr(8204), 'g');
    CREATE MACRO m1_normalize_text(value) AS nullif(trim(regexp_replace(regexp_replace(
        m1_collapse_joiners(m1_collapse_spaces(
            translate(nfc_normalize(value), 'يىك٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹', 'ییک01234567890123456789')
        )), '(^|[^\p{L}\p{M}\p{N}])\x{200C}', '\1', 'g'
    ), '\x{200C}($|[^\p{L}\p{M}\p{N}])', '\1', 'g')), '');

    CREATE MACRO m1_comparable_text(value) AS
        regexp_replace(lower(value), '[^\p{L}\p{N}\p{So}]', '', 'g');
    CREATE MACRO m1_word_count(value) AS
        len(regexp_extract_all(value, '[\p{L}\p{N}][\p{L}\p{N}\x{200C}]*'));
"""


def _write_parquet(con: duckdb.DuckDBPyConnection, table: str, destination: Path) -> int:
    columns = set(con.table(table).columns)
    text_columns = PRODUCT_TEXT if table == "products" else REVIEW_TEXT
    normalization = ", ".join(
        f"m1_normalize_text({name}) AS {name}" for name in text_columns if name in columns
    )
    query = f"SELECT * REPLACE ({normalization}) FROM {table}"

    if table == "products":
        derived = {}
        if "product_text_norm" in columns:
            derived["product_text_lexical"] = (
                r"nullif(array_to_string(regexp_extract_all(lower(product_text_norm), '[\p{L}\p{N}]+'), ' '), '')"
            )
    else:
        aspects = [
            f"coalesce({name}, '')" for name in ("advantages_clean", "disadvantages_clean")
            if name in columns
        ]
        aspects_sql = " || ' ' || ".join(aspects) if aspects else "''"
        derived = {
            "text_key": "md5(nullif(m1_comparable_text(text_clean), ''))",
            "word_count": "m1_word_count(text_clean)",
            "rag_eligible": (
                "coalesce((m1_word_count(text_clean) >= 2 AND length(m1_comparable_text(text_clean)) >= 8) "
                f"OR regexp_matches({aspects_sql}, '[\\p{{L}}]'), false)"
            ),
        }
    replacements = ", ".join(f"{expr} AS {key}" for key, expr in derived.items() if key in columns)
    if replacements:
        query = f"SELECT * REPLACE ({replacements}) FROM ({query}) AS normalized"

    return con.execute(f"COPY ({query}) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
                       [str(destination)]).fetchone()[0]


def build_analytics_data(source_dir: Path, output_dir: Path) -> dict[str, int]:
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output folder already exists: {output_dir}")

    products, reviews = (source_dir / name for name in CLEAN_FILENAMES)
    for path in (products, reviews):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".m1_", dir=output_dir.parent) as staging:
        snapshot = Path(staging) / "snapshot"
        snapshot.mkdir()
        counts = {}
        with open_analytics_tables((products, reviews)) as con:
            con.execute("SET preserve_insertion_order = false")
            con.execute("SET temp_directory = ?", [str(Path(staging) / "spill")])
            con.execute("SELECT title_fa, category1, brand, price FROM products LIMIT 0")
            con.execute("SELECT product_id, rating_5, text_clean FROM reviews LIMIT 0")
            for table, key in (("products", "product_id"), ("reviews", "comment_id")):
                rows, unique_ids = con.execute(f"""
                    SELECT count(*), count(DISTINCT nullif(trim(CAST({key} AS VARCHAR)), ''))
                      FROM {table}
                """).fetchone()
                if unique_ids != rows:
                    raise ValueError(f"{table} contains missing or duplicate {key} values")
                counts[table] = rows

            orphan = con.execute("""
                SELECT r.comment_id FROM reviews r ANTI JOIN products p USING (product_id)
                LIMIT 1
            """).fetchone()
            if orphan is not None:
                raise ValueError(f"Review {orphan[0]} has no matching product")

            con.execute(TEXT_NORMALIZATION)
            for table, name in zip(counts, FINAL_FILENAMES):
                if _write_parquet(con, table, snapshot / name) != counts[table]:
                    raise RuntimeError(f"Unexpected row count in {name}")

        snapshot.rename(output_dir)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        counts = build_analytics_data(args.source_dir, args.output_dir)
    except (OSError, ValueError, RuntimeError, duckdb.Error) as exc:
        parser.exit(1, f"{exc}\n")
    for name, count in zip(FINAL_FILENAMES, counts.values()):
        print(f"{name}: {count:,} rows")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
