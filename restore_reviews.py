"""Restore the complete review table from the data included in this ZIP."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

from build_analytics_data import TEXT_NORMALIZATION


TEXT_REFERENCES = {
    "title_clean": "m1_normalize_text(title_raw)",
    "body_clean": "m1_normalize_text(body_raw)",
    "advantages_clean": "m1_normalize_text(advantages_raw)",
    "disadvantages_clean": "m1_normalize_text(disadvantages_raw)",
}
COMBINED_TEXT = "nullif(concat_ws(' . ', title_clean, body_clean, advantages_clean, disadvantages_clean), '')"
DERIVED_FIELDS = {
    "word_count": "m1_word_count(text_clean)",
    "text_key": "md5(nullif(m1_comparable_text(text_clean), ''))",
    "rag_eligible": r"""coalesce(
        (m1_word_count(text_clean) >= 2 AND length(m1_comparable_text(text_clean)) >= 8)
        OR regexp_matches(coalesce(advantages_clean, '') || ' ' || coalesce(disadvantages_clean, ''), '[\p{L}]'),
        false)""",
}


def decode_text_field(name: str, reference: str) -> str:
    # A null marker repeats the reference, 0 represents null, and 1 prefixes a literal.
    return f"CASE WHEN {name} IS NULL THEN {reference} WHEN {name} = '0' THEN NULL ELSE {name}[2:] END AS {name}"


def reviews_query(column_names: list[str]) -> str:
    text = ", ".join(decode_text_field(name, reference) for name, reference in TEXT_REFERENCES.items())
    combined = decode_text_field("text_clean", COMBINED_TEXT)
    projections = []
    for name in column_names:
        quoted = '"' + name.replace('"', '""') + '"'
        expression = DERIVED_FIELDS.get(name, quoted)
        projections.append(f"{expression} AS {quoted}")
    return f"""
        WITH components AS (SELECT * REPLACE ({text}) FROM review_payload),
             normalized AS (SELECT * REPLACE ({combined}) FROM components)
        SELECT {', '.join(projections)} FROM normalized
    """


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_reviews(project_dir: Path) -> Path:
    directory = project_dir / "data" / "analytics"
    output = directory / "reviews.parquet"
    if output.is_file():
        return output

    payload = directory / "reviews_payload.csv.zst"
    manifest = json.loads((directory / "reviews_payload.json").read_text(encoding="utf-8"))
    if file_sha256(payload) != manifest["sha256"]:
        raise ValueError("The review data in the ZIP did not match its checksum")

    columns = dict(manifest["columns"])
    stored_columns = {name: dtype for name, dtype in columns.items() if name not in DERIVED_FIELDS}
    print(f"Restoring {manifest['row_count']:,} reviews from the local bundle...", flush=True)
    with TemporaryDirectory(prefix=".restore_reviews_", dir=directory) as staging:
        partial = Path(staging) / output.name
        with duckdb.connect(config={"threads": 4, "memory_limit": "2GB"}) as con:
            con.execute(TEXT_NORMALIZATION)
            con.sql("""
                SELECT * FROM read_csv(?, columns=?, header=true, auto_detect=false,
                    delim=',', quote='"', escape='"', nullstr='', allow_quoted_nulls=false,
                    compression='zstd')
            """, params=[str(payload), stored_columns]).create_view("review_payload")
            rows = con.execute(
                f"COPY ({reviews_query(list(columns))}) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
                [str(partial)],
            ).fetchone()[0]
        if rows != manifest["row_count"]:
            raise ValueError(f"Unexpected review row count: {rows:,}")
        partial.replace(output)
    return output


if __name__ == "__main__":
    try:
        print(restore_reviews(Path(__file__).resolve().parent))
    except (OSError, ValueError, duckdb.Error) as error:
        raise SystemExit(str(error)) from error
