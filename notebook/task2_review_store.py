"""Product-scoped access to real Digikala reviews.

DuckDB/Parquet is the production path. The verified JSONL sample and recovery
payload are optional portable sources; neither is silently substituted for the
full catalogue. No raw reviews from a different product are returned.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Iterable
from zipfile import ZipFile, BadZipFile

ROOT = Path(__file__).resolve().parents[1]
MAX_ID = 2**63 - 1
DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
TEXT_ALIASES = {
    "title_raw": ("title_raw", "title", "title_clean"),
    "body_raw": ("body_raw", "body", "body_clean", "text_clean"),
    "advantages_raw": ("advantages_raw", "advantages", "advantages_clean"),
    "disadvantages_raw": ("disadvantages_raw", "disadvantages", "disadvantages_clean"),
}
FIELDS = tuple(TEXT_ALIASES)
VALID_LABELS = {"recommended", "not_recommended", "no_idea"}
NULL_TEXT = {"", "null", "none", "nan", "n/a", "-", "[]"}


class ReviewDataError(RuntimeError):
    """Expected error: missing, incompatible, or unavailable review data."""


def identifier(value) -> int:
    text = str(value).strip().translate(DIGITS)
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]{1,19}", text):
        raise ValueError("شناسهٔ محصول یا نظر معتبر نیست.")
    value = int(text)
    if not 1 <= value <= MAX_ID:
        raise ValueError("شناسهٔ محصول یا نظر معتبر نیست.")
    return value


def _text(value) -> str | None:
    if not isinstance(value, str) or value.strip().casefold() in NULL_TEXT:
        return None
    return value


def _rating(value):
    try:
        x = float(value)
        return x if 0 <= x <= 5 else None
    except (TypeError, ValueError):
        return None


def canonical_review(row: dict) -> dict:
    """Retain original text bytes after decoding, not a generated quotation."""
    cid = identifier(row.get("comment_id", row.get("id")))
    pid = identifier(row.get("product_id"))
    result = {"comment_id": cid, "product_id": pid}
    old_sources = row.get("source_fields") or {}
    sources = {}
    for canonical, aliases in TEXT_ALIASES.items():
        key = next((x for x in aliases if _text(row.get(x)) is not None), None)
        result[canonical] = _text(row.get(key)) if key else None
        if key: sources[canonical] = old_sources.get(canonical, key)
    result["source_fields"] = sources
    result["rating_5"] = _rating(row.get("rating_5", row.get("rate")))
    label = row.get("recommendation_status")
    result["recommendation_status"] = label if isinstance(label, str) and label in VALID_LABELS else None
    return result


def unique_product_reviews(rows: Iterable[dict], product_id: int) -> list[dict]:
    product_id = identifier(product_id)
    found: dict[int, dict] = {}
    for row in rows:
        try:
            review = canonical_review(dict(row))
        except (ValueError, TypeError):
            continue
        if review["product_id"] != product_id:
            continue
        old = found.get(review["comment_id"])
        # Prefer the more informative record if duplicate IDs occur in an input.
        if old is None or sum(len(review.get(f) or "") for f in FIELDS) > sum(len(old.get(f) or "") for f in FIELDS):
            found[review["comment_id"]] = review
    return list(found.values())


@dataclass
class ReviewBatch:
    product_id: int
    reviews: list[dict]
    scope: str
    complete_for_product: bool
    source: str
    title: str | None = None
    status: str = "ok"


class MemoryReviewStore:
    def __init__(self, reviews, *, complete=False, scope="provided_reviews"):
        if hasattr(reviews, "to_dict"):
            reviews = reviews.to_dict(orient="records")
        self.rows = list(reviews)
        self.complete = complete
        self.scope = scope

    def get(self, product_id):
        pid = identifier(product_id)
        return ReviewBatch(pid, unique_product_reviews(self.rows, pid), self.scope, self.complete, "provided_reviews")


class JsonlReviewStore:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        manifest_path = self.path.with_name("manifest.json")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        expected_hash = self.manifest.get("reviews_sha256")
        if expected_hash and hashlib.sha256(self.path.read_bytes()).hexdigest() != expected_hash:
            raise ReviewDataError("محتوای نمونه با گزارش صحت داده مطابقت ندارد.")
        self.products = {identifier(r["product_id"]): r for r in self.manifest.get("products", [])}
        self.by_product = defaultdict(list)
        with self.path.open(encoding="utf-8-sig") as fh:
            for line in fh:
                if line.strip():
                    row = canonical_review(json.loads(line))
                    self.by_product[row["product_id"]].append(row)
        self.complete = bool(self.manifest.get("complete_for_listed_product_ids"))
        if self.complete:
            if sum(map(len, self.by_product.values())) != self.manifest.get("row_count"):
                raise ReviewDataError("تعداد سطرهای نمونه با گزارش آن مطابقت ندارد.")
            if not set(self.by_product) <= self.products.keys():
                raise ReviewDataError("نمونه شامل محصول خارج از محدوده است.")
            for pid, info in self.products.items():
                rows = self.by_product[pid]
                if len(rows) != info.get("review_count") or len({r["comment_id"] for r in rows}) != len(rows):
                    raise ReviewDataError("تعداد یا یکتایی نظرهای محصول در نمونه معتبر نیست.")

    def get(self, product_id):
        pid = identifier(product_id)
        if self.complete and pid not in self.products:
            return ReviewBatch(pid, [], "verified_real_sample", False, self.path.name, status="outside_sample")
        info = self.products.get(pid, {})
        return ReviewBatch(pid, unique_product_reviews(self.by_product.get(pid, []), pid),
                           "verified_real_sample" if self.complete else "provided_jsonl",
                           self.complete and pid in self.products, self.path.name, info.get("title"))


def _duckdb():
    try:
        import duckdb
        return duckdb
    except ImportError:
        raise ReviewDataError("برای خواندن Parquet، duckdb را نصب کن؛ یا مسیر نمونه/بستهٔ بازیابی نظرها را تعیین کن.") from None


def _column_map(columns):
    cset = set(columns)
    result = {"comment_id": next((x for x in ("comment_id", "id") if x in cset), None),
              "product_id": "product_id" if "product_id" in cset else None,
              "rating_5": next((x for x in ("rating_5", "rate", "rating") if x in cset), None),
              "recommendation_status": "recommendation_status" if "recommendation_status" in cset else None}
    result.update({name: next((x for x in aliases if x in cset), None) for name, aliases in TEXT_ALIASES.items()})
    if not result["comment_id"] or not result["product_id"] or not any(result[f] for f in FIELDS):
        raise ReviewDataError("فایل نظرات دارای شناسه و متن واقعی نظر نیست؛ فایل سبکِ صرفاً امتیاز/برچسب برای این تسک کافی نیست.")
    return result


class ParquetReviewStore:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        duckdb = _duckdb()
        try:
            with duckdb.connect(config={"threads": 4, "memory_limit": "2GB"}) as con:
                rel = con.read_parquet(str(self.path), hive_partitioning=False)
                self.mapping = _column_map(rel.columns)
                self.columns = set(rel.columns)
                self.product_type = str(rel.types[rel.columns.index(self.mapping["product_id"])])
        except ReviewDataError:
            raise
        except (OSError, duckdb.Error) as exc:
            raise ReviewDataError("خواندن فایل Parquet نظرات ممکن نبود؛ سالم بودن فایل و سازگاری نسخهٔ DuckDB را بررسی کن.") from exc

    def get(self, product_id):
        pid = identifier(product_id)
        duckdb = _duckdb()
        projections = []
        for canonical in ("comment_id", "product_id", "rating_5", "recommendation_status"):
            original = self.mapping[canonical]
            # Names come only from the fixed alias list, never from user input.
            projections.append(f'"{original}" AS "{canonical}"' if original else f'NULL AS "{canonical}"')
        # Select actual named text columns so a row-level NULL raw field can
        # fall back to the cleaned source column and report its actual name.
        available_text = sorted({alias for aliases in TEXT_ALIASES.values() for alias in aliases if alias in self.columns})
        projections.extend(f'"{name}"' for name in available_text)
        pid_col = self.mapping["product_id"]
        condition = f'"{pid_col}" = ?'
        param = str(pid) if "CHAR" in self.product_type.upper() or "STRING" in self.product_type.upper() else pid
        try:
            with duckdb.connect(config={"threads": 4, "memory_limit": "2GB"}) as con:
                con.read_parquet(str(self.path), hive_partitioning=False).create_view("task2_reviews")
                result = con.execute(f'SELECT {", ".join(projections)} FROM task2_reviews WHERE {condition} ORDER BY comment_id', [param])
                names = [x[0] for x in result.description]
                rows = [dict(zip(names, row)) for row in result.fetchall()]
        except (OSError, duckdb.Error) as exc:
            raise ReviewDataError("خواندن نظرهای محصول از Parquet ناموفق بود.") from exc
        return ReviewBatch(pid, unique_product_reviews(rows, pid), "local_full_review_file", True, self.path.name)


class PayloadReviewStore:
    """Read the original zstd CSV payload (also directly from M1_Complete.zip).

    Initial access scans the payload once. Per-product rows are cached by source
    version so later questions about that product do not repeat the full scan.
    For regular use, the DuckDB/Parquet source is preferable.
    """
    def __init__(self, path, cache_dir=None):
        self.path = Path(path).expanduser().resolve()
        self.is_zip = self.path.suffix.lower() == ".zip"
        self.payload_member = "data/analytics/reviews_payload.csv.zst"
        if self.is_zip:
            try:
                with ZipFile(self.path) as archive:
                    self.manifest = json.loads(archive.read("data/analytics/reviews_payload.json"))
                    archive.getinfo(self.payload_member)
            except (OSError, KeyError, ValueError, BadZipFile) as exc:
                raise ReviewDataError("بستهٔ بازیابی نظرها فایل داده یا manifest معتبر ندارد.") from exc
        else:
            manifest_path = self.path.with_name("reviews_payload.json")
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        version = [str(self.path), self.path.stat().st_size, self.path.stat().st_mtime_ns, self.manifest.get("sha256"), "canonical-v2"]
        key = hashlib.sha256(json.dumps(version, ensure_ascii=False).encode()).hexdigest()[:24]
        self.cache_dir = Path(cache_dir or os.environ.get("QBC13_REVIEW_DATA_CACHE", ROOT / "cache" / "task2_reviews")) / key
        self.by_product = {}

    def _read_cached(self, pid):
        if pid in self.by_product: return self.by_product[pid]
        path = self.cache_dir / f"{pid}.json"
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("product_id") != pid or data.get("source_sha256") != self.manifest.get("sha256"):
                    return None
                check = hashlib.sha256(json.dumps(data["reviews"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                if check != data.get("reviews_sha256"): return None
                rows = unique_product_reviews(data["reviews"], pid)
                if len(rows) == data.get("row_count"):
                    self.by_product[pid] = rows
                    return rows
            except (ValueError, OSError, KeyError, TypeError):
                pass
        return None

    def _save(self, pid, rows):
        self.by_product[pid] = rows
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.cache_dir / f"{pid}.json"
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.cache_dir, delete=False) as fh:
                json.dump({"product_id": pid, "source_sha256": self.manifest.get("sha256"), "row_count": len(rows),
                           "reviews_sha256": hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest(), "reviews": rows}, fh, ensure_ascii=False)
                partial = Path(fh.name)
            partial.replace(path)
        except OSError:
            pass

    def _scan(self, pids):
        try:
            import zstandard
        except ImportError:
            raise ReviewDataError("برای خواندن بستهٔ بازیابی فشرده، zstandard را نصب کن.") from None
        pending = {str(pid): pid for pid in pids}
        by_product = defaultdict(list)
        digest = hashlib.sha256()
        class HashReader:
            def __init__(self, stream): self.stream = stream
            def read(self, size=-1):
                data = self.stream.read(size)
                digest.update(data)
                return data
        csv.field_size_limit(10_000_000)
        from contextlib import ExitStack
        with ExitStack() as stack:
            if self.is_zip:
                archive = stack.enter_context(ZipFile(self.path))
                compressed = stack.enter_context(archive.open(self.payload_member))
            else:
                compressed = stack.enter_context(self.path.open("rb"))
            reader = stack.enter_context(zstandard.ZstdDecompressor().stream_reader(HashReader(compressed), read_size=1024*1024))
            stream = stack.enter_context(io.TextIOWrapper(reader, encoding="utf-8", newline=""))
            rows = csv.reader(stream)
            header = next(rows)
            mapping = _column_map(header)
            pidx = header.index(mapping["product_id"])
            fields = {key: header.index(col) for key, col in mapping.items() if col}
            total = 0
            for row in rows:
                total += 1
                if row[pidx] in pending:
                    data = {key: row[idx] for key, idx in fields.items()}
                    data["source_fields"] = {key: col for key, col in mapping.items() if key in FIELDS and col}
                    by_product[pending[row[pidx]]].append(data)
        if self.manifest.get("row_count") is not None and total != self.manifest["row_count"]:
            raise ReviewDataError("تعداد سطرهای بستهٔ بازیابی با گزارش آن مطابق نیست.")
        if self.manifest.get("sha256") and digest.hexdigest() != self.manifest["sha256"]:
            raise ReviewDataError("checksum بستهٔ بازیابی نظرها معتبر نیست.")
        for pid in pids:
            self._save(pid, unique_product_reviews(by_product.get(pid, []), pid))

    def get_many(self, product_ids):
        pids = list(dict.fromkeys(identifier(x) for x in product_ids))
        pending = [pid for pid in pids if self._read_cached(pid) is None]
        if pending: self._scan(pending)
        return {pid: ReviewBatch(pid, self.by_product[pid], "recovered_full_review_payload", True, self.path.name) for pid in pids}

    def get(self, product_id):
        pid = identifier(product_id)
        return self.get_many([pid])[pid]


class SQLiteReviewStore:
    """Verified packaged slice: every clean source review for all listed IDs."""
    def __init__(self, path):
        from .catalog import open_catalog
        try:
            self.catalog = open_catalog(str(Path(path).expanduser().resolve()))
        except (OSError, ValueError, sqlite3.Error) as exc:
            raise ReviewDataError("پایگاه دادهٔ نظرات یا ساختار آن در دسترس نیست.") from exc

    def get(self, product_id):
        pid = identifier(product_id)
        try:
            title, rows = self.catalog.get_review_rows(pid)
        except sqlite3.Error as exc:
            raise ReviewDataError("خواندن نظرات از پایگاه دادهٔ محلی ناموفق بود.") from exc
        if title is None:
            return ReviewBatch(pid, [], "indexed_catalogue", False, self.catalog.path.name,
                               status="outside_catalogue")
        return ReviewBatch(pid, [canonical_review(r) for r in rows],
                           "indexed_catalogue", True, self.catalog.path.name, title=title)

    def get_many(self, product_ids):
        return {pid: self.get(pid) for pid in dict.fromkeys(identifier(x) for x in product_ids)}


def resolve_review_store(*, source=None, data_dir=None):
    explicit = source or os.environ.get("QBC13_REVIEW_SOURCE")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists(): raise ReviewDataError("مسیر تعیین‌شده برای فایل نظرات وجود ندارد.")
        if path.suffix.lower() == ".jsonl": return JsonlReviewStore(path)
        if path.suffix.lower() in {".sqlite3", ".sqlite", ".db"}: return SQLiteReviewStore(path)
        if path.suffix.lower() in {".zip", ".zst"}: return PayloadReviewStore(path)
        if path.suffix.lower() == ".parquet": return ParquetReviewStore(path)
        if not path.is_dir(): raise ReviewDataError("نوع فایل نظرات پشتیبانی نمی‌شود.")
        data_dir = path
    configured = data_dir or os.environ.get("QBC13_ANALYTICS_DIR")
    folders = [Path(configured)] if configured else [ROOT / "processed_repo_based", ROOT / "data" / "analytics", ROOT / "data" / "processed_repo_based", ROOT]
    if not configured and (ROOT / "storage" / "catalog.sqlite3").is_file():
        return SQLiteReviewStore(ROOT / "storage" / "catalog.sqlite3")
    errors = []
    for folder in folders:
        for name in ("comments_clean.parquet", "reviews.parquet"):
            path = folder / name
            if path.is_file():
                try: return ParquetReviewStore(path)
                except ReviewDataError as exc: errors.append(str(exc))
        payload = folder / "reviews_payload.csv.zst"
        if payload.is_file(): return PayloadReviewStore(payload)
    raise ReviewDataError(errors[-1] if errors else "فایل کامل نظرات پیدا نشد. مسیر comments_clean.parquet یا reviews.parquet کامل را تعیین کن.")
