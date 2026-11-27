"""Identify the bundled test fixture and resolve the actual reviews path."""

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from .search_qa import COMMENTS_PATH, ROOT

_SAMPLE_SIZE = 3037
_SAMPLE_SHA256 = "4162548e63677d1777f634764b357b5bb5f73010b13952c28aa0c5526281f4fc"

try:
    _primary = json.loads((ROOT / "storage/catalog_manifest.json").read_text(encoding="utf-8"))["demo"]["primary"]
    REAL_EXAMPLE = {"product_id": str(_primary["product_id"]), "title_fa": _primary["title_fa"]}
except (OSError, ValueError, KeyError, TypeError):
    REAL_EXAMPLE = {"product_id": "", "title_fa": ""}
SAMPLE_PRODUCT = REAL_EXAMPLE
SAMPLE_QUESTION = "ایرادهای پرتکرار این محصول چیست؟"
BUNDLED_STORE = ROOT / "storage" / "catalog.sqlite3"


def comments_path():
    # Preserve the original app's configuration; default to the new real store.
    value = (os.getenv("FOROUHAR_COMMENTS_PATH") or os.getenv("QBC13_REVIEW_SOURCE") or "").strip()
    if value:
        path = Path(value).expanduser()
        return path if path.is_absolute() else ROOT / path
    if COMMENTS_PATH.is_file() and not is_bundled_sample(COMMENTS_PATH):
        return COMMENTS_PATH
    return BUNDLED_STORE


@lru_cache(maxsize=8)
def _matches_sample(path, modified_ns, size):
    return hashlib.sha256(path.read_bytes()).hexdigest() == _SAMPLE_SHA256


def is_bundled_sample(path):
    path = Path(path)
    try:
        stat = path.stat()
        return stat.st_size == _SAMPLE_SIZE and _matches_sample(
            path.resolve(), stat.st_mtime_ns, stat.st_size
        )
    except OSError:
        return False
