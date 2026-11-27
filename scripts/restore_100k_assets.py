"""Restore the unchanged runtime assets from the team's validated 100K ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
REVISION = "89c3133b169c8d3793db8834f56f32fee33d9db0"
MANIFEST = "storage/catalog_manifest.json"
REQUIRED = {
    "storage/catalog.sqlite3": "catalog_sha256",
    "storage/bm25_index.pkl": "bm25_sha256",
    "storage/id_mapping.json": "mapping_file_sha256",
    "data/analytics_100000/products.parquet": "analytics_products_sha256",
    "data/analytics_100000/reviews.parquet": "analytics_reviews_sha256",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore(archive, root=ROOT):
    root = Path(root).resolve()
    with ZipFile(archive) as z:
        manifests = [name for name in z.namelist() if name == MANIFEST or name.endswith("/" + MANIFEST)]
        if len(manifests) != 1:
            raise ValueError("Expected exactly one 100K catalogue manifest in the archive.")
        member = manifests[0]
        prefix = member[:-len(MANIFEST)]
        payload = z.read(member)
        manifest = json.loads(payload)
        if (manifest.get("dataset_revision") != REVISION or manifest.get("product_count") != 100000
                or manifest.get("review_rows") != 1874457):
            raise ValueError("Archive is not the validated Forouhar 100K data revision.")
        with tempfile.TemporaryDirectory(prefix=".restore_100k_", dir=root) as work:
            staging = Path(work)
            pending = []
            for rel, field in REQUIRED.items():
                expected = manifest.get(field)
                if not isinstance(expected, str) or len(expected) != 64:
                    raise ValueError(f"Missing verified hash: {field}")
                dst = root / rel
                if dst.exists():
                    if sha(dst) != expected:
                        raise FileExistsError(f"Existing asset differs; not overwriting {rel}.")
                    continue
                temp = staging / rel
                temp.parent.mkdir(parents=True, exist_ok=True)
                with z.open(prefix + rel) as source, temp.open("wb") as out:
                    shutil.copyfileobj(source, out, 1024 * 1024)
                if sha(temp) != expected:
                    raise ValueError(f"Hash mismatch: {rel}")
                pending.append((temp, dst))
            destination = root / MANIFEST
            if destination.exists() and json.loads(destination.read_bytes()) != manifest:
                raise FileExistsError("Existing catalogue manifest differs; not overwriting.")
            # Publish only after all hashes and pre-existing paths have passed.
            for temp, dst in pending:
                dst.parent.mkdir(parents=True, exist_ok=True)
                temp.replace(dst)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
    return {"status": "PASS", "products": 100000, "reviews": 1874457,
            "dataset_revision": REVISION, "restored": len(pending), "api_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.archive), ensure_ascii=False, indent=2))
