"""Validate shipped indexes, or explicitly rebuild them over the same product IDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_rag import sha256, verify_files
from src.bm25_index import BM25IndexManager, mapping_ids
from src.catalog import open_catalog
from src.config import Config
from src.search_qa import load_mapping


def rebuild(*, bm25=False, dense=False):
    mapping = load_mapping()
    ids = mapping_ids(mapping)
    cat = open_catalog()
    db = cat.path
    with cat.connection() as con:
        by_id = {str(r["product_id"]): dict(r) for r in con.execute(
            "SELECT product_id,title_fa,product_text_lexical FROM products"
        )}
    if set(ids) != set(by_id):
        raise ValueError("Cannot rebuild over a different corpus. Rebuild the catalogue first.")
    rows = [{**by_id[pid], "product_id": pid} for pid in ids]
    storage = Path(Config.STORAGE_DIR)
    manifest = storage / "catalog_manifest.json"
    updated = json.loads(manifest.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="rebuild-indexes-", dir=storage) as tmp:
        folder = Path(tmp)
        if dense:
            from src.embedding import DenseEmbedder
            from src.faiss_index import FAISSIndexManager
            embedder = DenseEmbedder()
            vectors = embedder.generate_embeddings([row["title_fa"] for row in rows])
            if vectors.shape[1] != 384:
                raise ValueError("The encoder does not match the bundled 384-dimensional FAISS model.")
            FAISSIndexManager(folder / "faiss_index.bin", folder / "id_mapping.json").build_and_save_index(vectors, rows)
        if bm25:
            BM25IndexManager(folder / "bm25_index.pkl").build_from_records(
                rows, source_metadata={"dataset_revision": updated["dataset_revision"]}
            )
            updated["bm25_sha256"] = sha256(folder / "bm25_index.pkl")
        # Original metadata order is preserved byte-for-byte; only the explicitly
        # requested artifact is written after its full build has succeeded.
        if dense:
            updated["faiss_sha256"] = sha256(folder / "faiss_index.bin")
            updated["dense_available"] = True
            (folder / "faiss_index.bin").replace(Config.FAISS_INDEX_PATH)
        if bm25:
            (folder / "bm25_index.pkl").replace(Config.BM25_INDEX_PATH)
        updated["catalog_sha256"] = sha256(db)
        (folder / manifest.name).write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (folder / manifest.name).replace(manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rebuild-bm25", action="store_true", help="Replace only the local BM25 index with a newly built, aligned one.")
    p.add_argument("--rebuild-dense", action="store_true", help="Replace the vector index after re-encoding all catalogue titles.")
    a = p.parse_args()
    if a.rebuild_bm25 or a.rebuild_dense:
        rebuild(bm25=a.rebuild_bm25, dense=a.rebuild_dense)
    print(json.dumps(verify_files(), indent=2))


if __name__ == "__main__":
    main()
