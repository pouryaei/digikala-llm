"""Paths are relative to this package, never to the caller's working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Config:
    BASE_DIR = str(ROOT)
    DATA_DIR = str(ROOT / "processed_repo_based")
    STORAGE_DIR = str(ROOT / "storage")
    PRODUCTS_PARQUET = str(Path(DATA_DIR) / "products_clean.parquet")
    PARQUET_FILE = PRODUCTS_PARQUET
    COMMENTS_PARQUET = str(Path(DATA_DIR) / "comments_clean.parquet")
    CATALOG_PATH = str(Path(STORAGE_DIR) / "catalog.sqlite3")
    FAISS_INDEX_PATH = str(Path(STORAGE_DIR) / "faiss_index.bin")
    MAPPING_FILE_PATH = str(Path(STORAGE_DIR) / "id_mapping.json")
    BM25_INDEX_PATH = str(Path(STORAGE_DIR) / "bm25_index.pkl")
    CACHE_DIR = str(Path(STORAGE_DIR) / "retrieval_cache")
    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    EMBEDDING_MODEL_NAME = MODEL_NAME
    DEVICE = "auto"
    BATCH_SIZE = 64
    SAMPLE_SIZE = 100_000
    TOP_K = 5
    BM25_K1 = 1.5
    BM25_B = .75
    RRF_K = 60
    HYBRID_FETCH_K = 50


class DenseUnavailable(RuntimeError):
    """Optional local Dense Search dependencies or model are unavailable."""
