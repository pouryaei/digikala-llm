"""Load and query the bundled product FAISS index without import-time work."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from .bm25_index import mapping_ids
from .config import Config, DenseUnavailable


def _faiss():
    try:
        import faiss
        return faiss
    except (ImportError, OSError, RuntimeError) as exc:
        raise DenseUnavailable("برای خواندن شاخص برداری، faiss-cpu را از requirements-dense.txt نصب کن.") from exc


class FAISSIndexManager:
    def __init__(self, index_path=Config.FAISS_INDEX_PATH, mapping_path=Config.MAPPING_FILE_PATH):
        self.index_path = str(index_path)
        self.mapping_path = str(mapping_path)
        self.index = None
        self.id_mapping = []

    def load_index(self):
        index_path, mapping_path = Path(self.index_path), Path(self.mapping_path)
        if not (index_path.is_file() and mapping_path.is_file()):
            return False
        mapping = json.loads(mapping_path.read_text(encoding="utf-8-sig"))
        ids = mapping_ids(mapping)
        faiss = _faiss()
        try:
            index = faiss.read_index(str(index_path))
        except RuntimeError as exc:
            raise ValueError("فایل شاخص FAISS معتبر نیست.") from exc
        if index.ntotal != len(ids) or index.d <= 0 or index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("شاخص FAISS با شناسه‌ها یا معیار شباهت مورد انتظار مطابقت ندارد.")
        self.index, self.id_mapping = index, mapping
        return True

    def build_and_save_index(self, embeddings, df_sample, add_batch_size=50000):
        import numpy as np
        faiss = _faiss()
        cols = ["product_id", "title_fa"]
        if hasattr(df_sample, "to_dict"):
            df_sample = df_sample[cols].to_dict(orient="records")
        mapping = [{"product_id": str(r["product_id"]), "title_fa": str(r["title_fa"])} for r in df_sample]
        ids = mapping_ids(mapping)
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        if (vectors.ndim != 2 or len(vectors) != len(ids) or not vectors.shape[1]
                or not 1 <= add_batch_size or not np.isfinite(vectors).all()):
            raise ValueError("Embedding matrix must match the IDs and contain finite vectors.")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
            raise ValueError("FAISS cosine vectors must be unit-normalized.")
        index = faiss.IndexFlatIP(vectors.shape[1])
        for start in range(0, len(vectors), add_batch_size):
            index.add(vectors[start:start+add_batch_size])
        index_path, mapping_path = Path(self.index_path), Path(self.mapping_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        files = []
        try:
            for path in (index_path, mapping_path):
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
                    files.append(Path(tmp.name))
            faiss.write_index(index, str(files[0]))
            files[1].write_text(json.dumps(mapping, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            files[0].replace(index_path)
            files[1].replace(mapping_path)
        finally:
            for path in files:
                path.unlink(missing_ok=True)
        self.index, self.id_mapping = index, mapping
        return index, mapping

    def search(self, query_vector, top_k=Config.TOP_K):
        import numpy as np
        if self.index is None:
            raise ValueError("FAISS index is not loaded.")
        if top_k <= 0:
            return []
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.shape != (1, self.index.d) or not np.isfinite(vector).all() or not np.any(vector):
            raise ValueError("Query embedding is empty or does not match the FAISS dimension.")
        scores, positions = self.index.search(np.ascontiguousarray(vector), min(top_k, self.index.ntotal))
        return [{**self.id_mapping[int(i)], "score": float(s)}
                for i, s in zip(positions[0], scores[0]) if 0 <= i < len(self.id_mapping)]
