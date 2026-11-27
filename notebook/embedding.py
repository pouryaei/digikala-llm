"""One lazy instance of the same local encoder used to build this FAISS index."""
from __future__ import annotations

import os
from pathlib import Path
from threading import RLock

from .config import Config, DenseUnavailable


class DenseEmbedder:
    def __init__(self, model_name=Config.MODEL_NAME, device=Config.DEVICE):
        configured_path = os.environ.get("QBC13_EMBEDDING_MODEL_DIR")
        self.model_name = str(Path(configured_path).expanduser().resolve()) if configured_path else model_name
        self.local_path = bool(configured_path)
        self.revision = Config.MODEL_REVISION if self.model_name == Config.MODEL_NAME else None
        self.device = os.environ.get("QBC13_EMBEDDING_DEVICE", device)
        self._model = None
        self._lock = RLock()

    @property
    def model(self):
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except (ImportError, OSError, RuntimeError) as exc:
                raise DenseUnavailable("برای جست‌وجوی برداری وابستگی‌های requirements-dense.txt را نصب کن.") from exc
            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.local_path and not Path(self.model_name).is_dir():
                raise DenseUnavailable("پوشهٔ تعیین‌شده در QBC13_EMBEDDING_MODEL_DIR پیدا نشد.")
            try:
                self._model = SentenceTransformer(
                    self.model_name, device=device, revision=self.revision,
                    local_files_only=self.local_path or os.environ.get("QBC13_EMBEDDING_LOCAL_ONLY") == "1",
                    trust_remote_code=False,
                )
            except (OSError, ValueError, RuntimeError, ImportError) as exc:
                raise DenseUnavailable("بارگذاری مدل محلی بردارساز ناموفق بود؛ پوشهٔ مدل یا اتصال دانلود را بررسی کن.") from exc
            return self._model

    @property
    def dimension(self):
        return self.model.get_sentence_embedding_dimension()

    def generate_embeddings(self, texts, batch_size=Config.BATCH_SIZE):
        import numpy as np
        values = list(texts)
        if not values:
            return np.empty((0, 0), dtype=np.float32)
        if any(not isinstance(t, str) or not t.strip() for t in values):
            raise ValueError("Embedding input must contain only nonempty text.")
        return np.ascontiguousarray(self.model.encode(
            values, batch_size=batch_size, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True,
        ), dtype=np.float32)

    get_embeddings = generate_embeddings

    def get_query_embedding(self, query):
        return self.generate_embeddings([query])[0]
