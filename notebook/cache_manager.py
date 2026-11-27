"""Small in-process cache; no archived DiskCache state is required."""
from collections import OrderedDict
from threading import RLock

from .config import Config


class SearchCache:
    def __init__(self, cache_dir=Config.CACHE_DIR, max_items=128):
        self.max_items = max(1, int(max_items))
        self._vectors = OrderedDict()
        self._lock = RLock()

    def get_query_embedding(self, query_text, embedder_model):
        key = (id(embedder_model), query_text)
        with self._lock:
            if key not in self._vectors:
                self._vectors[key] = embedder_model.encode(
                    [query_text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
                )[0]
                if len(self._vectors) > self.max_items:
                    self._vectors.popitem(last=False)
            self._vectors.move_to_end(key)
            return self._vectors[key].copy()

    def clear(self):
        with self._lock:
            self._vectors.clear()


# Preserve the previous import. This object does not create files or load a model.
search_cache = SearchCache()
