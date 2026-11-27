"""Product retrieval with Dense Search, BM25, and reciprocal-rank fusion."""
from typing import Any, Dict, List

try:
    from src.config import Config
except ImportError:
    class Config:
        TOP_K = 5
        RRF_K = 60
        HYBRID_FETCH_K = 50

def _normalize_query(query: Any) -> str:
    """Return query unchanged; do not repeat EDA/text normalization."""
    return "" if query is None else str(query)


def _id_key(value: Any):
    """Canonical hashable ID so e.g. 42 and '42' fuse as the same product."""
    if value is None:
        return None
    return str(value)


def reciprocal_rank_fusion(dense_results: List[Dict[str, Any]],
                           bm25_results: List[Dict[str, Any]],
                           top_k: int = getattr(Config, "TOP_K", 5),
                           rrf_k: int = getattr(Config, "RRF_K", 60),
                           dense_weight: float = 1.0, bm25_weight: float = 1.0,
                           id_key: str = "product_id") -> List[Dict[str, Any]]:
    """Fuse ranked lists; scores are reciprocal rank, not raw model scores."""
    if top_k <= 0:
        return []
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative.")
    fused: Dict[str, float] = {}
    metadata: Dict[str, Dict[str, Any]] = {}
    debug: Dict[str, Dict[str, Any]] = {}
    for label, results, weight in (("dense", dense_results, dense_weight),
                                   ("bm25", bm25_results, bm25_weight)):
        seen = set()
        rank = 0
        for item in results:
            key = _id_key(item.get(id_key))
            if key is None or key in seen:
                continue
            seen.add(key)
            rank += 1
            fused[key] = fused.get(key, 0.0) + float(weight) / (rrf_k + rank)
            metadata.setdefault(key, item.copy())
            info = debug.setdefault(key, {"dense_rank": None, "bm25_rank": None,
                                          "dense_score": None, "bm25_score": None})
            info[f"{label}_rank"] = rank
            info[f"{label}_score"] = item.get("score")
    ordered = sorted(fused, key=lambda key: -fused[key])[:top_k]
    final = []
    for key in ordered:
        item = metadata[key].copy()
        item["score"] = float(fused[key])
        item.update(debug[key])
        final.append(item)
    return final


class HybridSearchEngine:
    """Run dense, BM25, or RRF hybrid search using existing manager APIs."""
    def __init__(self, embedder, faiss_manager, bm25_manager, cache=None,
                 rrf_k: int = getattr(Config, "RRF_K", 60),
                 fetch_k: int = getattr(Config, "HYBRID_FETCH_K", 50)):
        self.embedder = embedder
        self.faiss_manager = faiss_manager
        self.bm25_manager = bm25_manager
        self.cache = cache
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k

    def _encode_query(self, query: str):
        clean_q = _normalize_query(query)
        if self.cache is not None and hasattr(self.cache, "get_query_embedding"):
            return self.cache.get_query_embedding(clean_q, self.embedder.model)
        return self.embedder.generate_embeddings([clean_q])[0]

    def search(self, query: str, top_k: int = getattr(Config, "TOP_K", 5),
               mode: str = "hybrid", dense_weight: float = 1.0,
               bm25_weight: float = 1.0, *, allowed_ids=None) -> List[Dict[str, Any]]:
        mode = mode.lower().strip()
        if mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"Unknown search mode: {mode!r}. Choose dense, bm25, or hybrid.")
        if top_k <= 0 or not query.strip() or (allowed_ids is not None and not allowed_ids):
            return []
        candidate_k = max(self.fetch_k, top_k)

        def lexical_results():
            kwargs = {"allowed_ids": allowed_ids} if allowed_ids is not None else {}
            return self.bm25_manager.search(query, top_k=candidate_k, **kwargs)

        if mode == "bm25":
            return self._deduplicate(lexical_results(), top_k)
        q_vec = self._encode_query(query)
        depth = candidate_k if allowed_ids is None else len(self.faiss_manager.id_mapping)
        dense = self.faiss_manager.search(q_vec, top_k=depth)
        if allowed_ids is not None:
            dense = [r for r in dense if _id_key(r.get("product_id")) in allowed_ids][:candidate_k]
        if mode == "dense":
            return self._deduplicate(dense, top_k)
        lexical = lexical_results()
        return reciprocal_rank_fusion(dense, lexical, top_k=top_k, rrf_k=self.rrf_k,
                                      dense_weight=dense_weight, bm25_weight=bm25_weight)

    @staticmethod
    def _deduplicate(results: List[Dict[str, Any]], top_k: int,
                     id_key: str = "product_id") -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        seen = set()
        output = []
        for item in results:
            doc_id = _id_key(item.get(id_key))
            if doc_id is not None and doc_id in seen:
                continue
            if doc_id is not None:
                seen.add(doc_id)
            output.append(item)
            if len(output) >= top_k:
                break
        return output
