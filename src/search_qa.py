"""Product search over matching original vectors, rebuilt BM25 and real metadata."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

from .bm25_index import BM25IndexManager, mapping_digest, mapping_ids
from .catalog import open_catalog
from .config import Config, ROOT, DenseUnavailable
from .rag_query import _check, search_request
from .product_preferences import recorded_budget, rerank

COMMENTS_PATH = Path(Config.COMMENTS_PARQUET)
PRODUCTS_PATH = Path(Config.PRODUCTS_PARQUET)
BM25_PATH = Path(Config.BM25_INDEX_PATH)
FAISS_PATH = Path(Config.FAISS_INDEX_PATH)
MAPPING_PATH = Path(Config.MAPPING_FILE_PATH)
MODES = ("bm25", "dense", "hybrid")


@lru_cache(maxsize=1)
def load_mapping():
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8-sig"))
    expected = open_catalog().metadata["mapping_sha256"]
    if mapping_digest(mapping) != expected:
        raise ValueError("نگاشت محصولات با کاتالوگ همراه بسته هماهنگ نیست.")
    return mapping


@lru_cache(maxsize=1)
def _load_bm25():
    bm25 = BM25IndexManager(BM25_PATH)
    if not bm25.load_index():
        raise FileNotFoundError("storage/bm25_index.pkl پیدا نشد.")
    if mapping_ids(bm25.id_mapping) != mapping_ids(load_mapping()):
        raise ValueError("شناسه‌ها یا ترتیب BM25 با نگاشت کاتالوگ یکسان نیست؛ شاخص هماهنگ را بازسازی کن.")
    return bm25


@lru_cache(maxsize=1)
def _load_dense():
    if not FAISS_PATH.is_file():
        raise DenseUnavailable("نمایهٔ برداری برای این مجموعه هنوز ساخته نشده است. حالت BM25 آمادهٔ اجراست؛ برای Dense یا Hybrid نمایهٔ هم‌اندازه را بسازید.")
    from .cache_manager import SearchCache
    from .embedding import DenseEmbedder
    from .faiss_index import FAISSIndexManager

    faiss = FAISSIndexManager(FAISS_PATH, MAPPING_PATH)
    if not faiss.load_index():
        raise FileNotFoundError("شاخص برداری یا نگاشت آن در storage پیدا نشد.")
    if mapping_ids(faiss.id_mapping) != mapping_ids(load_mapping()):
        raise ValueError("نگاشت شاخص برداری با کاتالوگ هم‌خوان نیست.")
    return DenseEmbedder(), faiss, SearchCache()


@lru_cache(maxsize=3)
def load_search_engine(mode="bm25"):
    from .hybrid_search import HybridSearchEngine
    if mode not in MODES:
        raise ValueError("search mode must be bm25, dense or hybrid.")
    bm25 = _load_bm25() if mode != "dense" else None
    embedder, faiss, cache = _load_dense() if mode != "bm25" else (None, None, None)
    return HybridSearchEngine(embedder, faiss, bm25, cache)


def search_products(query, *, top_k=5, mode="bm25", min_price=None, max_price=None, min_rating=None, min_reviews=0, source_price_unit=None,
                    allowed_product_ids=None):
    if not isinstance(query, str) or len(query) > 1000:
        raise ValueError("عبارت جست‌وجو باید متن حداکثر ۱۰۰۰ کاراکتری باشد.")
    if mode not in MODES or not 1 <= int(top_k) <= 50:
        raise ValueError("حالت جست‌وجو یا تعداد نتایج معتبر نیست.")
    request = search_request(query, min_price, max_price)
    if not request.text:
        return []
    min_rating = _check(min_rating, "min_rating", ceiling=100)
    if not isinstance(min_reviews, int) or min_reviews < 0:
        raise ValueError("min_reviews must be a non-negative integer.")
    cat = open_catalog()
    allowed = None if allowed_product_ids is None else set(map(str, allowed_product_ids))
    if request.scope_hint:
        scope_ids = cat.preference_scope_ids(request.scope_hint)
        allowed = scope_ids if allowed is None else allowed & scope_ids
    raw_budget = recorded_budget(request, cat, source_price_unit)
    if request.min_price is not None or request.max_price is not None or min_rating is not None or min_reviews:
        matches = cat.filtered_ids(min_price=request.min_price, max_price=request.max_price,
                                   min_rating=min_rating, min_reviews=min_reviews, source_price_unit=source_price_unit,
                                   price_in_recorded_units=raw_budget)
        allowed = matches if allowed is None else allowed & matches
    use_preferences = request.prefer_satisfaction or request.prefer_affordability
    candidate_k = max(int(top_k), min(100, max(50, int(top_k) * 5))) if use_preferences else int(top_k)
    engine = load_search_engine(mode)
    hits = engine.search(request.text, top_k=candidate_k, mode=mode, allowed_ids=allowed)
    relaxed = False
    if not hits and request.fallback_text:
        hits = engine.search(request.fallback_text, top_k=candidate_k, mode=mode, allowed_ids=allowed)
        relaxed = bool(hits)
    products = cat.get_products((hit["product_id"] for hit in hits), source_price_unit=source_price_unit)
    if use_preferences:
        stats = cat.review_stats(hit["product_id"] for hit in hits) if request.prefer_satisfaction else {}
        hits = rerank(hits, products, stats, request, int(top_k))
    return [{**hit, **products.get(int(hit["product_id"]), {}), "product_id": str(hit["product_id"]),
             **({"purpose_match_relaxed": True} if relaxed else {}),
             **({"budget_unit": "recorded"} if raw_budget else {})} for hit in hits]


def get_review_evidence(product_id, question, *, comments_path=None, top_k=8):
    """Legacy consumer: original exact snippets of the selected product only."""
    from .task2_evidence_selector import EvidenceSelector
    from .task2_review_store import resolve_review_store
    return EvidenceSelector(resolve_review_store(source=comments_path), top_k=top_k).select(question, product_id)


def answer_product_question(product_id, question, *, comments_path=None, top_k=8, service=None):
    """Adapter for the previous M4 UI: answer is still a structured dictionary."""
    from .rag import run_rag_pipeline
    result = run_rag_pipeline(question, product_id, source=comments_path, top_k=top_k,
                              llm=getattr(service, "llm", service))
    sources = result["evidence"]
    meta = dict(result["meta"])
    meta.setdefault("calls", meta.get("api_calls", 0))
    meta.setdefault("latency", meta.get("total_seconds", 0.0))
    for name, default in (("cached", False), ("citation_warning", False), ("model", None)):
        meta.setdefault(name, default)
    return {
        "product_id": str(result["product_id"]),
        "answer": {
            "summary": result.get("answer") or result.get("reason"),
            "positive_points": [{"point": e["text"], "comment_ids": [e["comment_id"]]} for e in sources if e["polarity"] == "positive"],
            "negative_points": [{"point": e["text"], "comment_ids": [e["comment_id"]]} for e in sources if e["polarity"] == "negative"],
            "insufficient_evidence": result["status"] != "answered", "inference": "",
        },
        "sources": sources,
        "meta": meta,
    }
