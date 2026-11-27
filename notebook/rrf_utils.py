"""Compatibility adapter for the project's earlier RRF helper."""
from .hybrid_search import reciprocal_rank_fusion


def calculate_rrf(list_a, list_b, k=60):
    return [
        {"product_id": item["product_id"], "fusion_score": item["score"]}
        for item in reciprocal_rank_fusion(list_a, list_b, top_k=len(list_a)+len(list_b), rrf_k=k)
    ]
