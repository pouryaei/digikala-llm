"""Small, deterministic reranker for explicitly requested shopping preferences."""
from __future__ import annotations

import math


def recorded_budget(request, catalog, source_price_unit=None):
    from .catalog import resolve_price_unit
    return bool((request.min_price is not None or request.max_price is not None)
                and not request.explicit_price_currency
                and not resolve_price_unit(source_price_unit)
                and not catalog.metadata.get("supplied_price_unit_confirmed"))


def _positive_price(row):
    value = row.get("price_recorded")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) and value > 0 else None


def rerank(hits, products, review_stats, request, top_k):
    """Only rerank already-relevant candidates. No user satisfaction is fabricated.

    A smoothed recommendation rate (2/4 prior) avoids ranking a single positive
    review like hundreds of consistent reviews. A recorded rating is the fallback
    when no labeled recommendation exists. Relative price uses a rank percentile,
    which is invariant to a uniform rial/toman conversion.
    """
    if not hits:
        return []
    prices = sorted(set(p for row in products.values() if (p := _positive_price(row)) is not None))
    affordability = {p: 1 - i / max(1, len(prices) - 1) for i,p in enumerate(prices)}
    scores = [float(r.get("score", 0)) for r in hits]
    lo, hi = min(scores), max(scores)
    ranked = []
    for index, hit in enumerate(hits):
        pid = int(hit["product_id"])
        product = products.get(pid, {})
        _, positive, valid = review_stats.get(pid, (0,0,0))
        if valid:
            satisfaction = (positive + 2) / (valid + 4)
            positive_signal = positive > 0 and positive / valid >= .5
        elif product.get("rating_100") is not None and (product.get("rating_count") or 0) > 0:
            n = product["rating_count"]
            satisfaction = (product["rating_100"] / 100 * n + 2) / (n + 4)
            positive_signal = product["rating_100"] >= 70 and n >= 2
        else:
            satisfaction = 0.0
            positive_signal = False
        # Relevance stays important, and equal lexical scores remain equal.
        relevance = 1.0 if hi == lo else .5 + .5 * (scores[index] - lo) / (hi - lo)
        cheap = affordability.get(_positive_price(product), 0.0)
        if request.prefer_affordability and request.prefer_satisfaction:
            value = .5 * relevance + .3 * satisfaction + .2 * cheap
        elif request.prefer_satisfaction:
            value = .55 * relevance + .45 * satisfaction
        else:
            value = .55 * relevance + .45 * cheap
        ranked.append((value, index, hit, positive_signal))
    # A cheap product with no positive evidence must not outrank an otherwise
    # relevant, demonstrably well-reviewed product when satisfaction is requested.
    ranked.sort(key=lambda row: (request.prefer_satisfaction and not row[3], -row[0], row[1]))
    if request.scope_hint == "travel":
        # For a broad travel request, first return different available product
        # families, then fill any remaining slots using the same ranking.
        first, extra, seen = [], [], set()
        for row in ranked:
            category = products.get(int(row[2]["product_id"]), {}).get("category2")
            (extra if category in seen else first).append(row)
            seen.add(category)
        ranked = first + extra
    return [{**hit, "preference_score": round(value, 6), "positive_feedback_signal": positive_signal}
            for value, _, hit, positive_signal in ranked[:top_k]]
