"""Orchestrate product selection, review retrieval and grounded generation."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from time import perf_counter

from .bm25_index import tokenize_query
from .catalog import open_catalog
from .rag_query import explicit_product_ids
from .review_qa import ask_reviews
from .search_qa import load_mapping, search_products
from .task2_evidence_selector import EvidenceSelector
from .task2_review_store import identifier, resolve_review_store

_TITLE_FILLER = set("مدل طرح سری کد سایز عددی بسته عدد عددی مجموعه به همراه است این برای".split())


@lru_cache(maxsize=1)
def _title_index():
    rows, freq = [], Counter()
    for item in load_mapping():
        pid = identifier(item["product_id"])
        phrase = " ".join(tokenize_query(item["title_fa"]))
        terms = frozenset(phrase.split()) - _TITLE_FILLER
        rows.append((pid, phrase, terms))
        freq.update(terms)
    return rows, freq


def _resolve_title(question):
    """Accept an exact unique name or one unambiguous high-coverage title.
    
    Product resolution and review relevance are separate. A high FAISS/BM25
    rank by itself never authorizes mixing or assigning reviews.
    """
    rows, freq = _title_index()
    qterms = frozenset(tokenize_query(question)) - _TITLE_FILLER
    phrase = " " + " ".join(tokenize_query(question)) + " "
    exact = []
    for pid, title, terms in rows:
        if len(terms) < 2:
            continue
        needle = f" {title} "
        start = phrase.find(needle)
        while start >= 0:
            exact.append((pid, start, start + len(needle)))
            start = phrase.find(needle, start + 1)
    if exact:
        # Drop a shorter match only at the same span, not if its name appears
        # separately in a comparison question.
        longest = list(dict.fromkeys(
            pid for pid, start, end in exact
            if not any(s <= start and end <= e and (s,e) != (start,end) for _, s, e in exact)
        ))
        if len(longest) == 1:
            return longest[0], "exact_title"
        return None, longest[:5]
    close = []
    for pid, _, terms in rows:
        shared = qterms & terms
        if (len(shared) >= min(3, max(2, len(terms))) and len(terms) >= 2
                and len(shared)/len(terms) >= .72
                and any(freq[t] <= 35 for t in shared)):
            # Distinctive terms prevent automatically interpreting a category
            # or a generic model number as a specific product.
            score = sum(1 / (freq[t] ** .5) for t in shared) * len(shared)/len(terms)
            close.append((score, pid))
    close.sort(key=lambda pair: (-pair[0], pair[1]))
    if close and (len(close) == 1 or close[0][0] >= 1.65 * close[1][0]):
        return close[0][1], "distinctive_title"
    return None, [pid for _, pid in close[:5]]


def find_product_candidates(query, top_k=5, mode="bm25", **filters):
    return search_products(query, top_k=top_k, mode=mode, **filters)


def _question(query):
    if not isinstance(query, str) or not query.strip() or len(query.strip()) > 1000:
        raise ValueError("سؤال باید متن غیرخالی و حداکثر ۱۰۰۰ کاراکتری باشد.")
    return query.strip()


def _needs_product(query, candidate_ids, search_mode, *, multiple_ids=False):
    catalogue = open_catalog()
    products = catalogue.get_products(candidate_ids)
    if candidate_ids:
        candidates = [products.get(pid, {"product_id": pid, "title_fa": None}) for pid in candidate_ids]
    else:
        # A generic review question without a product name must not give
        # misleading category matches on words like «نظر» or «کیفیت».
        from .task2_evidence_selector import STOP_WORDS, QUERY_THEMES
        generic = STOP_WORDS | {"ایرادهای", "چیست", "چطوره", "چطورن", "کیفیت", "کارایی", "عملکرد", "خریدش", "نظراتش"}
        generic |= {word for terms in QUERY_THEMES.values() for term in terms for word in term.split()}
        salient = " ".join(word for word in tokenize_query(query) if word not in generic and not word.isdecimal())
        candidates = find_product_candidates(salient, mode=search_mode) if salient else []
    return {
        "status": "needs_product", "product_id": None,
        "answer_mode": "needs_product", "answer": None, "evidence": [],
        "reason": ("چند شناسه در سؤال آمده است؛ یکی را انتخاب کن یا از مقایسه استفاده کن." if multiple_ids
                   else "محصول از روی سؤال به‌طور مطمئن مشخص نشد. نام دقیق یا شناسهٔ محصول را وارد کن."),
        "candidates": candidates, "meta": {"api_calls": 0, "api_attempts": 0},
    }


def resolve_rag_product(query, product_id=None, *, search_mode="bm25"):
    """Product resolution shared by the original and experimental QA paths."""
    resolved = "selected_id"
    if product_id is None:
        explicit = explicit_product_ids(query)
        if len(explicit) > 1:
            return None, None, None, _needs_product(query, explicit, search_mode, multiple_ids=True)
        if explicit:
            product_id, resolved = explicit[0], "question_id"
        else:
            product_id, resolved = _resolve_title(query)
            if product_id is None:
                return None, None, None, _needs_product(query, resolved, search_mode)
    pid = identifier(product_id)
    product = open_catalog().get_products((pid,)).get(pid)
    return pid, product, resolved, None


def run_rag_pipeline(user_query, product_id=None, *, source=None, search_mode="bm25",
                     generation_mode=None, llm=None, top_k=8, use_cache=True):
    """Run one QA request. A product ID may be passed, selected or resolved."""
    started = perf_counter()
    query = _question(user_query)
    if search_mode not in {"bm25", "dense", "hybrid"}:
        raise ValueError("Invalid search mode.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 20:
        raise ValueError("تعداد شواهد باید بین ۱ و ۲۰ باشد.")
    pid, product, resolved, unresolved = resolve_rag_product(query, product_id, search_mode=search_mode)
    if unresolved is not None:
        return unresolved
    result = ask_reviews(pid, query, source=source, top_k=top_k, generation_mode=generation_mode,
                         llm=llm, use_cache=use_cache, product_title=product["title_fa"] if product else None)
    result["product"] = product
    result["product_resolution"] = resolved
    result["status"] = "answered" if result["answer"] is not None else "insufficient_evidence"
    result["meta"]["rag_total_seconds"] = round(perf_counter()-started, 6)
    return result


def compare_products(product_ids, query=None, *, source=None, top_k=3, source_price_unit=None):
    """Compare catalogue facts and product-specific evidence, never a guessed winner."""
    from .review_qa import _stats
    started = perf_counter()
    ids = [identifier(pid) for pid in product_ids]
    if not 2 <= len(ids) <= 5 or len(ids) != len(set(ids)):
        raise ValueError("برای مقایسه، شناسهٔ ۲ تا ۵ محصول متفاوت را وارد کن.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 20:
        raise ValueError("تعداد شواهد باید بین ۱ و ۲۰ باشد.")
    query = _question(query) if query is not None else None
    products = open_catalog().get_products(ids, source_price_unit=source_price_unit)
    store = resolve_review_store(source=source)
    batches = store.get_many(ids) if hasattr(store, "get_many") else {pid: store.get(pid) for pid in ids}
    selector = EvidenceSelector(store, top_k=top_k)
    items = []
    for pid in ids:
        batch = batches[pid]
        product = products.get(pid)
        if product and not batch.title:
            batch.title = product["title_fa"]
        positive = selector.select_batch("نقاط قوت و مزایای این محصول چیست؟", batch)
        negative = selector.select_batch("ایرادها و نقاط ضعف این محصول چیست؟", batch)
        specific = selector.select_batch(query, batch) if query else None
        items.append({"product_id": pid, "facts": product, "review_stats": _stats(positive),
                      "strengths": [e for e in positive.evidence if e["polarity"] == "positive"],
                      "weaknesses": [e for e in negative.evidence if e["polarity"] == "negative"],
                      "strength_themes": positive.theme_counts, "weakness_themes": negative.theme_counts,
                      "related_reviews": specific.evidence if specific else [],
                      "related_review_note": specific.reason if specific else None,
                      "review_scope": batch.scope, "review_data_status": batch.status})
    notes = []
    missing_facts = [item["product_id"] for item in items if item["facts"] is None]
    missing_reviews = [item["product_id"] for item in items if not item["review_stats"]["text_review_count"]]
    if missing_facts:
        notes.append("اطلاعات کاتالوگ برای این شناسه‌ها موجود نیست: " + "، ".join(map(str, missing_facts)) + ".")
    if missing_reviews:
        notes.append("نظر متنی در دادهٔ در دسترس برای این شناسه‌ها موجود نیست: " + "، ".join(map(str, missing_reviews)) + ".")
    prices = {pid: products[pid]["price_toman"] for pid in ids if pid in products and not products[pid]["price_conflict"]
              and products[pid]["price_toman"] is not None}
    if len(prices) == len(ids) and len(set(prices.values())) > 1:
        low = min(prices.values())
        best = [str(pid) for pid in ids if prices[pid] == low]
        notes.append(f"کمترین قیمت ثبت‌شده در داده: {low:,.0f} تومان، شناسهٔ {'، '.join(best)}. قیمت روز بررسی نشده است.")
    valid_stats = [item for item in items if item["review_stats"]["valid_label_count"] > 0]
    if len(valid_stats) == len(ids) and len({item["review_stats"]["recommendation_rate"] for item in valid_stats}) > 1:
        ordered = sorted(valid_stats, key=lambda x: -x["review_stats"]["recommendation_rate"])
        labels = [f"{item['product_id']}: {item['review_stats']['recommended_count']} از {item['review_stats']['valid_label_count']}"
                  for item in ordered]
        notes.append("به‌ترتیب نسبت پیشنهاد خرید در نظرات برچسب‌دار: " + "؛ ".join(labels)
                     + ". تفاوت حجم نظرها را در نظر بگیر؛ این ترتیب به‌تنهایی توصیهٔ خرید نیست.")
    if len(prices) != len(ids) and any(p["price_recorded"] is not None and p["price_toman"] is None for p in products.values()):
        notes.append("عدد قیمت در داده موجود است، اما واحدش تأیید نشده؛ از آن برای محاسبهٔ قیمت تومانی استفاده نشد.")
    if not notes:
        notes.append("اطلاعات ثبت‌شده و شواهد هر محصول جدا آمده است؛ انتخاب نهایی به نیاز کاربر بستگی دارد.")
    return {"status": "compared", "products": items, "comparison_notes": notes,
            "insufficient_evidence": bool(missing_reviews or missing_facts),
            "meta": {"api_calls": 0, "api_attempts": 0, "total_seconds": round(perf_counter()-started, 6)}}
