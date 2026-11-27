import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from digikala_ai.utils.normalization import normalize_persian_text

class BM25Searcher:
    def __init__(self, products: List[Dict[str, Any]]):
        """
        مقداردهی اولیه اینکس BM25 با لیست محصولات کاتالوگ.
        هر محصول باید حداقل دارای شناسه و عنوان (title) باشد.
        """
        self.products = products
        self.corpus_tokens = []

        for p in products:
            # ترکیب عنوان و ویژگی‌ها برای متن قابل جستجو
            raw_text = f"{p.get('title', '')} {p.get('brand_title', '')} {p.get('category_title', '')}"
            normalized = normalize_persian_text(raw_text)
            tokens = normalized.split()
            self.corpus_tokens.append(tokens)

        # ساخت اینکس BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        جستجوی پرس‌وجو در اینکس و بازگرداندن top_k محصول برتر به همراه امتیاز.
        """
        norm_query = normalize_persian_text(query)
        query_tokens = norm_query.split()

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            res = self.products[idx].copy()
            res['bm25_score'] = float(scores[idx])
            results.append(res)

        return results
