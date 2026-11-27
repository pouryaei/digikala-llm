"""Optional structured LLM comparison; the default RAG comparison is local."""
from __future__ import annotations

import json
import re

from .cache import get_cached_response, make_cache_key, save_response
from .citation_validator import clean_comparison_answer, validate_comparison_citations
from .comparison_schema import ComparisonAnswer
from .llm_client import LLMClient
from .prompts import COMPARISON_PROMPT_VERSION, build_comparison_messages, prepare_comparison_reviews


def parse_comparison_response(response):
    if not isinstance(response, str):
        raise ValueError("Comparison output is not text.")
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", response, flags=re.I).strip()
    start, end = text.find("{"), text.rfind("}")
    return ComparisonAnswer.model_validate(json.loads(text[start:end+1] if 0 <= start <= end else text))


class ComparisonService:
    def __init__(self, llm=None, *, use_cache=True):
        self.llm = llm if llm is not None else LLMClient()
        self.use_cache = use_cache

    def compare(self, question, products, reviews_by_product):
        ids = [str(int(p["product_id"])) for p in products]
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError("At least two different products are required.")
        supplied = {str(int(pid)): reviews for pid, reviews in reviews_by_product.items()}
        if not set(ids) <= supplied.keys():
            raise ValueError("Provide a review list (possibly empty) for each product.")
        # There must be a unique product owner for every supporting comment.
        owners = {}
        for pid in ids:
            for row in supplied[pid]:
                cid = int(row["comment_id"])
                if str(row.get("product_id", pid)) != pid or (cid in owners and owners[cid] != pid):
                    raise ValueError("Review assigned to the wrong product.")
                owners[cid] = pid
        prepared = prepare_comparison_reviews({pid: supplied[pid] for pid in ids})
        key = make_cache_key(
            self.llm.model, COMPARISON_PROMPT_VERSION + "_rag2", question,
            {"products": products, "reviews_by_product": prepared},
            config={"client": type(self.llm).__qualname__, "generation": getattr(self.llm, "cache_config", {})},
        )
        usage = {"calls": 0, "api_attempts": 0, "latency": 0.0, "prompt_tokens": 0,
                 "completion_tokens": 0, "usage_known": True}
        if self.use_cache and (cached := get_cached_response(key)) is not None:
            try:
                previous = ComparisonAnswer.model_validate(cached["answer"])
                if (sorted(str(p.product_id) for p in previous.products) == sorted(ids)
                        and validate_comparison_citations(previous, prepared)["is_valid"]):
                    return {"answer": previous.model_dump(),
                            "meta": {**cached["meta"], **usage, "cached": True}}
            except (ValueError, TypeError, KeyError):
                pass
        system, user = build_comparison_messages(question, products, prepared)
        answer, check, feedback = None, None, ""
        for _ in range(2):
            try:
                out = self.llm.generate(system, user + feedback)
            except Exception as exc:
                usage["api_attempts"] += int(getattr(exc, "attempts", 1))
                usage["usage_known"] = False
                break
            usage["calls"] += 1
            usage["api_attempts"] += int(out.get("api_attempts", 1))
            usage["latency"] += float(out.get("latency", 0))
            usage["prompt_tokens"] += int(out.get("prompt_tokens", 0))
            usage["completion_tokens"] += int(out.get("completion_tokens", 0))
            usage["usage_known"] &= bool(out.get("usage_known", True))
            try:
                candidate = parse_comparison_response(out["text"])
            except (ValueError, TypeError, KeyError):
                feedback = "\nپاسخ قبلی معتبر نبود؛ فقط JSON مطابق ساختار خواسته‌شده بده."
                continue
            if sorted(str(p.product_id) for p in candidate.products) != sorted(ids):
                feedback = "\nفقط محصول‌های ورودی را یک بار در خروجی بیاور."
                continue
            check = validate_comparison_citations(candidate, prepared)
            answer = candidate
            if check["is_valid"]:
                break
            feedback = "\nفقط شناسهٔ نظرهای همان محصول را برای نقاط قوت و ضعف استفاده کن."

        warning = check is None or not check["is_valid"]
        if answer is None:
            result_answer = {"summary": "مدل مقایسهٔ قابل‌اعتبارسنجی ارائه نکرد.",
                             "products": [{"product_id": int(pid), "strengths": [], "weaknesses": []} for pid in ids],
                             "criteria": [], "conclusion": "", "inference": "", "insufficient_evidence": True}
        elif warning:
            result_answer = clean_comparison_answer(answer, prepared)
            # The claims in a free-text conclusion cannot be kept after rejected citations.
            result_answer.update(summary="برخی ارجاع‌های مدل نامعتبر بود؛ فقط نکته‌های با شناسهٔ معتبر باقی مانده‌اند.",
                                 conclusion="", inference="", insufficient_evidence=True)
        else:
            result_answer = answer.model_dump()
        result = {"answer": result_answer, "meta": {**usage, "cached": False, "citation_warning": warning,
                                                     "citation_check": check, "model": self.llm.model}}
        if self.use_cache and not warning:
            save_response(key, result)
        return result
