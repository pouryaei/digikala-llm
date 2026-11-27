"""LLM answering with retry, citation checks, and version-aware caching."""
from __future__ import annotations

from .llm_client import LLMClient, LLMUnavailable
from .prompts import PROMPT_VERSION, build_messages, prepare_reviews
from .response_schema import ProductAnswer, parse_response
from .citation_validator import validate_citations, clean_answer
from .cache import make_cache_key, get_cached_response, save_response


class LLMService:
    def __init__(self, llm=None, *, require_quotes=False, use_cache=True):
        self.llm = llm if llm is not None else LLMClient()
        self.require_quotes = bool(require_quotes)
        self.use_cache = use_cache

    def answer(self, question, reviews):
        reviews = prepare_reviews(reviews)
        if not reviews:
            return {"answer": ProductAnswer(summary="شواهد کافی در دسترس نیست.", insufficient_evidence=True).model_dump(),
                    "meta": {"latency": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "api_attempts": 0, "cached": False, "citation_warning": False, "model": self.llm.model, "usage_known": True}}
        config = {"require_quotes": self.require_quotes, "generation": getattr(self.llm, "cache_config", {})}
        key = make_cache_key(self.llm.model, PROMPT_VERSION, question, reviews, config=config)
        if self.use_cache:
            cached = get_cached_response(key)
            if cached is not None:
                # Stored token counts describe the original request, not current cost.
                try:
                    a = ProductAnswer.model_validate(cached["answer"])
                    if validate_citations(a, reviews, self.require_quotes)["is_valid"]:
                        old = cached.get("meta", {})
                        cached["meta"] = {**old, "cached": True, "calls": 0, "api_attempts": 0, "latency": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "cache_original_prompt_tokens": old.get("prompt_tokens", 0), "cache_original_completion_tokens": old.get("completion_tokens", 0)}
                        return cached
                except (KeyError, ValueError, TypeError):
                    pass
        system, user = build_messages(question, reviews, legacy_markers=not self.require_quotes)
        usage = {"latency": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "api_attempts": 0, "usage_known": True}
        answer = None
        check = None
        feedback = ""
        for _ in range(2):
            try:
                out = self.llm.generate(system, user + feedback)
            except Exception as exc:
                usage["usage_known"] = False
                usage["api_attempts"] += int(getattr(exc, "attempts", 1))
                return {"answer": ProductAnswer(summary="مدل زبانی در دسترس نبود.", insufficient_evidence=True).model_dump(),
                        "meta": {**usage, "cached": False, "citation_warning": True, "model": self.llm.model, "failure": type(exc).__name__}}
            usage["calls"] += 1
            usage["api_attempts"] += int(out.get("api_attempts", 1))
            usage["latency"] += float(out.get("latency", 0))
            usage["prompt_tokens"] += int(out.get("prompt_tokens", 0))
            usage["completion_tokens"] += int(out.get("completion_tokens", 0))
            usage["usage_known"] = usage["usage_known"] and bool(out.get("usage_known", True))
            try:
                candidate = parse_response(out["text"])
            except (ValueError, TypeError, KeyError):
                feedback = "\nخروجی قبلی JSON معتبر نبود. فقط ساختار JSON خواسته‌شده را برگردان."
                continue
            answer = candidate
            check = validate_citations(answer, reviews, self.require_quotes)
            if check["is_valid"]:
                break
            feedback = "\nپاسخ قبلی ارجاع یا نقل‌قول نامعتبر داشت. فقط شناسه‌های همین ورودی و نقل‌قول دقیق استفاده کن."
        if answer is None:
            answer = ProductAnswer(summary="خروجی ساختاریافتهٔ مدل معتبر نبود.", insufficient_evidence=True)
            check = {"is_valid": False}
        warning = not check["is_valid"]
        if warning: answer = clean_answer(answer, reviews)
        result = {"answer": answer.model_dump(), "meta": {**usage, "cached": False, "citation_warning": warning, "model": self.llm.model, "citation_check": check}}
        if self.use_cache and not warning: save_response(key, result)
        return result
