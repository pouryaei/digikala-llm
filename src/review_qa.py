"""Task 2 review question answering, ready for the existing Streamlit contract.

Example: from src.review_qa import ask_reviews
         ask_reviews(product_id=10133811, query="ایرادهای پرتکرار چیست؟")
"""
from __future__ import annotations

import os
from time import perf_counter
from .task2_review_store import ReviewDataError, identifier, resolve_review_store
from .task2_evidence_selector import EvidenceSelector, Selection


def _stats(s: Selection):
    return {
        "total_reviews": s.total_reviews, "text_review_count": s.text_review_count,
        "relevant_review_count": s.relevant_review_count, "valid_label_count": s.valid_label_count,
        "recommended_count": s.recommended_count, "not_recommended_count": s.not_recommended_count,
        "no_idea_count": s.no_idea_count,
        "recommendation_rate": s.recommended_count / s.valid_label_count if s.valid_label_count else None,
        "complete_for_product": s.complete_for_product,
    }


def _format_number(n): return f"{int(n):,}"


def _citation(ids): return "، ".join(f"نظر {i}" for i in ids)


def _extractive_answer(s):
    total = _format_number(s.total_reviews)
    relevant = _format_number(s.relevant_review_count)
    paragraphs = [f"در دادهٔ واقعیِ در دسترس برای این محصول، {total} نظر ثبت شده است."]
    if not s.complete_for_product:
        paragraphs.append("آمار زیر فقط دربارهٔ دادهٔ بارگذاری‌شده است و نمایندهٔ همهٔ نظرات محصول نیست.")
    if s.intent == "value":
        if s.valid_label_count:
            ratio = s.recommended_count / s.valid_label_count
            paragraphs.append(f"از {s.valid_label_count:,} نظر دارای برچسب معتبر، {s.recommended_count:,} مورد خرید را پیشنهاد کرده‌اند ({ratio:.1%})، {s.not_recommended_count:,} مورد پیشنهاد نکرده‌اند و {s.no_idea_count:,} مورد نظر مشخصی ثبت نکرده‌اند.")
            if ratio > .5:
                paragraphs.append("در میان نظرهای دارای برچسب، اکثریت خرید را پیشنهاد کرده‌اند.")
            paragraphs.append("این درصد یکی از نشانه‌های تجربهٔ کاربران است؛ ارزش خرید قطعی به نیاز و قیمت فعلی شما نیز بستگی دارد.")
        else:
            paragraphs.append("برچسب معتبری برای محاسبهٔ نرخ پیشنهاد خرید موجود نیست. فقط می‌توان به تجربه‌های متنی زیر رجوع کرد.")
    elif s.intent in ("positive", "negative") and s.theme_counts:
        denom = s.text_review_count
        polarity_label = "مثبت" if s.intent == "positive" else "منفی"
        paragraphs.append(f"در {denom:,} نظر دارای متن، موضوع‌های {polarity_label} زیر با قواعد مشخص روی فیلدهای مزایا/معایب و جمله‌های دارای نشانهٔ روشنِ رضایت یا نارضایتی شناسایی شدند:")
        for t in s.theme_counts:
            paragraphs.append(f"• {t['theme']}: {t['count']:,} نظر از {t['denominator']:,} ({_citation(t['evidence_ids'])}؛ نمونهٔ شاهد).")
        paragraphs.append("هر نظر در هر موضوع یک‌بار شمرده می‌شود. این شمارش فقط اشاره‌های صریحی را پوشش می‌دهد که قواعد شناسایی کرده‌اند.")
    elif s.intent in ("positive", "negative"):
        paragraphs.append(f"{relevant} نظر با نشانه‌های متنی مرتبط بازیابی شد. برای رتبه‌بندی قطعیِ فراوانی ویژگی‌ها شواهد صریح کافی نداریم؛ نمونه‌های واقعی زیر را ببینید.")
    else:
        topics = "، ".join(s.requested_themes)
        if topics:
            paragraphs.append(f"{relevant} نظر با شواهد مرتبط پیدا شد. نتیجهٔ بررسی موضوع‌های پرسش:")
            for facet in s.facet_coverage:
                theme = facet["theme"]
                if not facet["evidence_ids"]:
                    paragraphs.append(f"• {theme}: شاهد صریح کافی برای این بخش از سؤال در شواهد انتخاب‌شده پیدا نشد.")
                    continue
                paragraphs.append(f"• {theme}: در {facet['matched_review_count']:,} نظر نشانهٔ مرتبط پیدا شد ({_citation(facet['evidence_ids'])}).")
                for t in s.theme_counts:
                    if t["theme"] == theme:
                        kind = "مثبت" if t["polarity"] == "positive" else "منفی"
                        paragraphs.append(f"  نشانهٔ روشن {kind} در {t['count']:,} نظر از {t['denominator']:,} نظر دارای متن ({_citation(t['evidence_ids'])}).")
            if s.theme_counts:
                paragraphs.append("این شمارش قاعده‌محور است و همهٔ معنای ضمنی نظرها را پوشش نمی‌دهد.")
        else:
            paragraphs.append(f"{relevant} نظر دارای متن مرتبط پیدا شد. چند تجربهٔ واقعی کاربران:")
    # The first quotations are deliberately the real contiguous excerpts; the
    # UI also exposes the remaining evidence and comment/product identifiers.
    for e in s.evidence[:min(4, len(s.evidence))]:
        quote = e["text"].strip()
        if len(quote) > 230: quote = quote[:230].rstrip()
        paragraphs.append(f"• «{quote}» ({_citation([e['comment_id']])})")
    polarities = {e.get("polarity") for e in s.evidence}
    statuses = {e.get("recommendation_status") for e in s.evidence}
    if "mixed" in polarities or {"positive", "negative"} <= polarities or {"recommended", "not_recommended"} <= statuses:
        paragraphs.append("تجربه‌ها یکسان نیستند؛ نمونه‌های مثبت و منفی یا نظرهای ترکیبی را جداگانه بررسی کنید.")
    if s.relevant_review_count <= 2:
        paragraphs.append("تعداد شواهد مرتبط کم است؛ این چند تجربه برای تعمیم به همهٔ خریداران کافی نیست.")
    return "\n".join(paragraphs)


def _safe_model_points(response, evidence):
    """Only render model points whose references and exact quote are in retrieved text.

    This is a structural check, not a claim of automatic semantic faithfulness.
    Uncited summary/inference are deliberately never displayed as facts.
    """
    from .phase4_llm.response_schema import ProductAnswer
    from .phase4_llm.citation_validator import validate_citations
    answer = ProductAnswer.model_validate(response["answer"])
    if response["meta"].get("citation_warning") or answer.insufficient_evidence:
        return []
    if not validate_citations(answer, evidence, require_quotes=True)["is_valid"]:
        return []
    result = []
    for label, points in (("نکتهٔ مثبت", answer.positive_points), ("نکتهٔ منفی", answer.negative_points), ("مشاهده", answer.observations)):
        for point in points:
            # Do not expose generated quotes that exceed the input or unsupported IDs.
            result.append(f"• {label}: {point.point.strip()} ({_citation(point.comment_ids)}؛ شاهد: «{point.quote.strip()}»)")
    return result[:6]


def _insufficient(product_id, reason, *, scope="unavailable", stats=None, started=None):
    return {"product_id": product_id, "answer_mode": "insufficient_evidence", "answer": None,
            "reason": reason, "evidence": [], "scope": scope, "stats": stats or {},
            "meta": {"generation_mode": "none", "api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                     "usage_known": True, "total_seconds": round(perf_counter()-started, 6) if started is not None else 0.0}}


def ask_reviews(product_id, query, *, source=None, data_dir=None, store=None, generation_mode=None, llm=None, top_k=8, use_cache=True, product_title=None):
    started = perf_counter()
    pid = identifier(product_id)
    if not isinstance(query, str) or not query.strip() or len(query.strip()) > 1000:
        raise ValueError("سؤال باید متن غیرخالی و حداکثر ۱۰۰۰ کاراکتر باشد.")
    query = query.strip()
    mode = (generation_mode or os.environ.get("QBC13_REVIEW_GENERATION_MODE", "extractive")).lower().strip()
    if mode not in {"extractive", "llm"}: raise ValueError("generation_mode must be extractive or llm")
    try:
        store = store if store is not None else resolve_review_store(source=source, data_dir=data_dir)
        selector = EvidenceSelector(store, top_k=top_k)
        batch = store.get(pid)
        if product_title and not batch.title:
            from dataclasses import replace
            batch = replace(batch, title=product_title)
        if batch.product_id != pid:
            raise ReviewDataError("شناسهٔ نظرات با محصول انتخاب‌شده یکسان نیست.")
        selection = selector.select_batch(query, batch)
    except ReviewDataError as exc:
        return _insufficient(pid, str(exc), started=started)
    except (OSError, ValueError, KeyError):
        return _insufficient(pid, "خواندن دادهٔ نظرها ناموفق بود؛ مسیر و ساختار فایل را بررسی کن.", started=started)
    retrieval_time = perf_counter() - started
    stats = _stats(selection)
    if selection.insufficient:
        result = _insufficient(pid, selection.reason, scope=selection.scope, stats=stats, started=started)
        result["meta"]["retrieval_seconds"] = round(retrieval_time, 6)
        return result
    evidence = selection.evidence
    answer = _extractive_answer(selection)
    actual_mode = "extractive"
    meta = {"generation_mode": "extractive", "retrieval_seconds": round(retrieval_time, 6),
            "api_calls": 0, "api_attempts": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "usage_known": True, "cached": False, "citation_check": "exact source excerpts and product-scoped IDs"}
    if mode == "llm" or llm is not None:
        try:
            from .phase4_llm.service import LLMService
            result = LLMService(llm=llm, require_quotes=True, use_cache=use_cache).answer(query, evidence)
            gen_meta = result["meta"]
            meta.update({"api_calls": gen_meta.get("calls", 0), "api_attempts": gen_meta.get("api_attempts", 0),
                         "prompt_tokens": gen_meta.get("prompt_tokens", 0), "completion_tokens": gen_meta.get("completion_tokens", 0),
                         "usage_known": gen_meta.get("usage_known", False), "model_seconds": gen_meta.get("latency", 0),
                         "cached": gen_meta.get("cached", False), "model": gen_meta.get("model")})
            points = _safe_model_points(result, evidence)
            if points:
                answer += "\n\nجمع‌بندی مدل زبانی از همین شواهد (استنتاج محدود):\n" + "\n".join(points)
                actual_mode = "grounded"
                meta["generation_mode"] = "llm"
                meta["citation_check"] = "verified retrieved IDs and verbatim model quotes; semantic assessment remains separate"
            else:
                meta["fallback_reason"] = "LLM did not return supported points with valid quotes"
        except Exception as exc:
            # Initialization failure (no SDK/key) or validation failure: retain
            # the already verified deterministic answer; do not expose secrets.
            meta["fallback_reason"] = type(exc).__name__
    meta["total_seconds"] = round(perf_counter()-started, 6)
    return {"product_id": pid, "answer_mode": actual_mode, "answer": answer, "evidence": evidence,
            "scope": selection.scope, "source": selection.source, "stats": stats,
            "theme_counts": selection.theme_counts, "facet_coverage": selection.facet_coverage, "intent": selection.intent, "requested_themes": selection.requested_themes,
            "meta": meta}
