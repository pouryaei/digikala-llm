"""Explicit live M4 smoke test on sample reviews; importing this file costs zero."""
import argparse,json

REVIEWS=[{"comment_id":125,"text":"کیفیت دوربین بسیار خوب است."},
         {"comment_id":289,"text":"باتری عملکرد متوسطی دارد."}]
QUESTION="آیا خرید این محصول پیشنهاد می‌شود؟"


def main(check_cache=False):
    try:
        from src.phase4_llm.service import LLMService
        service=LLMService()
        result=service.answer(QUESTION,REVIEWS)
        print("تست قرارداد روی دادهٔ نمونه (نه نظرات واقعی):")
        print(json.dumps(result,ensure_ascii=False,indent=2))
        meta=result["meta"]
        if meta.get("failure") or meta.get("citation_warning"):
            print("پاسخ معتبر دریافت نشد؛ تست Cache اجرا نشد.");return 1
        if check_cache:
            again=service.answer(QUESTION,REVIEWS)
            print("Cache hit:",again["meta"].get("cached",False))
            if not again["meta"].get("cached"):return 1
        return 0
    except Exception as exc:
        print(f"تست M4 ناموفق بود ({type(exc).__name__}). تنظیمات محلی API و وابستگی‌ها را بررسی کن.")
        return 1


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--check-cache",action="store_true",help="Check a second identical request after a valid answer")
    raise SystemExit(main(p.parse_args().check_cache))
