"""Explicit live comparison smoke test on sample inputs; safe to import."""
import json

products = [
    {
        "product_id": 101,
        "title": "گوشی A",
        "price": 15000000,
        "brand": "Brand A",
        "rate": 4.2
    },
    {
        "product_id": 202,
        "title": "گوشی B",
        "price": 18000000,
        "brand": "Brand B",
        "rate": 4.5
    }
]

reviews_a = [
    {
        "comment_id": 125,
        "text": "باتری گوشی عملکرد بسیار خوبی دارد."
    },
    {
        "comment_id": 126,
        "text": "کیفیت دوربین متوسط است."
    }
]

reviews_b = [
    {
        "comment_id": 289,
        "text": "کیفیت دوربین بسیار خوب است."
    },
    {
        "comment_id": 290,
        "text": "قیمت گوشی کمی بالاست."
    }
]


reviews_by_product = {101: reviews_a, 202: reviews_b}
def main():
    try:
        from src.phase4_llm.comparison_service import ComparisonService
        service = ComparisonService()
        result = service.compare(
            question="این دو گوشی را از نظر تجربه کاربران و قیمت مقایسه کن.",
            products=products, reviews_by_product=reviews_by_product,
        )
        print("تست قرارداد مقایسه روی دادهٔ نمونه:")
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 1 if result.get("meta",{}).get("citation_warning") else 0
    except Exception as exc:
        print(f"تست مقایسه ناموفق بود ({type(exc).__name__}). کلید و تنظیمات سرویس را بررسی کن.")
        return 1


if __name__=="__main__":
    raise SystemExit(main())
