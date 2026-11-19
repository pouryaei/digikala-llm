import asyncio
from typing import Any

from httpx import ASGITransport, AsyncClient

from .mock_api import app

REVIEW_QUESTION = {"query": "نظر کاربران چیست؟"}


def assert_fields(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), actual
        for name, value in expected.items():
            assert_fields(actual[name], value)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), actual
        for actual_item, expected_item in zip(actual, expected):
            assert_fields(actual_item, expected_item)
    else:
        assert actual == expected, (actual, expected)


async def main() -> None:
    cases = [
        ("GET", "/health", None, 200, {"status": "ok"}),
        ("GET", "/api/categories", None, 200, {"items": ["ظروف", "نمونه"]}),
        ("GET", "/api/products/1", None, 200, {"product": {"product_id": 1, "price": None}}),
        ("GET", "/api/products/999", None, 404, {"code": "product_not_found"}),
        ("GET", "/api/products/0", None, 422, {"code": "invalid_request"}),
        ("POST", "/api/search", {"query": "کوله", "limit": 5}, 200,
         {"items": [{"product_id": 1}], "applied_filters": {}}),
        ("POST", "/api/search", {"query": "ماگ", "filters": {"category": "ظروف", "brand": "دیگر"}}, 200,
         {"items": [{"product_id": 2}], "applied_filters": {"category": "ظروف", "brand": "دیگر"}}),
        ("POST", "/api/search", {"query": "کوله", "filters": {"category": "ظروف"}}, 200, {"items": []}),
        ("POST", "/api/search", {"query": "   "}, 422, {"code": "invalid_request"}),
        ("POST", "/api/search", {"query": "کوله", "filters": {"min_price": 0}}, 422,
         {"code": "unsupported_filter"}),
        ("POST", "/api/search", {"query": "کوله", "filters": {"seller": "نمونه"}}, 422,
         {"code": "invalid_request"}),
        ("POST", "/api/search", {"query": "کوله", "filters": {"min_price": 100, "max_price": 20}}, 422,
         {"code": "invalid_request"}),
        ("POST", "/api/reviews/ask", {**REVIEW_QUESTION, "product_id": 1}, 200,
         {"product_id": 1, "answer_mode": "extractive", "answer": "جیب‌ها جادار هستند.",
          "evidence": [{"comment_id": 101, "product_id": 1, "text": "جیب‌ها جادار هستند."}]}),
        ("POST", "/api/reviews/ask", {**REVIEW_QUESTION, "product_id": 2}, 200,
         {"product_id": 2, "answer_mode": "insufficient_evidence", "answer": None, "evidence": []}),
        ("POST", "/api/reviews/ask", {**REVIEW_QUESTION, "product_id": 999}, 404,
         {"code": "product_not_found"}),
        ("POST", "/api/reviews/ask", {**REVIEW_QUESTION, "product_id": "1"}, 422,
         {"code": "invalid_request"}),
        ("POST", "/api/products/compare", {"product_ids": [2, 1]}, 200,
         {"items": [{"product_id": 2}, {"product_id": 1}]}),
        ("POST", "/api/products/compare", {"product_ids": [1, 1]}, 422,
         {"code": "invalid_request"}),
        ("POST", "/api/analytics", {"query": "تعداد محصولات چقدر است؟"}, 200,
         {"scope": "fixture", "metric": "product_count", "value": 2}),
        ("POST", "/api/analytics", {"query": "شمارش", "metric": "review_count"}, 200,
         {"scope": "fixture", "metric": "review_count", "value": 4}),
        ("POST", "/api/analytics", {"query": "نرخ", "metric": "recommendation_rate"}, 200,
         {"recommended_count": 3, "valid_label_count": 4, "value": 0.75}),
        ("POST", "/api/analytics",
         {"query": "نرخ", "metric": "recommendation_rate", "filters": {"category": "ظروف"}}, 200,
         {"recommended_count": 0, "valid_label_count": 0, "value": None}),
        ("POST", "/api/analytics", {"query": "وضع چطوره؟"}, 422, {"code": "unknown_metric"}),
        ("POST", "/api/predict", {"text": "محصول معمولی بود."}, 200,
         {"recommendation_status": "no_idea"}),
    ]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        for method, path, payload, status, expected in cases:
            response = await client.request(method, path, json=payload)
            assert response.status_code == status, (path, response.status_code, response.text)
            body = response.json()
            if status == 200:
                assert body["mode"] == "mock"
                assert_fields(body["data"], expected)
            else:
                assert_fields(body["detail"], expected)
                assert isinstance(body["detail"]["message"], str)
            print(method, path, response.status_code)

        response = await client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert len(schema["paths"]) == 8
        schemas = schema["components"]["schemas"]
        for operations in schema["paths"].values():
            for operation in operations.values():
                content = operation["responses"]["200"]["content"]["application/json"]
                model = content["schema"]["$ref"].split("/")[-1]
                payload = schemas[model]["properties"]["data"]
                if "$ref" in payload:
                    assert payload["$ref"].split("/")[-1] in schemas
                else:
                    assert len(payload["oneOf"]) == 2
                    assert payload["discriminator"]["propertyName"] == "metric"
        for path in ("/api/search", "/api/reviews/ask", "/api/products/{id}"):
            operation = next(iter(schema["paths"][path].values()))
            response_schema = operation["responses"]["422"]["content"]["application/json"]["schema"]
            assert response_schema["$ref"].endswith("/ErrorResponse")
        print("GET /openapi.json 200")
    print(f"PASS: {len(cases) + 1} in-process mock checks. QBC13 gateway requests: 0.")


if __name__ == "__main__":
    asyncio.run(main())
