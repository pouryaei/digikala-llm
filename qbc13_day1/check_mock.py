import asyncio

from httpx import ASGITransport, AsyncClient

from .mock_api import app


async def main() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        cases = [
            ("GET", "/health", None, 200),
            ("GET", "/api/categories", None, 200),
            ("GET", "/api/products/1", None, 200),
            ("GET", "/api/products/999", None, 404),
            ("POST", "/api/search", {"query": "کوله", "limit": 5}, 200),
            ("POST", "/api/search", {"query": "   "}, 422),
            ("POST", "/api/reviews/ask", {"query": "نظر کاربران چیست؟", "product_id": 1}, 200),
            ("POST", "/api/products/compare", {"product_ids": [1, 1]}, 200),
            ("POST", "/api/analytics", {"query": "تعداد محصولات چقدر است؟"}, 200),
            ("POST", "/api/predict", {"text": "محصول معمولی بود."}, 200),
            ("GET", "/openapi.json", None, 200),
        ]
        for method, path, payload, expected in cases:
            response = await client.request(method, path, json=payload)
            assert response.status_code == expected, (path, response.status_code, response.text)
            if expected == 200 and path != "/openapi.json":
                body = response.json()
                assert body["mode"] == "mock" and isinstance(body["data"], dict)
                if path == "/api/reviews/ask":
                    assert body["data"]["evidence"][0]["comment_id"] == 101
                    assert body["data"]["evidence"][0]["product_id"] == 1
            print(method, path, response.status_code)
    print(f"PASS: {len(cases)} in-process mock checks. QBC13 gateway requests: 0.")


if __name__ == "__main__":
    asyncio.run(main())
