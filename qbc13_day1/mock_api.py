from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StringConstraints

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
PositiveId = Annotated[int, Field(gt=0)]
Label = Literal["recommended", "not_recommended", "no_idea"]

PRODUCT = {
    "product_id": 1,
    "title_fa": "کولهٔ نمونه",
    "category": "نمونه",
}
EVIDENCE = {"comment_id": 101, "product_id": 1, "text": "جیب‌ها جادار هستند."}


class Query(BaseModel):
    query: Question


class Search(Query):
    limit: int = Field(default=5, ge=1, le=20)


class ReviewQuestion(Query):
    product_id: PositiveId


class Comparison(BaseModel):
    product_ids: list[PositiveId] = Field(min_length=2, max_length=5)


class Prediction(BaseModel):
    text: Question


class MockResponse(BaseModel):
    mode: Literal["mock"] = "mock"
    data: dict[str, Any]


app = FastAPI(title="QBC13 Day 1 Mock", version="0.1.0")


def mock(**data: Any) -> MockResponse:
    return MockResponse(data=data)


def product(product_id: int) -> dict[str, Any]:
    if product_id != PRODUCT["product_id"]:
        raise HTTPException(status_code=404, detail="Mock product not found")
    return PRODUCT


@app.get("/health", response_model=MockResponse)
def health() -> MockResponse:
    return mock(status="ok")


@app.post("/api/search", response_model=MockResponse)
def search(request: Search) -> MockResponse:
    items = [PRODUCT] if "کوله" in request.query else []
    return mock(items=items[:request.limit])


@app.post("/api/reviews/ask", response_model=MockResponse)
def ask(request: ReviewQuestion) -> MockResponse:
    product(request.product_id)
    return mock(answer=EVIDENCE["text"], evidence=[EVIDENCE])


@app.post("/api/products/compare", response_model=MockResponse)
def compare(request: Comparison) -> MockResponse:
    return mock(items=[product(product_id) for product_id in request.product_ids])


@app.post("/api/analytics", response_model=MockResponse)
def analytics(request: Query) -> MockResponse:
    return mock(query=request.query, metric="product_count", value=1, scope="fixture")


@app.post("/api/predict", response_model=MockResponse)
def predict(request: Prediction) -> MockResponse:
    label: Label = "no_idea"
    return mock(recommendation_status=label)


@app.get("/api/categories", response_model=MockResponse)
def categories() -> MockResponse:
    return mock(items=[PRODUCT["category"]])


@app.get("/api/products/{id}", response_model=MockResponse)
def get_product(id: PositiveId) -> MockResponse:
    return mock(product=product(id))
