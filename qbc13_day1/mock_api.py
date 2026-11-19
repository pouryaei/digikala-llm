from typing import Annotated, Generic, Literal, NoReturn, TypeVar

from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StringConstraints, model_validator

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
PositiveId = Annotated[StrictInt, Field(gt=0, le=2**63 - 1)]
ProductPathId = Annotated[int, Path(gt=0, le=2**63 - 1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
Rate = Annotated[float, Field(ge=0, le=1)]
Label = Literal["recommended", "not_recommended", "no_idea"]
Metric = Literal["product_count", "review_count", "recommendation_rate"]
AnswerMode = Literal["grounded", "extractive", "insufficient_evidence"]


class Filters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Question | None = None
    brand: Question | None = None
    min_price: NonNegativeInt | None = None
    max_price: NonNegativeInt | None = None

    @model_validator(mode="after")
    def valid_price_range(self) -> "Filters":
        if self.min_price is not None and self.max_price is not None and self.min_price > self.max_price:
            raise ValueError("min_price must not exceed max_price")
        return self


class Query(BaseModel):
    query: Question


class Search(Query):
    limit: int = Field(default=5, ge=1, le=20)
    filters: Filters = Field(default_factory=Filters)


class ReviewQuestion(Query):
    product_id: PositiveId


class Comparison(BaseModel):
    product_ids: list[PositiveId] = Field(min_length=2, max_length=5)

    @model_validator(mode="after")
    def unique_product_ids(self) -> "Comparison":
        if len(set(self.product_ids)) != len(self.product_ids):
            raise ValueError("product_ids must be distinct")
        return self


class Analytics(Query):
    metric: Metric | None = None
    filters: Filters = Field(default_factory=Filters)


class Prediction(BaseModel):
    text: Question


class Product(BaseModel):
    product_id: PositiveId
    title_fa: str
    category: str | None
    brand: str | None
    price: NonNegativeInt | None = None
    price_unit: str | None = None
    rating: float | None = None
    rating_count: NonNegativeInt | None = None
    comment_count: NonNegativeInt
    recommended_count: NonNegativeInt
    valid_label_count: NonNegativeInt
    recommendation_rate: Rate | None


class Evidence(BaseModel):
    comment_id: PositiveId
    product_id: PositiveId
    text: str


class HealthData(BaseModel):
    status: Literal["ok"]


class CategoriesData(BaseModel):
    items: list[str]


class ProductData(BaseModel):
    product: Product


class ProductListData(BaseModel):
    items: list[Product]


class SearchData(ProductListData):
    applied_filters: dict[str, str]


class ReviewQAData(BaseModel):
    product_id: PositiveId
    answer: str | None
    answer_mode: AnswerMode
    evidence: list[Evidence]


class AnalyticsCountData(BaseModel):
    metric: Literal["product_count", "review_count"]
    scope: Literal["fixture"]
    applied_filters: dict[str, str]
    value: NonNegativeInt


class AnalyticsRateData(BaseModel):
    metric: Literal["recommendation_rate"]
    scope: Literal["fixture"]
    applied_filters: dict[str, str]
    recommended_count: NonNegativeInt
    valid_label_count: NonNegativeInt
    value: Rate | None


AnalyticsData = Annotated[AnalyticsCountData | AnalyticsRateData, Field(discriminator="metric")]


class PredictionData(BaseModel):
    recommendation_status: Label


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    detail: ErrorDetail


Data = TypeVar("Data", bound=BaseModel)


class MockResponse(BaseModel, Generic[Data]):
    mode: Literal["mock"] = "mock"
    data: Data


PRODUCTS = {
    1: Product(
        product_id=1, title_fa="کولهٔ نمونه", category="نمونه", brand="آزمایشی",
        comment_count=4, recommended_count=3, valid_label_count=4, recommendation_rate=0.75,
    ),
    2: Product(
        product_id=2, title_fa="ماگ نمونه", category="ظروف", brand="دیگر",
        comment_count=0, recommended_count=0, valid_label_count=0, recommendation_rate=None,
    ),
}
SEARCH_TERMS = {1: "کوله", 2: "ماگ"}
EVIDENCE = Evidence(comment_id=101, product_id=1, text="جیب‌ها جادار هستند.")
VALIDATION_ERRORS = {422: {"model": ErrorResponse}}
PRODUCT_ERRORS = {**VALIDATION_ERRORS, 404: {"model": ErrorResponse}}
app = FastAPI(title="QBC13 Day 1 Mock", version="0.1.0")


def fail(status: int, code: str, message: str) -> NoReturn:
    detail = ErrorDetail(code=code, message=message)
    raise HTTPException(status_code=status, detail=detail.model_dump())


@app.exception_handler(RequestValidationError)
async def invalid_request(_: Request, __: RequestValidationError) -> JSONResponse:
    body = ErrorResponse(detail=ErrorDetail(code="invalid_request", message="ورودی درخواست معتبر نیست."))
    return JSONResponse(status_code=422, content=body.model_dump())


def product(product_id: int) -> Product:
    result = PRODUCTS.get(product_id)
    if result is None:
        fail(404, "product_not_found", "محصولی با این شناسه پیدا نشد.")
    return result


def applied_filters(filters: Filters) -> dict[str, str]:
    if filters.min_price is not None or filters.max_price is not None:
        fail(422, "unsupported_filter", "فیلتر قیمت تا تأیید واحد قیمت در دسترس نیست.")
    return {
        name: value for name, value in (("category", filters.category), ("brand", filters.brand))
        if value is not None
    }


def filtered_products(filters: dict[str, str]) -> list[Product]:
    return [
        item for item in PRODUCTS.values()
        if all(getattr(item, field) == value for field, value in filters.items())
    ]


def analytics_metric(request: Analytics) -> Metric:
    if request.metric is not None:
        return request.metric
    if "نرخ پیشنهاد" in request.query:
        return "recommendation_rate"
    if "تعداد نظر" in request.query:
        return "review_count"
    if "تعداد محصول" in request.query:
        return "product_count"
    fail(422, "unknown_metric", "معیار تحلیلی را مشخص کنید.")


@app.get("/health", response_model=MockResponse[HealthData])
def health() -> MockResponse[HealthData]:
    return MockResponse[HealthData](data=HealthData(status="ok"))


@app.post("/api/search", response_model=MockResponse[SearchData], responses=VALIDATION_ERRORS)
def search(request: Search) -> MockResponse[SearchData]:
    filters = applied_filters(request.filters)
    items = [
        item for item in filtered_products(filters)
        if SEARCH_TERMS[item.product_id] in request.query
    ][:request.limit]
    return MockResponse[SearchData](data=SearchData(items=items, applied_filters=filters))


@app.post("/api/reviews/ask", response_model=MockResponse[ReviewQAData], responses=PRODUCT_ERRORS)
def ask(request: ReviewQuestion) -> MockResponse[ReviewQAData]:
    product(request.product_id)
    evidence = [EVIDENCE] if request.product_id == EVIDENCE.product_id else []
    return MockResponse[ReviewQAData](data=ReviewQAData(
        product_id=request.product_id,
        answer=EVIDENCE.text if evidence else None,
        answer_mode="extractive" if evidence else "insufficient_evidence",
        evidence=evidence,
    ))


@app.post("/api/products/compare", response_model=MockResponse[ProductListData], responses=PRODUCT_ERRORS)
def compare(request: Comparison) -> MockResponse[ProductListData]:
    items = [product(product_id) for product_id in request.product_ids]
    return MockResponse[ProductListData](data=ProductListData(items=items))


@app.post("/api/analytics", response_model=MockResponse[AnalyticsData], responses=VALIDATION_ERRORS)
def analytics(request: Analytics) -> MockResponse[AnalyticsData]:
    filters = applied_filters(request.filters)
    items = filtered_products(filters)
    metric = analytics_metric(request)
    if metric == "recommendation_rate":
        numerator = sum(item.recommended_count for item in items)
        denominator = sum(item.valid_label_count for item in items)
        data: AnalyticsCountData | AnalyticsRateData = AnalyticsRateData(
            metric=metric, scope="fixture", applied_filters=filters,
            recommended_count=numerator, valid_label_count=denominator,
            value=numerator / denominator if denominator else None,
        )
    else:
        count = len(items) if metric == "product_count" else sum(item.comment_count for item in items)
        data = AnalyticsCountData(metric=metric, scope="fixture", applied_filters=filters, value=count)
    return MockResponse[AnalyticsData](data=data)


@app.post("/api/predict", response_model=MockResponse[PredictionData], responses=VALIDATION_ERRORS)
def predict(request: Prediction) -> MockResponse[PredictionData]:
    return MockResponse[PredictionData](data=PredictionData(recommendation_status="no_idea"))


@app.get("/api/categories", response_model=MockResponse[CategoriesData])
def categories() -> MockResponse[CategoriesData]:
    items = sorted({item.category for item in PRODUCTS.values() if item.category is not None})
    return MockResponse[CategoriesData](data=CategoriesData(items=items))


@app.get("/api/products/{id}", response_model=MockResponse[ProductData], responses=PRODUCT_ERRORS)
def get_product(id: ProductPathId) -> MockResponse[ProductData]:
    return MockResponse[ProductData](data=ProductData(product=product(id)))
