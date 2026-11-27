"""Optional standalone API for Task 2; the rest of the team backend is unchanged."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.review_qa import ask_reviews

app = FastAPI(title="Forouhar Task 2 — Review QA", version="2.0")


class ReviewQuestion(BaseModel):
    product_id: int = Field(strict=True, gt=0, le=2**63-1)
    query: str = Field(min_length=1, max_length=1000)


@app.get("/health")
def health():
    return {"status": "ok", "service": "review_qa"}


@app.post("/api/reviews/ask")
def review_qa(payload: ReviewQuestion):
    try:
        data = ask_reviews(payload.product_id, payload.query)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return {"data": data, "mode": "serving"}
