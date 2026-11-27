"""Structured Task 2 response (backwards compatible with the original schema)."""
from __future__ import annotations

import json
import re
from pydantic import BaseModel, Field


class Point(BaseModel):
    point: str
    comment_ids: list[int] = Field(default_factory=list)
    quote: str = ""


class ProductAnswer(BaseModel):
    summary: str = ""
    positive_points: list[Point] = Field(default_factory=list)
    negative_points: list[Point] = Field(default_factory=list)
    observations: list[Point] = Field(default_factory=list)
    insufficient_evidence: bool = False
    inference: str = ""


def parse_response(response: str) -> ProductAnswer:
    if not isinstance(response, str): raise ValueError("LLM response is not text")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.I).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end >= start: text = text[start:end+1]
    return ProductAnswer.model_validate(json.loads(text))
