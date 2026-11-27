from typing import List
from pydantic import BaseModel, Field


class ComparisonPoint(BaseModel):
    point: str
    comment_ids: List[int] = Field(default_factory=list)


class ProductComparison(BaseModel):
    product_id: int
    strengths: List[ComparisonPoint] = Field(default_factory=list)
    weaknesses: List[ComparisonPoint] = Field(default_factory=list)


class CriterionComparison(BaseModel):
    criterion: str
    differences: str
    comment_ids: List[int] = Field(default_factory=list)


class ComparisonAnswer(BaseModel):
    summary: str

    products: List[ProductComparison] = Field(default_factory=list)
    criteria: List[CriterionComparison] = Field(default_factory=list)
    conclusion: str

    insufficient_evidence: bool = False

    inference: str = ""