"""Small deterministic parsers for monetary constraints and explicit product IDs."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .task2_review_store import DIGITS, identifier

_MONEY = r"([0-9]+(?:[,٬][0-9]{3})*(?:[.٫][0-9]+)?)\s*(میلیارد|میلیون|هزار)?(?:\s*(تومان|تومن|ریال))?(?!\w)"
_RANGE = re.compile(r"(?<!\w)بین\s+" + _MONEY + r"\s+(?:و|تا)\s+" + _MONEY)
_CAP = re.compile(r"(?<!\w)(تا\s+سقف|کمتر\s+از|بیشتر\s+از|حداکثر|حداقل|بالای|زیر|تا)\s+" + _MONEY)
_SCALE = {None: 1, "هزار": 1000, "میلیون": 1000000, "میلیارد": 1000000000}
_PRODUCT_ID = re.compile(r"(?:#\s*|(?<!\w)(?:شناسه|کد)\s+(?:محصول|کالا)\s*[:：]?\s*)([0-9]{1,19})(?![\w])")
_MONEY_NORM = str.maketrans("يك", "یک")


@dataclass
class SearchRequest:
    text: str
    min_price: float | None = None
    max_price: float | None = None


def explicit_product_ids(question):
    return list(dict.fromkeys(identifier(x) for x in _PRODUCT_ID.findall(question.translate(DIGITS).translate(_MONEY_NORM))))


def _amount(number, scale, currency):
    value = float(number.replace(",", "").replace("٬", "").replace("٫", ".")) * _SCALE[scale]
    return value / 10 if currency == "ریال" else value


def _check(value, name, *, ceiling=None):
    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0 or (
        ceiling is not None and float(value) > ceiling
    ):
        raise ValueError(f"{name} مقدار معتبر ندارد.")
    return float(value)


def search_request(query, min_price=None, max_price=None):
    # Parse on normalized digits only when a real monetary unit is provided.
    clean = query.strip().translate(DIGITS).translate(_MONEY_NORM)
    clean = re.sub(r"(?<!\w)(کم|بیش)[\u200c\s]+تر(?!\w)", r"\1تر", clean)
    lower, upper = None, None
    spans = []
    for match in _RANGE.finditer(clean):
        a, scale_a, curr_a, b, scale_b, curr_b = match.groups()
        if not (scale_a or curr_a or scale_b or curr_b):
            continue
        lower = _amount(a, scale_a or (scale_b if not curr_a else None), curr_a or curr_b)
        upper = _amount(b, scale_b or (scale_a if not curr_b else None), curr_b or curr_a)
        spans.append(match.span())
    for match in _CAP.finditer(clean):
        if any(start <= match.start() < end for start, end in spans):
            continue
        lead, number, scale, currency = match.groups()
        if not (scale or currency):  # e.g. model 3 to 5, not a budget
            continue
        value = _amount(number, scale, currency)
        if lead in {"حداقل", "بالای", "بیشتر از"}:
            lower = value if lower is None else max(lower, value)
        else:
            upper = value if upper is None else min(upper, value)
        spans.append(match.span())
    for start, end in sorted(spans, reverse=True):
        clean = clean[:start] + " " + clean[end:]
    lower = _check(min_price if min_price is not None else lower, "min_price")
    upper = _check(max_price if max_price is not None else upper, "max_price")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("حداقل قیمت از حداکثر بیشتر است.")
    return SearchRequest(" ".join(clean.split()), lower, upper)
