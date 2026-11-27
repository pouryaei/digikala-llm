"""Small deterministic parsers for monetary constraints and explicit product IDs."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata

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
    prefer_satisfaction: bool = False
    prefer_affordability: bool = False
    # Explicit monetary units (or CLI price arguments) require a verified source
    # currency. An amount without a currency may be applied in recorded units,
    # provided that this is stated in the result.
    explicit_price_currency: bool = False
    fallback_text: str | None = None
    scope_hint: str | None = None


_USERS = r"(?:خریدار(?:ان|ها)?|کاربر(?:ان|ها)?|مشتری(?:ان|ها)?)"
_OPINIONS = r"(?:نظر(?:ات|های|ها)?|بازخورد(?:های|ها)?)"
_POSITIVE = r"(?:خوب|مثبت|عالی)"
_ENDING = r"(?:\s+(?:باشن|باشند|باشه|باشد|هستن|هستند|اند|بودن|بودند))?"
_SATISFACTION = tuple(re.compile(p) for p in (
    rf"(?<!\w)(?:با\s+)?رضایت\s+(?:{_USERS}|بالا|زیاد|خوب)(?!\w)",
    rf"(?<!\w)(?:با\s+)?(?:امتیاز|نرخ\s+پیشنهاد\s+خرید)\s+(?:بالا|زیاد|خوب)(?!\w)",
    rf"(?<!\w)(?:با\s+)?{_OPINIONS}\s+{_POSITIVE}\s+{_USERS}(?!\w)",
    rf"(?<!\w)(?:که\s+)?{_OPINIONS}\s+{_USERS}(?:\s+[\w]+){{0,6}}?\s+{_POSITIVE}{_ENDING}(?!\w)",
    rf"(?<!\w)(?:که\s+)?{_USERS}(?:\s+[\w]+){{0,6}}?\s+راضی{_ENDING}(?!\w)",
))
_AFFORDABILITY = tuple(re.compile(p) for p in (
    r"(?<!\w)(?:که\s+)?(?:خیلی\s+)?(?:گرون|گران)\s+(?:نباشه|نباشد|نباشن|نباشند|نیست)(?!\w)",
    r"(?<!\w)(?:با\s+)?(?:قیمت\s+(?:مناسب|پایین|کم)|مقرون\s+به\s+صرفه|ارزان(?:\s+قیمت)?|ارزون|اقتصادی|کم\s+قیمت)(?!\w)",
))


def _preferences(text):
    # Normalize only the intent-bearing query. Ordinary title queries keep their
    # existing BM25 behavior and need not be re-indexed.
    candidate = unicodedata.normalize("NFKC", text).replace("\u200c", " ")
    candidate = re.sub(r"[\u064b-\u065f\u0670]", "", candidate)
    # These broad use-case requests do not name a specific item. Restrict them
    # to the matching catalogue families rather than returning books about travel
    # or small cosmetic cases for an everyday bag.
    scope = None
    if re.search(r"(?<!\w)(?:وسیله|وسایل|لوازم|تجهیزات)(?:\s+ی)?(?:\s+کاربردی)?\s+(?:برای\s+)?سفر(?!\w)", candidate):
        scope = "travel"
    elif (re.search(r"(?<!\w)کیف(?!\w)", candidate) and re.search(r"(?<!\w)روزمره(?!\w)", candidate)
          and not re.search(r"(?:کیف\s+(?:پول|آرایش|گوشی|موبایل)|لپ\s*تاپ)", candidate)):
        scope = "everyday_bag"
    elif re.search(r"(?<!\w)کوله(?!\w)", candidate) and re.search(r"(?<!\w)لپ\s*تاپ(?!\w)", candidate):
        scope = "backpack"
    satisfied = affordable = False
    for pattern in _SATISFACTION:
        candidate, n = pattern.subn(" ", candidate)
        satisfied |= bool(n)
    for pattern in _AFFORDABILITY:
        candidate, n = pattern.subn(" ", candidate)
        affordable |= bool(n)
    if not (satisfied or affordable or scope):
        return text, False, False, None, None
    candidate = re.sub(r"(?<!\w)(?:می\s*(?:خوام|خواهم|خوایم)|معرفی\s+کن|چند\s+وسیله(?:\s+ی)?(?:\s+کاربردی)?)(?!\w)", " ", candidate)
    candidate = re.sub(r"(?<!\w)(?:برای\s+)?استفاده(?:\s+ی)?\s+روزمره(?!\w)", " روزمره ", candidate)
    # Keep the stated purpose on the first attempt. If it is absent from the
    # catalogue vocabulary, a disclosed fallback can still find the product type.
    candidate = re.sub(r"(?:^|\s)(?:که|و|هم|کن|باشن|باشه)(?=\s|$)", " ", candidate)
    candidate = " ".join(candidate.strip(" .،؟!").split())
    fallback = re.sub(r"(?<!\w)روزمره(?!\w)", " ", candidate)
    fallback = " ".join(fallback.split())
    if scope == "travel":
        candidate = re.sub(r"(?<!\w)(?:وسیله|وسایل|لوازم|تجهیزات)(?:\s+ی)?(?:\s+کاربردی)?(?=\s+(?:برای\s+)?سفر)", " ", candidate)
        candidate, fallback = " ".join(candidate.split()) or "سفر", None
    elif scope == "backpack":
        fallback = re.sub(r"(?<!\w)(?:برای\s+)?لپ\s*تاپ(?!\w)|(?<!\w)سبک(?!\w)", " ", candidate)
        fallback = " ".join(fallback.split())
        if fallback in {"کوله", "یک کوله"}:
            fallback += " پشتی"
    return candidate, satisfied, affordable, fallback if fallback and fallback != candidate else None, scope


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
    explicit_currency = min_price is not None or max_price is not None
    for match in _RANGE.finditer(clean):
        a, scale_a, curr_a, b, scale_b, curr_b = match.groups()
        if not (scale_a or curr_a or scale_b or curr_b):
            continue
        lower = _amount(a, scale_a or (scale_b if not curr_a else None), curr_a or curr_b)
        upper = _amount(b, scale_b or (scale_a if not curr_b else None), curr_b or curr_a)
        explicit_currency |= bool(curr_a or curr_b)
        spans.append(match.span())
    for match in _CAP.finditer(clean):
        if any(start <= match.start() < end for start, end in spans):
            continue
        lead, number, scale, currency = match.groups()
        if not (scale or currency):  # e.g. model 3 to 5, not a budget
            continue
        value = _amount(number, scale, currency)
        explicit_currency |= bool(currency)
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
    text, satisfied, affordable, fallback, scope = _preferences(" ".join(clean.split()))
    return SearchRequest(text, lower, upper, satisfied, affordable, explicit_currency, fallback, scope)
