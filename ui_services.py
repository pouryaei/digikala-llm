"""Connect Streamlit to the team's API and optional Python service functions."""

from __future__ import annotations

from collections.abc import Mapping
import importlib
from inspect import signature
import logging
from numbers import Integral, Real
import os
import sqlite3
from pathlib import Path
import re
from time import perf_counter
from urllib.parse import urlsplit

import duckdb
import requests

import local_catalog
from analytics_service import open_analytics_tables, resolve_analytics_files
from src.config import DenseUnavailable
from src.task2_review_store import ReviewDataError


ROOT = Path(__file__).resolve().parent
MAX_ID = 2**63 - 1
VALID_LABELS = {"recommended", "not_recommended", "no_idea"}
LOGGER = logging.getLogger(__name__)


class APIError(Exception):
    def __init__(self, message: str, *, unavailable: bool = False):
        super().__init__(message)
        self.unavailable = unavailable


def product_id_value(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        identifier = int(value)
    elif isinstance(value, str) and re.fullmatch(r"[0-9]{1,19}", value.strip()):
        identifier = int(value)
    else:
        return None
    return identifier if 1 <= identifier <= MAX_ID else None


def backend_url():
    url = os.environ.get("QBC13_BACKEND_URL", "http://127.0.0.1:8000").strip()
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or port == 0
    ):
        return None
    return url.rstrip("/")


def _request_api(method, path, payload=None):
    url = backend_url()
    if url is None:
        raise APIError("آدرس QBC13_BACKEND_URL معتبر نیست.", unavailable=True)
    try:
        response = requests.request(method, url + path, json=payload, timeout=(3.5, 45), allow_redirects=False)
    except requests.Timeout:
        raise APIError("زمان انتظار برای پاسخ سرویس تمام شد.", unavailable=True) from None
    except requests.ConnectionError:
        raise APIError("ارتباط با سرویس برقرار نشد. اجرای FastAPI و آدرس سرویس را بررسی کنید.", unavailable=True) from None
    except requests.RequestException:
        raise APIError("ارسال درخواست به سرویس ناموفق بود.", unavailable=True) from None

    if not 200 <= response.status_code < 300:
        messages = {
            401: "سرویس درخواست را بدون دسترسی معتبر نمی‌پذیرد.",
            403: "دسترسی به این مسیر مجاز نیست.",
            404: "محصول یا مسیر در سرویس پیدا نشد.",
            422: "سرویس ورودی را نپذیرفت. مقادیر فرم را بررسی کنید.",
            429: "سرویس درخواست‌های زیادی دریافت کرده است.",
            501: "این قابلیت هنوز در سرویس فعال نشده است.",
            503: "سرویس یا مدل پیش‌بینی فعلاً آماده نیست.",
        }
        status = response.status_code
        raise APIError(f"{messages.get(status, 'سرویس نتوانست درخواست را انجام دهد.')} (HTTP {status})",
                       unavailable=status == 429 or status >= 500)
    try:
        body = response.json()
    except ValueError:
        raise APIError("سرویس پاسخ JSON معتبر برنگرداند.", unavailable=True) from None
    if not isinstance(body, dict) or not isinstance(body.get("data", body), dict):
        raise APIError("ساختار پاسخ با قرارداد سرویس سازگار نیست.", unavailable=True)
    return {"data": body.get("data", body), "mode": body.get("mode")}


def _python_callable(setting: str, default_name: str, alternatives=()):
    specification = os.environ.get(setting, "").strip()
    if not specification:
        specification = next((f"{module}:{default_name}" for module, filename in alternatives
                              if (ROOT / filename).is_file()), "")
    if not specification:
        return None
    module_name, separator, name = specification.partition(":")
    name = name if separator else default_name
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module_name) or not re.fullmatch(r"[A-Za-z_]\w*", name):
        raise APIError(f"مقدار {setting} باید مسیر ماژول و نام تابع پایتون باشد.", unavailable=True)
    try:
        function = getattr(importlib.import_module(module_name), name)
        if not callable(function):
            raise TypeError("Expected a callable")
    except Exception as error:
        LOGGER.warning("Unable to load %s (%s)", setting, type(error).__name__)
        raise APIError(f"تابع تیم از مسیر {setting} بارگذاری نشد. فایل و وابستگی‌های آن را بررسی کنید.", unavailable=True) from None
    return function


def _invoke(function, *args, **kwargs):
    try:
        result = function(*args, **kwargs)
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        return result
    except Exception as error:
        LOGGER.warning("Team service failed (%s)", type(error).__name__)
        raise APIError("تابع تیم نتوانست پاسخ معتبر تولید کند.", unavailable=True) from None


def _entry(value, mode="team_function"):
    if isinstance(value, list):
        value = {"items": value}
    if not isinstance(value, Mapping) or not isinstance(value.get("data", value), Mapping):
        raise APIError("تابع تیم ساختار پاسخ مورد انتظار را برنگرداند.", unavailable=True)
    return {"data": dict(value.get("data", value)), "mode": value.get("mode", mode)}


def _prediction_score(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1:
        raise APIError("مدل مقدار احتمال نامعتبر برگرداند.", unavailable=True)
    return float(value)


def _validate_prediction_entry(entry):
    data = entry["data"]
    valid = data.get("valid", True)
    if not isinstance(valid, bool):
        raise APIError("وضعیت معتبر بودن ورودی در پاسخ مدل نامعتبر است.", unavailable=True)
    if not valid:
        # M2 uses no_idea as INVALID_LABEL for rejected input, not as a prediction.
        return {**entry, "data": {**data, "valid": False, "recommendation_status": None}}

    label = data.get("recommendation_status", data.get("label"))
    if not isinstance(label, str) or label not in VALID_LABELS:
        raise APIError("مدل برچسب معتبر پیش‌بینی را برنگرداند. خروجی مدل را بررسی کنید.", unavailable=True)
    normalized = {**data, "valid": True, "recommendation_status": str(label)}
    confidence = data.get("confidence", data.get("probability"))
    probabilities = data.get("probabilities")
    if probabilities is not None:
        if not isinstance(probabilities, Mapping) or not probabilities or not set(probabilities) <= VALID_LABELS:
            raise APIError("ساختار احتمال کلاس‌ها در پاسخ مدل معتبر نیست.", unavailable=True)
        scores = {str(key): _prediction_score(value) for key, value in probabilities.items()}
        normalized["probabilities"] = scores
        if confidence is None:
            confidence = scores.get(label)
    if confidence is not None:
        normalized["confidence"] = _prediction_score(confidence)
    return {**entry, "data": normalized}


def _prediction_text(text, title):
    return " ".join(part for part in (title, text) if isinstance(part, str) and part.strip())


def _invoke_prediction(function, text, title):
    """Keep optional title support without breaking existing text-only team functions."""
    kwargs = {}
    if title:
        try:
            signature(function).bind(text, title=title)
        except (TypeError, ValueError):
            text = _prediction_text(text, title)
        else:
            kwargs["title"] = title
    try:
        returned = function(text, **kwargs)
        if hasattr(returned, "model_dump"):
            returned = returned.model_dump()
    except FileNotFoundError:
        raise APIError(
            "فایل مدل آموزش‌دیده پیدا نشد. پوشهٔ m2_outputs/xgboost یا sentiment_model.joblib را بررسی کنید "
            "یا مسیر DIGIKALA_MODEL_PATH را تنظیم کنید.", unavailable=True,
        ) from None
    except ImportError:
        raise APIError(
            "وابستگی‌های مدل بارگذاری نشدند. xgboost، joblib و نسخهٔ scikit-learn سازگار با آموزش را نصب کنید.",
            unavailable=True,
        ) from None
    except Exception as error:
        LOGGER.warning("Prediction function failed (%s)", type(error).__name__)
        raise APIError("اجرای مدل پیش‌بینی ناموفق بود. فایل مدل و نسخهٔ وابستگی‌ها را بررسی کنید.",
                       unavailable=True) from None
    if isinstance(returned, str):
        returned = {"recommendation_status": str(returned)}
    return _validate_prediction_entry(_entry(returned))


def predict_recommendation(text: str, title: str | None = None) -> dict:
    """Use the configured local model; otherwise try the team's API."""
    direct_error = None
    try:
        function = _python_callable("QBC13_PREDICT_FUNCTION", "predict_recommendation", (
            ("ml_service", "ml_service.py"), ("src.ml_service", "src/ml_service.py"),
        ))
        if function is not None:
            return _invoke_prediction(function, text, title)
    except APIError as error:
        if not error.unavailable:
            raise
        direct_error = error

    try:
        # The existing HTTP contract accepts one text field, not a separate title.
        entry = _validate_prediction_entry(_request_api("POST", "/api/predict", {
            "text": _prediction_text(text, title),
        }))
    except APIError as error:
        if direct_error is not None:
            raise APIError(f"{direct_error} سرویس تیم نیز پاسخ قابل استفاده نداد: {error}",
                           unavailable=True) from None
        raise
    if direct_error is not None:
        entry["notice"] = "مدل محلی در دسترس نبود؛ نتیجه از سرویس پیش‌بینی تیم دریافت شد."
    return entry


def _team_request(method, path, payload):
    if path == "/api/search" and method == "POST":
        function = _python_callable("QBC13_SEARCH_FUNCTION", "search_products")
        if function:
            return _entry(_invoke(function, query=payload["query"], filters=payload.get("filters", {}), limit=payload.get("limit", 5)))
    elif path == "/api/reviews/ask" and method == "POST":
        function = _python_callable("QBC13_REVIEW_FUNCTION", "ask_reviews")
        if function:
            return _entry(_invoke(function, product_id=payload["product_id"], query=payload["query"]))
    elif path == "/api/predict" and method == "POST":
        return predict_recommendation(payload["text"], title=payload.get("title"))
    return _request_api(method, path, payload)


def _local_request(method, path, payload):
    if method == "POST" and path == "/api/search":
        return local_catalog.search_products(payload["query"], payload.get("filters"), payload.get("limit", 5))
    if method == "POST" and path == "/api/reviews/ask":
        return local_catalog.ask_reviews(payload["product_id"], payload["query"])
    if method == "GET" and path == "/api/categories":
        return local_catalog.get_categories()
    if method == "GET" and (product_match := re.fullmatch(r"/api/products/([0-9]+)", path)):
        return local_catalog.get_product(int(product_match[1]))
    return None


def _validate_review_entry(entry, product_id):
    data = entry["data"]
    if product_id_value(data.get("product_id")) != product_id:
        raise APIError("پاسخ سرویس به محصول دیگری مربوط بود.", unavailable=True)
    if data.get("answer_mode") == "insufficient_evidence":
        return {**entry, "data": {**data, "answer": None, "evidence": []}}
    answer = data.get("answer")
    items = data.get("evidence")
    if not isinstance(answer, str) or not answer.strip() or not isinstance(items, list) or not items:
        raise APIError("مدل پاسخ مستند و قابل نمایش برنگرداند.", unavailable=True)

    evidence = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise APIError("ساختار ارجاع‌های پاسخ نامعتبر است.", unavailable=True)
        identifier = product_id_value(item.get("comment_id", item.get("review_id")))
        if identifier is None or product_id_value(item.get("product_id")) != product_id:
            raise APIError("پاسخ مدل ارجاع نامعتبر یا ارجاع به محصول دیگری داشت.", unavailable=True)
        evidence[identifier] = {**item, "comment_id": identifier, "product_id": product_id}

    # When the local corpus is installed, look up references instead of trusting a supplied quote.
    if entry["mode"] not in ("local", "mock") and data.get("scope") != "fixture":
        try:
            with open_analytics_tables(resolve_analytics_files()) as con:
                rows = con.execute("""
                    SELECT try_cast(comment_id AS BIGINT), title_raw, body_raw, advantages_raw, disadvantages_raw
                    FROM reviews WHERE try_cast(product_id AS BIGINT) = ?
                      AND try_cast(comment_id AS BIGINT) IN (SELECT unnest(?::BIGINT[]))
                """, [product_id, list(evidence)]).fetchall()
        except (OSError, FileNotFoundError, duckdb.Error):
            rows = None
        if rows is not None:
            originals = {identifier: [text for text in fields if isinstance(text, str) and text.strip()]
                         for identifier, *fields in rows}
            if evidence.keys() - originals.keys() or any(not originals[key] for key in evidence):
                raise APIError("برخی شناسه‌های ارجاع در نظرهای واقعی این محصول پیدا نشدند.", unavailable=True)
            for key, item in evidence.items():
                quote = item.get("text")
                if not isinstance(quote, str) or not quote.strip() or not any(quote.strip() in text for text in originals[key]):
                    item["text"] = "\n".join(originals[key])[:1500]

    if any(not isinstance(item.get("text"), str) or not item["text"].strip() for item in evidence.values()):
        raise APIError("متن شواهد قابل نمایش نیست.", unavailable=True)
    return {**entry, "data": {**data, "evidence": list(evidence.values())}}


def _bundled_request(method, path, payload):
    if not (ROOT / "storage" / "catalog.sqlite3").is_file():
        return None
    from src.dashboard_rag import dispatch
    try:
        data = dispatch(method, path, payload or {})
    except (ValueError, ReviewDataError, DenseUnavailable) as exc:
        raise APIError(str(exc)) from None
    except (OSError, sqlite3.Error, KeyError):
        raise APIError("فایل‌های داده یا شاخص بسته قابل خواندن نیستند.") from None
    if data is None:
        return None
    return {"data": data, "mode": "local"}


def _team_explicit(path):
    return bool(
        os.environ.get("QBC13_BACKEND_URL", "").strip() or
        (path == "/api/search" and os.environ.get("QBC13_SEARCH_FUNCTION", "").strip()) or
        (path == "/api/reviews/ask" and os.environ.get("QBC13_REVIEW_FUNCTION", "").strip())
    )


def api_call(method, path, payload=None):
    started = perf_counter()
    is_review_request = method == "POST" and path == "/api/reviews/ask"
    # New bundles work without FastAPI, while explicitly configured team services
    # and apps without the bundled store retain their previous behavior.
    if not _team_explicit(path):
        local_bundle = _bundled_request(method, path, payload)
        if local_bundle is not None:
            if is_review_request:
                local_bundle = _validate_review_entry(local_bundle, payload["product_id"])
            return {**local_bundle, "elapsed": perf_counter() - started}
    try:
        entry = _team_request(method, path, payload)
        if is_review_request:
            entry = _validate_review_entry(entry, payload["product_id"])
    except APIError as error:
        if not error.unavailable:
            raise
        try:
            local_bundle = _bundled_request(method, path, payload)
            data = local_bundle["data"] if local_bundle is not None else _local_request(method, path, payload)
        except LookupError as missing:
            raise APIError(str(missing)) from None
        except (OSError, ValueError, duckdb.Error) as local_error:
            LOGGER.warning("Local fallback unavailable (%s)", type(local_error).__name__)
            raise APIError(f"{error} دادهٔ محلی هم در دسترس نیست.") from None
        if data is None:
            if path == "/api/predict":
                raise APIError(f"مدل پیش‌بینی پاسخ نداد. {error} اجرای سرویس تیم یا فایل ml_service.py و مدل آموزش‌دیدهٔ آن را بررسی کنید.") from None
            raise error
        entry = {
            "data": data, "mode": "local",
            "notice": "سرویس تیم پاسخ قابل استفاده نداد. نتیجه از دادهٔ واقعی محلی بازیابی شده است.",
        }
        if is_review_request:
            entry = _validate_review_entry(entry, payload["product_id"])
    return {**entry, "elapsed": perf_counter() - started}
