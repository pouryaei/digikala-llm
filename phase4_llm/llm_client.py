"""Opt-in OpenAI-compatible client for the QBC13 gateway; offline retrieval never calls it."""
from __future__ import annotations

import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://193.93.169.12/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def is_configured_key(value):
    key = (value or "").strip().casefold()
    return bool(key) and not key.startswith(("your", "<", "{", "replace", "put_", "put ", "insert", "example", "placeholder")) and not any(marker in key for marker in ("your_api", "your-api", "your_group", "your_qbc", "placeholder", "api_key_here", "group_key", "گروه"))


class LLMUnavailable(RuntimeError):
    def __init__(self, message, *, attempts=0):
        super().__init__(message)
        self.attempts = attempts


class LLMClient:
    def __init__(self, model=None, timeout=30, max_retries=2, json_mode=True, max_tokens=None):
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env", override=False)
        except ImportError:
            pass
        key = os.getenv("QBC13_API_KEY")
        # Do not send an unrelated personal OpenAI credential to the course gateway.
        if not is_configured_key(key):
            raise LLMUnavailable("QBC13_API_KEY is not configured with a real group key")
        try:
            from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError, BadRequestError
        except ImportError:
            raise LLMUnavailable("The optional openai SDK is not installed") from None
        self.model = model or os.getenv("OPENAI_MODEL") or os.getenv("QBC13_LLM_MODEL") or DEFAULT_MODEL
        self.base_url = os.getenv("QBC13_API_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        self.timeout = float(timeout)
        self.max_retries = min(max(1, int(max_retries)), 3)
        self.json_mode = bool(json_mode)
        self.max_tokens = min(max(150, int(max_tokens or os.getenv("QBC13_REVIEW_MAX_TOKENS", "700"))), 1800)
        self.client = OpenAI(api_key=key, base_url=self.base_url, timeout=self.timeout, max_retries=0)
        self.rate_errors = (RateLimitError, APITimeoutError, APIConnectionError)
        self.bad_request = BadRequestError

    @property
    def cache_config(self):
        return {"base_url": self.base_url, "max_tokens": self.max_tokens, "temperature": 0, "json_mode": self.json_mode}

    def generate(self, system, user):
        started = time.perf_counter()
        attempts = 0
        json_mode = self.json_mode
        for n in range(self.max_retries):
            attempts += 1
            try:
                kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0, max_tokens=self.max_tokens, **kwargs,
                )
                content = response.choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    raise LLMUnavailable("Provider returned no text", attempts=attempts)
                usage = response.usage
                return {"text": content, "latency": time.perf_counter()-started,
                        "prompt_tokens": int(usage.prompt_tokens or 0) if usage else 0,
                        "completion_tokens": int(usage.completion_tokens or 0) if usage else 0,
                        "usage_known": usage is not None, "api_attempts": attempts}
            except self.bad_request:
                if json_mode and n + 1 < self.max_retries:
                    json_mode = False
                    continue
                raise LLMUnavailable("Provider rejected the request", attempts=attempts) from None
            except self.rate_errors:
                if n + 1 == self.max_retries:
                    raise LLMUnavailable("Provider timeout or rate limit", attempts=attempts) from None
                time.sleep(min(1.0 * 2**n, 2.0))
            except LLMUnavailable:
                raise
            except Exception:
                # Never disclose a provider body, URL credentials, or key to UI.
                raise LLMUnavailable("LLM provider is temporarily unavailable", attempts=attempts) from None
        raise LLMUnavailable("LLM provider is temporarily unavailable", attempts=attempts)
