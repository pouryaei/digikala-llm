import argparse
import json
import os
import time
from decimal import Decimal
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

URL = "https://193.93.169.12/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"
ESTIMATED_MAX_INPUT = 1000
MAX_OUTPUT = 32
DAY1_SPEND_CAP = Decimal("0.10")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate one probe, or explicitly send it once.")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--input-rate", type=Decimal, default=Decimal("0.15"))
    parser.add_argument("--output-rate", type=Decimal, default=Decimal("0.60"))
    args = parser.parse_args()
    if any(not rate.is_finite() or rate < 0 for rate in (args.input_rate, args.output_rate)):
        parser.error("Rates must be finite nonnegative USD per million tokens.")
    estimate = (
        ESTIMATED_MAX_INPUT * args.input_rate + MAX_OUTPUT * args.output_rate
    ) / 1_000_000
    print(f"Model: {MODEL} | reference estimate: ${estimate:.6f} | output cap: {MAX_OUTPUT}")
    if not args.send:
        print("DRY RUN. Sent requests: 0.")
        return 0

    key = os.getenv("QBC13_API_KEY", "").strip()
    if not key:
        print("QBC13_API_KEY is missing. Sent requests: 0.")
        return 2
    if estimate > DAY1_SPEND_CAP:
        print("Reference estimate exceeds the Day 1 test cap. Sent requests: 0.")
        return 2

    request = Request(
        URL,
        data=json.dumps({
            "model": MODEL,
            "messages": [
                {"role": "system", "content": 'Return only a JSON object: {"ok": true}.'},
                {"role": "user", "content": "Confirm that the connection works."},
            ],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": MAX_OUTPUT,
        }).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    opener = build_opener(NoRedirect())
    start = time.monotonic()
    try:
        with opener.open(request, timeout=20) as response:
            data = json.load(response)
            print("HTTP:", response.status)
            print("Model returned:", data.get("model"))
            print("Request ID:", response.headers.get("x-request-id", "not reported"))
            print("Usage:", json.dumps(data.get("usage"), ensure_ascii=True))
            try:
                parsed = json.loads(data["choices"][0]["message"]["content"])
                valid = isinstance(parsed, dict) and parsed.get("ok") is True
            except (KeyError, IndexError, TypeError, ValueError):
                valid = False
            print("Expected JSON:", valid)
            return 0 if valid else 1
    except HTTPError as error:
        print("HTTP:", error.code)
        if error.headers.get("Retry-After"):
            print("Retry-After:", error.headers["Retry-After"])
        return 1
    except (URLError, TimeoutError, ValueError) as error:
        print("Probe error:", type(error).__name__)
        return 1
    finally:
        print(f"Latency: {1000 * (time.monotonic() - start):.0f} ms")


if __name__ == "__main__":
    raise SystemExit(main())
