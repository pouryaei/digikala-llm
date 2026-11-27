"""Run a single explicit LLM gateway connection smoke test (not on import)."""
from pathlib import Path
import argparse,json,os


def run_test(env_file=None):
    try:
        from dotenv import load_dotenv
        path=Path(env_file) if env_file else Path(__file__).resolve().parent/".env"
        if path.is_file():load_dotenv(path,override=False)
        from src.phase4_llm.llm_client import LLMClient, is_configured_key
        if not is_configured_key(os.getenv("QBC13_API_KEY")):
            print("QBC13_API_KEY واقعی تنظیم نشده است؛ کلید گروه را در .env محلی یا محیط اجرا قرار بده.")
            return 1
        client=LLMClient()
        # generate expects system and user, not just one positional argument.
        response=client.generate("به فارسی و فقط با یک جملهٔ کوتاه پاسخ بده.","سلام، این یک آزمون اتصال است.")
        if not isinstance(response,dict) or not response.get("text"):
            print("پاسخ متنی معتبری از سرویس دریافت نشد.");return 1
        print("اتصال موفق:",response["text"])
        print(json.dumps({k:response.get(k) for k in ("latency","prompt_tokens","completion_tokens","api_attempts")},ensure_ascii=False))
        return 0
    except Exception as e:
        print(f"تست اتصال ناموفق بود ({type(e).__name__}). وابستگی‌ها، کلید و آدرس سرویس را بررسی کن.")
        return 1


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--env-file",help="Optional local configuration file; never share its contents")
    raise SystemExit(run_test(p.parse_args().env_file))
