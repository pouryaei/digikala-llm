"""Load a user's existing local Forouhar interface without editing its file."""
from functools import lru_cache
import importlib.util
from pathlib import Path

from .config import ROOT


@lru_cache(maxsize=4)
def _load(path, modified):
    import ui_services
    spec = importlib.util.spec_from_file_location("_forouhar_user_dashboard", path)
    if spec is None or spec.loader is None:
        raise ValueError("فایل اصلی رابط پایتون معتبر نیست.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "main", None)) or not callable(getattr(module, "api_call", None)):
        raise ValueError("فایل انتخابی باید نسخهٔ رابط فروهر با تابع‌های main و api_call باشد.")
    # Use the existing page rendering and styling, including assets, untouched.
    module.api_call = ui_services.api_call
    module.APIError = ui_services.APIError
    module.backend_url = ui_services.backend_url
    return module


def load_original(path):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".py" or path == ROOT / "app.py":
        raise ValueError("مسیر فایل اصلی رابط فروهر را تعیین کنید؛ فایل app.py داخل این بسته نباشد.")
    return _load(path, path.stat().st_mtime_ns)
