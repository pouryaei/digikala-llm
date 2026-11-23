import re
import unicodedata

_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_ENGLISH_DIGITS = "0123456789"

_DIGIT_TRANSLATION_TABLE = str.maketrans(
    _PERSIAN_DIGITS + _ARABIC_INDIC_DIGITS,
    _ENGLISH_DIGITS * 2
)

def normalize_digits(text: str) -> str:
    """تبدیل تمام ارقام فارسی و عربی‌هندی به ارقام استاندارد انگلیسی."""
    if not isinstance(text, str):
        return ""
    return text.translate(_DIGIT_TRANSLATION_TABLE)

def normalize_persian_text(text: str) -> str:
    """
    تابع جامع نرمال‌سازی متن فارسی برای استفاده در BM25 و Dense Retrieval.
    - یکسان‌سازی فرم‌های یونیکد (NFC)
    - یکسان‌سازی کاف و ی (عربی به فارسی)
    - تبدیل ارقام به ASCII
    - مدیریت فاصله‌ها
    """
    if not isinstance(text, str):
        return ""

    # 1. استانداردسازی یونیکد
    text = unicodedata.normalize("NFC", text)

    # 2. یکسان‌سازی حروف عربی و فارسی
    text = text.replace("\u0643", "\u06a9")  # کاف عربی -> کاف فارسی
    text = text.replace("\u064a", "\u06cc").replace("\u0649", "\u06cc")  # یای عربی -> یای فارسی

    # 3. نرمال‌سازی ارقام
    text = normalize_digits(text)

    # 4. اصلاح فاصله‌ها
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text
