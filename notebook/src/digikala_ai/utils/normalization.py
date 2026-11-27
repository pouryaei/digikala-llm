"""
Advanced Persian Text Normalization Module
Designed for Digikala Comments Classification & Information Retrieval.
Optimized for high-throughput and vocabulary consolidation (Macro-F1 boost).
"""

import re
import unicodedata

# 1. نگاشت جامع ارقام (فارسی و عربی-هندی به ارقام انگلیسی/ASCII)
_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_ENGLISH_DIGITS = "0123456789"

_DIGIT_MAP = str.maketrans(
    _PERSIAN_DIGITS + _ARABIC_INDIC_DIGITS,
    _ENGLISH_DIGITS * 2
)

# 2. نگاشت جامع استانداردسازی حروف (کاف، ی، همزه، تاء مربوطه و الف‌های متفرقه)
_CHAR_MAP = str.maketrans({
    '\u0643': '\u06a9',  # ك عربی -> ک فارسی
    '\u064a': '\u06cc',  # ي عربی -> ی فارسی
    '\u0649': '\u06cc',  # ى عربی بدون نقطه -> ی فارسی
    '\u0626': '\u06cc',  # ئ -> ی
    '\u0624': '\u0648',  # ؤ -> و
    '\u0629': '\u0647',  # ة -> ه
    '\u0622': '\u0627',  # آ -> ا (یکدست‌سازی برای مهار تنوع در TF-IDF)
    '\u0623': '\u0627',  # أ -> ا
    '\u0625': '\u0627',  # إ -> ا
    '\u0671': '\u0627',  # ٱ -> ا
})

# 3. کاراکترهای کنترلی پنهان یونیکد و جداکننده‌های خطی برای حذف کامل
_INVISIBLE_CHARS_PATTERN = re.compile(
    r'[\u200d\u200e\u200f\ufeff\u00ad\u202a-\u202e]'
)

# 4. اعراب، تنوین، تشدید و سکون (Diacritics / Tashkeel)
_DIACRITICS_PATTERN = re.compile(
    r'[\u064b-\u0652\u0670\u0654\u0655]'
)

# 5. کشیدگی حروف و تکرار اغراق‌آمیز کاراکترها در زبان محاوره‌ای (مانند: عاااااالی -> عالی)
# تکرار بیش از ۲ کاراکتر یکسان را به ۲ عدد کاهش می‌دهد
_ELONGATION_PATTERN = re.compile(r'(.)\1{2,}', flags=re.DOTALL)

# 6. اصلاح فضاهای خالی اطراف نیم‌فاصله
_ZWNJ = '\u200c'
_ZWNJ_EXTRA_SPACES = re.compile(r'\s*' + _ZWNJ + r'\s*')


def normalize_digits(text: str) -> str:
    """تبدیل تمام ارقام به ارقام استاندارد ASCII."""
    if not isinstance(text, str):
        return ""
    return text.translate(_DIGIT_MAP)


def normalize_persian_text(
    text: str,
    unify_alif: bool = True,
    reduce_elongation: bool = True,
    keep_zwnj: bool = False
) -> str:
    """
    نرمال‌ساز حرفه‌ای متن فارسی، بهینه‌شده برای تحلیل احساسات و طبقه‌بندی نظرات:
    1. فرم یونیکد NFC
    2. حذف کاراکترهای کنترلی پنهان
    3. حذف اعراب و تشدید
    4. یکسان‌سازی حروف متفرقه عربی و الف‌ها
    5. تبدیل ارقام به معادل استاندارد
    6. حذف کشیدگی حروف (Elongation reduction)
    7. مدیریت نیم‌فاصله (ZWNJ) و فاصله‌های زاید
    """
    if not isinstance(text, str) or not text:
        return ""

    # ۱. استانداردسازی شکل کدبندی کاراکترها
    text = unicodedata.normalize("NFC", text)

    # ۲. حذف کاراکترهای نامرئی/کنترلی
    text = _INVISIBLE_CHARS_PATTERN.sub('', text)

    # ۳. حذف اعراب، تنوین و تشدید
    text = _DIACRITICS_PATTERN.sub('', text)

    # ۴. نگاشت حروف هم‌صدا/عربی به معادل استاندارد فارسی
    text = text.translate(_CHAR_MAP) if unify_alif else text.replace("\u0643", "\u06a9").replace("\u064a", "\u06cc")

    # ۵. استانداردسازی ارقام
    text = text.translate(_DIGIT_MAP)

    # ۶. حذف کشیدگی کلمات محاوره‌ای (مانند: عاااااالی -> عالی)
    if reduce_elongation:
        text = _ELONGATION_PATTERN.sub(r'\1\1', text)

    # ۷. مدیریت نیم‌فاصله
    if keep_zwnj:
        text = _ZWNJ_EXTRA_SPACES.sub(_ZWNJ, text)
    else:
        # در مدل‌های Bag-of-Words / TF-IDF، تبدیل ZWNJ به فاصله معمولاً دقت توکنایزر را بالاتر می‌برد
        text = text.replace(_ZWNJ, ' ')

    # ۸. تجمیع فاصله‌ها و حذف فضاهای خالی ابتدا و انتها
    text = re.sub(r'\s+', ' ', text).strip()

    return text
