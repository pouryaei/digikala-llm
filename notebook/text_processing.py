"""
Text Normalization and Tokenization Module
Derived from QBC13_EDA_and_Cleaning (2).ipynb

"""
import re
import html
import unicodedata
from typing import List, Optional

# نگاشت کاراکترهای عربی و ارقام به فارسی یکدست
CHAR_MAP = str.maketrans({
    "ي": "ی", "ى": "ی", "ئ": "ی",
    "ك": "ک",
    "آ": "ا", "أ": "ا", "إ": "ا",
    "ة": "ه",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "\u200c": " ",  # ZWNJ (نیم‌فاصله) به فاصله
    "\u200b": " ",  # Zero-width space به فاصله
    "\xa0": " ",    # Non-breaking space به فاصله
})

WEB_MARKUP = re.compile(r"<[^>]+>|&[a-z]+;|[#*_~`]", re.IGNORECASE)
REMOVE_CHARS = re.compile(r"[\u200d\u200e\u200f\ufeff\xad]")
LEXICAL_TOKEN = re.compile(r"[^\W_]+")
NULL_TEXT = {"", "null", "none", "nan"}

def normalize_persian_text(value: Optional[str]) -> Optional[str]:
    """نرمال‌سازی رسمی متن فارسی طبق استاندارد نوت‌بوک EDA"""
    if value is None:
        return None
    value = str(value)
    if "&" in value:
        value = html.unescape(value)
    value = unicodedata.normalize("NFKC", value).translate(CHAR_MAP)
    value = WEB_MARKUP.sub(" ", value)
    value = REMOVE_CHARS.sub("", value)
    value = " ".join(value.split())
    return value if value.lower() not in NULL_TEXT else None

def tokenize_persian_lexical(value, include_joined_compounds=False):
    """
    توکنایزر واژگانی سازگار با ستون product_text_lexical
    پوشش‌دهنده هم حالت چندکلمه‌ای و هم پیوسته (مانند دست چین و دستچین)
    """
    norm = normalize_persian_text(value)
    if not norm:
        return []
    
    tokens = LEXICAL_TOKEN.findall(norm.lower())
    if not tokens:
        return []
    
    if include_joined_compounds and len(tokens) > 1:
        # کلمات دو به دو متوالی را بدون فاصله هم اضافه می‌کنیم
        compounds = [tokens[i] + tokens[i+1] for i in range(len(tokens) - 1)]
        return tokens + compounds
    
    return tokens
