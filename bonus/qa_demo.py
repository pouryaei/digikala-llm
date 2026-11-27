"""Ready-to-replay QA scenarios from the bundled 100K snapshot."""
DEMO_CASES = [
    ("QA-D1", "آمار مستقیم از داده", "چند نفر خرید این محصول را پیشنهاد کرده‌اند؟", 1696173, "simple_question_local_answer_sufficient"),
    ("QA-D2", "شاهد مرتبط پیدا نشد", "کاربران درباره باتری و شارژ این محصول چه گفته‌اند؟", 1696173, "missing_or_insufficient_evidence"),
    ("QA-D3", "پوشش ناقص معیارها", "با توجه به کیفیت و باتری آیا این محصول ارزش خرید دارد؟", 1696173, "requested_facets_uncovered"),
    ("QA-D4", "جمع‌بندی چند معیار با شاهد", "کیفیت ساخت و راحتی این محصول را با استناد به تجربه کاربران جمع‌بندی کن.", 1696173, "explicit_opt_in_complex_grounded_synthesis"),
    ("QA-D5", "درخواست انتخاب محصول", "ایرادهای پرتکرار این محصول چیست؟", None, "product_clarification_no_api"),
]
