"""Persian, product-scoped review retrieval and auditable evidence selection.

Counts are deterministic; the LLM never estimates the number of complaints or
recommendations. The displayed quotation is always a substring of a real field.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any

from .task2_review_store import FIELDS, MemoryReviewStore, ReviewBatch, identifier
from .task2_text_processing import normalize_persian_text

TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
STOP_WORDS = set("""
یک یه این آن اون کالا محصول محصولات برای که با از در به را رو و یا چه چی کدام کدوم
هست است بود بوده هستند باشه باشد می میخوام بخوام خرید خریداران خریدار کاربران کاربر
مردم نظر نظرات نظرها بازخورد گفته گفتند گفتن نوشته نوشتهاند درباره درمورد مورد میگن شناسه کد
بیشتر چه چیزی چیزهایی ویژگی ویژگیها نکته نکات نقاط قوت ضعف مثبت منفی ایراد ایرادها
مشکل مشکلات شکایت معایب مزایا پرتکرار پرتکرارترین راضی رضایت ناراضی نارضایتی
چیست چگونه چطور چطوره آیا ایا خوب بد بهتر عالی مناسب شده نیست خیلی درسته تجربه
بر اساس برحسب توجه ارزش پیشنهاد توصیه می ارزه داره دارن دارد داشتند بودند نداشتند شما
کسی چیزی استفاده حرفی حرف زده زدن داده دادن چند چقدر میشه شدهاند میکند میکنه کند کنه
کردن کرد کرده کردند میشود میشد بشه شدن شده بودم بودن بودهاند بوده هستند بوده است
""".split())
NO_CONTENT = {"", "nan", "null", "none", "ندارد", "نداره", "نداشت", "هیچ", "هیچی", "هیچ چیز", "هیچکدام", "هیچ کدام", "موردی نیست", "منفی ندارد", "نکته منفی ندارد", "نکته منفی نداره", "موردی ندیدم", "هیچ مشکلی ندارد", "ندیدم", "نظری ندارم"}
# Theme labels are explicit retrieval rules, not model-invented product attributes.
THEMES = {
    "کیفیت و جنس": ("کیفیت", "جنس", "دوخت", "ساخت", "چرم", "پارچه", "محکم", "نازک", "مقاومت", "دوام", "شکننده", "خراب", "پاره"),
    "اندازه و تناسب": ("سایز", "اندازه", "کوچک", "بزرگ", "تنگ", "گشاد", "تنخور", "تن خور", "قواره", "جا دار", "جادار", "حجم"),
    "راحتی و وزن": ("راحت", "سبک", "سنگین", "نرم", "خشک", "پاخور", "کف پا", "پا درد", "راحتی"),
    "ظاهر و رنگ": ("ظاهر", "رنگ", "زیبا", "خوشگل", "شیک", "طرح", "عکس", "بدقواره", "لک", "چروک"),
    "قیمت و ارزش خرید": ("قیمت", "گران", "گرون", "ارزان", "تخفیف", "ارزش خرید", "می ارزه", "صرفه", "هزینه", "نمی ارزه"),
    "کارایی و عملکرد": ("کارایی", "عملکرد", "کارکرد", "قدرت", "سرعت", "اثرگذار", "اثربخش", "تاثیر", "موثر", "مؤثر", "پاک کنندگی", "پاک کننده", "نرم کنندگی", "نرم کننده", "نرم شدن", "نرم کردن", "نرم شد", "نرم شدند", "نرم می کند", "نرم میکنه", "نرم می کنه", "نرم میکند", "نرم میکنند", "نرم می کنند", "خشک نمی کند", "خشک نمیکنه", "خشک نمی کنه", "خشک نمیکند", "پوست خشک", "موهای خشک", "مو خشک", "مرطوب", "ترک پا", "خشکی", "کاربرد"),
    "بسته‌بندی و ارسال": ("بسته بندی", "بستهبندی", "ارسال", "تحویل", "پست", "تاخیر", "جعبه", "بسته", "شکسته", "سالم رسید"),
    "اصالت و مطابقت": ("اصالت", "اصل", "تقلب", "فیک", "مطابق", "مغایرت", "فرق", "دست دوم", "مارک", "برند"),
    "باتری و شارژ": ("باتری", "شارژ", "شارژدهی", "باتری دهی", "برق"),
    "صدا": ("صدا", "نویز", "بلندی صدا", "میکروفون", "بیس"),
    "مزه و بو": ("مزه", "طعم", "بو", "خوشمزه", "بدبو", "عطر", "رایحه"),
    "محتوا و چاپ": ("محتوا", "ترجمه", "چاپ", "کاغذ", "صفحه", "متن", "خواندن", "خواندنی", "مطالعه", "نگارش", "ویراستاری"),
}
NORMAL_THEMES = {label: tuple(dict.fromkeys(normalize_persian_text(word).lower() for word in words)) for label, words in THEMES.items()}
# Query-side aliases are narrower than document-side themes; otherwise a word
# such as «محصول» can accidentally turn a broad question into a facet query.
QUERY_THEMES = {
    **NORMAL_THEMES,
    "کارایی و عملکرد": tuple(normalize_persian_text(x).lower() for x in ("کارایی", "عملکرد", "کارکرد", "قدرت", "اثرگذاری", "اثربخشی", "تاثیر", "موثر", "مؤثر", "نرم کنندگی", "نرم شدن", "نرم کردن", "نرم شد", "نرم شدند", "نرم می کند", "نرم میکنه", "نرم می کنه", "نرم میکند", "پوست خشک", "موهای خشک", "مو خشک", "ترک پا", "خشکی", "مرطوب", "سرعت")),
    "راحتی و وزن": tuple(normalize_persian_text(x).lower() for x in ("راحتی", "راحت", "سبک", "سنگین", "وزن", "نرمی", "پاخور")),
    "اصالت و مطابقت": tuple(normalize_persian_text(x).lower() for x in ("اصالت", "تقلب", "فیک", "اصل", "مطابقت", "مغایرت", "دست دوم")),
}
POS = re.compile(r"(?<!نا)راضی|رضایت|عالی|خوب|مناسب|راحت|خوشگل|زیبا|با کیفیت|باکیفیت|ارزش خرید|می ارزه|محکم|دوست داشتم|حرف نداشت|بی نظیر|بینظیر|خوش پا")
NEG = re.compile(r"ناراض|نارضایتی|افتضاح|بد(?:\b|ه)|ضعیف|نامناسب|نمی\s*ارزه|مرجوع|خراب|تقلب|گران|گرون|شکست|پاره|نخرید|توصیه نمی|پیشنهاد نمی|ارسال.{0,15}اشتباه|اشتباه|غلط|قیمت.{0,12}بالا|خوب\s*(?:نیست|نبود)|راضی\s*(?:نیست|نبود)|ارزش(?: خرید)?\s*(?:ندار|نداش)|کیفیت.{0,8}(?:پایین|ندار)|(?:خیلی|بیش از حد|زیادی|فوق العاده)\s*(?:نازک|کوچک|تنگ|خشک)|(?:سایز|اندازه|قواره|جنس).{0,18}(?:اشتباه|نامناسب|کوچک|تنگ|بزرگ|نازک)|(?:برام|برای من|پام)\s*(?:کوچک|تنگ|بزرگ)|(?:لک|لکه).{0,8}(?:داشت|بود)")
VALUE = re.compile(r"ارزش\s*خرید|می\s*ارزه|میرزه|نمی\s*ارزه|(?:آیا |ایا )?بخرم|پیشنهاد(?:\s*خرید)?|توصیه")
NEG_QUESTION = re.compile(r"شکایت|ایراد|معایب|نکات منفی|نقاط ضعف|ناراض|نارضایتی|مشکل|نقص")
POS_QUESTION = re.compile(r"(?<!نا)راضی|رضایت|مزایا|نقاط قوت|نکات مثبت|خوششان|دوست داشت")
GENERAL_QUESTION = re.compile(r"نظر کاربران|نظر خریداران|نظرات|نظرها|بازخورد|تجربه|چی گفتن|چه گفتند|چی میگن|چه نظری")


def norm(value: str | None) -> str:
    return (normalize_persian_text(value) or "").lower()


_SUFFIX = r"(?:ها(?:ی|یش|یشان|یتان|یمان)?|هایی|شان|تان|مان|ش|ت|م|ی|ه|تر(?:ین|ی)?|است|ست)?"
_LIST_ITEMS = re.compile(r"""["]((?:\\.|[^"\\])*)["]|[']((?:\\.|[^'\\])*)[']""")
_DENIED_POSITIVE = re.compile(r"(?<!\w)(?:خوب|عالی|راضی|راحت|مناسب|محکم|نرم|باکیفیت|با کیفیت)\s+(?:نیست\w*|نبود\w*|نشد\w*)(?!\w)")


@lru_cache(maxsize=2048)
def _term_pattern(terms):
    return re.compile(r"(?<![^\W_])(?:" + "|".join(map(re.escape, terms)) + ")" + _SUFFIX + r"(?![^\W_])")


def _has_term(text, term):
    # Whole words and common suffixes; never match «بو» inside «بودند».
    return bool(_term_pattern((term,)).search(text))


def _themes(text, mapping):
    return {label for label, terms in mapping.items() if _term_pattern(terms).search(text)}


def _list_segments(raw):
    """Extract contiguous spans of list-form attributes without changing the source text."""
    text = raw.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return [raw]
    segments = [
        re.sub(r"(?:\\[rnt])+$", "", m.group(1) if m.group(1) is not None else m.group(2)).strip()
        for m in _LIST_ITEMS.finditer(text)
    ]
    return [s for s in segments if useful(s)] or [raw]


def _tokens(text):
    return set(TOKEN.findall(text))


def useful(raw):
    text = norm(raw)
    # Strip list representation/punctuation for placeholder detection only.
    cleaned = " ".join(TOKEN.findall(text))
    return bool(cleaned and cleaned not in NO_CONTENT and text not in NO_CONTENT)


def polarity(field, text, status=None):
    if field == "advantages_raw": return "positive"
    if field == "disadvantages_raw":
        plain = " ".join(TOKEN.findall(text))
        if re.fullmatch(r"(?:بد (?:نیست|نبود)|مشکل(?:ی)? (?:ندارد|نداره|نداشت)|ایراد(?:ی)? (?:ندارد|نداره|نداشت))", plain):
            return "unknown"
        return "negative"
    # «بد نیست» is not evidence of dissatisfaction; other words can still
    # make the same sentence a mixed experience.
    mild = bool(re.search(r"(?<!\w)بد\s*(?:نیست|نبود)(?!\w)", text))
    neg_text = re.sub(r"(?<!\w)بد\s*(?:نیست|نبود)(?!\w)", " ", text)
    denied = bool(_DENIED_POSITIVE.search(text))
    pos = bool(POS.search(_DENIED_POSITIVE.sub(" ", text))) or mild
    neg = bool(NEG.search(neg_text)) or denied
    if pos and neg: return "mixed"
    if neg: return "negative"
    if pos: return "positive"
    return "unknown"


@dataclass(frozen=True)
class QuerySpec:
    intent: str
    themes: tuple[str, ...]
    terms: tuple[str, ...]
    topical: bool


def parse_question(question, title=None):
    text = norm(question)
    if not text: raise ValueError("سؤال را وارد کن.")
    # The title may be repeated by the user. It is not evidence or a query facet.
    topical_text = text
    if title:
        title_text = norm(title)
        if title_text and title_text in text:
            topical_text = text.replace(title_text, " ")
    requested = _themes(topical_text, QUERY_THEMES)
    # Value questions need numeric recommendation evidence plus relevant comments.
    if VALUE.search(topical_text) and not NEG_QUESTION.search(topical_text):
        intent = "value"
    elif NEG_QUESTION.search(topical_text):
        intent = "negative"
    elif requested:
        intent = "aspect"
    elif POS_QUESTION.search(topical_text):
        intent = "positive"
    elif GENERAL_QUESTION.search(topical_text):
        intent = "general"
    else:
        intent = "aspect"
    title_tokens = _tokens(norm(title)) if title else set()
    query_terms = {t for t in _tokens(topical_text) - STOP_WORDS - title_tokens if len(t) > 1 and not t.isdecimal()}
    # A request such as «جوهرش» should also retrieve «جوهر» in the source.
    possessives = {t[:-1] for t in query_terms if len(t) >= 5 and t.endswith("ش") and t[:-1] not in STOP_WORDS and t[:-1] not in title_tokens}
    terms = tuple(sorted(query_terms | possessives))
    # Broad positive/negative/value/general questions need examples even without
    # named facets. An explicit topical question, however, never pulls unrelated reviews.
    topical = bool(requested and intent != "value") or (intent in ("aspect", "general") and bool(terms))
    if intent == "aspect" and not requested and not terms:
        topical = True
    return QuerySpec(intent, tuple(sorted(requested)), terms, topical)


def _match_strength(text, spec):
    relevant_themes = _themes(text, NORMAL_THEMES).intersection(spec.themes)
    tokens = _tokens(text)
    matched = sum(t in tokens or _has_term(text, t) for t in spec.terms)
    # If a known facet is requested, unrelated product-title words cannot suffice.
    if spec.themes and spec.intent != "value" and not relevant_themes:
        return 0, 0
    return len(relevant_themes), matched


def _excerpt(raw, spec, max_chars=470):
    raw = raw.strip()
    if len(raw) <= max_chars: return raw
    # Score contiguous source windows, preserving an exact substring even if
    # the relevant phrase appears at the end of a long review.
    windows = [raw[i:i+max_chars].strip() for i in range(0, len(raw), max_chars//2)]
    return max(windows, key=lambda x: (sum(_match_strength(norm(x), spec)), len(x)))


@dataclass
class Selection:
    product_id: int
    question: str
    intent: str
    requested_themes: list[str]
    total_reviews: int
    valid_label_count: int
    recommended_count: int
    not_recommended_count: int
    no_idea_count: int
    text_review_count: int
    relevant_review_count: int
    evidence: list[dict]
    theme_counts: list[dict]
    scope: str
    complete_for_product: bool
    source: str
    title: str | None
    facet_coverage: list[dict]
    reason: str | None = None

    @property
    def insufficient(self):
        return not bool(self.evidence)


class EvidenceSelector:
    """Compatible with the previous DataFrame-based selector or a ReviewStore.

    The production store filters by product_id before loading text. A single
    best source field per comment is selected; duplicate IDs never appear.
    """
    def __init__(self, comments, lexical_search=None, dense_search=None, text_col="text_clean", top_k=8):
        self.store = comments if hasattr(comments, "get") and not isinstance(comments, (list, tuple, dict)) and not hasattr(comments, "to_dict") else MemoryReviewStore(comments)
        self.top_k = min(max(1, int(top_k)), 20)
        # Legacy constructor args kept to avoid breaking other modules. Review
        # retrieval is always product-scoped, rather than using a product-title FAISS index.
        self.lexical_search = lexical_search
        self.dense_search = dense_search
        self.text_col = text_col

    def select_details(self, question, product_id, top_k=None):
        pid = identifier(product_id)
        batch = self.store.get(pid)
        return self.select_batch(question, batch, top_k)

    def select_batch(self, question, batch: ReviewBatch, top_k=None):
        spec = parse_question(question, batch.title)
        limit = self.top_k if top_k is None else min(max(1, int(top_k)), 20)
        rows = batch.reviews
        labels = Counter(row.get("recommendation_status") for row in rows)
        total_with_text = sum(any(useful(row.get(f)) for f in FIELDS) for row in rows)
        by_cid = {}
        themes_by_polarity = {"positive": defaultdict(set), "negative": defaultdict(set)}
        theme_examples = defaultdict(list)
        facet_matches = defaultdict(set)
        facet_examples = defaultdict(list)
        for row in rows:
            cid = row["comment_id"]
            if row["product_id"] != batch.product_id: continue
            status = row.get("recommendation_status")
            best = None
            for field in FIELDS:
                raw = row.get(field)
                if not useful(raw): continue
                text = norm(raw)
                if field in ("advantages_raw", "disadvantages_raw"):
                    parts = _list_segments(raw)
                    part_polarities = {polarity(field, norm(part), status) for part in parts if useful(part)}
                    pol = "mixed" if {"positive", "negative"} <= part_polarities else "negative" if "negative" in part_polarities else "positive" if "positive" in part_polarities else "unknown"
                else:
                    pol = polarity(field, text, status)
                doc_themes = _themes(text, NORMAL_THEMES)
                # Count an explicit advantage/disadvantage, or an individual
                # text clause with an unambiguous sentiment cue and aspect.
                # Never assign the review-level recommendation to every aspect.
                if field in ("advantages_raw", "disadvantages_raw"):
                    clauses = _list_segments(raw)
                else:
                    clauses = re.split(r"[،؛.!؟\n\r]+|(?<!\w)(?:ولی|اما|متاسفانه|متأسفانه|فقط)(?!\w)", raw)
                for clause in clauses:
                    quote = clause.strip()
                    if not useful(quote): continue
                    qtext = norm(quote)
                    qpol = polarity(field, qtext, status)
                    if qpol not in ("positive", "negative"): continue
                    clause_themes = _themes(qtext, NORMAL_THEMES)
                    for theme in clause_themes:
                        themes_by_polarity[qpol][theme].add(cid)
                        theme_spec = QuerySpec("aspect", (theme,), (), True)
                        shown = _excerpt(quote, theme_spec)
                        if theme not in _themes(norm(shown), NORMAL_THEMES): continue
                        theme_examples[(qpol, theme)].append({
                            "comment_id": cid, "product_id": batch.product_id, "text": shown,
                            "source": row.get("source_fields", {}).get(field, field),
                            "polarity": qpol, "recommendation_status": status, "rating_5": row.get("rating_5"),
                            "score": 5.0 + (2 if field in ("advantages_raw", "disadvantages_raw") else 0) + min(len(qtext), 320)/320,
                            "themes": sorted(clause_themes),
                        })
                theme_hit, token_hit = _match_strength(text, spec)
                if spec.topical and not (theme_hit or token_hit):
                    continue
                if spec.intent == "positive" and pol not in ("positive", "mixed"):
                    continue
                if spec.intent == "negative" and pol not in ("negative", "mixed"):
                    continue
                if spec.intent == "value" and pol == "unknown" and status not in ("recommended", "not_recommended") and not theme_hit and not token_hit:
                    continue
                if spec.intent == "aspect" and not (theme_hit or token_hit):
                    continue
                explicit_bonus = 3.0 if (spec.intent == "positive" and field == "advantages_raw") or (spec.intent == "negative" and field == "disadvantages_raw") else 0.0
                value_bonus = 2.0 if spec.intent == "value" and "قیمت و ارزش خرید" in doc_themes else 0.0
                polarity_bonus = 1.5 if (spec.intent == "positive" and pol == "positive") or (spec.intent == "negative" and pol == "negative") else 0.0
                if spec.intent == "value" and status in ("recommended", "not_recommended"): polarity_bonus = 1.0
                score = 5.0*theme_hit + 1.25*token_hit + explicit_bonus + value_bonus + polarity_bonus + min(len(text), 320)/320 + (0.5 if field == "body_raw" else 0.0)
                if field in ("advantages_raw", "disadvantages_raw") and spec.intent in ("aspect", "general"):
                    score += 0.6
                candidates = _list_segments(raw) if field in ("advantages_raw", "disadvantages_raw") else [raw]
                # Show the strongest literal list item, rather than a Python list
                # representation. Its span remains a substring of the original field.
                if spec.intent in ("positive", "negative") and field in ("advantages_raw", "disadvantages_raw"):
                    candidates = [part for part in candidates if polarity(field, norm(part), status) in (spec.intent, "mixed")]
                    if not candidates: continue
                best_source = max(candidates, key=lambda part: (*_match_strength(norm(part), spec), len(norm(part))))
                evidence = {
                    "comment_id": cid, "product_id": batch.product_id, "text": _excerpt(best_source, spec),
                    "source": row.get("source_fields", {}).get(field, field), "polarity": pol,
                    "recommendation_status": status, "rating_5": row.get("rating_5"),
                    "score": round(score, 5), "themes": sorted(doc_themes),
                }
                # Coverage is measured per comment; displayed evidence must
                # contain the actual facet, not merely another field of that ID.
                for theme in doc_themes.intersection(spec.themes):
                    facet_matches[theme].add(cid)
                    if theme in _themes(norm(evidence["text"]), NORMAL_THEMES):
                        facet_examples[theme].append(evidence)
                if not best or (evidence["score"], len(evidence["text"])) > (best["score"], len(best["text"])):
                    best = evidence
            if best: by_cid[cid] = best
        ranked = sorted(by_cid.values(), key=lambda e: (-e["score"], e["comment_id"]))
        selected = []
        used = set()
        def add(e):
            if e and e["comment_id"] not in used and len(selected) < limit:
                selected.append(e); used.add(e["comment_id"])
        theme_counts = []
        wanted = ["positive"] if spec.intent == "positive" else ["negative"] if spec.intent == "negative" else ["positive", "negative"] if spec.intent == "aspect" and spec.themes else []
        for pol in wanted:
            eligible_themes = [(theme, ids) for theme, ids in themes_by_polarity[pol].items() if not spec.themes or theme in spec.themes]
            for theme, ids in sorted(eligible_themes, key=lambda kv: (-len(kv[1]), kv[0]))[:3]:
                examples = sorted(theme_examples[(pol, theme)], key=lambda e: (-e["score"], e["comment_id"]))
                # Prefer a quote that actually contains this theme in the correct
                # explicit field; a different field in the same comment is not a witness.
                sample = next((e for e in examples if e["comment_id"] not in used), None)
                if sample:
                    add(sample)
                    if sample["comment_id"] not in by_cid:
                        by_cid[sample["comment_id"]] = sample
                theme_counts.append({"theme": theme, "polarity": pol, "count": len(ids), "denominator": total_with_text, "source": "explicit attributes and clear-sentiment text clauses", "support_ids": sorted(ids)})
        if spec.intent == "aspect":
            # Do not hide evidence for a second requested facet just because
            # all high scores concern the first facet.
            for theme in spec.themes:
                if not any(theme in _themes(norm(e["text"]), NORMAL_THEMES) for e in selected):
                    ranked_facet = sorted(facet_examples[theme], key=lambda e: (-e["score"], e["comment_id"]))
                    add(next((e for e in ranked_facet if e["comment_id"] not in used), None))
        if spec.intent in ("aspect", "value", "general"):
            # Preserve contrasting experiences where evidence exists. Ranking by
            # recommendation label alone would suppress minority experiences.
            for pol in ("positive", "negative"):
                desired_label = "recommended" if pol == "positive" else "not_recommended"
                if any(e["polarity"] == pol for e in selected): continue
                option = next((e for e in ranked if e["polarity"] == pol and e["comment_id"] not in used), None)
                if option is None and spec.intent == "value":
                    option = next((e for e in ranked if e["recommendation_status"] == desired_label and e["comment_id"] not in used), None)
                add(option)
        for e in ranked: add(e)
        for t in theme_counts:
            ids = set(t.pop("support_ids"))
            t["evidence_ids"] = [e["comment_id"] for e in selected if e["comment_id"] in ids and e["polarity"] == t["polarity"] and t["theme"] in _themes(norm(e["text"]), NORMAL_THEMES)][:3]
        theme_counts = [t for t in theme_counts if t["evidence_ids"]]
        # Count themes found in requested aspect witnesses as well, including
        # the explicitly classified sentence samples that ranked before a raw field.
        for theme in spec.themes:
            facet_matches[theme].update(themes_by_polarity["positive"].get(theme, set()))
            facet_matches[theme].update(themes_by_polarity["negative"].get(theme, set()))
        facet_coverage = [{"theme": theme, "matched_review_count": len(facet_matches[theme]),
                           "evidence_ids": [e["comment_id"] for e in selected if theme in _themes(norm(e["text"]), NORMAL_THEMES)][:3]}
                          for theme in spec.themes]
        reason = None
        if not selected:
            if batch.status == "outside_sample": reason = "این محصول در محدودهٔ نمونهٔ ارزیابی نیست؛ فایل کامل نظرات را تعیین کن."
            elif batch.status == "outside_catalogue": reason = "این شناسه خارج از مجموعهٔ محصولات بسته است. برای پاسخ‌گویی از فایل کامل نظرات استفاده کن."
            elif not rows: reason = "برای این شناسه در دادهٔ در دسترس هیچ نظری پیدا نشد."
            elif not total_with_text: reason = "نظرات این محصول متن قابل استناد ندارند."
            else: reason = "در متن نظرات این محصول شاهد مرتبط و قابل استناد برای این سؤال پیدا نشد."
        return Selection(batch.product_id, question, spec.intent, list(spec.themes), len(rows), sum(labels[x] for x in ("recommended", "not_recommended", "no_idea")), labels["recommended"], labels["not_recommended"], labels["no_idea"], total_with_text, len(by_cid), selected, theme_counts, batch.scope, batch.complete_for_product, batch.source, batch.title, facet_coverage, reason)

    def select(self, question, product_id=None, top_k=None):
        if product_id is None: raise ValueError("برای پرسش از نظرات باید شناسهٔ محصول مشخص باشد.")
        return self.select_details(question, product_id, top_k).evidence

    def compare(self, question, product_ids, top_k=None):
        return {pid: self.select(question, pid, top_k) for pid in product_ids}
