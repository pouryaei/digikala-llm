"""BM25 over the actual product corpus with consistent Persian tokenization.

The public BM25IndexManager interface is preserved. Both the original
rank_bm25 pickle and the versioned data-only payload are readable. When
rank_bm25 is installed we use its scorer; otherwise the same Okapi formula
is evaluated by the local reference scorer (no model download required).
"""
from __future__ import annotations

from collections import Counter
import hashlib
import html
import json
import math
import os
from pathlib import Path
import pickle
import re
import tempfile
import unicodedata

try:
    from rank_bm25 import BM25Okapi as _RankBM25
except ImportError:
    _RankBM25 = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "storage" / "bm25_index.pkl"
try:
    from src.config import Config
except ImportError:
    class Config:
        BM25_INDEX_PATH = str(DEFAULT_PATH)
        BM25_K1 = 1.5
        BM25_B = .75
        TOP_K = 5

# Same character/lexical policy as QBC13_EDA_and_Cleaning, cell "قواعد مشترک".
_CHAR_MAP = str.maketrans("يىك٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "ییک01234567890123456789")
_WEB_MARKUP = re.compile(r"https?://\S+|www\.\S+|</?[a-z][^>]*>", re.I)
_REMOVE_CHARS = re.compile(r"[ـ\u064b-\u065f\u0670\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_LEXICAL_TOKEN = re.compile(r"[^\W_]+")
_NULL_TEXT = {"", "none", "nan", "null", "n/a", "-"}
INDEX_FORMAT = 2
TOKENIZATION_VERSION = "qbc13_eda_lexical_v1"
# Query-only guard: generic Persian request words are not product features.
# Keep actual model numbers and descriptive terms. Use a minimum coverage of
# 60% of distinct remaining terms (ceil), with a title-anchor fallback for
# requests that add purpose words after the product name.
_SEARCH_FILLER = frozenset("""
از به با و یا در برای را رو این اون آن یک یه می میخوام میخواهم می‌خواهم بخوام
میخواهیم میخوایم میشه بخرم خرید خریدن محصول کالا محصولات کالای مدل کد طرح
خواهش لطفا لطفاً دنبال نیاز هستم هستیم است هست هستند بود باشه باشد که چه
چطور چگونه آیا ایا کاربران خریداران نظر نظرات کیفیت کارایی مناسب خوب عالی
راضی رضایت گفته توصیه پیشنهاد چند چقدر واقعی
""".split())



def tokenize_query(query):
    if query is None:
        return []
    value = str(query)
    if "&" in value:
        value = html.unescape(value)
    value = unicodedata.normalize("NFKC", value).translate(_CHAR_MAP)
    value = _WEB_MARKUP.sub(" ", value)
    value = _REMOVE_CHARS.sub("", value)
    value = " ".join(value.replace("\u200b", " ").split()).lower()
    return [] if value in _NULL_TEXT else _LEXICAL_TOKEN.findall(value)


def product_key(value):
    if value is None or isinstance(value, bool): raise ValueError("Missing or invalid product_id")
    key = str(value).strip()
    if not key or key.casefold() in _NULL_TEXT: raise ValueError("Missing product_id")
    return key


def mapping_ids(mapping):
    if not isinstance(mapping, list) or not mapping:
        raise ValueError("Expected a nonempty list of product metadata")
    ids = [product_key(row["product_id"]) for row in mapping]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate product_id in index mapping")
    return ids


def mapping_digest(mapping):
    # IDs and their order define the corpus alignment; text is hashed separately.
    payload = json.dumps(mapping_ids(mapping), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _okapi_state(document_tokens, k1=1.5, b=.75, epsilon=.25):
    if not document_tokens or not any(document_tokens):
        raise ValueError("BM25 corpus has no searchable text")
    doc_freqs = [dict(Counter(tokens)) for tokens in document_tokens]
    doc_len = [len(tokens) for tokens in document_tokens]
    n = len(doc_freqs)
    frequencies = Counter()
    for freq in doc_freqs:
        frequencies.update(freq.keys())
    # Identical Okapi IDF and negative-IDF floor convention to rank_bm25.
    raw_idf = {token: math.log(n-count+.5) - math.log(count+.5) for token, count in frequencies.items()}
    average_idf = sum(raw_idf.values())/len(raw_idf)
    idf = {token: val if val >= 0 else epsilon*average_idf for token,val in raw_idf.items()}
    return {"k1":float(k1),"b":float(b),"epsilon":float(epsilon),"corpus_size":n,
            "avgdl":sum(doc_len)/n,"doc_freqs":doc_freqs,"idf":idf,"doc_len":doc_len,
            "tokenizer":None,"average_idf":average_idf}


class _ReferenceScorer:
    """Data-only reference implementation of the same BM25Okapi scoring formula."""
    def __init__(self, state): self.__dict__.update(state)
    def get_scores(self, query_tokens):
        scores = [0.0]*self.corpus_size
        if self.avgdl <= 0:return scores
        lengths = [self.k1*(1-self.b+self.b*dl/self.avgdl) for dl in self.doc_len]
        # Repeated query tokens count repeatedly, as in rank_bm25.
        for token,qfreq in Counter(query_tokens).items():
            idf = self.idf.get(token,0.0)
            if not idf:continue
            for i,freqs in enumerate(self.doc_freqs):
                f=freqs.get(token,0)
                if f: scores[i] += qfreq*idf*f*(self.k1+1)/(f+lengths[i])
        return scores


class _LegacyStateCarrier:
    pass


class _IndexUnpickler(pickle.Unpickler):
    """Legacy index contains only the known rank_bm25 class and built-in data."""
    def find_class(self, module, name):
        if (module,name)==("rank_bm25","BM25Okapi"):
            return _LegacyStateCarrier
        raise pickle.UnpicklingError(f"Unexpected class in BM25 index: {module}.{name}")


def _read_index(path):
    with open(path,"rb") as handle:
        return _IndexUnpickler(handle).load()


def _make_scorer(state):
    if _RankBM25 is not None:
        obj = _RankBM25.__new__(_RankBM25)
        obj.__dict__.update(state)
        return obj
    return _ReferenceScorer(state)


def _state_valid(state, ids):
    n = len(ids)
    if state.get("corpus_size") != n or len(state.get("doc_freqs",[])) != n or len(state.get("doc_len",[])) != n:
        raise ValueError("BM25 corpus size and product metadata disagree")
    if not math.isfinite(float(state.get("avgdl",0))) or state["avgdl"] <= 0:
        raise ValueError("Invalid average document length in BM25 index")
    if not isinstance(state.get("idf"),dict) or not state["idf"]:
        raise ValueError("BM25 IDF dictionary is missing")
    if any(not math.isfinite(float(v)) for v in state["idf"].values()):
        raise ValueError("Non-finite BM25 IDF")
    if any(sum(freq.values()) != size for freq,size in zip(state["doc_freqs"],state["doc_len"])):
        raise ValueError("BM25 document frequency counts and lengths disagree")
    if not math.isclose(sum(state["doc_len"])/n, state["avgdl"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("BM25 average document length does not match rows")


class BM25IndexManager:
    """Compatible with existing search_qa and HybridSearchEngine call sites."""
    def __init__(self, index_path=getattr(Config,"BM25_INDEX_PATH",str(DEFAULT_PATH)), k1=getattr(Config,"BM25_K1",1.5), b=getattr(Config,"BM25_B",.75)):
        self.index_path=str(index_path)
        self.k1=float(k1);self.b=float(b);self.bm25=None;self.id_mapping=[]
        self.document_tokens=[];self.text_col=None;self.metadata={}

    def build_from_records(self, records, text_col="product_text_lexical", *, source_metadata=None):
        data=list(records)
        mapping=[{k:r[k] for k in ("product_id","title_fa","id","comment_clean") if k in r} for r in data]
        ids=mapping_ids(mapping)
        if any(text_col not in r for r in data):
            raise ValueError(f"Missing lexical source column: {text_col}")
        tokens=[tokenize_query(r[text_col]) for r in data]
        if _RankBM25 is not None:
            obj=_RankBM25(tokens,k1=self.k1,b=self.b)
            state={k:v for k,v in vars(obj).items() if k in ("k1","b","epsilon","corpus_size","avgdl","doc_freqs","idf","doc_len","tokenizer","average_idf")}
        else:
            state=_okapi_state(tokens,self.k1,self.b)
        _state_valid(state,ids)
        self.bm25=_make_scorer(state);self.id_mapping=mapping;self.document_tokens=tokens;self.text_col=text_col
        self.metadata={"format_version":INDEX_FORMAT,"tokenization_version":TOKENIZATION_VERSION,
                       "mapping_sha256":mapping_digest(mapping),"corpus_size":len(mapping),**(source_metadata or {})}
        # The corrected artifact contains only serialized data, not executable
        # objects. Installed rank_bm25 is rehydrated from validated state at load.
        payload={"format_version":INDEX_FORMAT,"bm25_state":state,"id_mapping":mapping,
                 "document_tokens":tokens,"k1":self.k1,"b":self.b,"text_col":text_col,"metadata":self.metadata}
        path=Path(self.index_path);path.parent.mkdir(parents=True,exist_ok=True)
        temp=None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent,suffix=".tmp",delete=False) as handle:
                temp=Path(handle.name);pickle.dump(payload,handle,protocol=4)
            temp.replace(path)
        finally:
            if temp is not None:temp.unlink(missing_ok=True)
        return self.bm25,self.id_mapping

    def build_and_save_index(self, df_sample, text_col=None):
        if text_col is None:
            text_col="product_text_lexical" if "product_text_lexical" in df_sample.columns else "title_fa"
        if text_col not in df_sample.columns:raise ValueError(f"Missing {text_col}")
        columns=list(dict.fromkeys([c for c in ("product_id","title_fa","id","comment_clean",text_col) if c in df_sample.columns]))
        frame=df_sample[columns].copy();frame[text_col]=frame[text_col].fillna("").astype(str)
        return self.build_from_records(frame.to_dict(orient="records"),text_col)

    def load_index(self):
        path=Path(self.index_path)
        if not path.is_file():return False
        payload=_read_index(path)
        if not isinstance(payload,dict):raise ValueError("Invalid BM25 index payload")
        self.id_mapping=payload["id_mapping"];ids=mapping_ids(self.id_mapping)
        state=payload.get("bm25_state")
        if state is None:
            legacy=payload.get("bm25")
            if not isinstance(legacy,_LegacyStateCarrier):raise ValueError("Unsupported BM25 index format")
            state=vars(legacy)
        _state_valid(state,ids)
        self.bm25=_make_scorer(state);self.k1=state["k1"];self.b=state["b"]
        self.text_col=payload.get("text_col");self.metadata=payload.get("metadata",{})
        self.document_tokens=payload.get("document_tokens") or [list(freq.keys()) for freq in state["doc_freqs"]]
        if len(self.document_tokens)!=len(ids):raise ValueError("BM25 token and mapping counts differ")
        expected=self.metadata.get("mapping_sha256")
        if expected is not None and expected!=mapping_digest(self.id_mapping):raise ValueError("BM25 mapping checksum mismatch")
        # Existing 500-item artifacts still load for diagnostics, but callers
        # should validate their alignment before comparing dense and lexical modes.
        return True

    def search(self,query,top_k=getattr(Config,"TOP_K",5),*,allowed_ids=None):
        if self.bm25 is None:raise ValueError("BM25 index is not loaded or built")
        if top_k<=0:return []
        tokens=[t for t in tokenize_query(query) if t not in _SEARCH_FILLER]
        if not tokens:return []
        token_set=set(tokens);scores=self.bm25.get_scores(tokens)
        required=max(1, math.ceil(.60*len(token_set)))
        leading=set(list(dict.fromkeys(tokens))[:2])
        matched=[]
        for i,frequencies in enumerate(self.bm25.doc_freqs):
            if allowed_ids is not None and product_key(self.id_mapping[i]["product_id"]) not in allowed_ids:
                continue
            overlap=len(token_set.intersection(frequencies))
            if overlap>=required:
                matched.append(i)
                continue
            # Preserve explicit product names followed by a use-case request
            # (e.g. purpose words absent from catalogue descriptions).
            title_anchor=(len(token_set)>2 and len(leading)==2 and leading.issubset(frequencies)
                          and leading.issubset(tokenize_query(self.id_mapping[i].get("title_fa",""))))
            if title_anchor:
                matched.append(i)
        matched.sort(key=lambda i:(-float(scores[i]),i))
        return [{**self.id_mapping[i],"score":float(scores[i])} for i in matched[:top_k]]
