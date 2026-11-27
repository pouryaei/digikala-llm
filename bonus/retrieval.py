"""BM25 + character TF-IDF hybrid retrieval, with explicit filters and guardrails.

The extra representation is sparse lexical/fuzzy retrieval, not an embedding
model. The original BM25 index/service is left unchanged.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from src.bm25_index import tokenize_query, _SEARCH_FILLER
from src.catalog import open_catalog
from src.dashboard_rag import _matching_ids, _dash_products, search as baseline_search
from src.product_preferences import recorded_budget
from src.rag_query import search_request
from src.search_qa import search_products
from src.task2_review_store import identifier

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"bonus_artifacts"/"retrieval"
VERSION="char-hybrid-1"
# General shopping vocabulary; variants are retrieval hints, not assertions
# about device compatibility or the technical characteristics of a product.
ALIASES={
    "هدست":"هدفون","هندزفری":"هدفون","headset":"هدفون","headphone":"هدفون","headphones":"هدفون",
    "wireless":"بیسیم","bluetooth":"بیسیم","بلوتوث":"بیسیم","بلوتوثی":"بیسیم",
    "بچه":"کودک","بچگانه":"کودک","کودکان":"کودک","اطفال":"کودک","مهد":"کودک",
    "دخترانه":"کودک","پسرانه":"کودک",
    "گلس":"محافظ","glass":"محافظ","galaxy":"گلکسی","fit":"فیت",
    "samsung":"سامسونگ","موهای":"مو","موی":"مو","hair":"مو",
    "dry":"خشک","مرطوبکننده":"مرطوب","ضدآفتاب":"آفتاب"
}
FILLER=set(_SEARCH_FILLER)|set("یه لطفا لطفن خواهم خوام میخوام استفاده معرفی کن کاربری کاربردی برای مناسب از برند مدل کد طرح عدد بسته روزانه".split())


def canonical_tokens(text):
    tokens=tokenize_query(text)
    out=[]
    i=0
    while i<len(tokens):
        if tokens[i:i+2]==["بی","سیم"]:
            out.append("بیسیم");i+=2;continue
        if tokens[i:i+2]==["ضد","آفتاب"]:
            out.append("آفتاب");i+=2;continue
        out.append(ALIASES.get(tokens[i],tokens[i]));i+=1
    return out


def index_text(text):
    return " ".join(canonical_tokens(text))


def query_terms(text):
    # Deduplicate synonyms to count one intent once.
    return list(dict.fromkeys(t for t in canonical_tokens(text) if t not in FILLER))


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def build_index():
    import joblib
    import sklearn
    from scipy.sparse import save_npz
    from sklearn.feature_extraction.text import TfidfVectorizer
    path=ROOT/"storage/catalog_manifest.json"
    catalog_manifest=json.loads(path.read_text(encoding="utf8"))
    cat=open_catalog()
    with cat.connection() as con:
        rows=con.execute("SELECT product_id,title_fa,brand FROM products ORDER BY product_id").fetchall()
    ids=np.array([r[0] for r in rows],dtype=np.int64)
    docs=[index_text(str(r[1])+" "+str(r[2] or "")) for r in rows]
    started=perf_counter()
    vectorizer=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,4),max_features=80000,min_df=2,
                               sublinear_tf=True,dtype=np.float32,norm="l2")
    matrix=vectorizer.fit_transform(docs).tocsr()
    ART.mkdir(parents=True,exist_ok=True)
    joblib.dump(vectorizer,ART/"vectorizer.joblib",compress=3)
    save_npz(ART/"matrix.npz",matrix,compressed=True)
    np.save(ART/"ids.npy",ids,allow_pickle=False)
    manifest={"version":VERSION,"catalog_sha256":catalog_manifest["catalog_sha256"],
              "products":len(rows),"features":matrix.shape[1],"nnz":int(matrix.nnz),
              "build_seconds":round(perf_counter()-started,3),"scikit_learn":sklearn.__version__,
              "files":{n:sha(ART/n) for n in ("vectorizer.joblib","matrix.npz","ids.npy")},
              "method":"character word-boundary TF-IDF, 2–4 grams, title+brand, normalized vocabulary"}
    (ART/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    return manifest


@lru_cache(maxsize=1)
def load_index():
    import joblib
    from scipy.sparse import load_npz
    manifest=json.loads((ART/"manifest.json").read_text(encoding="utf8"))
    cat_hash=json.loads((ROOT/"storage/catalog_manifest.json").read_text(encoding="utf8"))["catalog_sha256"]
    if manifest["version"]!=VERSION or manifest["catalog_sha256"]!=cat_hash:
        raise ValueError("Character index is not aligned to this release")
    for name,expected in manifest["files"].items():
        if sha(ART/name)!=expected:raise ValueError("Character index checksum mismatch")
    v=joblib.load(ART/"vectorizer.joblib")
    matrix=load_npz(ART/"matrix.npz").tocsr()
    ids=np.load(ART/"ids.npy",allow_pickle=False)
    if len(ids)!=matrix.shape[0] or len(ids)!=manifest["products"] or len(set(ids.tolist()))!=len(ids):
        raise ValueError("Character index ID mismatch")
    return v,matrix,ids,{int(pid):i for i,pid in enumerate(ids)}


def _matches(token,words):
    if token in words:return 1.0
    if token.isdecimal() or len(token)<4:return 0.0
    # Small edit/transposition robustness; don't match arbitrary short words.
    return max((SequenceMatcher(None,token,w,autojunk=False).ratio()
                for w in words if len(w)>=4 and abs(len(w)-len(token))<=2),default=0.0)


def _coverage(terms,words):
    scores=[_matches(t,words) for t in terms]
    return sum(v>=.74 for v in scores)/len(terms) if terms else 0.,scores


def route_retrieval(query,filters=None):
    req=search_request(query)
    try:
        identifier(query)
        return "bm25","explicit_product_id"
    except ValueError:
        pass
    if req.prefer_affordability or req.prefer_satisfaction or req.scope_hint:
        return "bm25","existing_preference_and_scope_policy"
    terms=query_terms(req.text)
    if len(terms)<=2 or filters:
        return "bm25","simple_or_structured_request"
    return "hybrid_char","long_or_multi_constraint_query"


def search(query,filters=None,limit=5,method="auto"):
    started=perf_counter()
    req=search_request(query)
    actual,reason=route_retrieval(query,filters) if method=="auto" else (method,"user_selected")
    if actual not in ("bm25","hybrid_char"):
        raise ValueError("Unknown retrieval method")
    if not isinstance(limit,int) or not 1<=limit<=20:
        raise ValueError("limit must be 1..20")
    filters=dict(filters or {})
    if filters.keys()-{"category","brand","min_price","max_price"}:
        raise ValueError("Unknown filter")
    for key in ("min_price","max_price"):
        if key in filters and (isinstance(filters[key],bool) or not isinstance(filters[key],(int,float))
                               or not math.isfinite(filters[key]) or filters[key]<0):
            raise ValueError("بازه قیمت معتبر نیست.")
    # Keep original handling of explicit IDs / meaningful preference-aware
    # ranking and fallback notices. Hybrid does not weaken these contracts.
    try:
        explicit=identifier(query)
    except ValueError:
        explicit=None
    prechecked_empty=False
    if actual=="bm25" or explicit is not None or req.prefer_satisfaction or req.prefer_affordability or req.scope_hint:
        result=baseline_search(query,filters,limit)
        if (method=="auto" and reason=="simple_or_structured_request" and not filters
                and not result["items"] and len(query_terms(req.text))>=2):
            actual,reason="hybrid_char","bm25_empty_hybrid_rescue"
            prechecked_empty=True
        else:
            return {**result,"retrieval_method":"bm25","route_reason":reason if actual=="bm25" else "protected_existing_search_policy",
                    "retrieval_seconds":perf_counter()-started}
    terms=query_terms(req.text)
    if not terms:
        return {"items":[],"applied_filters":filters,"scope":"indexed_catalogue","search_notes":[],
                "retrieval_method":"hybrid_char","route_reason":reason,"retrieval_seconds":perf_counter()-started}
    cat=open_catalog()
    allowed=_matching_ids(filters)
    notes=["ترکیب بازیابی واژگانی BM25 و شباهت کاراکتری؛ تطابق تقریبی تضمین مشخصات یا سازگاری کالا نیست."]
    if req.min_price is not None or req.max_price is not None:
        raw=recorded_budget(req,cat)
        money=cat.filtered_ids(min_price=req.min_price,max_price=req.max_price,
                               price_in_recorded_units=raw)
        allowed=money if allowed is None else allowed&money
        notes.append("فیلتر بودجه در واحد ثبت‌شده اعمال شد؛ واحد تومان تأیید نشده." if raw else "فیلتر بودجه با واحد منبع تأییدشده اعمال شد.")
    if allowed is not None and not allowed:
        return {"items":[],"applied_filters":filters,"scope":"indexed_catalogue","search_notes":[],
                "retrieval_method":"hybrid_char","route_reason":reason,"retrieval_seconds":perf_counter()-started}
    # Both models use the same allowed product IDs. No filter is relaxed.
    lexical=[] if prechecked_empty else search_products(query,top_k=50,mode="bm25",allowed_product_ids=allowed)
    if method=="auto" and any(query_terms(row.get("title_fa",""))==terms for row in lexical[:limit]):
        # Exact catalogue-title intent is already solved by the cheaper
        # lexical model. Keep the same order; skip the second representation.
        selected=[int(row["product_id"]) for row in lexical[:limit]]
        info=_dash_products(selected)
        return {"items":[info[pid] for pid in selected if pid in info],"applied_filters":filters,
                "scope":"indexed_catalogue","search_notes":["تطابق عنوان دقیق؛ اجرای مدل دوم لازم نبود.",*notes[1:]],
                "retrieval_method":"bm25","route_reason":"exact_title_confidence",
                "retrieval_seconds":perf_counter()-started}
    # If the optional index was not built, fail clearly; never relabel BM25.
    v,matrix,ids,pos=load_index()
    lex_rank={int(r["product_id"]):i+1 for i,r in enumerate(lexical)}
    sparse_q=v.transform([" ".join(terms)])
    sims=np.asarray((matrix@sparse_q.T).toarray()).ravel()
    if allowed is not None:
        mask=np.zeros(len(ids),dtype=bool)
        chosen=[pos[int(pid)] for pid in allowed if int(pid) in pos]
        mask[chosen]=True;sims[~mask]=0
    depth=min(80,len(sims))
    indices=np.argpartition(-sims,depth-1)[:depth]
    indices=sorted(indices,key=lambda i:(-float(sims[i]),int(ids[i])))
    char_rank={int(ids[i]):j+1 for j,i in enumerate(indices) if sims[i]>.10}
    candidates=set(lex_rank)|set(char_rank)
    products=cat.get_products(candidates)
    numbers={t for t in terms if t.isdecimal()}
    # Device manufacturer named in the query is a hard anchor, not a
    # category synonym. A Huawei accessory is not evidence of Samsung fit.
    makers=set(terms)&{"سامسونگ","هواوی","اپل","شیائومی","سونی","لنوو"}
    ranked=[]
    for pid,row in products.items():
        title_words=set(canonical_tokens(row["title_fa"]+" "+str(row.get("brand") or "")))
        fields=title_words|set(canonical_tokens(" ".join(str(row.get(k) or "") for k in ("category1","category2","sub_category"))))
        coverage,match=_coverage(terms,fields)
        tc,_=_coverage(terms,title_words)
        cosine=float(sims[pos[pid]])
        # Require primary anchor + majority of intent; preserve abstention
        # on out-of-catalog requests and explicit numeric model identifiers.
        if match[0]<.74 or coverage<.60 or not numbers<=fields or not makers<=fields:
            continue
        if pid not in lex_rank and cosine<.16:
            continue
        rrf=((1/(60+lex_rank[pid])) if pid in lex_rank else 0)+((1/(60+char_rank[pid])) if pid in char_rank else 0)
        combined=.46*tc+.25*coverage+.24*cosine+.05*(rrf/(2/61))
        ranked.append((combined,int(pid),tc,coverage,cosine,lex_rank.get(pid),char_rank.get(pid)))
    ranked.sort(key=lambda r:(-r[0],r[5] or 9999,r[1]))
    selected=ranked[:limit]
    if not selected and lexical and not makers and not numbers:
        # A conservative fallback for natural use-case words absent from
        # titles. Do not relax explicit manufacturer or number constraints.
        base=baseline_search(query,filters,limit)
        return {**base,"retrieval_method":"hybrid_char","route_reason":reason+"; lexical_fallback",
                "search_notes":[*base.get("search_notes",[]),"بازرتبه‌بندی شاهد کافی نداشت؛ نتایج واژگانی قبلی با همان فیلترها نمایش داده شدند."],
                "candidate_counts":{"bm25":len(lex_rank),"char":len(char_rank),"guarded":0},
                "retrieval_seconds":perf_counter()-started}
    info=_dash_products([r[1] for r in selected])
    items=[{**info[pid],"hybrid_score":round(s,5),"title_coverage":round(tc,3),
            "char_similarity":round(cos,4),"bm25_rank":lr,"char_rank":cr}
           for s,pid,tc,cov,cos,lr,cr in selected if pid in info]
    return {"items":items,"applied_filters":filters,"scope":"indexed_catalogue",
            "search_notes":notes,
            "retrieval_method":"hybrid_char","route_reason":reason,"candidate_counts":{"bm25":len(lex_rank),"char":len(char_rank),"guarded":len(ranked)},
            "retrieval_seconds":perf_counter()-started}


if __name__=="__main__":
    print(json.dumps(build_index(),ensure_ascii=False,indent=2))
