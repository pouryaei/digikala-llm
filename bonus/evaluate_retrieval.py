"""Measured comparison of BM25, sparse hybrid, and adaptive retrieval."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
import statistics
from time import perf_counter
from unittest.mock import patch
import os

from bonus.retrieval import search, load_index, query_terms, index_text
from evaluation.run_evaluation import database, catalogue, independent_reference, ranked_metrics
from src.search_qa import load_search_engine

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports"/"bonus"
STRESS=OUT/"retrieval_stress_cases.json"
METHODS=("bm25","hybrid_char","auto")
SEED=271828


def _mutate(token,rng):
    options=[i for i in range(1,len(token)-1) if token[i]!=token[i+1]]
    if not options:return token
    i=rng.choice(options)
    return token[:i]+token[i+1]+token[i]+token[i+2:]


def make_stress(products,count=40):
    """Freeze a deterministic, title-derived typo stress set before evaluation."""
    if STRESS.is_file():
        return json.loads(STRESS.read_text(encoding="utf8"))["cases"]
    rng=random.Random(SEED)
    ids=sorted(products)
    rng.shuffle(ids)
    title_counts=Counter(index_text(p["title_fa"]) for p in products.values())
    cases=[]
    for pid in ids:
        title=products[pid]["title_fa"]
        terms=query_terms(title)
        if not 3<=len(terms)<=7 or title_counts[index_text(title)]!=1:
            continue
        choices=[i for i,t in enumerate(terms) if len(t)>=4 and t.isalpha()]
        if len(choices)<2:continue
        perturbed=terms.copy()
        changed=0
        for i in rng.sample(choices,2):
            perturbed[i]=_mutate(terms[i],rng);changed+=perturbed[i]!=terms[i]
        if changed!=2:continue
        cases.append({"case_id":f"T{len(cases)+1:02}","kind":"known_item","split":"stress",
                      "query":" ".join(perturbed),"target_product_id":pid,
                      "source_title":title,"filters":{},"expected":"answerable","perturbed_terms":2})
        if len(cases)==count:break
    if len(cases)!=count:raise ValueError("Insufficient deterministic typo cases")
    OUT.mkdir(parents=True,exist_ok=True)
    STRESS.write_text(json.dumps({"seed":SEED,"policy":"Unique real catalogue titles; deterministic two adjacent-character transpositions in two salient alphabetic tokens, fixed before model evaluation. Controlled stress, not human-written queries.",
                                  "cases":cases},ensure_ascii=False,indent=2)+"\n", encoding="utf8")
    return cases


def latency(values):
    values=sorted(values)
    return {"median_ms":statistics.median(values),"mean_ms":statistics.mean(values),
            "p95_ms":values[max(0,int((len(values)*.95+.999999))-1)]}


def summary(rows):
    ans=[r for r in rows if r["reference_count"]>0]
    empty=[r for r in rows if r["reference_count"]==0]
    hits=sum(r["hit"] for r in ans)
    all_n=sum(len(r["ids"]) for r in ans)
    grades=sum(r["relevant"] for r in ans)
    return {"cases":len(rows),"answerable":len(ans),"hit_at_5":hits/len(ans) if ans else None,
            "p_at_5":grades/(5*len(ans)) if ans else None,
            "returned_precision":grades/all_n if all_n else None,
            "mrr_at_5":statistics.mean(r["rr"] for r in ans) if ans else None,
            "ndcg_at_5":statistics.mean(r["ndcg"] for r in ans) if ans else None,
            "no_result_cases":len(empty),"correct_no_result":sum(not r["ids"] and not r.get("error") for r in empty),
            "hybrid_runs":sum(r["actual_method"]=="hybrid_char" for r in rows),
            "latency":latency([r["ms"] for r in rows]),"errors":sum(r.get("error") is not None for r in rows)}


def evaluate():
    with database() as con:products=catalogue(con)
    legacy=json.loads((ROOT/"evaluation/results/final/evaluation.json").read_text(encoding="utf8"))["search"]
    cases=[{**r,"benchmark":"original_diagnostic"} for r in legacy]
    cases.extend({**r,"benchmark":"typo_stress"} for r in make_stress(products))
    load_index();load_search_engine("bm25")
    result={m:[] for m in METHODS}
    for ix,case in enumerate(cases):
        refs=independent_reference(case,products)
        for method in METHODS:
            started=perf_counter();error=None
            try:
                answer=search(case["query"],case.get("filters"),5,method)
                ids=[int(p["product_id"]) for p in answer["items"]]
                actual=answer["retrieval_method"]
            except (ValueError, OSError) as e:
                ids=[];actual=method;error=str(e)[:160]
            ms=(perf_counter()-started)*1000
            rel=[1 if pid in refs else 0 for pid in ids]
            # Metric definitions independent of the retrieval implementation.
            import math
            dcg=sum(1/math.log2(i+2) for i,x in enumerate(rel) if x)
            ideal=sum(1/math.log2(i+2) for i in range(min(5,len(refs))))
            rr=next((1/(i+1) for i,x in enumerate(rel) if x),0)
            result[method].append({"case_id":case["case_id"],"query":case["query"],
                "benchmark":case["benchmark"],"split":case["split"],"kind":case["kind"],
                "reference_count":len(refs),"target_product_id":case.get("target_product_id"),
                "ids":ids,"relevant":sum(rel),"hit":int(any(rel)),"rr":rr,
                "ndcg":dcg/ideal if ideal else 0,"ms":ms,"actual_method":actual,
                "error":error})
        if (ix+1)%20==0:print("evaluated",ix+1,"/",len(cases),flush=True)
    groups={}
    for key in ("original_diagnostic","typo_stress","all","original_discovery"):
        groups[key]={}
        for method in METHODS:
            rows=result[method]
            if key=="original_discovery":rows=[r for r in rows if r["benchmark"]=="original_diagnostic" and r["kind"]=="discovery"]
            elif key!="all":rows=[r for r in rows if r["benchmark"]==key]
            groups[key][method]=summary(rows)
    report={"method":"Same fixed 100K catalogue and filters. BM25 vs BM25+character TF-IDF/RRF/coverage rerank vs deterministic adaptive router.",
            "measurement":"Sequential warm-process wall time; index loaded before timing. No API, human judgement or embedding download.",
            "original_set_limitation":"Original 47 cases are previously inspected diagnostic cases; some metadata-only reference labels do not establish exact compatibility.",
            "stress_limitation":"40 frozen controlled title-derived typo cases; not representative human-written discovery. First results were inspected; subsequent general fallback/anchor and exact-title router guards were rerun diagnostically.",
            "groups":groups,"cases":result}
    (OUT/"retrieval.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    return report


if __name__=="__main__":
    with patch.dict(os.environ,{"QBC13_UI_SEARCH_MODE":"bm25","QBC13_SOURCE_PRICE_UNIT":"",
                                "QBC13_API_KEY":"","QBC13_REVIEW_GENERATION_MODE":"extractive"}):
        r=evaluate()
    print(json.dumps(r["groups"],ensure_ascii=False,indent=2))
