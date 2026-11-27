"""Section 4: reproducible evaluation of the bundled, actual 100k system.

Default OFFLINE. No API credential, external LLM, judge API or training required.
Run from any directory: python -X utf8 evaluation/run_evaluation.py
Use --core-only for the fixed retrieval/QA benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sqlite3
import statistics
import subprocess
import sys
from time import perf_counter
import unicodedata
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE_PATH = ROOT / "evaluation/cases.json"
OUT = ROOT / "evaluation/results"
CAT = ROOT / "storage/catalog.sqlite3"
MODEL = ROOT / "m2_outputs/best_model.joblib"
FIELDS = ("title_raw", "body_raw", "advantages_raw", "disadvantages_raw")
LABELS = ("no_idea", "not_recommended", "recommended")
OFFLINE_ENV = {k: "" for k in ("QBC13_BACKEND_URL", "QBC13_SEARCH_FUNCTION", "QBC13_REVIEW_FUNCTION",
    "QBC13_PREDICT_FUNCTION", "QBC13_ANALYTICS_DIR", "FOROUHAR_COMMENTS_PATH", "QBC13_REVIEW_SOURCE",
    "QBC13_API_KEY", "QBC13_SOURCE_PRICE_UNIT")}
OFFLINE_ENV.update(QBC13_REVIEW_GENERATION_MODE="extractive", QBC13_UI_SEARCH_MODE="bm25",
                   DIGIKALA_MODEL_PATH=str(MODEL))

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=str)+"\n", encoding="utf-8")

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",",":"),default=str).encode()).hexdigest()

def file_hash(path):
    with path.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()

def avg(values):
    xs=[x for x in values if x is not None]
    return statistics.mean(xs) if xs else None

def fraction(a,b):
    return a/b if b else None

@contextmanager
def database():
    con=sqlite3.connect(CAT.resolve().as_uri()+"?mode=ro", uri=True)
    con.row_factory=sqlite3.Row
    try: yield con
    finally: con.close()

def timed(f):
    t=perf_counter()
    result=f()
    return result,perf_counter()-t

def distribution(times):
    ordered=sorted(times)
    if not ordered: return {"n":0}
    # nearest-rank P95; with a small n this may be the maximum
    return {"n":len(ordered),"mean_ms":1000*statistics.mean(ordered),
            "median_ms":1000*statistics.median(ordered),
            "p95_ms":1000*ordered[math.ceil(.95*len(ordered))-1],
            "min_ms":1000*ordered[0],"max_ms":1000*ordered[-1]}

def normalized(value):
    if value is None: return ""
    value=unicodedata.normalize("NFKC",str(value)).casefold()
    value=value.translate(str.maketrans("يىكآأإ۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩","ییکااا01234567890123456789"))
    return " ".join(re.findall(r"[^\W_]+",value))

def contains(text,phrase):
    phrase=normalized(phrase)
    return bool(phrase) and f" {phrase} " in f" {text} "

def catalogue(con):
    rows={}
    for row in con.execute("SELECT product_id,title_fa,category1,brand,price_recorded,rating_100,rating_count FROM products"):
        r=dict(row); pid=r["product_id"]
        rows[pid]={**r,"title_norm":normalized(r["title_fa"]),"category_norm":normalized(r["category1"]),"brand_norm":normalized(r["brand"])}
    return rows

def independent_reference(case, products):
    """Conservative catalogue gold defined by fixed title/category/brand/price rules."""
    if case.get("target_product_id"):
        pid=int(case["target_product_id"])
        if pid not in products: raise ValueError(f"Known-item reference absent: {pid}")
        return {pid}
    c=case["criteria"]; f=case["filters"]
    cats={normalized(cat) for cat in c["categories"]}
    prefixes=[normalized(x) for x in c["title_prefix_any"]]
    groups=[[normalized(x) for x in g] for g in c["title_has_all_groups"]]
    exclusions=[normalized(x) for x in c["title_exclude_any"]]
    brand=normalized(c["brand"]) if c["brand"] else None
    fcat=normalized(f["category"]) if f.get("category") else None
    fbrand=normalized(f["brand"]) if f.get("brand") else None
    result=set()
    for pid,p in products.items():
        title=p["title_norm"]; price=p["price_recorded"]
        if (cats and p["category_norm"] not in cats or fcat and p["category_norm"]!=fcat or
            brand and p["brand_norm"]!=brand or fbrand and p["brand_norm"]!=fbrand):
            continue
        if "min_price" in f and (price is None or price<f["min_price"]): continue
        if "max_price" in f and (price is None or price>f["max_price"]): continue
        if prefixes and not any(title==x or title.startswith(x+" ") for x in prefixes): continue
        if groups and not all(any(" "+x+" " in " "+title+" " for x in g) for g in groups): continue
        if exclusions and any(" "+x+" " in " "+title+" " for x in exclusions): continue
        result.add(pid)
    if case.get("expected")=="empty" and result:
        raise ValueError(f"{case['case_id']} no-result reference disagrees with catalogue: {len(result)}")
    return result

def ranked_metrics(grades, all_grades, k):
    g=list(grades)[:k]; gold=list(all_grades)
    direct=sum(x==2 for x in g); total_direct=sum(x==2 for x in gold)
    dcg=sum((2**x-1)/math.log2(i+2) for i,x in enumerate(g))
    ideal=sorted(gold,reverse=True)[:k]
    idcg=sum((2**x-1)/math.log2(i+2) for i,x in enumerate(ideal))
    return {"returned":len(g),"direct":direct,"partial":sum(x==1 for x in g),
            "strict_precision_returned":fraction(direct,len(g)),
            "strict_precision_at_k":direct/k,
            "relaxed_precision_returned":fraction(sum(x>0 for x in g),len(g)),
            "direct_recall_at_k":fraction(direct,total_direct),
            "hit_at_k":int(direct>0),
            "mrr_at_k":next((1/(i+1) for i,x in enumerate(g) if x==2),0.0),
            "ndcg_at_k":fraction(dcg,idcg)}

def independent_stats(rows):
    counts=Counter(r.get("recommendation_status") for r in rows)
    valid=sum(counts[x] for x in LABELS)
    return {"total_reviews":len(rows),"valid_label_count":valid,"recommended_count":counts["recommended"],
            "not_recommended_count":counts["not_recommended"],"no_idea_count":counts["no_idea"],
            "recommendation_rate":fraction(counts["recommended"],valid)}

def matched_stats(actual,expected):
    for k,v in expected.items():
        found=actual.get(k)
        if isinstance(v,float):
            if found is None or not math.isclose(found,v,rel_tol=1e-9,abs_tol=1e-10): return False
        elif found!=v: return False
    return True

def exact_witness(e,by_id,pid):
    try: cid=int(e["comment_id"]); row=by_id[cid]
    except (ValueError,TypeError,KeyError):return False
    return int(e.get("product_id",-1))==pid and row["product_id"]==pid and bool(e.get("text")) and any(e["text"] in (row.get(f) or "") for f in FIELDS)

def grounding(result,rows,pid):
    by_id={r["comment_id"]:r for r in rows}
    evidence=result.get("evidence",[])
    ids=[int(e["comment_id"]) for e in evidence]
    exact=sum(exact_witness(e,by_id,pid) for e in evidence)
    answer=result.get("answer") or ""
    references={int(x) for x in re.findall(r"(?:\(|،\s*)نظر\s+(\d+)",answer)}
    # Independently check the narrative's literal excerpts, not just evidence cards.
    quotes=re.findall(r"«(.*?)»\s*\(نظر\s+(\d+)\)",answer,re.S)
    quotes_ok=sum(bool(q.strip() and int(cid) in by_id and any(q in (by_id[int(cid)].get(f) or "") for f in FIELDS)) for q,cid in quotes)
    stat_match=matched_stats(result.get("stats",{}),independent_stats(rows))
    themes=result.get("theme_counts",[])
    facets=result.get("facet_coverage",[])
    theme_valid=all(0<t["count"]<=t["denominator"]<=len(rows) and t["evidence_ids"] and set(t["evidence_ids"])<=set(ids) for t in themes)
    facet_valid=all(0<=t["matched_review_count"]<=len(rows) and set(t["evidence_ids"])<=set(ids) for t in facets)
    return {"evidence":len(evidence),"evidence_exact_and_scoped":exact,
            "citation_ids_valid":references<=set(ids),"distinct_evidence":len(ids)==len(set(ids)),
            "quoted_spans":len(quotes),"quoted_spans_exact":quotes_ok,
            "stats_match_source":stat_match,"theme_bounds_and_references":theme_valid,"facet_bounds_and_references":facet_valid,
            "pass":exact==len(evidence) and references<=set(ids) and len(ids)==len(set(ids)) and quotes_ok==len(quotes) and stat_match and theme_valid and facet_valid}

def token_usage(meta):
    # Missing provider usage is not treated as a known zero-token paid request.
    calls=int(meta.get("api_calls",meta.get("calls",0)) or 0)
    attempts=int(meta.get("api_attempts",calls) or 0)
    known=bool(meta.get("usage_known",calls==0 and attempts==0)) and (calls==0 or "prompt_tokens" in meta and "completion_tokens" in meta)
    return {"api_calls":calls,"api_attempts":attempts,"prompt_tokens":int(meta.get("prompt_tokens",0) or 0),
            "completion_tokens":int(meta.get("completion_tokens",0) or 0),"usage_known":known}

def summarize_search(cases):
    groups={}
    for label in ("all","dev","holdout","title_derived"):
        rows=[r for r in cases if label=="all" or r["split"]==label]
        known=[r for r in rows if r["kind"]=="known_item"]
        pos=[r for r in rows if r["kind"]!="known_item" and r["reference_count"]>0]
        neg=[r for r in rows if not r["reference_count"]]
        groups[label]={"cases":len(rows),"known_item_cases":len(known),"known_hit_at_5":avg([r["metrics"]["hit_at_k"] for r in known]),
            "known_mrr_at_5":avg([r["metrics"]["mrr_at_k"] for r in known]),
            "discovery_answerable":len(pos),"discovery_p_at_5":avg([r["metrics"]["strict_precision_at_k"] for r in pos]),
            "discovery_precision_returned_micro":fraction(sum(r["metrics"]["direct"] for r in pos),sum(r["metrics"]["returned"] for r in pos)),
            "discovery_hit_at_5":avg([r["metrics"]["hit_at_k"] for r in pos]),
            "discovery_ndcg_at_5":avg([r["metrics"]["ndcg_at_k"] for r in pos]),
            "no_result_cases":len(neg),"correct_no_result":sum(r["correct_empty"] for r in neg),
            "no_result_accuracy":avg([r["correct_empty"] for r in neg]),
            "all_filter_and_catalogue_checks":all(r["catalogue_and_filters_valid"] for r in rows),
            "latency":distribution([r["seconds"] for r in rows])}
    return groups

def summarize_qa(cases):
    groups={}
    for label in ("all","dev","holdout"):
        rows=[r for r in cases if label=="all" or r["split"]==label]
        positive=[r for r in rows if r["expected"]=="answered"]
        negative=[r for r in rows if r["expected"]=="insufficient_evidence"]
        answered=[r for r in rows if r["result"]["status"]=="answered"]
        n=sum(r["metrics"]["returned"] for r in rows)
        groups[label]={"cases":len(rows),"answerable":len(positive),"unanswerable":len(negative),"answered":len(answered),
            "status_accuracy":avg([r["correct_status"] for r in rows]),
            "answerable_with_direct_evidence":sum(r["result"]["status"]=="answered" and r["metrics"]["direct"]>0 for r in positive),
            "answerable_hit_rate":avg([r["result"]["status"]=="answered" and r["metrics"]["direct"]>0 for r in positive]),
            "correct_abstention":sum(r["correct_status"] for r in negative),
            "abstention_accuracy":avg([r["correct_status"] for r in negative]),
            "review_strict_precision_micro":fraction(sum(r["metrics"]["direct"] for r in rows),n),
            "review_relaxed_precision_micro":fraction(sum(r["metrics"]["direct"]+r["metrics"]["partial"] for r in rows),n),
            "review_macro_recall_at_8":avg([r["metrics"]["direct_recall_at_k"] for r in positive]),
            "review_macro_ndcg_at_8":avg([r["metrics"]["ndcg_at_k"] for r in positive]),
            "evidence_total":n,"exact_scoped_evidence":sum(r["grounding"]["evidence_exact_and_scoped"] for r in rows),
            "quote_total":sum(r["grounding"]["quoted_spans"] for r in rows),
            "quotes_exact":sum(r["grounding"]["quoted_spans_exact"] for r in rows),
            "grounding_structural_pass":sum(r["grounding"]["pass"] for r in rows),
            "latency":distribution([r["seconds"] for r in rows])}
    return groups

def result_fingerprint(result):
    return digest({"status":result.get("status"),"answer":result.get("answer"),"reason":result.get("reason"),"evidence":[[e["comment_id"],e["product_id"],e["text"]] for e in result.get("evidence",[])]})

def evaluate_core(cases):
    from ui_services import api_call, APIError
    from src.rag import run_rag_pipeline
    search_cases=list(cases["search"])
    old=read_json(ROOT/"queries.json")
    with database() as con:
        products=catalogue(con)
        if len(products)!=100000: raise ValueError("This benchmark is aligned to the bundled 100k catalogue")
        for q in old["queries"]:
            if q.get("target_product_id"):
                search_cases.append({"case_id":"TITLE_"+q["query_id"],"split":"title_derived","query":q["query"],"filters":{},"kind":"known_item","target_product_id":int(q["target_product_id"])})
        search_rows=[]
        for case in search_cases:
            reference=independent_reference(case,products)
            started=perf_counter()
            try:
                entry,secs=timed(lambda:api_call("POST","/api/search",{"query":case["query"],"filters":case["filters"],"limit":5}))
                items=entry["data"]["items"]
            except APIError as exc:
                items=[]; secs=perf_counter()-started; error=str(exc)
            else:error=None
            grades=[2 if int(p["product_id"]) in reference else 0 for p in items]
            metrics=ranked_metrics(grades,[2]*len(reference),5)
            out=[]
            consistent=True
            for p,grade in zip(items,grades):
                pid=int(p["product_id"]); real=products.get(pid); f=case["filters"]
                ok=bool(real and p["title_fa"]==real["title_fa"] and p.get("price")==real["price_recorded"] and p.get("brand")==real["brand"] and p.get("category")==real["category1"])
                if real:
                    ok=ok and (not f.get("category") or normalized(f["category"])==real["category_norm"]) and (not f.get("brand") or normalized(f["brand"])==real["brand_norm"])
                    ok=ok and ("min_price" not in f or real["price_recorded"] is not None and real["price_recorded"]>=f["min_price"]) and ("max_price" not in f or real["price_recorded"] is not None and real["price_recorded"]<=f["max_price"])
                consistent=consistent and ok
                out.append({"product_id":pid,"title":p["title_fa"],"category":p.get("category"),"brand":p.get("brand"),"price":p.get("price"),"grade":grade})
            search_rows.append({**case,"reference_count":len(reference),"reference_sample_ids":sorted(reference)[:12],
                     "metrics":metrics,"results":out,"seconds":secs,"error":error,"correct_empty":not reference and not items and error is None,
                     "catalogue_and_filters_valid":consistent,"usage":token_usage({})})
        print("Search complete",len(search_rows),flush=True)
        qa_rows=[]
        stress=[]
        for case in [*cases["qa"],*cases["stress_qa"]]:
            rows=[dict(r) for r in con.execute("SELECT * FROM reviews WHERE product_id=? ORDER BY comment_id",(case["product_id"],))]
            if "qrels" in case:
                if digest(rows)!=case["source_review_ids_sha256"] or set(map(str,(r["comment_id"] for r in rows)))!=set(case["qrels"]):raise ValueError(f"Gold labels stale for {case['case_id']}")
            entry,secs=timed(lambda:api_call("POST","/api/reviews/ask",{"product_id":case["product_id"],"query":case["question"]}))
            result=entry["data"]
            result["status"]="answered" if result.get("answer") is not None else "insufficient_evidence"
            # Exclude machine-specific source paths from the reproducible result.
            result.pop("source",None)
            row={**case,"seconds":secs,"result":result,"fingerprint":result_fingerprint(result),
                 "grounding":grounding(result,rows,case["product_id"]),"usage":token_usage(result["meta"])}
            if "qrels" in case:
                grades=[case["qrels"].get(str(e["comment_id"]),0) for e in result["evidence"]]
                row["grades_in_rank_order"]=grades
                row["metrics"]=ranked_metrics(grades,list(case["qrels"].values()),8)
                row["correct_status"]=result["status"]==case["expected"]
                qa_rows.append(row)
            else:stress.append(row)
        edges=[]
        for case in cases["edge_qa"]:
            result,secs=timed(lambda:run_rag_pipeline(case["query"],case["product_id"],generation_mode="extractive",source=CAT))
            edges.append({**case,"status":result["status"],"resolved_product_id":result.get("product_id"),"reason":result.get("reason"),
                          "correct":result["status"]==case["expected"] and ("expected_product_id" not in case or result.get("product_id")==case["expected_product_id"]),
                          "seconds":secs,"usage":token_usage(result.get("meta",{}))})
    print("QA complete",len(qa_rows),flush=True)
    return {"search":search_rows,"qa":qa_rows,"stress_qa":stress,"edge_qa":edges,
            "search_summary":summarize_search(search_rows),"qa_summary":summarize_qa(qa_rows)}


def classification(y_true,y_pred):
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    if not y_true: return None
    return {"n":len(y_true),"accuracy":float(accuracy_score(y_true,y_pred)),
            "macro_f1":float(f1_score(y_true,y_pred,labels=list(LABELS),average="macro",zero_division=0)),
            "report":classification_report(y_true,y_pred,labels=list(LABELS),output_dict=True,zero_division=0),
            "confusion_matrix_labels":list(LABELS),
            "confusion_matrix":confusion_matrix(y_true,y_pred,labels=list(LABELS)).tolist()}

def evaluate_ml(out):
    from verify_ml_integration import verify_recorded_test
    import ml_service, sklearn, joblib
    rows,verified=verify_recorded_test()
    actual=[r["recommendation_status"] for r in rows]
    recorded=classification(actual,[r["predicted_status"] for r in rows])
    majority=Counter(actual).most_common(1)[0][0]
    baseline=classification(actual,[majority]*len(actual))
    # Only replay IDs in the 100k bundle, never substitute a random test set
    # or assert the whole 15k is available. Original training texts are absent.
    ids=[int(r["comment_id"]) for r in rows]
    found={}
    with database() as con:
        for i in range(0,len(ids),500):
            subset=ids[i:i+500]
            marks=",".join("?" for _ in subset)
            for p in con.execute(f"SELECT comment_id,product_id,recommendation_status,title_raw,body_raw FROM reviews WHERE comment_id IN ({marks})",subset):
                found[str(p["comment_id"])]=dict(p)
    aligned=[(r,found[r["comment_id"]]) for r in rows if r["comment_id"] in found]
    if not aligned or any(g["recommendation_status"]!=s["recommendation_status"] for s,g in aligned):
        raise ValueError("Recorded test labels disagree with matched 100k source")
    _,load_seconds=timed(ml_service.load_model)
    results=[]
    t=perf_counter()
    for i in range(0,len(aligned),512):
        block=aligned[i:i+512]
        results.extend(ml_service.predict_recommendation_batch([r["body_raw"] for _,r in block],[r["title_raw"] for _,r in block]))
    batch_seconds=perf_counter()-t
    output=[]; failure_records=[]
    for (saved,raw),pred in zip(aligned,results):
        truth=raw["recommendation_status"]
        row={"comment_id":int(saved["comment_id"]),"product_id":raw["product_id"],"true_status":truth,
                "recorded_prediction":saved["predicted_status"],"replayed_prediction":pred["label"] if pred["valid"] else None,
                "valid":pred["valid"],"reason":pred["reason"],"confidence":pred["confidence"]}
        output.append(row)
        if pred["valid"] and truth!=row["replayed_prediction"]:
            failure_records.append({**row,"raw_title":raw["title_raw"],"raw_body_excerpt":(raw["body_raw"] or "")[:260]})
    out.mkdir(parents=True,exist_ok=True)
    with (out/"ml_replay.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    valid=[x for x in output if x["valid"]]
    overlap_recorded=classification([x["true_status"] for x in output],[x["recorded_prediction"] for x in output])
    replay=classification([x["true_status"] for x in valid],[x["replayed_prediction"] for x in valid])
    replay_all=classification([x["true_status"] for x in output],[x["replayed_prediction"] or "__rejected__" for x in output])
    majority_overlap=classification([x["true_status"] for x in valid],[majority]*len(valid))
    thresholds=[]
    for threshold in (.5,.6,.7,.8):
        keeping=[x for x in valid if x["confidence"]>=threshold]
        met=classification([x["true_status"] for x in keeping],[x["replayed_prediction"] for x in keeping]) if keeping else None
        thresholds.append({"threshold":threshold,"accepted":len(keeping),"coverage_of_matched":len(keeping)/len(output),"macro_f1_on_accepted":met["macro_f1"] if met else None,
                           "accuracy_on_accepted":met["accuracy"] if met else None})
    failures=[]
    for true,pred in ((a,b) for a in LABELS for b in LABELS if a!=b):
        cases=[x for x in failure_records if x["true_status"]==true and x["replayed_prediction"]==pred]
        if cases:
            # Show a high-confidence mistake per direction; not a selected-success demo.
            failures.append(max(cases,key=lambda row:row["confidence"]))
    print("ML original",recorded["n"],recorded["macro_f1"],"actual raw replay",len(output),len(valid),replay["macro_f1"],flush=True)
    return {"model_sha256":file_hash(MODEL),"runtime":{"scikit_learn":sklearn.__version__,"joblib":joblib.__version__},
            "recorded_test":recorded,"recorded_metrics_verified":verified["matches_supplied_reports"],
            "majority_class":majority,"majority_recorded_baseline":baseline,
            "overlap_with_bundled_reviews":len(output),"missing_from_bundle":len(rows)-len(output),
            "recorded_metrics_for_overlap":overlap_recorded,"valid_replay":len(valid),
            "rejected_count":len(output)-len(valid),"rejected_reasons":dict(Counter(x["reason"] for x in output if not x["valid"])),
            "replay_metrics_valid_only":replay,
            "replay_macro_f1_with_rejections_counted_as_misses":replay_all["macro_f1"],
            "replay_accuracy_with_rejections_counted_as_misses":replay_all["accuracy"],
            "majority_replay_baseline":majority_overlap,
            "same_label_as_original_on_valid":sum(x["replayed_prediction"]==x["recorded_prediction"] for x in valid),
            "load_seconds":load_seconds,"batch_seconds":batch_seconds,"confidence_tradeoff":thresholds,
            "examples_wrong":failures,"errors_by_true_and_pred":{f"{a}->{b}":n for (a,b),n in Counter((x["true_status"],x["replayed_prediction"]) for x in failure_records).items()},
            "limitations":"15k metrics are recomputed from fixed supplied predictions; replay only the intersection with the 100k corpus using raw title+body and service preprocessing. Training-time inputs/full split are not available in this ZIP. No retraining or threshold selection on the test set."}

def rows_for(con,pid):
    return [dict(r) for r in con.execute("SELECT * FROM reviews WHERE product_id=? ORDER BY comment_id",(pid,))]

def evaluate_comparison():
    from ui_services import api_call
    ids_sets=[[1696173,2896502],[8217413,11405303,4176458],[1696173,2896502,8217413,11405303,4176458]]
    records=[]
    with database() as con:
        for i,pids in enumerate(ids_sets,1):
            entry,secs=timed(lambda:api_call("POST","/api/products/compare",{"product_ids":pids}))
            result=entry["data"]; items=result["items"]; evidence=result["evidence"]
            source={p:{r["comment_id"]:r for r in rows_for(con,p)} for p in pids}
            exact=sum(int(e["product_id"]) in source and exact_witness(e,source[int(e["product_id"])],int(e["product_id"])) for e in evidence)
            fact_ok=True; count_ok=True
            for item in items:
                real=con.execute("SELECT * FROM products WHERE product_id=?",(item["product_id"],)).fetchone()
                fact_ok=fact_ok and item["title_fa"]==real["title_fa"] and item["price"]==real["price_recorded"] and item["brand"]==real["brand"] and item["rating"]==real["rating_100"]
                stats=independent_stats(list(source[int(item["product_id"])].values()))
                actual={**item,"total_reviews":item["comment_count"]}
                count_ok=count_ok and matched_stats(actual,{k:v for k,v in stats.items() if k not in ("not_recommended_count","no_idea_count")})
            ids_match=[x["product_id"] for x in items]==pids
            records.append({"case_id":f"C{i:02d}","ids":pids,"seconds":secs,"facts_match":fact_ok,"counts_match":count_ok,
                            "ids_match":ids_match,"evidence_count":len(evidence),"exact_evidence":exact,
                            "pass":ids_match and fact_ok and count_ok and exact==len(evidence),"result":result,"usage":token_usage(result.get("meta",{}))})
    print("Comparison",[(r["case_id"],r["pass"]) for r in records],flush=True)
    return records

def manager_request(category,question):
    from manager_analytics import answer_manager_question,ManagerScope
    return answer_manager_question(question,ManagerScope(category))

def evaluate_managers():
    from manager_analytics import QUESTION_EXAMPLES
    from analytics_service import open_analytics_tables,resolve_analytics_files,get_category_review_distribution
    records=[]
    with database() as con:
        for category in ("کفش مردانه","کتاب شعر و ادبیات"):
            row=con.execute("SELECT count(DISTINCT p.product_id),count(r.comment_id) FROM products p JOIN reviews r ON p.product_id=r.product_id WHERE p.category1=?",(category,)).fetchone()
            scope_products,scope_reviews=row
            for question in QUESTION_EXAMPLES.values():
                answer,secs=timed(lambda:manager_request(category,question))
                counts_ok=(answer.product_count,answer.review_count)==(scope_products,scope_reviews)
                # Compare aggregate counts independently against the read-only SQLite store.
                for x in answer.rows:
                    if "product_id" in x or "brand" in x:
                        if "product_id" in x:
                            raw=con.execute("""SELECT COUNT(*) n,SUM(r.recommendation_status='recommended') recommended,
                                SUM(r.recommendation_status IN ('recommended','not_recommended','no_idea')) valid
                                FROM reviews r JOIN products p USING(product_id) WHERE r.product_id=? AND p.category1=?""",(int(x["product_id"]),category)).fetchone()
                            counts_ok=counts_ok and raw["n"]==x["review_count"] and (raw["recommended"] or 0)==x["recommended_count"] and (raw["valid"] or 0)==x["valid_label_count"]
                        else:
                            # Brand spellings in the dashboard are grouped by Persian-normalized key;
                            # match that key rather than an arbitrary representative string.
                            def group_key(value):
                                if value is None: return None
                                value=unicodedata.normalize("NFC",value).translate(str.maketrans("يىك٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹","ییک01234567890123456789"))
                                return re.sub(r"[\s\u200b\u200c]+"," ",value).strip()
                            con.create_function("group_key",1,group_key)
                            raw=con.execute("""SELECT COUNT(*) n,SUM(r.recommendation_status='recommended') recommended,
                                SUM(r.recommendation_status IN ('recommended','not_recommended','no_idea')) valid
                                FROM reviews r JOIN products p USING(product_id) WHERE p.category1=? AND lower(group_key(p.brand))=?""",(category,x["brand_key"])).fetchone()
                            counts_ok=counts_ok and raw["n"]==x["review_count"] and (raw["recommended"] or 0)==x["recommended_count"] and (raw["valid"] or 0)==x["valid_label_count"]
                        if x["valid_label_count"]:
                            counts_ok=counts_ok and math.isclose(100*x["recommended_count"]/x["valid_label_count"],x["recommendation_pct"],rel_tol=1e-9,abs_tol=1e-9)
                    elif "mentions" in x:
                        # Rule-based topic totals need semantic re-annotation; bounds and denominator only.
                        counts_ok=counts_ok and 0<x["mentions"]<=scope_reviews and math.isclose(100*x["mentions"]/scope_reviews,x["review_share_pct"],rel_tol=1e-9,abs_tol=1e-9)
                ev=[e for values in answer.evidence.values() for e in values]
                exact=0
                if ev:
                    # Manager quotations originate from cleaned Parquet fields (not necessarily raw bytes).
                    with open_analytics_tables(resolve_analytics_files()) as a:
                        raw_rows=a.execute("""SELECT CAST(r.comment_id AS VARCHAR),CAST(r.product_id AS VARCHAR),
                            p.category1,r.title_clean,r.body_clean,r.disadvantages_clean
                            FROM reviews r JOIN products p USING(product_id)
                            WHERE CAST(r.comment_id AS VARCHAR) IN (SELECT unnest(?::VARCHAR[]))""",
                            [list(dict.fromkeys(e.comment_id for e in ev))]).fetchall()
                    byid={r[0]:r for r in raw_rows}
                    for e in ev:
                        source=byid.get(e.comment_id)
                        if source and source[1]==e.product_id and source[2]==category and any(e.quote in (s or "") for s in source[3:]):
                            exact+=1
                records.append({"category":category,"question":question,"intent":answer.intent,"seconds":secs,"product_count":answer.product_count,
                    "review_count":answer.review_count,"row_count":len(answer.rows),"scope_and_numeric_checks":bool(counts_ok),
                    "evidence_count":len(ev),"exact_clean_scoped_evidence":exact,"pass":counts_ok and exact==len(ev),
                    "result":asdict(answer),"usage":token_usage({})})
    distribution_frame,secs=timed(get_category_review_distribution)
    summary={"categories":len(distribution_frame),"reviews":int(distribution_frame["review_count"].sum()),
             "matches_bundle":int(distribution_frame["review_count"].sum())==1874457,"seconds":secs}
    print("Manager",[(r["intent"],r["category"],r["pass"]) for r in records],flush=True)
    return {"cases":records,"category_distribution":summary}

def probe(name):
    from ui_services import api_call
    if name=="search_plain": return api_call("POST","/api/search",{"query":"کفش پیاده روی مردانه","filters":{},"limit":5})
    if name=="search_filter": return api_call("POST","/api/search",{"query":"خودکار","filters":{"category":"نوشت افزار","max_price":200000},"limit":5})
    if name=="search_empty": return api_call("POST","/api/search",{"query":"بلیط هواپیما از تهران به پاریس","filters":{},"limit":5})
    if name=="qa_6_reviews": return api_call("POST","/api/reviews/ask",{"product_id":8217413,"query":"سایزبندی کفش چطور است؟"})
    if name=="qa_200_reviews": return api_call("POST","/api/reviews/ask",{"product_id":1696173,"query":"ایرادهای پرتکرار این محصول چیست؟"})
    if name=="qa_no_evidence": return api_call("POST","/api/reviews/ask",{"product_id":1508262,"query":"باتری این کتاب چند ساعت شارژ نگه می‌دارد؟"})
    if name=="compare_2": return api_call("POST","/api/products/compare",{"product_ids":[1696173,2896502]})
    if name=="compare_5": return api_call("POST","/api/products/compare",{"product_ids":[1696173,2896502,8217413,11405303,4176458]})
    if name=="predict_short": return api_call("POST","/api/predict",{"text":"کیفیتش عالیه پیشنهاد می‌کنم"})
    if name=="predict_long": return api_call("POST","/api/predict",{"text":"کفش را برای استفاده روزمره خریدم. سایز با اندازه من مناسب بود اما بعد از چند هفته استفاده دوخت کناره کمی باز شد. بسته بندی سالم بود و محصول به موقع رسید. با توجه به قیمت انتظار کیفیت بالاتری داشتم."})
    if name=="manager_complaints": return manager_request("کفش مردانه","پرتکرارترین شکایت کاربران در این دسته چیست؟")
    if name=="manager_low_recommendation": return manager_request("کفش مردانه","کدام محصولات نظر زیادی دارند اما درصد پیشنهاد خرید آن‌ها پایین است؟")
    if name=="manager_two_products":
        from manager_analytics import answer_manager_question,ManagerScope
        return answer_manager_question("پرتکرارترین شکایت کاربران در این دسته چیست؟",ManagerScope(product_ids=("1696173","2896502")))
    if name=="analytics_distribution":
        from analytics_service import get_category_review_distribution
        return get_category_review_distribution()
    raise ValueError(name)

SCENARIOS=("search_plain","search_filter","search_empty","qa_6_reviews","qa_200_reviews","qa_no_evidence","compare_2","compare_5","predict_short","predict_long","manager_complaints","manager_low_recommendation","manager_two_products","analytics_distribution")

def evaluate_latency(repetitions=5,cold_repetitions=3):
    records=[]
    for name in SCENARIOS:
        # Discard one warmup (same process), then keep every measured observation.
        probe(name)
        for repeat in range(1,repetitions+1):
            _,seconds=timed(lambda:probe(name))
            records.append({"scenario":name,"cache_state":"warm_process","repeat":repeat,"seconds":seconds})
    for name in ("search_plain","qa_200_reviews","predict_short"):
        for repeat in range(1,cold_repetitions+1):
            t=perf_counter()
            call=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--cold-probe",name],
                cwd=ROOT,env={**os.environ,**OFFLINE_ENV,"PYTHONUTF8":"1"},encoding="utf-8",capture_output=True,check=True,timeout=90)
            process_wall=perf_counter()-t
            one=json.loads(call.stdout.strip().splitlines()[-1])
            records.append({"scenario":name,"cache_state":"cold_process","repeat":repeat,"seconds":one["seconds"],"process_wall_seconds":process_wall})
    summary=[]
    for state in ("warm_process","cold_process"):
        for name in SCENARIOS:
            selected=[r for r in records if r["scenario"]==name and r["cache_state"]==state]
            if selected:
                summary.append({"scenario":name,"cache_state":state,**distribution([r["seconds"] for r in selected]),
                    **({"process_total":distribution([r["process_wall_seconds"] for r in selected])} if state=="cold_process" else {})})
    print("Latency",len(records),"measurements",flush=True)
    return {"method":"Sequential Python service wall-clock seconds, excluding browser rendering and network. Warm = one untimed call per scenario; cold = new Python process incl service imports/cache load, OS cache not flushed. P95 nearest rank; small sample not production load/SLA.","observations":records,"summary":summary}

def attach_adjudications(report):
    path=ROOT/"evaluation/answer_assessments.json"
    if not path.exists():return {"reviewed":0,"status":"no frozen answer assessments available"}
    data=read_json(path); refs={r["case_id"]:r for r in data["cases"]}
    matched=[]
    for case in report["qa"]:
        ref=refs.get(case["case_id"])
        if ref and ref["fingerprint"]==case["fingerprint"]:
            case["assessment"]=ref
            matched.append(ref)
    n=len(matched)
    return {"reviewed":n,"total_cases":len(report["qa"]),"unreviewed":len(report["qa"])-n,
            "quality_mean_0_to_2":avg([r["quality_score"] for r in matched]),
            "quality_full":sum(r["quality_score"]==2 for r in matched),
            "quality_partial":sum(r["quality_score"]==1 for r in matched),
            "quality_failed":sum(r["quality_score"]==0 for r in matched),
            "semantic_grounding_mean_0_to_2":avg([r["grounding_score"] for r in matched]),
            "semantic_grounding_full":sum(r["grounding_score"]==2 for r in matched),
            "semantic_grounding_partial":sum(r["grounding_score"]==1 for r in matched),
            "semantic_grounding_failed":sum(r["grounding_score"]==0 for r in matched),
            "excerpt_count":sum(len(r["evidence_grades"]) for r in matched),
            "excerpt_grade2":sum(sum(e["grade"]==2 for e in r["evidence_grades"]) for r in matched),
            "excerpt_grade1":sum(sum(e["grade"]==1 for e in r["evidence_grades"]) for r in matched),
            "method":data["method"]}

def write_csvs(out,report):
    # Keep compact case results alongside complete JSON and reviewable evidence.
    def write(name,rows):
        if not rows:return
        with (out/name).open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write("search_cases.csv",[{"case_id":r["case_id"],"split":r["split"],"query":r["query"],"kind":r["kind"],"reference_count":r["reference_count"],
            "returned":r["metrics"]["returned"],"direct":r["metrics"]["direct"],"hit_at_5":r["metrics"]["hit_at_k"],
            "mrr_at_5":r["metrics"]["mrr_at_k"],"precision_at_5":r["metrics"]["strict_precision_at_k"],"ndcg_at_5":r["metrics"]["ndcg_at_k"],"correct_empty":r["correct_empty"],
            "latency_ms":1000*r["seconds"],"filter_grounding":r["catalogue_and_filters_valid"]} for r in report["search"]])
    write("qa_cases.csv",[{"case_id":r["case_id"],"split":r["split"],"product_id":r["product_id"],"question":r["question"],"expected":r["expected"],
            "actual":r["result"]["status"],"evidence":r["metrics"]["returned"],"direct":r["metrics"]["direct"],"partial":r["metrics"]["partial"],"recall_at_8":r["metrics"]["direct_recall_at_k"],
            "ndcg_at_8":r["metrics"]["ndcg_at_k"],"grounding":r["grounding"]["pass"],"quality_score_0_2":r.get("assessment",{}).get("quality_score"),
            "semantic_grounding_0_2":r.get("assessment",{}).get("grounding_score"),"seconds":r["seconds"]} for r in report["qa"]])
    write("qa_evidence.csv",[{"case_id":r["case_id"],"rank":i+1,"comment_id":e["comment_id"],"product_id":e["product_id"],
            "quote":e["text"],"review_grade_0_2":g,"excerpt_grade_0_2":next((x["grade"] for x in r.get("assessment",{}).get("evidence_grades",[]) if x["comment_id"]==e["comment_id"]),None),
            "source":e.get("source"),"polarity":e.get("polarity")} for r in report["qa"] for i,(e,g) in enumerate(zip(r["result"]["evidence"],r["grades_in_rank_order"]))])
    if "latency" in report:
        write("latency.csv",[{"scenario":r["scenario"],"cache_state":r["cache_state"],"repeat":r["repeat"],"seconds":r["seconds"],
              "process_wall_seconds":r.get("process_wall_seconds")} for r in report["latency"]["observations"]])
        write("latency_summary.csv",[{k:r[k] for k in ("scenario","cache_state","n","mean_ms","median_ms","p95_ms","min_ms","max_ms")} for r in report["latency"]["summary"]])
    usage_rows=[{"scenario":key,"case_id":r.get("case_id",r.get("intent",str(i+1))),**r["usage"]} for key in ("search","qa","stress_qa","edge_qa","comparison") for i,r in enumerate(report.get(key,[]))]
    usage_rows.extend({"scenario":"manager","case_id":r["intent"]+" / "+r["category"],**r["usage"]} for r in report.get("manager",{}).get("cases",[]))
    write("api_usage.csv",usage_rows)

def load_baseline():
    import importlib.util
    import src
    for name,filename in (("src.bm25_index","bm25_index_v1.py"),("src.task2_evidence_selector","task2_evidence_selector_v1.py")):
        spec=importlib.util.spec_from_file_location(name,ROOT/"evaluation/baseline"/filename)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=OUT/"rerun")
    parser.add_argument("--core-only",action="store_true")
    parser.add_argument("--baseline",action="store_true",help="Use frozen v1 policy snapshots for reproducible core comparison.")
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--cold-repeats",type=int,default=3)
    parser.add_argument("--cold-probe",choices=SCENARIOS,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not 1<=args.repeats<=30 or not 1<=args.cold_repeats<=10:parser.error("repeat bounds exceeded")
    with patch.dict(os.environ,OFFLINE_ENV):
        import requests
        # Default offline benchmark never calls the paid gateway or external backend.
        with patch("requests.request",side_effect=AssertionError("Offline benchmark forbids external HTTP")):
            if args.cold_probe:
                _,secs=timed(lambda:probe(args.cold_probe));print(json.dumps({"seconds":secs}));return
            if args.baseline:
                if not args.core_only:parser.error("--baseline requires --core-only")
                load_baseline()
            cases=read_json(CASE_PATH)
            report=evaluate_core(cases)
            report["benchmark"]={"version":cases["version"],"cases_sha256":file_hash(CASE_PATH),"variant":"baseline_v1" if args.baseline else "current",
                                 "products":100000,"reviews":1874457,"qa_reference_pairs":sum(len(r["qrels"]) for r in cases["qa"]),
                                 "runtime":{"python":platform.python_version(),"platform":platform.platform(),"processor":platform.processor(),"logical_cpus":os.cpu_count()}}
            report["answer_quality"]=attach_adjudications(report)
            if not args.core_only:
                report["comparison"]=evaluate_comparison()
                report["manager"]=evaluate_managers()
                report["prediction"]=evaluate_ml(args.output_dir)
                report["latency"]=evaluate_latency(args.repeats,args.cold_repeats)
            usage=[r["usage"] for key in ("search","qa","stress_qa","edge_qa","comparison") for r in report.get(key,[])]
            usage.extend(r["usage"] for r in report.get("manager",{}).get("cases",[]))
            report["cost"]={"api_calls":sum(r["api_calls"] for r in usage),"api_attempts":sum(r["api_attempts"] for r in usage),
                            "prompt_tokens":sum(r["prompt_tokens"] for r in usage),"completion_tokens":sum(r["completion_tokens"] for r in usage),
                            "usage_known":all(r["usage_known"] for r in usage),"api_cost_usd":0.0,
                            "live_llm_evaluated":False,"note":"Offline extractive/BM25/sklearn/SQL only, no paid request. Latency repeats and model replay are also local. CPU/RAM/storage/electricity not included. API model price must be supplied for a live run; no guessed rate."}
            # Machine-readable results and compact tables.
            args.output_dir.mkdir(parents=True,exist_ok=True)
            dump(args.output_dir/"evaluation.json",report)
            write_csvs(args.output_dir,report)
            print("Saved",args.output_dir,flush=True)

if __name__=="__main__":
    main()
