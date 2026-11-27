"""Reproduce actual review QA cases without paid API calls.

Optional --full-source verifies all selected product reviews directly from the
original recovered payload; the bundled sample is the same complete product
scope. The checks cover evidence membership, verbatim quotation, counts,
expected evidence/no-evidence behavior, and scoped latency. They are not a
substitute for human semantic relevance labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.review_qa import ask_reviews
from src.task2_review_store import FIELDS, JsonlReviewStore, PayloadReviewStore
from src.task2_text_processing import normalize_persian_text


def normalized(text): return (normalize_persian_text(text) or "").lower()


def actual_cases(manifest):
    clues = {"Q16": ["جنس", "چرم", "کیفیت"], "Q17": ["جنس", "نازک", "کیفیت"], "Q18": ["کیفیت", "پاک"], "Q19": ["کیفیت", "جنس"]}
    out = [{"case_id": q["query_id"], "product_id": int(q["target_product_id"]), "question": q["query"],
            "expected_mode": "insufficient_evidence" if q["query_id"] in {"Q15", "Q20"} else "extractive",
            "clues": clues.get(q["query_id"], []), "group": "existing_qa_query"} for q in manifest["qa_queries"]]
    def case(cid, pid, q, expected="extractive", clues=None, intent=None, group="official_examples"):
        return {"case_id": cid, "product_id": pid, "question": q, "expected_mode": expected,
                "clues": clues or [], "expected_intent": intent, "group": group}
    out.extend([
        case("P1",10133811,"مردم بیشتر از چه چیزی در این محصول راضی بودند؟",intent="positive"),
        case("P2",10133811,"ایرادهای پرتکرار این محصول چیست؟",intent="negative"),
        case("P3",10133811,"خریداران درباره کیفیت این محصول چه گفته‌اند؟",clues=["کیفیت","جنس","دوخت"],intent="aspect"),
        case("P4",10133811,"آیا با توجه به تجربه کاربران ارزش خرید دارد؟",intent="value"),
        case("P5",1096725,"آیا این کرم به نرم شدن پوست خشک پا کمک می‌کند؟",clues=["نرم","خشکی","ترک"],group="topical_queries"),
        case("P6",1126863,"کیفیت چاپ و ترجمه این کتاب چطور است؟",clues=["چاپ","ترجمه","کاغذ","صفحه"],group="topical_queries"),
        case("P7",1126863,"باتری این محصول چند ساعت شارژ نگه می‌دارد؟","insufficient_evidence",group="insufficient_evidence"),
        case("P8",999999999,"کیفیت این محصول چگونه است؟","insufficient_evidence",group="outside_sample"),
        case("P9",10133811,"سایزبندی و اندازهٔ کفش چطور است؟",clues=["سایز","اندازه","تنگ","گشاد"],group="topical_queries"),
        case("P10",1096725,"نظر کاربران دربارهٔ قیمت و ارزش خرید این کرم چیست؟",clues=["قیمت","ارزش","گرون","گران"],group="topical_queries"),
    ])
    return out


def validate(result, batch):
    rows={r["comment_id"]:r for r in batch.reviews}
    evidence=result["evidence"]
    ids=[e["comment_id"] for e in evidence]
    evidence_checks=[]
    for e in evidence:
        review=rows.get(e["comment_id"])
        evidence_checks.append(bool(review and e["product_id"]==batch.product_id and any(e["text"] in (review.get(f) or "") for f in FIELDS)))
    references={int(i) for i in re.findall(r"(?:\(|،\s*)نظر\s+(\d+)",result.get("answer") or "")}
    labels=Counter(r.get("recommendation_status") for r in batch.reviews)
    valid=sum(labels[x] for x in ("recommended","not_recommended","no_idea"));s=result["stats"]
    stats_ok=(s.get("total_reviews")==len(rows) and s.get("valid_label_count")==valid and s.get("recommended_count")==labels["recommended"] and s.get("not_recommended_count")==labels["not_recommended"] and s.get("no_idea_count")==labels["no_idea"])
    ratio=s.get("recommendation_rate")
    stats_ok=stats_ok and ((ratio is None) if not valid else (ratio is not None and math.isclose(ratio,labels["recommended"]/valid)))
    themes_ok=all(0<t["count"]<=t["denominator"] and bool(t["evidence_ids"]) and set(t["evidence_ids"])<=set(ids) for t in result.get("theme_counts",[]))
    return {"evidence_count":len(ids),"exact_evidence_count":sum(evidence_checks),"all_evidence_exact_and_scoped":all(evidence_checks),"no_duplicate_ids":len(ids)==len(set(ids)),"answer_references_in_evidence":references<=set(ids),"stats_match_source":stats_ok,"theme_references_valid":themes_ok}


def evaluate(sample_file, output_dir, full_source=None):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    sample=JsonlReviewStore(sample_file);manifest=sample.manifest
    full_meta={"verified_sample_row_count":manifest["row_count"],"sample_product_count":len(manifest["products"]),
               "source_total_clean_reviews":manifest["source_total_clean_reviews"],"source_payload_sha256":manifest["source_payload_sha256"],
               "selection":manifest["selection"],"full_payload_rechecked_in_this_run":False}
    all_cases=actual_cases(manifest)
    with tempfile.TemporaryDirectory(prefix="task2_eval_") as temp:
        store=sample
        if full_source:
            # Each retained product has all its reviews in the sample. Validate
            # the final production payload adapter against the exact original rows.
            start=time.perf_counter();store=PayloadReviewStore(full_source,cache_dir=temp)
            batches=store.get_many(list(sample.products));cold=time.perf_counter()-start
            for pid,batch in batches.items():
                actual={r["comment_id"]:r for r in batch.reviews};expected={r["comment_id"]:r for r in sample.get(pid).reviews}
                if actual!=expected: raise AssertionError(f"Full source and sample differ for product {pid}")
            start=time.perf_counter();warm=PayloadReviewStore(full_source,cache_dir=temp)
            warm.get_many(list(sample.products));warm_time=time.perf_counter()-start
            full_meta.update({"full_payload_rechecked_in_this_run":True,"all_selected_product_rows_match_original":True,
                              "cold_scan_seconds":round(cold,6),"warm_load_seconds":round(warm_time,6),"validated_products":len(batches),
                              "validated_review_rows":sum(len(b.reviews) for b in batches.values())})
        records=[]
        for case in all_cases:
            pid=case["product_id"]
            selected_store=sample if case["group"]=="outside_sample" else store
            result=ask_reviews(pid,case["question"],store=selected_store,generation_mode="extractive")
            batch=selected_store.get(pid)
            checks=validate(result,batch)
            matches=case["clues"]
            keyword_check=any(normalized(word) in normalized(e["text"]) for word in matches for e in result["evidence"]) if matches else None
            correct_mode=result["answer_mode"]==case["expected_mode"]
            intent_ok=case.get("expected_intent") is None or result.get("intent")==case.get("expected_intent")
            all_ok=correct_mode and intent_ok and all(checks[k] for k in ("all_evidence_exact_and_scoped","no_duplicate_ids","answer_references_in_evidence","stats_match_source","theme_references_valid")) and keyword_check is not False
            record={**case,"product_title":sample.products.get(pid,{}).get("title"),"checks":checks,"expected_mode_match":correct_mode,"expected_intent_match":intent_ok,
                    "keyword_relevance_check":keyword_check,"checked_pass":all_ok,"result":result}
            records.append(record)
            print(case["case_id"],result["answer_mode"],"evidence",len(result["evidence"]),"PASS" if all_ok else "FAIL",flush=True)
    timings=sorted(r["result"]["meta"]["total_seconds"] for r in records)
    summary={"case_count":len(records),"checked_pass_count":sum(r["checked_pass"] for r in records),
             "answered":sum(r["result"]["answer_mode"]!="insufficient_evidence" for r in records),
             "insufficient_evidence":sum(r["result"]["answer_mode"]=="insufficient_evidence" for r in records),
             "expected_mode_matches":sum(r["expected_mode_match"] for r in records),
             "total_evidence":sum(r["checks"]["evidence_count"] for r in records),
             "exact_scoped_evidence":sum(r["checks"]["exact_evidence_count"] for r in records),
             "stats_match_cases":sum(r["checks"]["stats_match_source"] for r in records),
             "keyword_checks":sum(r["keyword_relevance_check"] is not None for r in records),
             "keyword_checks_passed":sum(r["keyword_relevance_check"] is True for r in records),
             "median_query_seconds":round(statistics.median(timings),6),"p95_query_seconds":round(timings[math.ceil(.95*len(timings))-1],6),
             "max_query_seconds":round(max(timings),6),"api_calls":0,"prompt_tokens":0,"completion_tokens":0,"api_cost_usd":0.0,
             "mode":"deterministic extractive / real reviews, no live LLM", "human_semantic_evaluation_completed":False}
    report={"source":full_meta,"summary":summary,"cases":records}
    (output_dir/"task2_evaluation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    with (output_dir/"task2_case_results.csv").open("w",encoding="utf-8-sig",newline="") as fp:
        keys=["case_id","group","product_id","question","expected_mode","actual_mode","evidence_count","exact_evidence_count","all_evidence_exact_and_scoped","stats_match_source","expected_mode_match","keyword_relevance_check","checked_pass","latency_seconds","api_calls","evidence_ids","reason"]
        w=csv.DictWriter(fp,fieldnames=keys);w.writeheader()
        for x in records:
            r=x["result"];c=x["checks"]
            w.writerow({"case_id":x["case_id"],"group":x["group"],"product_id":x["product_id"],"question":x["question"],"expected_mode":x["expected_mode"],"actual_mode":r["answer_mode"],
                        "evidence_count":c["evidence_count"],"exact_evidence_count":c["exact_evidence_count"],"all_evidence_exact_and_scoped":c["all_evidence_exact_and_scoped"],"stats_match_source":c["stats_match_source"],"expected_mode_match":x["expected_mode_match"],"keyword_relevance_check":x["keyword_relevance_check"],"checked_pass":x["checked_pass"],"latency_seconds":r["meta"]["total_seconds"],"api_calls":r["meta"]["api_calls"],"evidence_ids":";".join(str(e["comment_id"]) for e in r["evidence"]),"reason":r.get("reason","")})
    with (output_dir/"task2_human_review.csv").open("w",encoding="utf-8-sig",newline="") as fp:
        keys=["case_id","product_id","question","answer","evidence","human_relevance_0_2","human_grounding_0_2","human_completeness_0_2","reviewer_notes"]
        w=csv.DictWriter(fp,fieldnames=keys);w.writeheader()
        for x in records:
            r=x["result"]
            w.writerow({"case_id":x["case_id"],"product_id":x["product_id"],"question":x["question"],"answer":r.get("answer") or r.get("reason"),"evidence":"\n".join(f"{e['comment_id']}: {e['text']}" for e in r["evidence"])})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if summary["checked_pass_count"] != summary["case_count"]:
        raise SystemExit("One or more real-data checks failed; inspect task2_case_results.csv")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample",default=str(ROOT/"data"/"task2_demo"/"reviews.jsonl"))
    p.add_argument("--output-dir",default=str(ROOT/"reports"))
    p.add_argument("--full-source",help="Forouhar_M1_Complete.zip or full reviews_payload.csv.zst")
    a=p.parse_args();evaluate(a.sample,a.output_dir,a.full_source)
