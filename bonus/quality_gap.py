"""New manager question: discrepancies between rating and review recommendation.

A triage flag, not proof of poor product quality or causal evidence.
"""
from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
from time import perf_counter

from src.catalog import open_catalog
from src.task2_review_store import SQLiteReviewStore
from src.config import Config

ROOT=Path(__file__).resolve().parents[1]


def wilson_upper(success,n,z=1.96):
    if n<=0:return None
    p=success/n
    return (p+z*z/(2*n)+z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


@lru_cache(maxsize=1)
def catalogue_feedback():
    with open_catalog().connection() as c:
        # Denominator includes all three valid recommendation labels.
        rows=c.execute("""
            SELECT p.product_id,p.title_fa,p.category1,p.brand,p.rating_100,p.rating_count,
              count(r.comment_id) AS total,
              sum(CASE WHEN r.recommendation_status IN ('recommended','not_recommended','no_idea') THEN 1 ELSE 0 END) AS valid,
              sum(CASE WHEN r.recommendation_status='recommended' THEN 1 ELSE 0 END) AS positive
            FROM products p JOIN reviews r USING(product_id)
            WHERE p.rating_100 IS NOT NULL AND p.rating_count>0 GROUP BY p.product_id
        """).fetchall()
    return tuple(dict(r) for r in rows)


def evidence(product_id,limit=3):
    rows=SQLiteReviewStore(Config.CATALOG_PATH).get(product_id).reviews
    out=[]
    for row in rows:
        if row["recommendation_status"]!="not_recommended":continue
        for field in ("disadvantages_raw","body_raw","title_raw"):
            text=row.get(field)
            if isinstance(text,str) and text.strip():
                out.append({"comment_id":row["comment_id"],"product_id":product_id,"text":text.strip()[:400]})
                break
        if len(out)>=limit:break
    return out


def audit(category=None,*,min_labels=20,min_ratings=10,min_rating=70,gap=.15,limit=20,include_evidence=True):
    start=perf_counter()
    if not isinstance(min_labels,int) or min_labels<5 or not isinstance(min_ratings,int) or min_ratings<1:
        raise ValueError("حداقل تعداد برچسب باید ۵ و حداقل رأی ۱ باشد.")
    if not 0<=min_rating<=100 or not 0<=gap<=1 or not 1<=limit<=50:
        raise ValueError("Invalid audit thresholds")
    total=naive=low_support=0; eligible=[]; flagged=[]
    for row in catalogue_feedback():
        if category and row["category1"]!=category:continue
        if not row["valid"] or row["rating_100"]<min_rating:continue
        total+=1
        p=row["positive"]/row["valid"]
        raw_gap=row["rating_100"]/100-p
        enough=row["valid"]>=min_labels and row["rating_count"]>=min_ratings
        if raw_gap>=gap:
            naive+=1
            if not enough:low_support+=1
        if not enough:continue
        upper=wilson_upper(row["positive"],row["valid"])
        conservative=row["rating_100"]/100-upper
        item={"product_id":row["product_id"],"title":row["title_fa"],"category":row["category1"],"brand":row["brand"],
              "rating_100":row["rating_100"],"rating_count":row["rating_count"],"review_count":row["total"],
              "valid_labels":row["valid"],"recommended":row["positive"],"recommendation_rate":p,
              "wilson_upper":upper,"naive_gap":raw_gap,"conservative_gap":conservative}
        eligible.append(item)
        if conservative>=gap:flagged.append(item)
    flagged.sort(key=lambda r:(-r["conservative_gap"],-r["valid_labels"],r["product_id"]))
    top=flagged[:limit]
    if include_evidence:
        top=[{**row,"evidence":evidence(row["product_id"])} for row in top]
    return {"status":"answered","scope":category or "all 100k products","thresholds":{"min_labels":min_labels,
                "min_ratings":min_ratings,"min_rating_100":min_rating,"min_gap":gap},
            "rated_and_labeled_products":total,"eligible_products":len(eligible),"naive_flags":naive,
            "naive_flags_insufficient_support":low_support,"conservative_flags":len(flagged),
            "items":top,"seconds":perf_counter()-start,
            "note":"Rating and recommendation measure different user populations. Flags are for manual investigation, not proven defects; the Wilson interval is for the observed labeled recommendation proportion only."}


def evaluate():
    from src.task2_review_store import FIELDS
    report=audit()
    cat=open_catalog()
    for row in report["items"]:
        with cat.connection() as c:
            counts=c.execute("""
                SELECT count(*),sum(CASE WHEN recommendation_status='recommended' THEN 1 ELSE 0 END),
                    sum(CASE WHEN recommendation_status IN ('recommended','not_recommended','no_idea') THEN 1 ELSE 0 END)
                  FROM reviews WHERE product_id=?
            """,[row["product_id"]]).fetchone()
        if tuple(counts)!=(row["review_count"],row["recommended"],row["valid_labels"]):
            raise AssertionError("Audit source counts mismatch")
        if row["conservative_gap"]>row["naive_gap"]+1e-10:
            raise AssertionError("Conservative gap should be <= naive gap")
        source={x["comment_id"]:x for x in SQLiteReviewStore(Config.CATALOG_PATH).get(row["product_id"]).reviews}
        for e in row["evidence"]:
            actual=source[e["comment_id"]]
            if e["product_id"]!=actual["product_id"] or not any(e["text"] in (actual.get(f) or "").strip() for f in FIELDS):
                raise AssertionError("Audit citation mismatch")
    sensitivity=[]
    for n in (10,20,50):
        for g in (.10,.15,.20):
            r=audit(min_labels=n,gap=g,include_evidence=False)
            sensitivity.append({"min_labels":n,"gap":g,"eligible":r["eligible_products"],"flagged":r["conservative_flags"]})
    result={"new_problem":"Disagreement between catalogue rating and labeled recommendation; conservative investigation queue",
            "mentor_approval_confirmed":False,"report":report,"sensitivity":sensitivity,
            "source_checks":{"top_rows":len(report["items"]),"count_checks_pass":True,"exact_evidence_checks_pass":True},
            "limitation":"No ground-truth defect labels in this dataset. Evaluation validates arithmetic and evidence only; usefulness and mentor approval require real review."}
    path=ROOT/"reports/bonus/quality_gap.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    return result


if __name__=="__main__":
    r=evaluate()
    print(json.dumps({k:v for k,v in r["report"].items() if k!="items"},ensure_ascii=False,indent=2))
