"""Local, blinded human evaluation; only explicit reviewer submissions count."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import re
import sqlite3

from bonus.observability import STATE,timestamp
from src.config import Config
from src.task2_review_store import SQLiteReviewStore

ROOT=Path(__file__).resolve().parents[1]
DB=STATE/"human_reviews.sqlite3"


def cases():
    j=json.loads((ROOT/"evaluation/results/final/evaluation.json").read_text(encoding="utf8"))
    return j["qa"]


def display_case(case):
    """Never expose the assistant judgment before the reviewer submits."""
    result=case["result"]
    rows=SQLiteReviewStore(Config.CATALOG_PATH).get(case["product_id"]).reviews
    return {"case_id":case["case_id"],"product_id":case["product_id"],"product_title":case["product_title"],
            "question":case["question"],"answer":result.get("answer") or result.get("reason"),
            "evidence":result.get("evidence",[]),
            "source_reviews":[{"comment_id":r["comment_id"],"title":r.get("title_raw"),"body":r.get("body_raw"),
                               "advantages":r.get("advantages_raw"),"disadvantages":r.get("disadvantages_raw")} for r in rows]}


def _connect(path=DB):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(path)
    c.execute("""CREATE TABLE IF NOT EXISTS reviews(reviewer TEXT NOT NULL,case_id TEXT NOT NULL,
       fingerprint TEXT NOT NULL,quality INTEGER NOT NULL,grounding INTEGER NOT NULL,
       rationale TEXT NOT NULL,created_at TEXT NOT NULL,PRIMARY KEY(reviewer,case_id,fingerprint))""")
    return c


def submit(reviewer,case_id,quality,grounding,rationale,*,path=DB):
    if not isinstance(reviewer,str) or not re.fullmatch(r"[\w.-]{1,32}",reviewer.strip()):
        raise ValueError("یک شناسه مستعار کوتاه (حرف/عدد) برای داور وارد کنید.")
    if type(quality) is not int or quality not in (0,1,2) or type(grounding) is not int or grounding not in (0,1,2):
        raise ValueError("هر دو نمره مستقل را انتخاب کنید.")
    if not isinstance(rationale,str) or not rationale.strip() or len(rationale)>1000:
        raise ValueError("دلیل کوتاه و واقعی داوری را وارد کنید.")
    case=next((x for x in cases() if x["case_id"]==case_id),None)
    if case is None:raise ValueError("Unknown benchmark case")
    with _connect(path) as c:
        c.execute("INSERT OR REPLACE INTO reviews VALUES(?,?,?,?,?,?,?)",
                  (reviewer.strip(),case_id,case["fingerprint"],quality,grounding,rationale.strip(),timestamp()))
    return {"case_id":case_id,"reviewer":reviewer.strip(),"saved":True}


def records(path=DB):
    if not Path(path).is_file():return []
    refs={r["case_id"]:r for r in cases()}
    with _connect(path) as c:
        c.row_factory=sqlite3.Row
        rows=[dict(r) for r in c.execute("SELECT * FROM reviews ORDER BY case_id,reviewer")]
    return [r for r in rows if r["case_id"] in refs and r["fingerprint"]==refs[r["case_id"]]["fingerprint"]]


def agreement(rows=None):
    from sklearn.metrics import cohen_kappa_score
    rows=records() if rows is None else list(rows)
    refs={r["case_id"]:r["assessment"] for r in cases() if r.get("assessment")}
    rows=[r for r in rows if r["case_id"] in refs]
    output={"human_submissions":len(rows),"unique_cases":len({r["case_id"] for r in rows}),
            "reviewers":len({r["reviewer"] for r in rows}),"total_cases":len(cases()),
            "quality_exact":None,"grounding_exact":None,"quality_weighted_kappa":None,"grounding_weighted_kappa":None}
    if not rows:return output
    import math
    for key in ("quality","grounding"):
        actual=[r[key] for r in rows]; expected=[refs[r["case_id"]][key+"_score"] for r in rows]
        output[key+"_exact"]=sum(a==b for a,b in zip(actual,expected))/len(actual)
        if len(actual)>=2 and len(set(actual+expected))>=2:
            v=float(cohen_kappa_score(actual,expected,labels=[0,1,2],weights="quadratic"))
            output[key+"_weighted_kappa"]=v if math.isfinite(v) else None
    return output


def export_csv(rows=None):
    rows=records() if rows is None else list(rows)
    refs={r["case_id"]:r["assessment"] for r in cases() if r.get("assessment")}
    fields=["reviewer","case_id","fingerprint","quality","grounding","rationale","created_at","assistant_quality","assistant_grounding"]
    out=io.StringIO();w=csv.DictWriter(out,fieldnames=fields);w.writeheader()
    for r in rows:
        ref=refs.get(r["case_id"],{})
        # Neutralize spreadsheet formula interpretation for human-entered text.
        d=dict(r)
        for key in ("reviewer","rationale"):
            if str(d.get(key,"")).lstrip().startswith(("=","+","-","@")):d[key]="'"+d[key]
        w.writerow({**d,"assistant_quality":ref.get("quality_score"),"assistant_grounding":ref.get("grounding_score")})
    return out.getvalue().encode("utf-8-sig")
