"""Stream the final selected classifier over a CSV/Parquet of your own reviews.

The default input is the real bundled 100k-product review Parquet.
Predictions never use rating, product ID, reviewer metadata or ground-truth label.
This is inference, not training or an independent test.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from time import perf_counter

import duckdb
import ml_service
from recommendation_ml.train_final import ROOT

DEFAULT_INPUT = ROOT / "data/analytics_100000/reviews.parquet"
DEFAULT_OUTPUT = ROOT / "results/final_model_predictions.csv"
DEFAULT_FIELDS = ("row_number","comment_id","product_id","predicted_status","confidence",
                  "p_recommended","p_not_recommended","p_no_idea","valid","reason","model_version")


def select_field(columns, alternatives, explicit=None, required=False):
    if explicit is not None:
        if explicit not in columns:
            raise ValueError(f"Column {explicit!r} not found in input")
        return explicit
    column = next((c for c in alternatives if c in columns), None)
    if column is None and required:
        raise ValueError(f"No review text column found. Available: {', '.join(columns)}. Pass --text-column.")
    return column


def infer(input_path, output_path, *, text_column=None, title_column=None, limit=None, batch_size=512):
    input_path, output_path = Path(input_path).expanduser().resolve(), Path(output_path).expanduser().resolve()
    summary_path = output_path.with_suffix(".summary.json")
    if (input_path == output_path or input_path == summary_path or output_path.exists() or summary_path.exists()
            or not input_path.is_file()):
        raise ValueError("Provide an existing input and fresh output CSV; source files are never overwritten")
    if output_path.suffix.lower() != ".csv":
        raise ValueError("Output must have a .csv extension")
    if not 1 <= batch_size <= 4096 or (limit is not None and limit < 1):
        raise ValueError("Provide batch size 1..4096 and an optional positive row limit")
    if input_path.suffix.lower()==".parquet":
        relation="read_parquet(?)"
    elif input_path.suffix.lower()==".csv":
        relation="read_csv_auto(?, all_varchar=true)"
    else:
        raise ValueError("Review input must be a CSV or Parquet")
    started=perf_counter()
    done=0
    counts=Counter()
    invalid=Counter()
    # Fail at startup if the selected final model is unavailable, rather than
    # silently writing empty/partial or fallback-classifier predictions.
    model=ml_service.load_model()
    version=getattr(model,"model_version","Character TF-IDF + Logistic Regression")
    def quote(name):
        return '"' + name.replace('"','""') + '"'
    with duckdb.connect(config={"threads":2,"memory_limit":"1GB"}) as con:
        columns=[r[0] for r in con.execute("DESCRIBE SELECT * FROM "+relation,[str(input_path)]).fetchall()]
        body=select_field(columns,("body_raw","body","body_clean","text_clean","text_title_body","text","review_text","model_input"),text_column,required=True)
        # A precombined text field already contains its title.
        title=(None if title_column is None and body in ("text_title_body","model_input")
               else select_field(columns,("title_raw","title","title_clean"),title_column))
        cid=select_field(columns,("comment_id","id","review_id"))
        pid=select_field(columns,("product_id",))
        expr=lambda col: "CAST("+quote(col)+" AS VARCHAR)" if col else "NULL"
        sql="SELECT "+",".join(expr(col) for col in (cid,pid,title,body))+" FROM "+relation
        parameters=[str(input_path)]
        if limit is not None:
            sql+=" LIMIT ?";parameters.append(limit)
        cursor=con.execute(sql,parameters)
        output_path.parent.mkdir(parents=True,exist_ok=True)
        partial=output_path.with_name(output_path.name+".partial")
        # Exclusive create; do not overwrite a partial from a previous run.
        with partial.open("x",encoding="utf-8-sig",newline="") as f:
            writer=csv.writer(f);writer.writerow(DEFAULT_FIELDS)
            while True:
                rows=cursor.fetchmany(batch_size)
                if not rows:break
                predicted=ml_service.predict_recommendation_batch([r[3] for r in rows],[r[2] for r in rows])
                for row,p in zip(rows,predicted):
                    done+=1
                    valid=p["valid"]
                    probs=p["probabilities"] if valid else {}
                    if valid:counts[p["label"]]+=1
                    else:invalid[p["reason"]]+=1
                    writer.writerow((done,row[0] or "",row[1] or "",p["label"] if valid else "",
                        p["confidence"] if valid else "",probs.get("recommended",""),
                        probs.get("not_recommended",""),probs.get("no_idea",""),
                        valid,p.get("reason") or "",p.get("model_version",version)))
                if done//10000 != (done-len(rows))//10000:
                    print(f"Processed {done:,} reviews",flush=True)
        # The final destination is still expected to be fresh.
        if output_path.exists():
            raise FileExistsError(output_path)
        partial.rename(output_path)
    summary={"input":str(input_path),"output":str(output_path),"model_version":version,
             "text_column":body,"title_column":title,"rows":done,"valid":sum(counts.values()),
             "invalid":sum(invalid.values()),"predicted_class_counts":dict(counts),
             "invalid_reasons":dict(invalid),"elapsed_seconds":round(perf_counter()-started,3),
             "note":"Inference on input reviews only. No labels used as model features; no evaluation or retraining."}
    with summary_path.open("x",encoding="utf8") as f:
        json.dump(summary,f,ensure_ascii=False,indent=2);f.write("\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=DEFAULT_INPUT)
    p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--text-column")
    p.add_argument("--title-column")
    p.add_argument("--limit",type=int)
    p.add_argument("--batch-size",type=int,default=512)
    a=p.parse_args()
    print(json.dumps(infer(a.input,a.output,text_column=a.text_column,title_column=a.title_column,
                           limit=a.limit,batch_size=a.batch_size),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
