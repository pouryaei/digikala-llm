"""Calculate API charges from measured provider usage and user-supplied prices.

Reads evaluator api_usage.csv, a JSON response/meta object, a list of such
objects, or JSONL. Does not make network requests or guess model prices.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

def load_records(path):
    if path.suffix.lower()==".csv":
        with path.open(encoding="utf-8-sig",newline="") as f:
            return list(csv.DictReader(f))
    text=path.read_text(encoding="utf-8-sig")
    if path.suffix.lower()==".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    raw=json.loads(text)
    if isinstance(raw,list): return raw
    if isinstance(raw,dict): return [raw]
    raise ValueError("Expected a list or JSON object of measured usage records")

def truth(value):
    return value is True or isinstance(value,str) and value.lower() in ("true","1","yes")

def calculate(records,input_rate=None,output_rate=None):
    if (input_rate is None)!=(output_rate is None) or any(x is not None and (x<0 or not __import__("math").isfinite(x)) for x in (input_rate,output_rate)):
        raise ValueError("Supply both nonnegative prices per 1M tokens or neither")
    inp=out=calls=attempts=unknown=0
    for r in records:
        m=r.get("meta",r)
        n=int(m.get("api_calls",m.get("calls",0)) or 0)
        a=int(m.get("api_attempts",n) or 0)
        i=int(m.get("prompt_tokens",0) or 0);o=int(m.get("completion_tokens",0) or 0)
        if min(n,a,i,o)<0 or a<n:raise ValueError("Invalid measured usage")
        complete=(n==0 and a==0 and i==0 and o==0) or (truth(m.get("usage_known",False)) and "prompt_tokens" in m and "completion_tokens" in m and a==n)
        unknown+=not complete
        calls+=n;attempts+=a;inp+=i;out+=o
    estimated=(inp*input_rate+out*output_rate)/1e6 if input_rate is not None else 0.0 if calls==attempts==inp==out==0 else None
    return {"records":len(records),"calls":calls,"attempts":attempts,"prompt_tokens_reported":inp,"completion_tokens_reported":out,
        "records_with_unknown_or_retry_usage":unknown,"input_usd_per_million":input_rate,"output_usd_per_million":output_rate,
        "cost_from_reported_tokens_usd":estimated,"complete_cost_estimate_usd":estimated if not unknown else None,
        "note":"Not a provider invoice. Rates must be from your actual gateway. Failed/retried requests with unknown usage cannot be priced as zero; cost from reported tokens may be incomplete. Cached answers consume zero current API tokens if reported as such."}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("usage_file",type=Path)
    p.add_argument("--input-usd-per-million",type=float)
    p.add_argument("--output-usd-per-million",type=float)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    result=calculate(load_records(a.usage_file),a.input_usd_per_million,a.output_usd_per_million)
    text=json.dumps(result,ensure_ascii=False,indent=2)+"\n"
    if a.output:a.output.write_text(text,encoding="utf-8")
    print(text,end="")
