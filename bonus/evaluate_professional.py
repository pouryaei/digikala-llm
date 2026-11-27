"""Offline fixture responses, real data and real graph: measures policy, not model quality."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from bonus.professional import ProfessionalWorkflow
from bonus.qa_demo import DEMO_CASES
ROOT=Path(__file__).resolve().parents[1]

class ControlledTransport:
    def __init__(self,stage,model,limit):self.stage=stage
    def generate(self,system,user):
        payload=json.JSONDecoder().raw_decode(user)[0]
        if self.stage=="router":answer={"intent":"qa"}
        elif self.stage=="qa":
            witness=payload["retrieved_reviews"][0]
            answer={"observations":[{"point":"نمونه کنترل ساختار، نه کیفیت مدل واقعی","quote":witness["text"],"comment_ids":[witness["comment_id"]]}]}
        else:
            refs=[];owners=set()
            for src in payload["sources"]:
                pid=src.get("product_id")
                if self.stage=="compare" and (not pid or pid in owners):continue
                refs.append({"source_id":src["source_id"],"quote":src["text"][:120]});owners.add(pid)
                if self.stage=="manager" or len(refs)==2:break
            answer={"points":[{"text":"نمونه کنترل ساختار ارجاع، نه کیفیت مدل واقعی","citations":refs}]}
        return {"text":json.dumps(answer,ensure_ascii=False),"api_attempts":1,
                "prompt_tokens":0,"completion_tokens":0,"usage_known":False,"latency":0.0}

def cases():
    rows=[("search","جستجوی مستقیم",{"intent":"search","query":"هدفون"},0)]
    for i,(_,label,q,pid,_) in enumerate(DEMO_CASES):
        rows.append((f"qa_{i}",label,{"intent":"qa","query":q,"product_ids":[pid] if pid else []},1 if i==3 else 0))
    return rows+[
        ("ambiguity","ابهام مسیر و جمع‌بندی",{"query":"نظرهای کاربران درباره کیفیت ساخت و راحتی این محصول را پیدا کن و جمع‌بندی کن.","product_ids":[1696173]},2),
        ("compare","مقایسه چندمعیاره",{"intent":"compare","query":"کیفیت ساخت و راحتی دو محصول را مقایسه کن و جمع بندی کن.","product_ids":[1696173,2896502]},1),
        ("manager","تحلیل مدیریتی",{"intent":"manager","query":"پرتکرارترین شکایت در این دسته چیست و چه اقداماتی برای بهبود پیشنهاد می‌کنی؟","category":"کفش مردانه"},1),
        ("predict","طبقه‌بندی نظر",{"intent":"predict","query":"پیش بینی این نظر: کیفیت عالی بود و پیشنهاد می کنم"},0)]

def run():
    from unittest.mock import patch
    from bonus.smart_policy import DEFAULT_MODELS
    settings={"QBC13_PRO_MAX_ATTEMPTS":"2","FOROUHAR_LANGFUSE_ENABLED":"0",
              **{"QBC13_"+k.upper()+"_MODEL":v for k,v in DEFAULT_MODELS.items()}}
    rows=[]
    with patch.dict(os.environ,settings):
        flow=ProfessionalWorkflow(engine="langgraph",export=False,client_factory=ControlledTransport)
        for cid,label,req,expected in cases():
            local=flow.run({**req,"allow_llm":False,"use_cache":False},record=False)
            first=flow.run({**req,"allow_llm":True,"use_cache":True},record=False)
            repeat=flow.run({**req,"allow_llm":True,"use_cache":True},record=False)
            t,r=first["trace"],repeat["trace"]; a,b=local["result"],first["result"]
            if t["api_attempts"]!=expected or t["status"]=="error" or any(a.get(k)!=b.get(k) for k in ("items","products","rows","evidence","probabilities")):
                raise ValueError("Policy/facts mismatch: "+cid)
            rows.append({"case_id":cid,"scenario":label,"route":t["route"],"status":t["status"],
                "local_attempts":local["trace"]["api_attempts"],"first_attempts":t["api_attempts"],
                "repeat_attempts":r["api_attempts"],"warm_cache_hit":r["cached"],"local_facts_preserved":True,
                "first_roles":[m["stage"] for m in t["model_usage"]],
                "decisions":[{"stage":d["stage"],"mode":d["mode"],"reason":d["reason"]} for d in t["decisions"]]})
    return {"mode":"offline_fixture","real_gateway_attempts":0,"cases":rows,"total_cases":len(rows),
            "first_simulated_attempts":sum(r["first_attempts"] for r in rows),
            "repeat_simulated_attempts":sum(r["repeat_attempts"] for r in rows),
            "zero_attempt_cases":sum(r["first_attempts"]==0 for r in rows),
            "warm_cache_hits":sum(r["warm_cache_hit"] for r in rows),
            "facts_preserved_cases":sum(r["local_facts_preserved"] for r in rows),
            "max_attempts_per_request":max(r["first_attempts"] for r in rows),
            "note":"Fixtures test decisions, cache and structural validators; actual language quality, usage and costs are not measured."}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=ROOT/"reports/bonus/professional_policy.json")
    a=p.parse_args();r=run();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({k:v for k,v in r.items() if k!="cases"},ensure_ascii=False,indent=2))
