"""Real offline workflow, router and cache experiments."""
from __future__ import annotations

import json
from pathlib import Path
import statistics
from tempfile import TemporaryDirectory
from time import perf_counter
from collections import Counter
from unittest.mock import patch
import os

from bonus.graph import Workflow, classify_intent, validate_request, generation_policy
from bonus.observability import ResultCache,TraceStore,digest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/bonus"

# These are intent-labelled requests, not examples of LLM output quality.
ROUTER_CASES=[
    ("کفش پیاده روی مردانه",[],"","search"),
    ("یک کیف ارزان میخوام که خریدارها راضی باشن",[],"","search"),
    ("ساعت ضد آب تا ۵ میلیون",[],"","search"),
    ("معرفی وسیله مناسب سفر",[],"","search"),
    ("هدفون بی سیم بلوتوثی",[],"","search"),
    ("ایرادهای پرتکرار این محصول چیست؟",[1696173],"","qa"),
    ("ارزش خرید این کالا چطور است؟",[1696173],"","qa"),
    ("نقاط قوت و ضعف از دید کاربران",[],"","qa"),
    ("کیفیت این محصول را بگو",[2896502],"","qa"),
    ("درباره کیفیت کفش چه گفته‌اند؟",[],"","qa"),
    ("این دو محصول را مقایسه کن",[1696173,2896502],"","compare"),
    ("کدام بهتر است و چرا؟",[1696173,2896502],"","compare"),
    ("مقایسه قیمت و نظر کاربران",[],"","compare"),
    ("پرتکرارترین شکایت کاربران در این دسته چیست؟",[],"کفش مردانه","manager"),
    ("بازخورد برندها را در دسته کفش مردانه مقایسه کن",[],"کفش مردانه","manager"),
    ("کدام محصولات پرنظر نرخ پیشنهاد خرید پایین دارند؟",[],"کفش مردانه","manager"),
    ("پیش بینی وضعیت پیشنهاد خرید این نظر را انجام بده",[],"","predict"),
    ("طبقه بندی متن نظر",[],"","predict"),
    ("برچسب این نظر چیست؟",[],"","predict"),
    ("اختلاف امتیاز ثبت شده و نرخ پیشنهاد خرید",[],"","audit"),
    ("محصولات دارای تعارض امتیاز و رضایت",[],"","audit"),
    ("شکاف رضایت را پیدا کن",[],"","audit"),
    ("سلام",[],"","clarify"),
    ("درود",[],"","clarify"),
]


def result_key(result):
    # Ignore only measured timing/usage metadata when comparing content.
    stable={k:v for k,v in result.items() if k not in ("meta","retrieval_seconds","seconds")}
    return digest(stable)


def med(values):return statistics.median(values)


def evaluate():
    OUT.mkdir(parents=True,exist_ok=True)
    routes=[]
    for q,ids,category,expected in ROUTER_CASES:
        req=validate_request({"query":q,"product_ids":ids,"category":category or None})
        actual,reason=classify_intent(req)
        routes.append({"query":q,"expected":expected,"actual":actual,"reason":reason,"correct":actual==expected})
    scenarios=[
        {"query":"هدفون","intent":"search","retrieval_mode":"auto"},
        {"query":"کوله پشتی بچه برای مهد کودک","intent":"search","retrieval_mode":"hybrid_char"},
        {"query":"ایرادهای پرتکرار این محصول چیست؟","intent":"qa","product_ids":[1696173]},
        {"query":"ارزش خرید این محصول را جمع‌بندی کن","intent":"qa","product_ids":[2896502]},
    ]
    caches=[];demos=[]
    with TemporaryDirectory() as tmp:
        trace_store=TraceStore(Path(tmp)/"traces.sqlite3")
        cache=ResultCache()
        runner=Workflow(cache=cache,store=trace_store,engine="auto",export=False)
        # Warm method/model initialization; compare result-cache only.
        for s in scenarios:runner.run({**s,"use_cache":False},record=False)
        for s in scenarios:
            cold=[];warm=[];contents=[]
            cache.clear()
            for _ in range(5):
                a=perf_counter();v=runner.run({**s,"use_cache":False},record=True);cold.append((perf_counter()-a)*1000)
                contents.append(result_key(v["result"]))
            priming=runner.run({**s,"use_cache":True},record=True)
            for _ in range(5):
                a=perf_counter();v=runner.run({**s,"use_cache":True},record=True);warm.append((perf_counter()-a)*1000)
                contents.append(result_key(v["result"]))
                if not v["trace"]["cached"]:raise AssertionError("Expected cache hit")
            same=len(set(contents+[result_key(priming["result"])]))==1
            if not same:raise AssertionError("Cached result changed")
            caches.append({"query":s["query"],"intent":s["intent"],"uncached_ms":cold,"cached_ms":warm,
                           "uncached_median_ms":med(cold),"cached_median_ms":med(warm),"speedup":med(cold)/max(1e-8,med(warm)),
                           "warm_hits":5,"outputs_identical":same})
        for req in [*scenarios,
                    {"query":"این دو محصول را مقایسه کن","product_ids":[1696173,2896502]},
                    {"query":"کیفیت عالی است و خرید را پیشنهاد می‌کنم","intent":"predict"},
                    {"query":"اختلاف امتیاز و پیشنهاد خرید محصولات"},
                    {"query":"سلام"}]:
            v=runner.run({**req,"use_cache":False})
            demos.append({"query":req["query"],"route":v["trace"]["route"],"status":v["trace"]["status"],
                          "trace":v["trace"],"api_calls":v["trace"]["api_calls"]})
        traces=trace_store.recent(1000)
        if len(traces)<len(caches)*11:raise AssertionError("Trace persistence failed")
    # Policy-only exposure analysis: no API key and no actual calls.
    prior=json.loads((ROOT/"evaluation/results/final/evaluation.json").read_text(encoding="utf8"))["qa"]
    decisions=Counter(generation_policy({"query":r["question"],"allow_llm":True},r["result"],model_available=True)[1] for r in prior)
    report={"router":{"cases":routes,"correct":sum(r["correct"] for r in routes),"total":len(routes),
                      "note":"Intent labels authored manually for a bounded routing test, not semantic answer annotations."},
            "cache":{"cases":caches,"total_warm_requests":sum(x["warm_hits"] for x in caches),
                     "uncached_global_median_ms":med([v for x in caches for v in x["uncached_ms"]]),
                     "cached_global_median_ms":med([v for x in caches for v in x["cached_ms"]]),
                     "outputs_identical":True,
                     "measurement":"5 repeated calls per scenario after process/index warmup, measured end-to-end including local trace recording; warm request cache primed once and not priced as API savings."},
            "workflow_runs":demos,"recorded_trace_count":len(traces),
            "executed_engines":sorted({r["engine"] for r in traces}),
            "actual_api_calls":sum(r["api_calls"] for r in traces),
            "llm_policy_simulation":{"case_count":len(prior),"decisions":dict(decisions),
                                     "note":"Policy eligibility only (key availability simulated). No network calls or measured monetary saving."},
            "langfuse_live_tested":False}
    (OUT/"router_cache.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    return report


if __name__=="__main__":
    with patch.dict(os.environ,{"QBC13_API_KEY":"","QBC13_SOURCE_PRICE_UNIT":"","QBC13_UI_SEARCH_MODE":"bm25",
                                "QBC13_REVIEW_GENERATION_MODE":"extractive","FOROUHAR_LANGFUSE_ENABLED":"0"}):
        r=evaluate()
    print(json.dumps({"router":{k:v for k,v in r["router"].items() if k!="cases"},
        "cache":{k:v for k,v in r["cache"].items() if k!="cases"},"engines":r["executed_engines"],"api_calls":r["actual_api_calls"]},ensure_ascii=False,indent=2))
