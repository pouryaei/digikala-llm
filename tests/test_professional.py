"""Tests use actual local data, LangGraph, Streamlit; every LLM is a fixture."""
import json, os
from unittest.mock import patch
import pytest
from bonus.professional import ProfessionalWorkflow,validate_synthesis,request_fingerprint
from bonus.smart_policy import RequestBudget,DEFAULT_MODELS
from bonus.observability import ResultCache,TraceStore,ROOT
from bonus.evaluate_professional import ControlledTransport,cases
from bonus.qa_demo import DEMO_CASES
from phase4_llm.llm_client import LLMUnavailable

class Factory:
    def __init__(self,invalid_qa=False,invalid_router=False,fail=False,invalid_supplement=False):
        self.calls=[]; self.invalid_qa=invalid_qa;self.invalid_router=invalid_router
        self.fail=fail;self.invalid_supplement=invalid_supplement
    def __call__(self,stage,model,limit):
        owner=self
        class Transport:
            def generate(self,system,user):
                owner.calls.append(stage)
                if owner.fail:raise LLMUnavailable("fixture failure",attempts=1)
                out=ControlledTransport(stage,model,limit).generate(system,user)
                data=json.loads(out["text"])
                if stage=="router" and owner.invalid_router:data={"intent":"unknown"}
                if stage=="qa" and owner.invalid_qa:data["observations"][0]["comment_ids"]=[-1]
                if stage in ("compare","manager") and owner.invalid_supplement:data["points"][0]["citations"][0]["source_id"]="fake"
                return {**out,"text":json.dumps(data,ensure_ascii=False),"prompt_tokens":50,"completion_tokens":10,"usage_known":True}
        return Transport()

@pytest.fixture(autouse=True)
def settings(monkeypatch):
    for k,v in {"QBC13_API_KEY":"","QBC13_PRO_MAX_ATTEMPTS":"2","QBC13_MODEL_TARIFFS_JSON":"{}","FOROUHAR_LANGFUSE_ENABLED":"0",
                **{"QBC13_"+r.upper()+"_MODEL":m for r,m in DEFAULT_MODELS.items()}}.items():monkeypatch.setenv(k,v)

def demo(i,allow=True):
    _,_,q,pid,_=DEMO_CASES[i]
    return {"intent":"qa","query":q,"product_ids":[pid] if pid else [],"allow_llm":allow}
def ambiguous():
    return {**demo(3),"intent":"auto","query":"نظرهای کاربران درباره کیفیت ساخت و راحتی این محصول را پیدا کن و جمع‌بندی کن."}
def runner(tmp_path,factory=None,engine="langgraph",cache=None):
    return ProfessionalWorkflow(engine=engine,export=False,store=TraceStore(tmp_path/"trace.sqlite"),
                                client_factory=factory,cache=cache)

@pytest.mark.parametrize("i,expected",[(0,0),(1,0),(2,0),(3,1),(4,0)])
def test_qa_real_graph(tmp_path,i,expected):
    f=Factory();w=runner(tmp_path,f);r=w.run(demo(i))
    assert r["trace"]["engine"]=="LangGraph" and r["trace"]["status"]!="error"
    assert r["trace"]["api_attempts"]==len(f.calls)==expected
    assert demo(i)["query"] not in json.dumps(w.store.recent(),ensure_ascii=False)
    if expected:assert r["result"]["answer_mode"]=="grounded" and r["trace"]["estimated_cost_usd"] is None

def test_budget_and_known_unknown_cost(monkeypatch):
    b=RequestBudget(authorized=True,available=True,models=DEFAULT_MODELS)
    assert b.choose("qa",value=False).mode=="local"
    assert b.choose("qa",value=True,evidence=False).mode=="local"
    b.reserve("router",DEFAULT_MODELS["router"]);b.reserve("qa",DEFAULT_MODELS["qa"])
    assert b.totals()["api_attempts"]==2 and b.totals()["estimated_cost_usd"] is None
    with pytest.raises(LLMUnavailable):b.reserve("qa",DEFAULT_MODELS["qa"])
    monkeypatch.setenv("QBC13_MODEL_TARIFFS_JSON",json.dumps({DEFAULT_MODELS["qa"]:{"input":1.0,"output":2.0}}))
    b=RequestBudget(authorized=True,available=True,models=DEFAULT_MODELS)
    b.reserve("qa",DEFAULT_MODELS["qa"]).update(status="success",usage_known=True,prompt_tokens=100,completion_tokens=50)
    assert b.totals()["estimated_cost_usd"]==pytest.approx(.0002)

def test_consent_and_cache_upgrade(tmp_path):
    f=Factory();w=runner(tmp_path,f)
    a=w.run(demo(3,False));assert not a["trace"]["api_attempts"] and a["result"]["answer_mode"]=="extractive"
    b=w.run(demo(3));assert b["trace"]["api_attempts"]==1
    c=w.run(demo(3,False))
    assert c["trace"]["cached"] and c["trace"]["api_attempts"]==0 and c["trace"]["estimated_cost_usd"]==0
    assert c["result"]["answer"]==b["result"]["answer"] and len(f.calls)==1

def test_shared_cap_including_repairs(tmp_path):
    f=Factory(invalid_qa=True);w=runner(tmp_path,f)
    r=w.run(ambiguous());assert f.calls==["router","qa"]
    assert r["trace"]["api_attempts"]==2 and r["result"]["answer_mode"]=="extractive"
    r2=w.run(ambiguous())
    assert not r2["trace"]["cached"] and r2["trace"]["api_attempts"]==2 and f.calls==["router","qa","qa","qa"]
    assert any(d["reason"]=="valid_route_cache" for d in r2["trace"]["decisions"])

def test_bad_route_fallback(tmp_path):
    f=Factory(invalid_router=True);r=runner(tmp_path,f).run(ambiguous())
    assert r["trace"]["route"]=="qa" and r["trace"]["api_attempts"]<=2
    assert any(d["reason"]=="router_failed_local_fallback" for d in r["trace"]["decisions"])

def test_full_cache_precedes_router(tmp_path):
    f=Factory();w=runner(tmp_path,f)
    assert w.run(ambiguous())["trace"]["api_attempts"]==2
    b=w.run(ambiguous());assert b["trace"]["cached"] and b["trace"]["api_attempts"]==0 and len(f.calls)==2

def test_model_config_and_expiry_invalidate(tmp_path,monkeypatch):
    f=Factory();w=runner(tmp_path,f);w.run(demo(3))
    assert w.run(demo(3))["trace"]["cached"]
    monkeypatch.setenv("QBC13_QA_MODEL","openai/gpt-4.1-mini")
    assert not w.run(demo(3))["trace"]["cached"]
    w=runner(tmp_path,f,cache=ResultCache(ttl=0));w.run(demo(3));assert not w.run(demo(3))["trace"]["cached"]

def test_missing_key_and_strict_consent(tmp_path):
    assert runner(tmp_path).run(demo(3))["trace"]["api_attempts"]==0
    f=Factory();r=runner(tmp_path,f).run({**demo(3),"allow_llm":"false"});assert not r["trace"]["api_attempts"] and not f.calls

def test_provider_failure_counts_unknown(tmp_path):
    r=runner(tmp_path,Factory(fail=True)).run(demo(3))
    assert r["trace"]["api_attempts"]==1 and r["trace"]["api_calls"]==0
    assert r["trace"]["estimated_cost_usd"] is None and r["result"]["answer_mode"]=="extractive"

def test_compare_manager_preserve_facts(tmp_path):
    f=Factory();w=runner(tmp_path,f)
    for _,_,req,_ in cases()[-3:-1]:
        local=w.run({**req,"allow_llm":False})["result"];r=w.run({**req,"allow_llm":True})
        key="products" if req["intent"]=="compare" else "rows"
        assert r["trace"]["api_attempts"]==1 and local[key]==r["result"][key]
        assert r["result"]["professional_synthesis"]["points"]
        assert w.run({**req,"allow_llm":True})["trace"]["cached"]
    assert f.calls==["compare","manager"]

def test_simple_search_and_prediction(tmp_path):
    f=Factory();w=runner(tmp_path,f)
    a=w.run({"intent":"search","query":"هدفون","allow_llm":True})
    assert a["result"]["items"] and not a["trace"]["api_attempts"]
    assert w.run({"intent":"search","query":"هدفون","allow_llm":True})["trace"]["cached"]
    b=w.run({"intent":"predict","query":"پیش بینی این نظر: کیفیت عالی بود و پیشنهاد می کنم","allow_llm":True})
    assert b["result"]["valid"] and "Head Fine-tuning" in b["result"]["model_version"] and not f.calls

def test_reject_invalid_sources_and_numbers():
    src=[{"source_id":"F1","text":"تعداد نظر: 12","product_id":"1"}]
    bad={"points":[{"text":"تعداد 99 نظر","citations":[{"source_id":"F1","quote":"تعداد نظر: 12"}]}]}
    with pytest.raises(ValueError):validate_synthesis(bad,src)
    bad["points"][0]["text"]="تعداد نظر"
    with pytest.raises(ValueError):validate_synthesis(bad,src,["2"])
    bad["points"][0]["citations"][0]["quote"]="ساختگی"
    with pytest.raises(ValueError):validate_synthesis(bad,src)

def test_python_langgraph_parity(tmp_path):
    a=runner(tmp_path,engine="python").run(demo(0,False))
    b=runner(tmp_path,engine="langgraph").run(demo(0,False))
    assert a["result"]["answer"]==b["result"]["answer"] and a["result"]["evidence"]==b["result"]["evidence"]

def test_real_streamlit_autonomous_policy_and_cache():
    from streamlit.testing.v1 import AppTest
    with patch("requests.request",side_effect=AssertionError("No API")):
        app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=60).run()
        app.button(key="open_professional").click().run()
        assert not app.exception
        app.selectbox(key="bonus_router_example").select("آمار مستقیم از داده").run()
        next(b for b in app.button if b.label=="اجرای گراف").click().run()
        assert not app.exception
        assert app.session_state["bonus_router_result"]["trace"]["professional"]
        assert not app.session_state["bonus_router_result"]["trace"]["api_attempts"]
        next(b for b in app.button if b.label=="اجرای گراف").click().run()
        assert app.session_state["bonus_router_result"]["trace"]["cached"]
