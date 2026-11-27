"""Central, deliberately local-first decision policy and shared request budget."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json, math, os, re
from time import perf_counter
from phase4_llm.llm_client import LLMUnavailable

POLICY_VERSION="professional-v1"
DEFAULT_MODELS={"router":"google/gemini-2.5-flash-lite","qa":"openai/gpt-4o-mini",
                "compare":"openai/gpt-4.1-mini","manager":"openai/gpt-4.1-mini"}
OUTPUT_LIMITS={"router":150,"qa":350,"compare":350,"manager":350}
MAX_ATTEMPTS=2
MAX_INPUT_CHARS=10000
CHAT_MODELS=frozenset(("openai/gpt-4o-mini","openai/gpt-4.1-mini","openai/gpt-4.1",
"openai/gpt-5-mini","openai/gpt-5-nano","google/gemini-2.5-flash","google/gemini-2.5-flash-lite",
"deepseek/deepseek-chat-v3-0324","deepseek/deepseek-chat-v3.1","qwen/qwen-plus","qwen/qwen-2.5-72b-instruct",
"qwen/qwen3-coder","meta-llama/llama-3.3-70b-instruct","google/gemma-3-27b-it"))

@dataclass(frozen=True)
class Decision:
    stage: str
    mode: str
    reason: str
    model: str|None=None
    would_benefit: bool=False
    def record(self): return asdict(self)

def configured_models():
    result={role:os.getenv("QBC13_"+role.upper()+"_MODEL",default).strip() for role,default in DEFAULT_MODELS.items()}
    if any(value not in CHAT_MODELS for value in result.values()):raise ValueError("Unallowed chat model")
    return result

def max_attempts():
    try:return min(MAX_ATTEMPTS,max(0,int(os.getenv("QBC13_PRO_MAX_ATTEMPTS","2"))))
    except (ValueError,TypeError):return 0

def tariffs():
    try:
        raw=json.loads(os.getenv("QBC13_MODEL_TARIFFS_JSON","{}"))
        result={}
        for model,entry in raw.items():
            if model not in CHAT_MODELS or not isinstance(entry,dict):continue
            pair=(float(entry["input"]),float(entry["output"]))
            if all(math.isfinite(x) and x>=0 for x in pair):result[model]=pair
        return result
    except (TypeError,ValueError,KeyError,AttributeError):return {}

class RequestBudget:
    """One instance per request, shared by all stages including schema repair."""
    def __init__(self,*,authorized,available,models,limit=None,simulated=False):
        self.authorized=authorized is True;self.available=available is True
        self.models=dict(models);self.limit=max_attempts() if limit is None else min(MAX_ATTEMPTS,max(0,int(limit)))
        self.records=[];self.simulated=bool(simulated);self.rates=tariffs()
    @property
    def remaining(self):return max(0,self.limit-len(self.records))
    def choose(self,stage,*,value,evidence=True,local_reason="local_result_sufficient"):
        if not evidence:return Decision(stage,"local","insufficient_or_uncovered_evidence")
        if not value or stage not in self.models:return Decision(stage,"local",local_reason)
        if not self.authorized:return Decision(stage,"local","llm_not_authorized",would_benefit=True)
        if not self.available:return Decision(stage,"local","group_key_unavailable",would_benefit=True)
        if not self.remaining:return Decision(stage,"local","request_budget_exhausted",would_benefit=True)
        return Decision(stage,"llm","additional_language_reasoning_needed",self.models[stage],True)
    def reserve(self,stage,model):
        if not(self.authorized and self.available and self.remaining and self.models.get(stage)==model):
            raise LLMUnavailable("Shared request budget unavailable",attempts=0)
        row={"stage":stage,"model":model,"status":"attempted","prompt_tokens":0,"completion_tokens":0,"usage_known":False,"latency_ms":0.0}
        self.records.append(row)
        return row
    def totals(self):
        rows=[dict(r) for r in self.records];known=all(r["usage_known"] and r["status"]=="success" for r in rows)
        total=0.0
        if rows:
            if self.simulated or not known or any(r["model"] not in self.rates for r in rows):total=None
            else:total=sum((r["prompt_tokens"]*self.rates[r["model"]][0]+r["completion_tokens"]*self.rates[r["model"]][1])/1e6 for r in rows)
        return {"api_attempts":len(rows),"api_calls":sum(r["status"]=="success" for r in rows),
                "prompt_tokens":sum(r["prompt_tokens"] for r in rows),"completion_tokens":sum(r["completion_tokens"] for r in rows),
                "usage_known":known,"estimated_cost_usd":total,"model_usage":rows,
                "attempt_limit":self.limit,"attempts_remaining":self.remaining,"simulated":self.simulated}

class BoundedClient:
    """Adapter: existing LLMClient configured with a single attempt per generate."""
    def __init__(self,stage,budget,transport,*,bridge=None,trace_context=None):
        self.stage,self.budget,self.transport=stage,budget,transport
        self.model=budget.models[stage];self.bridge,self.trace_context=bridge,trace_context
        self.cache_config={"policy":POLICY_VERSION,"max_tokens":OUTPUT_LIMITS[stage]}
    def generate(self,system,user):
        if len(system)+len(user)>MAX_INPUT_CHARS:raise LLMUnavailable("Prompt too large",attempts=0)
        row=self.budget.reserve(self.stage,self.model);start=perf_counter()
        def call(obs=None):
            try:
                out=self.transport.generate(system,user)
                if out.get("api_attempts",1)!=1:raise LLMUnavailable("Invalid transport count",attempts=1)
                row.update(status="success",prompt_tokens=max(0,int(out.get("prompt_tokens",0))),
                           completion_tokens=max(0,int(out.get("completion_tokens",0))),usage_known=bool(out.get("usage_known",False)))
                return out
            except Exception:
                row.update(status="failed",usage_known=False)
                raise LLMUnavailable("Bounded gateway attempt failed",attempts=1) from None
            finally:
                row["latency_ms"]=round((perf_counter()-start)*1000,3)
                if self.bridge:
                    usage={"input":row["prompt_tokens"],"output":row["completion_tokens"]} if row["usage_known"] else None
                    self.bridge.update(obs,{k:row[k] for k in ("stage","status","latency_ms","usage_known")},usage=usage,model=self.model)
        if self.bridge:
            with self.bridge.span("professional_llm_"+self.stage,safe_input={"stage":self.stage},
                    as_type="generation",trace_context=self.trace_context) as obs:return call(obs)
        return call()

_PATTERNS={
"search":r"پیدا کن|معرفی کن|جستجو|جست وجو|دنبال|بودجه|تا [0-9۰-۹]",
"qa":r"نظرهای|نظرات|خریدارها|خریداران|کاربران از|معایب|مزایا|نقاط قوت|نقاط ضعف|این محصول|دوام",
"compare":r"مقایسه|کدام بهتر|کدوم بهتر|در مقابل|بین .* و ",
"manager":r"مدیر|در دسته|در این دسته|دسته بندی|برندها|شکایت|نارضایتی|فروشگاه"}
_SYNTHESIS=re.compile(r"جمع بندی|در مجموع|چرا|ارزش خرید|تحلیل|تصمیم|کدوم|کدام|مناسب تر|اولویت|تفسیر|اقدام|راهکار|پیشنهاد مدیریتی|بهبود|تفاوت|چه کار|چه اقدام|دلیل|علت")
_NUMERIC=re.compile(r"چند نفر|چند درصد|چه درصد|تعداد|نرخ|کمترین قیمت|ارزان ترین|گران ترین|قیمت .* چقدر")

def route_needs_llm(request,route,reason,text):
    if reason in {"explicit_user_choice","explicit_prediction_intent","rating_recommendation_disagreement","insufficient_intent"}:return False,reason
    signals={role for role,pattern in _PATTERNS.items() if re.search(pattern,text)}
    if len(request.get("product_ids",[]))>=2 and signals<={"compare","qa"}:return False,"selected_products_make_route_clear"
    if request.get("category") and route=="manager" and signals<={"manager","qa"}:return False,"selected_category_makes_route_clear"
    if _NUMERIC.search(text) and not _SYNTHESIS.search(text):return False,"direct_numeric_request"
    if len(signals)>=2:return True,"conflicting_intent_signals"
    if reason=="shopping_discovery_default" and not signals and len(text.split())>=7:return True,"long_query_without_clear_intent"
    return False,"local_intent_sufficient"

def synthesis_value(stage,text):
    if _NUMERIC.search(text) and not _SYNTHESIS.search(text):return False
    return stage in ("compare","manager") and bool(_SYNTHESIS.search(text))
