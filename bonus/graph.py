"""Bounded, inspectable workflow over the existing Forouhar services.

Uses a real LangGraph StateGraph when installed; otherwise executes exactly the
same node/edge policy locally and labels that engine honestly.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import cached_property
import os
import re
from time import perf_counter
from typing import TypedDict
from uuid import uuid4
from sqlite3 import Error as SQLiteError

from bonus.observability import CACHE, LangfuseBridge, TraceStore, cache_key, timestamp
from src.bm25_index import tokenize_query
from src.config import Config
from src.rag_query import explicit_product_ids
from src.task2_review_store import identifier

ROUTES=("search","qa","compare","manager","predict","audit","clarify")
LABELS={"search":"جست‌وجو","qa":"پرسش از نظرات","compare":"مقایسه","manager":"تحلیل مدیر",
        "predict":"پیش‌بینی","audit":"تعارض امتیاز و رضایت","clarify":"نیاز به توضیح"}
QA_EDGES = {
    "qa_resolve": ("qa_retrieve", "finalize"),
    "qa_retrieve": ("qa_policy",),
    "qa_policy": ("qa_extractive", "qa_insufficient", "qa_generate", "finalize"),
    "qa_generate": ("qa_validate",),
    "qa_validate": ("qa_fallback", "finalize"),
    "qa_extractive": ("finalize",),
    "qa_insufficient": ("finalize",),
    "qa_fallback": ("finalize",),
}
EDGE_PAIRS = [("START", "route_intent"), ("route_intent", "cache_lookup"), ("cache_lookup", "finalize")]
EDGE_PAIRS += [("cache_lookup", r if r != "qa" else "qa_resolve") for r in ROUTES]
EDGE_PAIRS += [(source, dest) for source, choices in QA_EDGES.items() for dest in choices]
EDGE_PAIRS += [(r, "finalize") for r in ROUTES if r != "qa"] + [("finalize", "END")]


class GraphState(TypedDict,total=False):
    request:dict
    route:str
    route_reason:str
    cache_key:str
    cached:bool
    result:dict
    llm_policy:str
    events:list[dict]
    trace_context:dict
    resolution:tuple
    selection:object
    qa_started:float
    retrieval_seconds:float
    qa_destination:str
    model_response:dict
    model_outcome:str
    evidence_count:int
    uncovered_facets:int


def norm(text):
    return " ".join(tokenize_query(text))


def classify_intent(request):
    explicit=request.get("intent","auto")
    if explicit!="auto":
        if explicit not in ROUTES:raise ValueError("مسیر درخواست ناشناخته است.")
        return explicit,"explicit_user_choice"
    q=norm(request["query"])
    ids=request.get("product_ids",[])
    if re.search(r"تعارض|تناقض|اختلاف امتیاز|شکاف رضایت",q):
        return "audit","rating_recommendation_disagreement"
    if re.search(r"پیش بینی|طبقه بندی|کلاس برچسب|برچسب این نظر",q):
        return "predict","explicit_prediction_intent"
    if len(ids)>=2:
        return "compare","multi_product_comparison"
    if re.search(r"مدیر|در این دسته|در دسته|برندها|برند های|پرتکرارترین شکایت|محصولات پرنظر|محصولات با نظر",q) or request.get("category"):
        return "manager","population_level_analysis"
    if re.search(r"مقایسه|کدام بهتر|کدوم بهتر|در مقابل",q):
        return "compare","multi_product_comparison"
    if ids or re.search(r"ایرادهای|ایراد ها|معایب|نقاط قوت|نقاط ضعف|ارزش خرید|این محصول|این کالا|درباره کیفیت|درباره جنس|از چه چیزی راضی|کیفیت این",q):
        return "qa","product_review_question"
    if not q or q in {"سلام","درود","سلام خوبی","چه خبر"}:
        return "clarify","insufficient_intent"
    return "search","shopping_discovery_default"


SYNTHESIS = re.compile(r"جمع بندی|در مجموع|چرا|ارزش خرید|تحلیل|تصمیم بگیرم|بخرم|می ارزه")
NUMERIC = re.compile(r"چند (?:نفر|درصد|نظر)|درصد|نرخ|تعداد")
VALUE_FRAMING = re.compile(r"(?<!\w)(?:ارزش خرید|بخرم|می ارزه|نمی ارزه)(?!\w)")
EXPLICIT_PRICE = re.compile(r"(?<!\w)(?:قیمت|هزینه|تخفیف|گران|گرون|ارزان|صرفه)(?!\w)")


def coverage(request, result):
    """Count uncovered named facets in the actual evidence sent to the model."""
    requested = set(result.get("requested_themes", []))
    text = norm(request["query"])
    if VALUE_FRAMING.search(text) and not EXPLICIT_PRICE.search(text):
        requested.discard("قیمت و ارزش خرید")  # decision framing is not a price fact
    witnesses = {e["comment_id"] for e in result.get("evidence", [])}
    covered = {f["theme"] for f in result.get("facet_coverage", [])
               if witnesses.intersection(f.get("evidence_ids", []))}
    return len(requested), len(requested - covered)


def generation_policy(request, result, *, strategy="selective", model_available=None):
    if result.get("status") != "answered" or not result.get("evidence"):
        return False, "missing_or_insufficient_evidence"
    if not request.get("allow_llm", False):
        return False, "offline_by_default"
    if strategy == "selective":
        text = norm(request["query"])
        facets, missing = coverage(request, result)
        if missing:
            return False, "requested_facets_uncovered"
        if len({e["comment_id"] for e in result["evidence"]}) < 2:
            return False, "too_few_distinct_witnesses"
        if (NUMERIC.search(text) and not SYNTHESIS.search(text)) or (facets < 2 and not SYNTHESIS.search(text)):
            return False, "simple_question_local_answer_sufficient"
    if model_available is None:
        from phase4_llm.llm_client import is_configured_key
        model_available = is_configured_key(os.getenv("QBC13_API_KEY"))
    if not model_available:
        return False, "group_key_unavailable"
    return True, "explicit_opt_in_complex_grounded_synthesis" if strategy == "selective" else "always_generate_baseline"


def validate_request(request):
    r=dict(request)
    q=r.get("query")
    if not isinstance(q,str) or len(q)>1000:
        raise ValueError("متن درخواست باید حداکثر ۱۰۰۰ کاراکتر باشد.")
    if not q.strip():
        raise ValueError("درخواست را بنویسید.")
    ids=list(dict.fromkeys([identifier(x) for x in r.get("product_ids",[])]+explicit_product_ids(q)))
    if len(ids)>5:raise ValueError("حداکثر پنج شناسه محصول")
    r.update(query=q.strip(),product_ids=ids,intent=r.get("intent","auto"),
             retrieval_mode=r.get("retrieval_mode","auto"),filters=dict(r.get("filters") or {}),
             use_cache=bool(r.get("use_cache",True)),allow_llm=bool(r.get("allow_llm",False)))
    if r["retrieval_mode"] not in ("auto","bm25","hybrid_char"):
        raise ValueError("روش بازیابی ناشناخته")
    return r


def cache_destination(state):
    if state.get("cached"):return "finalize"
    return "qa_resolve" if state["route"]=="qa" else state["route"]


class Workflow:
    def __init__(self, *, cache=CACHE, store=None, engine="auto", export=True, generation_strategy="selective", llm_factory=None):
        if generation_strategy not in ("selective", "always"):
            raise ValueError("Unknown generation strategy.")
        self.cache = cache
        self.store = TraceStore() if store is None else store
        self.engine_choice = engine
        self.bridge = LangfuseBridge() if export else None
        self.generation_strategy = generation_strategy
        self.llm_factory = llm_factory
        raw = {"route_intent": self._route, "cache_lookup": self._cache_lookup, "search": self._search,
               "qa_resolve": self._qa_resolve, "qa_retrieve": self._qa_retrieve, "qa_policy": self._qa_policy,
               "qa_generate": self._qa_generate, "qa_validate": self._qa_validate, "qa_fallback": self._qa_fallback,
               "qa_extractive": self._qa_extractive, "qa_insufficient": self._qa_insufficient,
               "compare": self._compare, "manager": self._manager, "predict": self._predict, "audit": self._audit,
               "clarify": self._clarify, "finalize": self._finalize}
        self.nodes = {name: self._timed(name, fn) for name, fn in raw.items()}

    def _timed(self,name,fn):
        def wrapped(state):
            started=perf_counter()
            event={"node":name,"status":"ok"}
            def execute():
                try:
                    update=fn(state)
                    if update.get("result",{}).get("status")=="error":event["status"]="error"
                    return update
                except (ValueError,FileNotFoundError,ImportError) as exc:
                    event["status"]="error"
                    # Error text is rendered to the local user, not exported.
                    return {"result":{"status":"error","reason":str(exc)[:250]},"route":"clarify" if name=="route_intent" else state.get("route","clarify"),
                            "llm_policy":"node_failed"}
                except Exception as exc:
                    event["status"]="error"
                    return {"result":{"status":"error","reason":"اجرای سرویس ناموفق بود: "+type(exc).__name__},
                            "route":"clarify" if name=="route_intent" else state.get("route","clarify"),"llm_policy":"node_failed"}
            def run(obs=None):
                update = execute()
                event["ms"] = (perf_counter()-started)*1000
                for key in ("route", "cached", "llm_policy", "model_outcome", "evidence_count", "uncovered_facets"):
                    if key in update:
                        event[key] = update[key]
                result = update.get("result", {})
                if result.get("retrieval_method"):
                    event["retrieval_method"] = result["retrieval_method"]
                if self.bridge is not None:
                    m = result.get("meta", {})
                    usage = ({"input": int(m.get("prompt_tokens", 0)), "output": int(m.get("completion_tokens", 0))}
                             if name == "qa_generate" and not getattr(self,"professional_budgeted",False) and m.get("usage_known") and m.get("api_attempts") else None)
                    self.bridge.update(obs, {**event, "api_calls": m.get("api_calls", 0),
                                              "api_attempts": m.get("api_attempts", 0)}, usage=usage,
                                       model=m.get("model") if name == "qa_generate" and not getattr(self,"professional_budgeted",False) else None)
                return update
            if self.bridge is not None:
                with self.bridge.span(name, safe_input={"route": state.get("route", "pending")},
                                      as_type="generation" if name=="qa_generate" and not getattr(self,"professional_budgeted",False) else "span",
                                      trace_context=state.get("trace_context")) as obs:
                    update = run(obs)
            else:
                update = run()
            update["events"]=[*state.get("events",[]),event]
            return update
        return wrapped

    def _route(self,state):
        route,reason=classify_intent(state["request"])
        return {"route":route,"route_reason":reason}

    def _cache_lookup(self,state):
        r=state["request"]
        if not r["use_cache"] or r["allow_llm"] or state["route"] not in ("search","qa"):
            return {"cached":False}
        key=cache_key(r,state["route"])
        hit=self.cache.get(key)
        if hit is None:return {"cache_key":key,"cached":False}
        return {"cache_key":key,"cached":True,"result":hit,"llm_policy":"cached_local_result"}

    def _search(self,state):
        from bonus.retrieval import search
        r=state["request"]
        data=search(r["query"],r["filters"],5,r["retrieval_mode"])
        return {"result":{**data,"status":"answered" if data["items"] else "no_results"},"llm_policy":"not_needed_for_structured_retrieval"}

    def _qa_resolve(self, state):
        from src.rag import resolve_rag_product
        r = state["request"]
        started = perf_counter()
        pid = r["product_ids"][0] if len(r["product_ids"]) == 1 else None
        resolved = resolve_rag_product(r["query"], pid)
        if resolved[-1] is not None:
            return {"result": resolved[-1], "llm_policy": "product_clarification_no_api"}
        return {"resolution": resolved[:3], "qa_started": started}

    def _qa_retrieve(self, state):
        from src.review_qa import retrieve_review_selection
        pid, product, _ = state["resolution"]
        started = perf_counter()
        selection, failure = retrieve_review_selection(pid, state["request"]["query"], source=Config.CATALOG_PATH,
                                                       product_title=product["title_fa"] if product else None, started=started)
        info = {"retrieval_seconds": perf_counter()-started}
        if failure is not None:
            return {**info, "result": failure, "evidence_count": 0}
        return {**info, "selection": selection, "evidence_count": len(selection.evidence)}

    def _qa_policy(self, state):
        if state.get("result", {}).get("status") == "error":
            return {"qa_destination": "finalize"}
        s = state.get("selection")
        r = state["request"]
        if s is None or s.insufficient:
            return {"qa_destination": "qa_insufficient", "llm_policy": "missing_or_insufficient_evidence"}
        facts = {"status": "answered", "evidence": s.evidence, "requested_themes": s.requested_themes,
                 "facet_coverage": s.facet_coverage, "intent": s.intent}
        go, reason = generation_policy(r, facts, strategy=self.generation_strategy,
                                       model_available=True if self.llm_factory is not None else None)
        return {"qa_destination": "qa_generate" if go else "qa_extractive", "llm_policy": reason,
                "uncovered_facets": coverage(r, facts)[1]}

    def _local_qa(self, state):
        from src.review_qa import build_extractive_review
        pid, product, resolution = state["resolution"]
        result = state.get("result")
        if result is None:
            result = build_extractive_review(pid, state["selection"], started=state["qa_started"],
                                             retrieval_seconds=state["retrieval_seconds"])
        return {**result, "product": product, "product_resolution": resolution,
                "status": "answered" if result.get("answer") is not None else "insufficient_evidence"}

    def _qa_insufficient(self, state):
        return {"result": self._local_qa(state)}

    def _qa_extractive(self, state):
        return {"result": self._local_qa(state)}

    def _qa_generate(self, state):
        from phase4_llm.llm_client import LLMClient
        from phase4_llm.service import LLMService
        result = self._local_qa(state)
        meta = result["meta"]
        try:
            client = self.llm_factory() if self.llm_factory else LLMClient(max_retries=1, max_tokens=350)
            response = LLMService(llm=client, require_quotes=True, use_cache=False).answer(
                state["request"]["query"], result["evidence"])
            m = response["meta"]
            meta.update(api_calls=m.get("calls", 0), api_attempts=m.get("api_attempts", 0),
                        prompt_tokens=m.get("prompt_tokens", 0), completion_tokens=m.get("completion_tokens", 0),
                        usage_known=m.get("usage_known", False), model=m.get("model"), model_seconds=m.get("latency", 0))
            return {"result": result, "model_response": response}
        except Exception as exc:
            attempts = int(getattr(exc, "attempts", 0) or 0)
            meta.update(api_attempts=attempts, usage_known=not attempts)
            return {"result": result, "model_outcome": "model_unavailable"}

    def _qa_validate(self, state):
        from src.review_qa import _safe_model_points
        result = {**state["result"], "meta": dict(state["result"]["meta"])}
        outcome = state.get("model_outcome")
        response = state.get("model_response")
        if outcome is None:
            try:
                points = _safe_model_points(response, result["evidence"])
            except (ValueError, KeyError, TypeError):
                points = []
            if points:
                result["answer"] += "\n\nجمع‌بندی مدل زبانی از همین شواهد (استنتاج محدود):\n" + "\n".join(points)
                result["answer_mode"] = "grounded"
                result["meta"].update(generation_mode="llm",
                    citation_check="verified retrieved IDs and verbatim model quotes; semantic assessment remains separate")
                outcome = "citations_verified"
            elif response and response.get("meta", {}).get("failure"):
                outcome = "model_unavailable"
            else:
                outcome = "model_evidence_validation_failed"
        return {"result": result, "model_outcome": outcome}

    def _qa_fallback(self, state):
        result = {**state["result"], "meta": dict(state["result"]["meta"])}
        result["meta"]["fallback_reason"] = state.get("model_outcome", "model_unavailable")
        return {"result": result, "model_outcome": state.get("model_outcome", "model_unavailable")}

    def _compare(self,state):
        from src.rag import compare_products
        r=state["request"]
        if len(r["product_ids"])<2:
            return {"result":{"status":"needs_product","reason":"دو تا پنج شناسه محصول را انتخاب کنید."},"llm_policy":"clarification_no_api"}
        return {"result":compare_products(r["product_ids"],r["query"],source=Config.CATALOG_PATH),"llm_policy":"comparison_local"}

    def _manager(self,state):
        from manager_analytics import answer_manager_question, ManagerScope
        r=state["request"]
        answer=answer_manager_question(r["query"],ManagerScope(category=r.get("category") or None,product_ids=tuple(map(str,r["product_ids"]))))
        return {"result":{"status":"answered",**asdict(answer)},"llm_policy":"aggregate_sql_local"}

    def _predict(self,state):
        import ml_service
        r=state["request"]
        text=r["query"]
        if re.search(r"پیش\s*بینی|طبقه\s*بندی|برچسب",norm(text)) and re.search(r"[:：]",text):
            text=re.split(r"[:：]",text,maxsplit=1)[1].strip()
        if r.get("prediction_model")=="adapted":
            from bonus.adapt_classifier import predict_adapted
            answer=predict_adapted(text,r.get("title"))
        else:answer=ml_service.predict_recommendation(text,r.get("title"))
        return {"result":{"status":"answered" if answer["valid"] else "invalid_input",**answer},"llm_policy":"local_classifier"}

    def _audit(self,state):
        from bonus.quality_gap import audit
        return {"result":audit(category=state["request"].get("category")),"llm_policy":"local_statistical_audit"}

    def _clarify(self,state):
        if state.get("result",{}).get("status")=="error":return {}
        return {"result":{"status":"needs_clarification","reason":"نوع نیاز، نام محصول یا سؤال تحلیلی را مشخص کنید."},"llm_policy":"clarification_no_api"}

    def _finalize(self,state):
        result=state.get("result",{"status":"error"})
        if state.get("cache_key") and not state.get("cached") and result.get("status")=="answered":
            self.cache.put(state["cache_key"],result)
        return {}

    @staticmethod
    def _qa_next(node, state):
        if node == "qa_resolve":
            return "qa_retrieve" if state.get("resolution") is not None else "finalize"
        if node == "qa_policy":
            return state.get("qa_destination", "finalize")
        if node == "qa_validate":
            return "finalize" if state.get("model_outcome") == "citations_verified" or state.get("result", {}).get("status") == "error" else "qa_fallback"
        return QA_EDGES[node][0]

    @cached_property
    def _langgraph(self):
        from langgraph.graph import StateGraph,START,END
        graph=StateGraph(GraphState)
        for name,node in self.nodes.items():graph.add_node(name,node)
        graph.add_edge(START,"route_intent")
        graph.add_edge("route_intent","cache_lookup")
        destinations={r:r for r in ROUTES if r!="qa"}
        destinations.update(qa_resolve="qa_resolve", finalize="finalize")
        graph.add_conditional_edges("cache_lookup", cache_destination, destinations)
        for node, choices in QA_EDGES.items():
            if len(choices) == 1:
                graph.add_edge(node, choices[0])
            else:
                def branch(state, current=node):
                    return self._qa_next(current, state)
                graph.add_conditional_edges(node, branch, {dest: dest for dest in choices})
        for name in (r for r in ROUTES if r != "qa"):
            graph.add_edge(name, "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def _python(self, state):
        state = dict(state)
        for name in ("route_intent", "cache_lookup"):
            state.update(self.nodes[name](state))
        next_node = cache_destination(state)
        while next_node != "finalize":
            state.update(self.nodes[next_node](state))
            next_node = self._qa_next(next_node, state) if next_node in QA_EDGES else "finalize"
        state.update(self.nodes["finalize"](state))
        return state

    def run(self,request,*,record=True,flush_export=False):
        request=validate_request(request)
        initial={"request":request,"events":[],"cached":False}
        trace_id=uuid4().hex;started=perf_counter()
        engine="python-fallback"
        compiled=None
        if self.engine_choice!="python":
            try:
                compiled=self._langgraph;engine="LangGraph"
            except ImportError:
                if self.engine_choice=="langgraph":raise
            except Exception:
                if self.engine_choice=="langgraph":raise
                engine="python-fallback (LangGraph initialization failed)"
        def execute():
            try:
                return compiled.invoke(initial,config={"recursion_limit":16}) if compiled is not None else self._python(initial)
            except Exception as exc:
                # Do not retry the business graph when an engine error might
                # have happened after a real model request.
                return {"route":"clarify","route_reason":"workflow_engine_error",
                        "result":{"status":"error","reason":"خطای موتور گراف: "+type(exc).__name__,
                                  "meta":{"usage_known":False}},
                        "events":[{"node":"workflow_error","status":"error","ms":(perf_counter()-started)*1000}]}
        remote_id=None
        if self.bridge is not None:
            with self.bridge.span("forouhar-workflow",safe_input={"trace_id":trace_id,"engine":engine}) as obs:
                if obs is not None:remote_id=getattr(obs,"trace_id",None)
                # Carry only IDs through the graph, so parallel requests cannot mix parents.
                trace_context = self.bridge.trace_context(obs)
                if trace_context:
                    initial["trace_context"] = trace_context
                state = execute()
                result = state.get("result", {"status":"error"})
                self.bridge.update(obs, {"route": state.get("route"), "status": result.get("status"),
                    "cached": bool(state.get("cached")), "llm_policy": state.get("llm_policy"),
                    "evidence_count": state.get("evidence_count"), "model_outcome": state.get("model_outcome"),
                    "api_calls": result.get("meta", {}).get("api_calls", 0)})
            if flush_export:
                self.bridge.flush()
            elif self.bridge.status == "sdk_ready" and remote_id:
                self.bridge.status = "queued_unverified"
        else:
            state=execute();result=state.get("result",{"status":"error"})
        meta=result.get("meta",{})
        row={"trace_id":trace_id,"created_at":timestamp(),"route":state.get("route"),"route_reason":state.get("route_reason"),
             "engine":engine,"status":result.get("status","error"),"cached":bool(state.get("cached")),
             "elapsed_ms":(perf_counter()-started)*1000,"api_calls":int(meta.get("api_calls",0) or 0),
             "api_attempts":int(meta.get("api_attempts",meta.get("api_calls",0)) or 0),
             "prompt_tokens":int(meta.get("prompt_tokens",0) or 0),"completion_tokens":int(meta.get("completion_tokens",0) or 0),
             "usage_known":bool(meta.get("usage_known",True)),"retrieval_method":result.get("retrieval_method"),
             "llm_policy":state.get("llm_policy"),"evidence_count":state.get("evidence_count"),
             "uncovered_facets":state.get("uncovered_facets"),"model_outcome":state.get("model_outcome"),"langfuse_status":self.bridge.status if self.bridge else "disabled",
             "langfuse_trace_id":str(remote_id) if remote_id else None,"events":state.get("events",[])}
        if record:
            try:
                self.store.append(row)
            except (SQLiteError, OSError):
                row["local_trace_saved"] = False
        return {"result":result,"trace":row}
