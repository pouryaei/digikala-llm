"""The visible Forouhar bonus lab. Imported only from the optional Streamlit tab."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

import pandas as pd
import streamlit as st

from bonus.graph import Workflow, ROUTES, LABELS, EDGE_PAIRS
from bonus.observability import TraceStore,ResultCache,stats
from bonus.qa_demo import DEMO_CASES
from bonus.professional import ProfessionalWorkflow

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/"reports/bonus"
ROUTE_OPTIONS=["auto",*ROUTES[:-1]]
METHODS={"bm25":"BM25 اصلی","hybrid_char":"BM25 + Character TF-IDF","auto":"Router خودکار"}


def report(name):
    try:return json.loads((REPORT/f"{name}.json").read_text(encoding="utf8"))
    except (OSError,ValueError):return None


def pct(value):
    return "—" if value is None else f"{value:.1%}"


def number(value):
    return "—" if value is None else f"{value:,.0f}"


def show_products(items,prefix,open_review=None):
    if not items:
        st.info("محصول کافی با این درخواست و قیود پیدا نشد.")
        return
    rows=[]
    for r in items:
        rows.append({"شناسه":r.get("product_id"),"محصول":r.get("title_fa",r.get("title")),
                     "دسته":r.get("category"),"قیمت ثبت‌شده":r.get("price"),"نرخ پیشنهاد":pct(r.get("recommendation_rate")),
                     "نظرات برچسب‌دار":r.get("valid_label_count"),"امتیاز بازیابی":r.get("hybrid_score")})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    if open_review is not None:
        ids={str(p["product_id"]):p for p in items}
        chosen=st.selectbox("انتخاب محصول برای پرسش مستند",list(ids),
                            format_func=lambda pid:ids[pid].get("title_fa",pid),key=f"{prefix}_review_target")
        st.button("رفتن به پرسش از نظرات",key=f"{prefix}_open_review",
                  on_click=open_review,args=(int(chosen),ids[chosen].get("title_fa")))


def draw_graph(trace):
    seen={e["node"] for e in trace.get("events",[])}
    nodes={
        "START":"درخواست","route_intent":"تشخیص نوع سؤال","cache_lookup":"بررسی کش","search":"جست‌وجو",
        "qa_resolve":"تشخیص محصول","qa_retrieve":"بازیابی شواهد","qa_policy":"تصمیم استفاده از مدل",
        "qa_extractive":"پاسخ از داده","qa_insufficient":"شاهد ناکافی","qa_generate":"جمع‌بندی با مدل",
        "qa_validate":"اعتبارسنجی ارجاع","qa_fallback":"بازگشت به پاسخ از داده",
        "compare":"مقایسه","manager":"تحلیل مدیر",
        "predict":"پیش‌بینی","audit":"شکاف رضایت","clarify":"رفع ابهام","finalize":"خروجی و ثبت","END":"پایان"
    }
    if trace.get("professional"):
        nodes["route_intent"]="کش کامل؛ سپس مسیریابی"
        nodes["cache_lookup"]="نتیجه کش یا ادامه سرویس"
    lines=['digraph forouhar { rankdir=TB; graph [bgcolor="transparent",ranksep=0.28,nodesep=0.16];',
           'node [shape=box,style="rounded,filled",fontsize=10,fontname="Arial",color="#afbedc"];',
           'edge [color="#a9b7d3",arrowsize=.65];']
    for name,label in nodes.items():
        fill="#b6d7ff" if name in seen or name in ("START","END") else "#f3f4f8"
        lines.append(f'{name} [label="{label}",fillcolor="{fill}"];')
    executed=["START",*[e["node"] for e in trace.get("events",[])],"END"]
    active=set(zip(executed,executed[1:]))
    for a,b in EDGE_PAIRS:
        lines.append(f'{a} -> {b} [color="{"#2062ba" if (a,b) in active else "#c4cddd"}",penwidth={2 if (a,b) in active else 1}];')
    lines.append("}")
    try:st.graphviz_chart("\n".join(lines),use_container_width=True)
    except Exception:
        # Render the actual executed path without a graphviz installation.
        h=64*len(executed)+20
        parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="480" height="{h}" viewBox="0 0 480 {h}">']
        for i,name in enumerate(executed):
            y=10+64*i
            parts.append(f'<rect x="86" y="{y}" width="308" height="40" rx="10" fill="#e6f1ff" stroke="#346aba"/>')
            parts.append(f'<text x="240" y="{y+26}" text-anchor="middle" direction="rtl" font-size="15" fill="#152f6a">{nodes.get(name,name)}</text>')
            if i<len(executed)-1:
                parts.append(f'<path d="M240 {y+40} V{y+58} l-5 -7 M240 {y+58} l5 -7" stroke="#346aba" fill="none" stroke-width="2"/>')
        parts.append("</svg>")
        st.image("".join(parts))


POLICY_TEXT = {
    "product_clarification_no_api": "تا محصول مشخص نشود، مدل فراخوانی نمی‌شود.",
    "missing_or_insufficient_evidence": "برای این محصول و سؤال، شاهد مرتبط کافی پیدا نشد.",
    "requested_facets_uncovered": "برای بعضی معیارهای سؤال شاهد انتخاب نشده؛ فقط یافته‌های موجود نمایش داده می‌شوند.",
    "simple_question_local_answer_sufficient": "پاسخ مستقیم از داده کافی است؛ مدل فراخوانی نشد.",
    "too_few_distinct_witnesses": "برای جمع‌بندی مدل دست‌کم دو نظر مستقل لازم است.",
    "explicit_opt_in_complex_grounded_synthesis": "شواهد کافی و چند معیار یا درخواست جمع‌بندی؛ استفاده از مدل مجاز است.",
    "group_key_unavailable": "این سؤال واجد شرایط جمع‌بندی است، اما کلید گروه تنظیم نشده است.",
    "offline_by_default": "اجازهٔ صریح استفاده از مدل فعال نیست؛ فقط پاسخ محلی ساخته شد.",
}
OUTCOME_TEXT = {
    "citations_verified": "شناسه و نقل‌قول بررسی شد؛ درستی معنایی ادعا همچنان نیازمند ارزیابی است.",
    "model_evidence_validation_failed": "ارجاع یا نقل‌قول پذیرفته نشد؛ پاسخ استخراجی حفظ شد.",
    "model_unavailable": "مدل پاسخ قابل استفاده نداد؛ پاسخ استخراجی حفظ شد.",
}


PROFESSIONAL_REASONS={
 "valid_complete_response_cache":"پاسخ کامل معتبر در کش","cache_miss_or_disabled":"پاسخ کامل در کش نیست؛ پردازش محلی",
 "valid_route_cache":"مسیر معتبر از کش","explicit_user_choice":"سرویس با انتخاب کاربر مشخص است",
 "local_intent_sufficient":"هدف با قواعد محلی مشخص است","selected_products_make_route_clear":"شناسه‌ها مسیر را مشخص می‌کنند",
 "selected_category_makes_route_clear":"دسته مسیر را مشخص می‌کند","direct_numeric_request":"پرسش عددی مستقیم",
 "additional_language_reasoning_needed":"تفسیر یا رفع ابهام ارزش اضافه دارد","llm_not_authorized":"اجازه ارسال داده نشد",
 "group_key_unavailable":"کلید یا پیکربندی مدل موجود نیست","request_budget_exhausted":"سقف مشترک تلاش مصرف شد",
 "router_output_validated":"مسیر پیشنهادی اعتبارسنجی شد","router_failed_local_fallback":"مسیریابی ناموفق؛ مسیر محلی",
 "existing_retrieval_preserves_constraints":"جست‌وجوی واقعی و حفظ فیلترها","structured_local_answer_sufficient":"پاسخ ساختاریافته کافی است",
 "simple_question_local_answer_sufficient":"پاسخ استخراجی کافی است","insufficient_or_uncovered_evidence":"شاهد کافی یا پوشش معیار وجود ندارد",
 "citations_validated":"شناسه، نقل‌قول و عدد بررسی شد","invalid_generation_local_fallback":"خروجی نامعتبر؛ حفظ پاسخ محلی",
 "selected_head_fine_tuning_classifier":"مدل منتخب طبقه‌بندی محلی","aggregate_statistics_only":"آمار واقعی محلی",
 "clarification_no_api":"نیاز به تکمیل اطلاعات",
}

def professional_trace(trace):
    if not trace.get("professional"):return
    cost=trace.get("estimated_cost_usd")
    a,b,c=st.columns(3)
    a.metric("تلاش مصرف‌شده / سقف",f"{trace.get('api_attempts',0)} / {trace.get('attempt_limit',0)}")
    b.metric("Token گزارش‌شده",f"{trace.get('prompt_tokens',0)+trace.get('completion_tokens',0):,}")
    c.metric("هزینه این درخواست","نامعلوم" if cost is None else f"${cost:.6f}")
    if trace.get("cached"):
        st.success("پاسخ از کش معتبر؛ صفر فراخوانی تازه. منشأ اجرای قبلی: "+str(trace.get("cache_origin") or "محلی"))
    elif cost is None:st.caption("تعرفه واقعی یا اطلاعات مصرف کامل نیست؛ هزینه صفر فرض نشده است.")
    rows=[{"لایه":d["stage"],"تصمیم":d["mode"],"دلیل":PROFESSIONAL_REASONS.get(d["reason"],d["reason"]),
           "مدل":d.get("model") or "—"} for d in trace.get("decisions",[])]
    if rows:
        st.markdown("**تصمیم واقعی در هر لایه**")
        st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
    if trace.get("model_usage"):
        st.markdown("**مصرف هر تلاش درگاه**")
        st.dataframe(pd.DataFrame(trace["model_usage"]),hide_index=True,width="stretch")
    st.caption("اعتبارسنجی شناسه و نقل‌قول جای ارزیابی معنایی انسانی را نمی‌گیرد.")

def show_supplement(result):
    supplement=result.get("professional_synthesis")
    if not supplement:return
    st.markdown("**جمع‌بندی تکمیلی مدل از منابع واقعی**")
    st.caption(supplement["model"]+" · استنتاج محدود، نه واقعیت جدید")
    for point in supplement["points"]:
        st.write(point["text"])
        for ref in point["citations"]:
            label=f"نظر {ref['comment_id']} · محصول {ref.get('product_id')}" if ref.get("comment_id") else f"ردیف داده {ref['source_id']}"
            st.caption(label+" — «"+ref["quote"]+"»")


def trace_view(trace):
    st.caption(f"شناسه Trace: {trace['trace_id']}")
    a,b,c,d=st.columns(4)
    a.metric("مسیر",LABELS.get(trace.get("route"),"—"))
    b.metric("زمان پاسخ",f"{trace['elapsed_ms']:.1f} ms")
    c.metric("کش","HIT" if trace.get("cached") else "MISS / خاموش")
    d.metric("درخواست API",trace.get("api_calls",0))
    st.caption(f"موتور واقعی اجرا: {trace['engine']} · Langfuse: {trace.get('langfuse_status')} · سیاست مدل: {trace.get('llm_policy')}")
    if trace.get("llm_policy") in POLICY_TEXT:
        st.info(POLICY_TEXT[trace["llm_policy"]])
    if trace.get("model_outcome") in OUTCOME_TEXT:
        st.caption(OUTCOME_TEXT[trace["model_outcome"]])
    if trace.get("evidence_count") is not None:
        st.caption(f"تعداد شواهد: {trace['evidence_count']} · معیارهای بدون شاهد: {trace.get('uncovered_facets') or 0} · تلاش‌های Gateway: {trace.get('api_attempts', 0)}")
    if trace.get("langfuse_trace_id"):
        st.caption("شناسه برای جست‌وجو در Langfuse (دریافت توسط سرور هنوز تأیید نشده):")
        st.code(trace["langfuse_trace_id"], language=None)
    if trace.get("local_trace_saved") is False:
        st.warning("پاسخ آماده است، اما ثبت تاریخچهٔ محلی ناموفق بود.")
    professional_trace(trace)
    with st.expander("گراف و زمان واقعی هر گره",expanded=True):
        draw_graph(trace)
        rows=[{"گره":r["node"],"زمان (ms)":round(r["ms"],3),"وضعیت":r["status"]} for r in trace["events"]]
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        if rows:st.bar_chart(pd.DataFrame(rows).set_index("گره")[["زمان (ms)"]])


def show_result(result,route,open_review=None):
    if result.get("status")=="error":
        st.error(result.get("reason","خطای سرویس"));return
    if route=="search":
        st.caption(METHODS.get(result.get("retrieval_method"),"روش بازیابی"))
        for note in result.get("search_notes",[]):st.info(note)
        show_products(result.get("items",[]),"router_result",open_review)
    elif route=="qa":
        st.write(result.get("answer") or result.get("reason") or "برای پاسخ شواهد کافی نیست.")
        missing = [face["theme"] for face in result.get("facet_coverage", []) if not face.get("evidence_ids")]
        if missing:
            st.info("برای این موضوع‌ها در شواهد انتخاب‌شده شاهد مستقیم نداریم: " + "، ".join(missing))
        if result.get("candidates"):show_products(result["candidates"],"router_candidates",open_review)
        with st.expander("شواهد واقعی و شناسه نظر",expanded=bool(result.get("evidence"))):
            for e in result.get("evidence",[]):
                st.markdown(f"**نظر {e.get('comment_id')}** · محصول {e.get('product_id')}")
                st.write(e.get("text"))
    elif route=="compare":
        for item in result.get("products",[]):
            facts=item.get("facts") or {}
            with st.container(border=True):
                st.subheader(facts.get("title_fa",str(item.get("product_id"))))
                st.write(item.get("review_stats",{}))
                for e in item.get("strengths",[])+item.get("weaknesses",[]):
                    st.write(f"نظر {e.get('comment_id')}: {e.get('text')}")
        for n in result.get("comparison_notes",[]):st.info(n)
        if not result.get("products"):st.info(result.get("reason","محصول‌ها را مشخص کنید."))
    elif route in ("manager","audit"):
        st.write(result.get("summary") or result.get("note") or result.get("reason"))
        rows=result.get("rows",result.get("items",[]))
        if rows:
            st.dataframe(pd.DataFrame([{k:v for k,v in r.items() if k!="evidence"} for r in rows]),hide_index=True,use_container_width=True)
    elif route=="predict":
        if result.get("valid"):
            st.success(str(result["label"]))
            st.bar_chart(pd.DataFrame({"Probability":result.get("probabilities",{})}))
        else:st.info(f"ورودی معتبر نیست: {result.get('reason')}")
    else:st.info(result.get("reason") or "درخواست روشن‌تر لازم است.")


DEMO={
    "جست‌وجوی ترکیبی":("کوله پشتی بچه برای مهد کودک","",""),
    "پرسش مستند":("ایرادهای پرتکرار این محصول چیست؟","1696173",""),
    "مقایسه":("این دو محصول را مقایسه کن","1696173,2896502",""),
    "تحلیل مدیر":("پرتکرارترین شکایت کاربران در این دسته چیست؟","","کفش مردانه"),
    "پیش‌بینی":("پیش بینی وضعیت پیشنهاد خرید این نظر: کیفیت عالیه، پیشنهاد می‌کنم","",""),
    "مسئله جدید":("اختلاف امتیاز ثبت شده و نرخ پیشنهاد خرید محصولات","",""),
}

QA_DEMOS = {label: (q, str(pid) if pid is not None else "", "") for _, label, q, pid, _ in DEMO_CASES}
DEMO.update(QA_DEMOS)


def router_panel(open_review=None):
    st.subheader("دستیار حرفه‌ای · انتخاب خودکار مسیر و مدل")
    st.caption("یک مسیر هوشمند: کش معتبر، سرویس محلی، مدل زبانی فقط در صورت نیاز.")
    st.info("اگر نوع کار یا محصول را می‌دانید، مشخص کنید تا هزینه مسیریابی کم شود. برای مقایسه دو شناسه و برای تحلیل مدیر یک دسته انتخاب کنید.")
    if "professional_session_cache" not in st.session_state:
        st.session_state["professional_session_cache"]=ResultCache(capacity=128,ttl=600)
    installed=importlib.util.find_spec("langgraph") is not None
    st.caption("LangGraph قابل استفاده است." if installed else "LangGraph نصب نیست؛ موتور Python با همان گراف مشخص اجرا می‌شود. نتیجه به‌عنوان اجرای LangGraph معرفی نمی‌شود.")
    scenario=st.selectbox("سناریوی آماده برای Demo",list(DEMO),key="bonus_router_example")
    question,ids,category=DEMO[scenario]
    with st.form("bonus_router_form_"+scenario):
        q=st.text_area("درخواست فارسی",value=question,max_chars=1000)
        c1,c2=st.columns(2)
        mode=c1.selectbox("نوع درخواست (اختیاری)",ROUTE_OPTIONS,index=ROUTE_OPTIONS.index("qa") if scenario in QA_DEMOS else 0,key="bonus_intent_"+scenario,format_func=lambda x:"تشخیص خودکار" if x=="auto" else LABELS[x])
        chosen_ids=c2.text_input("شناسه محصولات (با ویرگول)",value=ids)
        cat=st.text_input("دسته برای تحلیل مدیر (اختیاری)",value=category)
        cache=st.checkbox("استفاده از کش معتبر (برای آزمایش می‌توانید خاموش کنید)",value=True,key="bonus_cache_"+scenario)
        live=st.checkbox("اجازه ارسال سؤال و شواهد لازم به API گروه؛ فقط برای این اجرا",value=False,key="bonus_live_"+scenario)
        st.caption("اجازه دادن به معنی تماس قطعی نیست. سقف ۲ تلاش مشترک است؛ مدل و تعرفه از تنظیمات محلی خوانده می‌شوند.")
        submitted=st.form_submit_button("اجرای گراف",type="primary")
    if submitted:
        try:
            from src.task2_review_store import identifier
            parsed=[identifier(x) for x in re.split(r"[,،;؛\s]+",chosen_ids.strip()) if x]
            with st.spinner("در حال اجرای گره‌های انتخاب‌شده..."):
                st.session_state["bonus_router_result"]=ProfessionalWorkflow(cache=st.session_state["professional_session_cache"]).run(
                    {"query":q,"intent":mode,"product_ids":parsed,"category":cat.strip() or None,
                     "retrieval_mode":"auto","use_cache":cache,"allow_llm":live})
        except Exception as e:st.error(f"درخواست اجرا نشد: {type(e).__name__}. داده یا تنظیمات را بررسی کنید.")
    entry=st.session_state.get("bonus_router_result")
    if entry:
        trace_view(entry["trace"])
        show_result(entry["result"],entry["trace"]["route"],open_review)
        show_supplement(entry["result"])
    st.divider()
    st.subheader("مانیتورینگ محلی")
    rows=TraceStore().recent(200)
    stat=stats(rows)
    a,b,c,d=st.columns(4)
    a.metric("Trace ثبت‌شده",stat["requests"])
    b.metric("Median latency",f"{stat['median_ms']:.1f} ms")
    c.metric("Cache hit rate",pct(stat["cache_rate"]))
    d.metric("خطا",stat["errors"])
    st.caption(f"API calls: {stat['api_calls']} · attempts: {stat['api_attempts']} · input tokens: {stat['prompt_tokens']} · output tokens: {stat['completion_tokens']}")
    measured=[r.get("estimated_cost_usd") if r.get("professional") else (0.0 if not r.get("api_attempts") else None) for r in rows]
    if all(v is not None for v in measured):
        st.caption(f"برآورد هزینه تاریخچه با تعرفه واقعی: ${sum(measured):.6f}")
    else:st.caption("هزینه کل تاریخچه نامعلوم است؛ برخی تعرفه‌ها یا اطلاعات مصرف کامل نیستند.")
    if rows:
        flat=[{k:r.get(k) for k in ("trace_id","created_at","route","engine","status","elapsed_ms","cached","api_calls","prompt_tokens","completion_tokens")} for r in rows]
        st.dataframe(pd.DataFrame(flat),hide_index=True,use_container_width=True)
        trace_ids={r["trace_id"]:r for r in rows}
        tid=st.selectbox("جزئیات Trace ثبت‌شده",list(trace_ids),key="bonus_trace_select")
        with st.expander("گره‌های همین درخواست"):
            st.dataframe(pd.DataFrame(trace_ids[tid]["events"]),hide_index=True,use_container_width=True)
        st.download_button("دریافت Traceها (بدون متن خصوصی درخواست)",json.dumps(rows,ensure_ascii=False,indent=2),"forouhar_traces.json","application/json")
    else:st.info("هنوز Trace زنده‌ای ثبت نشده؛ یکی از سناریوهای بالا را اجرا کن.")
    st.caption("ارسال به Langfuse فقط با تنظیم صریح و کلیدهای محلی فعال می‌شود. متن سؤال، نظر و کلیدها به مانیتورینگ بیرونی فرستاده نمی‌شوند.")
    base=os.getenv("LANGFUSE_BASE_URL","").strip()
    if os.getenv("FOROUHAR_LANGFUSE_ENABLED")=="1" and urlsplit(base).scheme in ("http","https") and urlsplit(base).hostname:
        st.markdown(f"[باز کردن داشبورد Langfuse]({base})")
    experiment=report("qa_graph_ab")
    if experiment and experiment.get("mode") == "offline_stub":
        with st.expander("آزمایش کنترل‌شدهٔ گراف پرسش‌وپاسخ", expanded=True):
            p = experiment["offline_parity"]
            a = experiment["comparison"]
            x, y, z = st.columns(3)
            x.metric("پاسخ آفلاین یکسان", f"{p['matched']} / {p['total']}")
            y.metric("فراخوانی شبیه‌سازی‌شده؛ مسیر همیشگی", a["always_model_attempts"])
            z.metric("فراخوانی شبیه‌سازی‌شده؛ مسیر انتخابی", a["selective_model_attempts"])
            st.caption("این اعداد مربوط به مدل آزمایشی‌اند؛ هزینه یا کیفیت مدل واقعی را نشان نمی‌دهند.")
            st.dataframe(pd.DataFrame([{"شناسه": row["case_id"], "سیاست": row["selective"]["policy"],
                "تلاش همیشگی": row["always"]["api_attempts"], "تلاش انتخابی": row["selective"]["api_attempts"]}
                for row in experiment["paired_cases"]]),hide_index=True,use_container_width=True)
    ev=report("professional_policy")
    if ev and ev.get("mode")=="offline_fixture":
        with st.expander("ارزیابی کنترل‌شده سیاست حرفه‌ای روی داده واقعی",expanded=True):
            a,b,c,d=st.columns(4)
            a.metric("سناریوها",ev["total_cases"])
            b.metric("بدون تلاش زبانی",ev["zero_attempt_cases"])
            c.metric("تلاش شبیه‌سازی؛ اجرای نخست",ev["first_simulated_attempts"])
            d.metric("تلاش شبیه‌سازی؛ تکرار",ev["repeat_simulated_attempts"])
            st.caption(f"حفظ داده محلی در {ev['facts_preserved_cases']} سناریو؛ کش کامل در {ev['warm_cache_hits']} تکرار. کیفیت و هزینه مدل واقعی اندازه‌گیری نشده است.")
            st.dataframe(pd.DataFrame([{"سناریو":r["scenario"],"مسیر":r["route"],"تلاش اول":r["first_attempts"],"تلاش تکرار":r["repeat_attempts"],"Cache HIT":r["warm_cache_hit"]} for r in ev["cases"]]),hide_index=True,width="stretch")
    result=report("router_cache")
    if result:
        with st.expander("نتایج آزمایش Router و Cache روی داده واقعی",expanded=True):
            rr=result["router"];st.metric("تشخیص Intent",f"{rr['correct']} / {rr['total']}")
            cc=result["cache"];b1,b2,b3=st.columns(3)
            b1.metric("بدون کش (میانه)",f"{cc['uncached_global_median_ms']:.1f} ms")
            b2.metric("با کش (میانه)",f"{cc['cached_global_median_ms']:.2f} ms")
            b3.metric("پاسخ‌های یکسان",f"{cc['total_warm_requests']} / {cc['total_warm_requests']}")
            table=[{"پرسش":r["query"],"بدون کش (ms)":round(r["uncached_median_ms"],3),
                    "با کش (ms)":round(r["cached_median_ms"],3),"سرعت نسبی":round(r["speedup"],1)} for r in cc["cases"]]
            st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)


def retrieval_panel(open_review=None):
    st.subheader("مقایسه A/B بازیابی")
    st.caption("همان داده و همان فیلترها؛ مدل دوم از شباهت کاراکتری برای غلط تایپی و اختلاف نگارش استفاده می‌کند.")
    with st.form("bonus_ab"):
        query=st.text_input("عبارت جست‌وجو",value="کوله پشتی بچه برای مهد کودک")
        c1,c2=st.columns(2);category=c1.text_input("دسته (اختیاری)");brand=c2.text_input("برند (اختیاری)")
        lo,hi=st.columns(2);minimum=lo.text_input("حداقل قیمت خام");maximum=hi.text_input("حداکثر قیمت خام")
        submitted=st.form_submit_button("مقایسه هر سه روش",type="primary")
    if submitted:
        try:
            from src.task2_review_store import DIGITS
            filters={k:v.strip() for k,v in (("category",category),("brand",brand)) if v.strip()}
            for k,value in (("min_price",minimum),("max_price",maximum)):
                if value.strip():filters[k]=float(value.translate(DIGITS).replace(",","").replace("٬",""))
            with st.spinner("بازیابی روی کاتالوگ واقعی..."):
                runner=Workflow()
                st.session_state["bonus_ab_result"]={m:runner.run({"query":query,"intent":"search","retrieval_mode":m,
                                                                "filters":filters,"use_cache":False}) for m in METHODS}
        except Exception as e:st.error(f"ورودی یا داده نامعتبر: {type(e).__name__}")
    entries=st.session_state.get("bonus_ab_result")
    if entries:
        for col,(method,entry) in zip(st.columns(3),entries.items()):
            with col:
                st.markdown(f"**{METHODS[method]}**")
                st.metric("زمان",f"{entry['trace']['elapsed_ms']:.1f} ms")
                st.caption(f"روش اجرا: {entry['trace'].get('retrieval_method') or 'ناموجود'}")
                if entry["result"].get("status")=="error":st.error(entry["result"].get("reason"))
                else:show_products(entry["result"].get("items",[]),"ab_"+method,open_review)
    r=report("retrieval")
    if r:
        st.divider();st.subheader("نتیجه کمی ارزیابی ثابت")
        group=st.selectbox("مجموعه ارزیابی",["original_discovery","typo_stress","original_diagnostic","all"],
                           format_func=lambda x:{"original_discovery":"کشف طبیعی (مجموعه قبلی)","typo_stress":"۴۰ پرسش غلط تایپی کنترل‌شده",
                                                "original_diagnostic":"تمام پرسش‌های قبلی","all":"ترکیب هر دو"}[x])
        data=r["groups"][group]
        df=pd.DataFrame([{"روش":METHODS[m],"Hit@5":q["hit_at_5"],"P@5":q["p_at_5"],"nDCG@5":q["ndcg_at_5"],
                         "Median ms":q["latency"]["median_ms"],"اجرای مدل دوم":q["hybrid_runs"]} for m,q in data.items()])
        st.dataframe(df,hide_index=True,use_container_width=True)
        st.bar_chart(df.set_index("روش")[["Hit@5","nDCG@5"]])
        st.caption("مجموعه غلط تایپی از عنوان‌های واقعی و تغییر کنترل‌شده ساخته شده است. هر دو مجموعه آزمایشی تشخیصی‌اند، نه آزمون انسانی مستقل.")


def hyperparameter_panel():
    st.divider()
    st.subheader("آزمایش تاریخی Hyperparameter Search مدل Logistic Regression")
    r=report("hyperparameter_search/report")
    if not r:
        st.info("جست‌وجو هنوز اجرا نشده است. دستور بازتولید در docs/BONUS_GUIDE.md قرار دارد.")
        return
    params=r["best_params"]
    baseline=r["baseline_validation"]
    best=r["best_validation"]
    test=r["test"]
    a,b,c,d=st.columns(4)
    a.metric("ترکیب‌های آزمایش‌شده",str(r["n_trials"]))
    b.metric("Validation Macro F1",f"{best['macro_f1']:.2%}",
             f"{100*r['validation_delta_macro_f1']:+.2f} pp")
    c.metric("Test Macro F1",f"{test['macro_f1']:.2%}")
    d.metric("Test Accuracy",f"{test['accuracy']:.2%}")
    weight=params["class_weight"] or "None"
    st.success(f"تنظیمات منتخب روی Validation: C = {params['C']:g} | class_weight = {weight}")
    st.caption("همان معماری Character TF-IDF + Logistic Regression؛ بردارساز فقط از Train یاد گرفته، "
               "انتخاب با Macro F1 روی Validation ثابت و ارزیابی Test تنها برای برنده انجام شده است. "
               "این گزارش مربوط به آزمایش قبلی LR است؛ مدل فعال نسخه نهایی Head Fine-tuning مدل LR است.")
    splits=r["audit"]["splits"]
    st.caption(f"Train: {number(splits['train']['rows'])} | Validation: {number(splits['validation']['rows'])} "
               f"| Test: {number(splits['test']['rows'])} — محصول و کلید متن مشترک بین این مجموعه‌ها: صفر.")
    trials=pd.DataFrame([{"rank":t["rank"],"C":t["C"],"class_weight":t["class_weight"] or "None",
                          "Validation Macro F1":f"{t['validation_macro_f1']:.2%}",
                          "Validation Accuracy":f"{t['validation_accuracy']:.2%}",
                          "Fit (s)":round(t["fit_seconds"],2)} for t in r["trials"]])
    st.dataframe(trials,hide_index=True,use_container_width=True)
    chart=pd.DataFrame([{"C":str(t["C"]),"class_weight":t["class_weight"] or "None",
                         "Macro F1":t["validation_macro_f1"]} for t in r["trials"]])
    st.bar_chart(chart.pivot(index="C",columns="class_weight",values="Macro F1").reindex(
        [str(v) for v in r["param_grid"]["C"]]))
    st.caption(f"مرجع منصفانه همان نمونه (C=1, balanced): {baseline['validation_macro_f1']:.2%} روی Validation.")
    reference=r.get("recorded_test_reference")
    if reference:
        original=reference["original"]
        st.markdown("**مقایسه روی شناسه و برچسب یکسان آزمون ثبت‌شده**")
        st.dataframe(pd.DataFrame([
            {"مدل":"مدل اصلی ذخیره‌شده","Macro F1":f"{original['macro_f1']:.2%}","Accuracy":f"{original['accuracy']:.2%}"},
            {"مدل":"نامزد منتخب جست‌وجو","Macro F1":f"{test['macro_f1']:.2%}","Accuracy":f"{test['accuracy']:.2%}"},
        ]),hide_index=True,use_container_width=True)
        delta=100*(test["macro_f1"]-original["macro_f1"])
        st.info(f"اختلاف Macro F1 نامزد و مدل اصلی روی Test: {delta:+.2f} pp. "
                "این نامزد Logistic Regression در نسخه قبلی فعال نشده بود.")
        st.caption("اندازه و نمونه آموزش مدل تاریخی با این آزمایش الزاماً برابر نیست؛ اثر خالص تغییر "
                   "ابرپارامترها را از جدول Validation بسنجید. Test ثبت‌شده، آزمون کور جدید نیست.")
    with st.expander("جزئیات روش، ارزیابی کلاس‌ها و کنترل نشت اطلاعات"):
        st.json({"method":r["method"],"param_grid":r["param_grid"],
                 "fixed_params":r["fixed_params"],"audit":r["audit"],
                 "production_model_replaced":r["production_model_replaced"],
                 "test_used_for_selection":r["test_used_for_selection"]})
        class_report=r["test_classification_report"]
        st.dataframe(pd.DataFrame([{"class":label,**class_report[label]}
                                   for label in r["test_confusion_matrix"]["labels"]]),
                     hide_index=True,use_container_width=True)
    directory=REPORT/"hyperparameter_search"
    for name,mime in (("report.json","application/json"),("trials.csv","text/csv")):
        file=directory/name
        if file.is_file():
            st.download_button(f"دریافت {name}",file.read_bytes(),f"hyperparameter_{name}",mime,
                               key="hp_"+name)


def xgb_report():
    try:
        return json.loads((ROOT/"m2_outputs/xgboost/report.json").read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None


def xgboost_panel():
    st.subheader("مدل XGBoost و جست‌وجوی ابرپارامترها")
    r=xgb_report()
    if not r:
        st.error("فایل نتیجه و مدل XGBoost یافت نشد.")
        return
    best=r["best_validation"];test=r["test"];params=r["best_params"]
    a,b,c,d=st.columns(4)
    a.metric("ترکیب‌های آزمایش‌شده",str(r["n_trials"]))
    b.metric("Validation Macro F1",f"{best['macro_f1']:.2%}")
    c.metric("Test Macro F1",f"{test['macro_f1']:.2%}")
    d.metric("Test Accuracy",f"{test['accuracy']:.2%}")
    st.success(f"بهترین تنظیم XGBoost روی Validation: max_depth={params['max_depth']} | "
               f"learning_rate={params['learning_rate']} | n_estimators={params['n_estimators']}")
    st.caption("این مدل در نسخه نهایی نامزد مقایسه است؛ مدل منتخب Head Fine-tuning است. "
               "بردارساز و chi2 فقط با Train fit شده‌اند. Test پس از انتخاب برنده ارزیابی شده است.")
    splits=r["audit"]["splits"]
    st.caption(f"Train: {number(splits['train']['rows'])} | Validation: {number(splits['validation']['rows'])} "
               f"| Test: {number(splits['test']['rows'])} | ویژگی ورودی XGBoost: {number(r['preprocessing']['selected_features'])}")
    trials=pd.DataFrame([{"rank":t["rank"],"max_depth":t["max_depth"],"learning_rate":t["learning_rate"],
                         "n_estimators":t["n_estimators"],"Validation Macro F1":f"{t['validation_macro_f1']:.2%}",
                         "Validation Accuracy":f"{t['validation_accuracy']:.2%}"} for t in r["trials"]])
    st.dataframe(trials,hide_index=True,width="stretch")
    chart=pd.DataFrame([{"ترکیب":f"d={t['max_depth']}, η={t['learning_rate']}, n={t['n_estimators']}",
                         "Validation Macro F1":t["validation_macro_f1"]} for t in r["trials"]])
    st.bar_chart(chart.set_index("ترکیب"))
    ref=r.get("recorded_test_reference")
    comparison=[{"مدل":"XGBoost (آزمایش تاریخی)","Macro F1":test["macro_f1"],"Accuracy":test["accuracy"]}]
    if ref:
        original=ref["original"]
        comparison.insert(0,{"مدل":"Logistic Regression تاریخی","Macro F1":original["macro_f1"],"Accuracy":original["accuracy"]})
        old=report("hyperparameter_search/report")
        old_ref=(old or {}).get("recorded_test_reference") or {}
        if old_ref.get("original_predictions_sha256")==ref["original_predictions_sha256"]:
            comparison.insert(1,{"مدل":"Logistic Regression با جست‌وجو","Macro F1":old["test"]["macro_f1"],"Accuracy":old["test"]["accuracy"]})
        adaptation=report("adaptation")
        if (adaptation and adaptation.get("original_test_reproduced_all")==ref["matched_recorded_test_ids_and_labels"]
                and adaptation.get("audit",{}).get("source_payload_sha256")==r.get("provenance",{}).get("source_payload_sha256")):
            a=adaptation["test"]["adapted"]
            comparison.insert(-1,{"مدل":"Head تکمیلی LR (تاریخی)","Macro F1":a["macro_f1"],"Accuracy":a["accuracy"]})
        delta=100*(test["macro_f1"]-original["macro_f1"])
        st.info(f"اختلاف XGBoost و Logistic Regression تاریخی در Test Macro F1: {delta:+.2f} واحد درصد. "
                "این مقایسه روی همان شناسه‌های آزمون است، نه الزاماً اندازه آموزش یکسان.")
    st.markdown("**مقایسه مدل‌ها روی Test**")
    comparison_df=pd.DataFrame(comparison)
    st.dataframe(comparison_df.assign(**{key:comparison_df[key].map(lambda v:f"{v:.2%}") for key in ("Macro F1","Accuracy")}),hide_index=True,width="stretch")
    st.bar_chart(comparison_df.set_index("مدل"))
    with st.expander("جزئیات تنظیمات، کنترل نشت اطلاعات و خطای کلاس‌ها"):
        st.json({"method":r["method"],"preprocessing":r["preprocessing"],"param_grid":r["param_grid"],
                 "fixed_params":r["fixed_params"],"audit":r["audit"],"test_used_for_selection":r["test_used_for_selection"]})
        class_report=r["test_classification_report"]
        st.dataframe(pd.DataFrame([{"class":label,**class_report[label]}
                                   for label in r["labels"]]),hide_index=True,width="stretch")
        st.caption("سطرها: برچسب واقعی، ستون‌ها: پیش‌بینی XGBoost")
        st.dataframe(pd.DataFrame(r["test_confusion_matrix"]["matrix"],index=r["labels"],columns=r["labels"]),width="stretch")
    directory=ROOT/"m2_outputs/xgboost"
    for name,mime in (("report.json","application/json"),("trials.csv","text/csv")):
        file=directory/name
        if file.is_file():
            st.download_button(f"دریافت گزارش XGBoost — {name}",file.read_bytes(),f"xgboost_{name}",mime,key="xgb_"+name)


def model_panel():
    from recommendation_ml.model_dashboard import render_model_lab
    render_model_lab()
    with st.expander("Hyperparameter Search مدل XGBoost (هشت ترکیب)"):
        xgboost_panel()
    with st.expander("Hyperparameter Search مدل Logistic Regression (هشت ترکیب)"):
        hyperparameter_panel()
    r=report("adaptation")
    if r:
        with st.expander("تنظیمات Head Fine-tuning مدل منتخب"):
            chosen=r["selected_on_validation_only"]
            st.success("تنظیمات منتخب: epoch="+str(chosen["epoch"])+" | learning_rate="+str(chosen["rate"]))
            st.caption("آموزش تکمیلی Head خطی با بردارساز ثابت؛ نه LoRA یا آموزش Transformer.")
            st.dataframe(pd.DataFrame([{"مدل":t["name"],"epoch":t.get("epoch"),"lr":t.get("rate"),
                "Validation Macro F1":t["validation"]["macro_f1"],
                "Validation Accuracy":t["validation"]["accuracy"]} for t in r["validation_trials"]]),hide_index=True)
            st.json(chosen)


def human_panel():
    from bonus.human_review import cases,display_case,submit,agreement,export_csv
    st.subheader("ارزیابی انسانی مستقل")
    st.caption("قبل از ثبت نمره، نمرات دستیار نمایش داده نمی‌شوند. داور باید سؤال، جواب، شواهد و منبع را بخواند.")
    options={r["case_id"]:r for r in cases()}
    selected=st.selectbox("نمونه ثابت ارزیابی",list(options),format_func=lambda x:x+" · "+options[x]["product_title"],key="bonus_human_case")
    data=display_case(options[selected])
    st.markdown("**پرسش**");st.write(data["question"])
    st.markdown("**پاسخ واقعی سیستم**");st.write(data["answer"])
    with st.expander("شناسه و متن شواهد انتخاب‌شده",expanded=True):
        for e in data["evidence"]:st.write(f"نظر {e['comment_id']}: {e['text']}")
    with st.expander("نظرهای واقعی منبع این محصول"):
        for row in data["source_reviews"]:
            st.markdown(f"**نظر {row['comment_id']}**")
            for k in ("title","body","advantages","disadvantages"):
                if row.get(k):st.write(row[k])
    with st.form("bonus_human_form_"+selected):
        alias=st.text_input("شناسه مستعار داور")
        quality=st.radio("کیفیت پاسخ", [0,1,2],index=None,format_func=lambda x:{0:"۰ - نامناسب",1:"۱ - ناقص",2:"۲ - کامل"}[x],horizontal=True)
        grounding=st.radio("پشتیبانی پاسخ با شواهد",[0,1,2],index=None,format_func=lambda x:{0:"۰ - بدون پشتیبانی",1:"۱ - جزئی",2:"۲ - مستند"}[x],horizontal=True)
        rationale=st.text_area("دلیل کوتاه داوری مستقل",max_chars=1000)
        saved=st.form_submit_button("ثبت داوری انسانی",type="primary")
    if saved:
        try:submit(alias,selected,quality,grounding,rationale);st.success("داوری واقعی ثبت شد.")
        except ValueError as e:st.error(str(e))
    a=agreement()
    c1,c2,c3=st.columns(3)
    c1.metric("داوری ثبت‌شده",a["human_submissions"])
    c2.metric("پوشش نمونه‌ها",f"{a['unique_cases']} / {a['total_cases']}")
    c3.metric("تعداد داور",a["reviewers"])
    if a["human_submissions"]:
        st.write({"توافق کیفیت":pct(a["quality_exact"]),"توافق شواهد":pct(a["grounding_exact"]),
                  "Weighted Kappa کیفیت":a["quality_weighted_kappa"],"Weighted Kappa شواهد":a["grounding_weighted_kappa"]})
    else:st.warning("هنوز داوری انسانی واقعی ثبت نشده؛ نتیجه انسانی ادعا نمی‌شود.")
    st.download_button("دریافت CSV داوری‌ها",export_csv(),"human_reviews.csv","text/csv")


def gap_panel(open_review=None):
    from bonus.quality_gap import audit
    from src.dashboard_rag import categories
    st.subheader("مسئله جدید: اختلاف امتیاز و رضایت")
    st.caption("محصولی ممکن است امتیاز تجمیعی بالا داشته باشد، اما در نظرهای برچسب‌دار کمتر پیشنهاد شده باشد؛ این فهرست برای بررسی دستی مدیر است.")
    with st.form("bonus_gap"):
        cat=st.selectbox("دسته",["همه دسته‌ها",*categories()["items"]])
        c1,c2,c3=st.columns(3)
        n=c1.slider("حداقل نظر برچسب‌دار",5,100,20)
        rating=c2.slider("حداقل امتیاز از ۱۰۰",50,100,70)
        gap=c3.slider("حداقل شکاف محافظه‌کارانه",0.05,0.5,0.15,0.05)
        go=st.form_submit_button("تحلیل کل دامنه",type="primary")
    if go:
        with st.spinner("در حال محاسبه شمارش واقعی..."):
            st.session_state["bonus_gap_result"]=audit(None if cat=="همه دسته‌ها" else cat,min_labels=n,min_rating=rating,gap=gap)
    r=st.session_state.get("bonus_gap_result") or (report("quality_gap") or {}).get("report")
    if not r:return
    a,b,c=st.columns(3)
    a.metric("پرچم روش ساده",number(r["naive_flags"]))
    b.metric("شواهد ناکافی در روش ساده",number(r["naive_flags_insufficient_support"]))
    c.metric("پرچم محافظه‌کارانه",number(r["conservative_flags"]))
    st.caption("آستانه نتیجه نمایش‌داده‌شده: "+str(r["thresholds"]))
    for item in r["items"]:
        with st.expander(f"{item['title']} — شکاف {100*item['conservative_gap']:.1f} واحد درصد"):
            c1,c2,c3=st.columns(3)
            c1.metric("امتیاز محصول",f"{item['rating_100']:.0f}/100")
            c2.metric("نرخ پیشنهاد",pct(item["recommendation_rate"]))
            c3.metric("نظر برچسب‌دار",number(item["valid_labels"]))
            st.caption(f"کران بالای Wilson: {pct(item['wilson_upper'])}")
            for e in item.get("evidence",[]):st.write(f"نظر {e['comment_id']}: {e['text']}")
            if open_review is not None:
                st.button("بررسی کامل نظرها",key="gap_"+str(item["product_id"]),on_click=open_review,args=(item["product_id"],item["title"]))
    st.warning("تأیید منتور و اعتبارسنجی سودمندی مدیریتی هنوز لازم است؛ پرچم به معنی اثبات نقص کالا نیست.")


def defense_panel():
    st.subheader("مستندات دفاع و فایل‌های ارزیابی")
    file=REPORT/"defense_fa.md"
    if file.is_file():st.markdown(file.read_text(encoding="utf8"))
    for name in ("retrieval.json","adaptation.json","hyperparameter_search/report.json",
                 "hyperparameter_search/trials.csv","router_cache.json","qa_graph_ab.json","professional_policy.json","quality_gap.json"):
        p=REPORT/name
        if p.is_file():
            mime="text/csv" if p.suffix==".csv" else "application/json"
            st.download_button(name,p.read_bytes(),p.name,mime,key="defense_"+name)

    for name,mime in (("report.json","application/json"),("trials.csv","text/csv")):
        p=ROOT/"m2_outputs/xgboost"/name
        if p.is_file():
            st.download_button(f"XGBoost: {name}",p.read_bytes(),f"xgboost_{name}",mime,key="defense_xgb_"+name)

    for name,mime in (("report.json","application/json"),("model_comparison.csv","text/csv")):
        p=ROOT/"reports/final_model"/name
        if p.is_file():
            st.download_button("مدل نهایی: "+name,p.read_bytes(),f"final_{name}",mime,key="defense_final_"+name)


def render_bonus(open_review=None):
    st.header("آزمایشگاه شواهد و امکانات امتیازی")
    st.caption("داده و قابلیت‌های اصلی حفظ شده‌اند؛ مدل فعال پیش‌بینی Head Fine-tuning مدل LR است. این بخش نتایج آزمایش‌ها و مسیر تصمیم را نمایش می‌دهد.")
    r=report("retrieval");m=report("adaptation")
    if r and m:
        group=r["groups"]["typo_stress"];p0=group["bm25"]["hit_at_5"];p1=group["hybrid_char"]["hit_at_5"]
        c1,c2,c3=st.columns(3)
        c1.metric("Hit@5 غلط تایپی",pct(p1),f"{100*(p1-p0):+.1f} pp")
        c2.metric("Macro F1 بهترین مدل",pct(m["test"]["adapted"]["macro_f1"]),f"{100*m['test']['delta_macro_f1']:+.2f} pp")
        c3.metric("دامنه اجرا","۱۰۰٬۰۰۰ محصول")
    names=("مسیریاب و مانیتورینگ","مقایسه بازیابی","مدل و آموزش تکمیلی","ارزیابی انسانی","تحلیل جدید","دفاع و گزارش")
    tabs=st.tabs(names,key="bonus_lab_tabs",on_change="rerun")
    functions=(lambda:router_panel(open_review),lambda:retrieval_panel(open_review),model_panel,human_panel,lambda:gap_panel(open_review),defense_panel)
    for tab,fn in zip(tabs,functions):
        if tab.open:
            with tab:
                try:fn()
                except Exception as exc:
                    st.error(f"این بخش اجرا نشد: {type(exc).__name__}. وجود داده، مدل و وابستگی‌ها را بررسی کنید.")
