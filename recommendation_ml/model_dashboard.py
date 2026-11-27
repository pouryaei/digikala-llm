"""Shared report views for the final dashboard and the model comparison lab."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/final_model"


def load_report():
    return json.loads((REPORT_DIR/"report.json").read_text(encoding="utf8"))


def selected(report):
    return next(row for row in report["models"] if row["id"] == report["selected_model"])


def comparison_frame(report):
    return pd.DataFrame([{
        "رتبه Validation":row["rank"],
        "مدل":row["name"],
        "منتخب":row["id"]==report["selected_model"],
        "Validation Macro F1":row["validation"]["macro_f1"],
        "Validation Accuracy":row["validation"]["accuracy"],
        "Test Macro F1":row["test"]["macro_f1"],
        "Test Accuracy":row["test"]["accuracy"],
        "Test Macro Precision":row["test"]["precision"],
        "Test Macro Recall":row["test"]["recall"],
    } for row in report["models"]])


def render_comparison(report, charts=True):
    df=comparison_frame(report)
    metrics=[c for c in df if c.startswith(("Validation ","Test "))]
    st.dataframe(df.assign(**{m:df[m].map(lambda x:f"{x:.2%}") for m in metrics}),
                 hide_index=True,width="stretch")
    if charts:
        st.caption("مقایسه Macro F1 روی Validation و Test ثابت")
        st.bar_chart(df.set_index("مدل")[["Validation Macro F1","Test Macro F1"]])
        st.caption("مقایسه Accuracy روی Test ثابت")
        st.bar_chart(df.set_index("مدل")[["Test Accuracy"]])
    st.caption("رتبه‌بندی فقط بر اساس Validation Macro F1 است. Test شامل همان ۱۵ هزار نظر ثبت‌شده است، "
               "نه آزمون کور جدید. اندازه و متن‌های آموزش تاریخی الزاماً با نمونه‌های جدید برابر نیستند.")


def render_class_metrics(row, labels):
    cr=row["classification_report"]
    st.dataframe(pd.DataFrame([{"Class":label,"Precision":cr[label]["precision"],
                               "Recall":cr[label]["recall"],"F1":cr[label]["f1-score"],
                               "Support":int(cr[label]["support"])} for label in labels]),
                 hide_index=True,width="stretch",column_config={
                     k:st.column_config.NumberColumn(format="%.4f") for k in ("Precision","Recall","F1")})
    st.caption("ماتریس خطا: سطر = برچسب واقعی، ستون = پیش‌بینی")
    st.dataframe(pd.DataFrame(row["confusion_matrix"],index=labels,columns=labels),width="stretch")


def render_final_prediction_report():
    with st.expander("نتایج ارزیابی بهترین مدل نهایی (Head Fine-tuning)"):
        try:
            result=load_report()
            row=selected(result)
        except (OSError,ValueError,KeyError,StopIteration):
            st.error("گزارش معتبر انتخاب مدل نهایی در دسترس نیست.")
            return
        st.success("مدل فعال: "+row["name"])
        st.caption("Character TF-IDF ثابت + Head آموزش‌دیده Logistic Regression. انتخاب مدل فقط با Validation انجام شده است.")
        a,b,c=st.columns(3)
        a.metric("Macro F1",f"{row['test']['macro_f1']:.4f}")
        b.metric("Accuracy",f"{row['test']['accuracy']:.2%}")
        c.metric("تعداد نمونهٔ آزمون",f"{result['fixed_test_count']:,}")
        render_class_metrics(row,result["labels"])
        st.markdown("**مقایسه هر چهار مدل**")
        render_comparison(result,charts=False)
        st.caption("نمودارها، اجرای عملی چهار مدل و نتایج جست‌وجوی تنظیمات در زیرتب مدل آزمایشگاه امتیازی موجودند.")


def render_model_lab():
    import ml_service
    st.subheader("انتخاب بهترین مدل و مقایسه چهار مدل")
    try:
        report=load_report();best=selected(report)
    except (OSError,ValueError,KeyError,StopIteration):
        st.error("گزارش مقایسه چهار مدل یافت نشد.")
        return
    st.success("بهترین مدل منتخب: "+best["name"])
    base=next(row for row in report["models"] if row["id"]=="lr_original")
    a,b,c,d=st.columns(4)
    a.metric("مدل‌های مقایسه‌شده",str(len(report["models"])))
    b.metric("Validation Macro F1",f"{best['validation']['macro_f1']:.2%}",
             f"{100*(best['validation']['macro_f1']-base['validation']['macro_f1']):+.2f} pp")
    c.metric("Test Macro F1",f"{best['test']['macro_f1']:.2%}",
             f"{100*(best['test']['macro_f1']-base['test']['macro_f1']):+.2f} pp")
    d.metric("Test Accuracy",f"{best['test']['accuracy']:.2%}",
             f"{100*(best['test']['accuracy']-base['test']['accuracy']):+.2f} pp")
    st.caption("مدل انتخاب‌شده در پیش‌بینی اصلی و مسیریاب پیش‌فرض است. Logistic Regression اولیه، نسخه تنظیم‌شده و XGBoost برای مقایسه باقی مانده‌اند.")
    render_comparison(report)
    names={row["id"]:row["name"] for row in report["models"]}
    with st.expander("عملکرد کلاس‌ها و ماتریس خطای مدل موردنظر"):
        model_id=st.selectbox("مدل",list(names),format_func=names.get,key="final_compare_class_model")
        render_class_metrics(next(row for row in report["models"] if row["id"]==model_id),report["labels"])
    st.divider()
    st.subheader("آزمایش یک نظر توسط هر چهار مدل")
    with st.form("bonus_model_trial"):
        title=st.text_input("عنوان نظر (اختیاری)",key="final_compare_title")
        body=st.text_area("متن نظر",value="کیفیت خوب بود، اما قیمتش زیاد است.",key="final_compare_text")
        go=st.form_submit_button("مقایسه چهار مدل")
    if go:
        st.session_state.pop("bonus_models_final",None)
        try:
            outputs=[]
            for row in report["models"]:
                started=perf_counter()
                ans=ml_service.predict_model(row["id"],body,title)
                outputs.append({"id":row["id"],"name":row["name"],"result":ans,
                                "elapsed_ms":(perf_counter()-started)*1000})
            st.session_state["bonus_models_final"]=outputs
        except Exception as exc:
            st.error(f"بارگذاری یا اجرای مدل‌ها ناموفق: {type(exc).__name__}")
    outputs=st.session_state.get("bonus_models_final")
    if outputs:
        records=[]; chart=[]
        for item in outputs:
            ans=item["result"];proba=ans.get("probabilities",{})
            records.append({"مدل":item["name"],"برچسب":ans["label"] if ans["valid"] else "ورودی نامعتبر",
                "اطمینان":f"{ans['confidence']:.2%}" if ans["valid"] else "—",
                **{k:f"{proba.get(k,0):.2%}" if ans["valid"] else "—" for k in report["labels"]},
                "زمان (ms)":round(item["elapsed_ms"],2)})
            if ans["valid"]:chart.append({"مدل":item["name"],**proba})
        st.dataframe(pd.DataFrame(records),hide_index=True,width="stretch")
        if chart:st.bar_chart(pd.DataFrame(chart).set_index("مدل"))
        st.caption("احتمال‌های پیش‌بینی، دقت آزمون نیستند؛ زمان اجرای نخست شامل بارگذاری مدل است.")
    with st.expander("روش انتخاب، اندازه داده و کنترل هم‌پوشانی"):
        st.json({"selection_rule":report["selection_rule"],"audit":report["audit"],
                 "test_used_for_selection":report["test_used_for_selection"],
                 "limitations":report["limitations"]})
    for name,mime in (("model_comparison.csv","text/csv"),("report.json","application/json")):
        file=REPORT_DIR/name
        st.download_button("دریافت "+name,file.read_bytes(),f"final_{name}",mime,key="final_model_"+name)
