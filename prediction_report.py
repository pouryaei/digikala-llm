"""Display the supplied M2 evaluation separately from individual predictions."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


OUTPUTS = Path(__file__).resolve().parent / "m2_outputs"
LABELS = ("no_idea", "not_recommended", "recommended")


def _render_xgboost_report(path):
    """Report the active model, not the historical LR's metrics."""
    path = Path(path)
    if path.is_dir():
        path = path / "report.json"
    with st.expander("نتایج ارزیابی مدل فعال XGBoost"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            metrics = report["test"]
            counts = report["audit"]["splits"]
            class_report = report["test_classification_report"]
            labels = report["labels"]
            params = report["best_params"]
            if report["model"] != "XGBoost" or tuple(labels) != LABELS:
                raise ValueError("Invalid active model report")
        except (OSError, ValueError, TypeError, KeyError):
            st.error("گزارش مدل XGBoost در دسترس نیست؛ بسته کامل مدل را بررسی کنید.")
            return
        st.caption("مدل فعال: Character TF-IDF → انتخاب ویژگی chi2 → XGBoost. "
                   "انتخاب تنظیمات تنها با Macro F1 مجموعه Validation انجام شده است.")
        a, b, c = st.columns(3)
        a.metric("Macro F1", f"{metrics['macro_f1']:.4f}")
        b.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        c.metric("تعداد نمونهٔ آزمون", f"{counts['test']['rows']:,}")
        st.caption(f"max_depth={params['max_depth']} | learning_rate={params['learning_rate']} "
                   f"| n_estimators={params['n_estimators']} — تعداد ترکیب‌ها: {report['n_trials']}")
        st.dataframe(pd.DataFrame([
            {"Class": label, "Precision": float(class_report[label]["precision"]),
             "Recall": float(class_report[label]["recall"]), "F1": float(class_report[label]["f1-score"]),
             "Support": int(class_report[label]["support"])}
            for label in labels
        ]), hide_index=True, width="stretch", column_config={
            key: st.column_config.NumberColumn(format="%.4f") for key in ("Precision", "Recall", "F1")
        })
        matrix = pd.DataFrame(report["test_confusion_matrix"]["matrix"], index=labels, columns=labels)
        st.caption("ماتریس خطای Test مدل XGBoost؛ سطر = برچسب واقعی، ستون = پیش‌بینی.")
        st.dataframe(matrix, width="stretch")
        if report.get("recorded_test_reference"):
            previous = report["recorded_test_reference"]["original"]
            st.caption(f"مدل Logistic Regression تاریخی روی همان شناسه‌های Test: "
                       f"Macro F1 = {previous['macro_f1']:.2%} | Accuracy = {previous['accuracy']:.2%}. "
                       "اندازه آموزش دو مدل الزاماً برابر نیست.")
        st.caption("جدول کامل Hyperparameter Search در آزمایشگاه شواهد و امتیازی موجود است.")


def render_prediction_report():
    import ml_service
    if ml_service.active_model_kind() == "lr_adapted":
        from recommendation_ml.model_dashboard import render_final_prediction_report
        render_final_prediction_report()
        return
    if ml_service.is_xgboost_active():
        _render_xgboost_report(ml_service.MODEL_PATH)
        return

    with st.expander("نتایج ارزیابی مدل تحویلی هانیه"):
        try:
            metrics = json.loads((OUTPUTS / "test_results.json").read_text(encoding="utf-8-sig"))
            report = json.loads((OUTPUTS / "classification_report.json").read_text(encoding="utf-8-sig"))
            size = int(report["macro avg"]["support"])
            accuracy = float(metrics["accuracy"])
            macro_f1 = float(metrics["macro_f1"])
            if size < 1 or not 0 <= accuracy <= 1 or not 0 <= macro_f1 <= 1:
                raise ValueError("Invalid test metrics")
            rows = [
                {"Class": label, "Precision": float(report[label]["precision"]),
                 "Recall": float(report[label]["recall"]), "F1": float(report[label]["f1-score"]),
                 "Support": int(report[label]["support"])}
                for label in LABELS
            ]
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            st.info("فایل‌های معتبر نتیجهٔ آزمون مدل در دسترس نیستند.")
            return

        st.caption("نتایج ذخیره‌شدهٔ مدل TF-IDF کاراکتری و Logistic Regression در مجموعهٔ آزمون؛ "
                   "این اعداد با احتمال یک پیش‌بینی جدید متفاوت‌اند.")
        f1_col, acc_col, count_col = st.columns(3)
        f1_col.metric("Macro F1", f"{macro_f1:.4f}")
        acc_col.metric("Accuracy", f"{accuracy:.2%}")
        count_col.metric("تعداد نمونهٔ آزمون", f"{size:,}")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            key: st.column_config.NumberColumn(format="%.4f") for key in ("Precision", "Recall", "F1")
        })
        matrix = OUTPUTS / "confusion_matrix.png"
        if matrix.is_file():
            st.image(str(matrix), caption="ماتریس خطای مجموعهٔ آزمونِ مدل تحویلی", width="stretch")

        try:
            current = json.loads((OUTPUTS.parent / "reports/model_replay_complete.json").read_text(encoding="utf-8"))
            replay = current["service_replay"]
            whole = current["model_replay"]
            st.caption(f"بازاجرای مدل با قالب آموزش روی {int(whole['n']):,} نظر واقعی: "
                       f"{int(whole['same_as_recorded']):,} پیش‌بینی منطبق. "
                       f"Macro F1 سرویس اصلاح‌شده روی {int(replay['valid']):,} ورودی معتبر: "
                       f"{float(replay['macro_f1_valid']):.4f}؛ "
                       f"{int(replay['rejected']):,} ورودی نامعتبر جداگانه رد شدند.")
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            try:
                check = json.loads((OUTPUTS / "integration_check.json").read_text(encoding="utf-8"))
                replay = check["deployment_replay"]
                st.caption(f"بازاجرای تاریخی قبل از اصلاح قالب متن: "
                           f"Macro F1 روی {int(replay['valid_count']):,} ورودی معتبر "
                           f"{float(replay['macro_f1_on_valid']):.4f} بود. "
                           "این بازاجرا جایگزین گزارش اولیهٔ آزمون نیست.")
            except (OSError, ValueError, TypeError, KeyError, OverflowError):
                pass

        comparison = OUTPUTS / "model_comparison.csv"
        if comparison.is_file():
            try:
                data = pd.read_csv(comparison)
                st.caption("مقایسهٔ مدل‌ها از فایل ارسالی، جدا از گزارش نهایی آزمون")
                st.dataframe(data, hide_index=True, width="stretch")
            except (OSError, ValueError, pd.errors.ParserError):
                st.info("جدول مقایسهٔ مدل‌ها قابل خواندن نیست.")
