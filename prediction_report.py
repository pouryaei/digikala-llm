"""Display the supplied M2 evaluation separately from individual predictions."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


OUTPUTS = Path(__file__).resolve().parent / "m2_outputs"
LABELS = ("no_idea", "not_recommended", "recommended")


def render_prediction_report():
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
            check = json.loads((OUTPUTS / "integration_check.json").read_text(encoding="utf-8"))
            replay = check["deployment_replay"]
            st.caption(f"بازاجرای تابع اصلی روی عنوان و متن خامِ همان شناسه‌های آزمون: "
                       f"Macro F1 روی {int(replay['valid_count']):,} ورودی معتبر برابر "
                       f"{float(replay['macro_f1_on_valid']):.4f} است؛ "
                       f"{int(replay['rejected_count']):,} ورودی در اعتبارسنجی رد شدند. "
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
