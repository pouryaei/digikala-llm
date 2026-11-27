"""Build editable Persian HTML report and presentation from the measured JSON."""
from pathlib import Path
import ast
import html
import json

DIR=Path(__file__).resolve().parent
ROOT=DIR.parent
r=json.loads((DIR/'results/final/evaluation.json').read_text(encoding='utf-8'))
b=json.loads((DIR/'results/baseline_core.json').read_text(encoding='utf-8'))
ab=json.loads((DIR/'results/search_ablation.json').read_text(encoding='utf-8'))
s=r["search_summary"]["all"];bs=b["search_summary"]["all"];q=r["qa_summary"]["all"];bq=b["qa_summary"]["all"]
a=r["answer_quality"];m=r["prediction"];mt=m["recorded_test"];mr=m["replay_metrics_valid_only"]
lookup={(x["scenario"],x["cache_state"]):x for x in r["latency"]["summary"]}
qa={x["case_id"]:x for x in r["qa"]};search={x["case_id"]:x for x in r["search"]}
sbase={x["case_id"]:x for x in b["search"]};qbase={x["case_id"]:x for x in b["qa"]}
FONT=next(ast.literal_eval(x.value) for x in ast.parse((ROOT/'app.py').read_text(encoding='utf-8')).body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="FONT_WOFF2" for t in x.targets))
e=html.escape
def ltr(x):return '<bdi dir="ltr" class="ltr">'+e(str(x))+'</bdi>'
def num(x,d=0):return ltr(f"{x:,.{d}f}")
def ratio(x):return ltr(f"{100*x:.1f}%") if x is not None else "—"
def score(x,d=3):return ltr(f"{x:.{d}f}") if x is not None else "—"
def table(headers,rows,cls=""):
 return '<table class="'+cls+'"><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'
def para(x):return '<p>'+x+'</p>'
def sub(x):return '<h3>'+x+'</h3>'
def bullet(xs):return '<ul>'+''.join('<li>'+x+'</li>' for x in xs)+'</ul>'
def note(x):return '<div class="note">'+x+'</div>'
def big(value,label):return '<div class="metric"><div class="value">'+value+'</div><div>'+label+'</div></div>'
def metrics(*xs):return '<div class="metrics">'+''.join(xs)+'</div>'
def comparison_rows():
 return [
  [ltr("Hit@5")+" شناسه‌محور",ratio(bs["known_hit_at_5"]),ratio(s["known_hit_at_5"])],
  [ltr("MRR@5")+" شناسه‌محور",score(bs["known_mrr_at_5"]),score(s["known_mrr_at_5"])],
  [ltr("Precision@5")+" کشف",ratio(bs["discovery_p_at_5"]),ratio(s["discovery_p_at_5"])],
  ["دقت نتایج واقعاً برگشتی (کشف)",ratio(bs["discovery_precision_returned_micro"]),ratio(s["discovery_precision_returned_micro"])],
  ["درخواست بدون مصداق؛ خروجی خالی",f'{bs["correct_no_result"]} از {bs["no_result_cases"]}',f'{s["correct_no_result"]} از {s["no_result_cases"]}'],
  ["دقت سخت‌گیرانه نظرهای برگشتی",ratio(bq["review_strict_precision_micro"]),ratio(q["review_strict_precision_micro"])],
  [ltr("Recall@8")+" نظر مستقیم",ratio(bq["review_macro_recall_at_8"]),ratio(q["review_macro_recall_at_8"])],
  [ltr("nDCG@8")+" نظر",score(bq["review_macro_ndcg_at_8"]),score(q["review_macro_ndcg_at_8"])],
  ["تشخیص پاسخ‌پذیری / امتناع",f'{int(bq["status_accuracy"]*bq["cases"]+.1)}/{bq["cases"]}',f'{int(q["status_accuracy"]*q["cases"]+.1)}/{q["cases"]}']
 ]
def latrow(key,label,state="warm_process"):
 x=lookup[(key,state)];return [label,"گرم" if state=="warm_process" else "سرد",str(x["n"]),num(x["median_ms"],2),num(x["p95_ms"],2)]
def mlrows():
 return [[ltr(c),num(mt["report"][c]["support"]),score(mt["report"][c]["precision"],4),score(mt["report"][c]["recall"],4),score(mt["report"][c]["f1-score"],4),score(mr["report"][c]["f1-score"],4)] for c in mt["confusion_matrix_labels"]]
def confmatrix():
 return table(["واقعی ↓ / پیش‌بینی",*[ltr(x) for x in mt["confusion_matrix_labels"]]],[[ltr(c),*[num(n) for n in row]] for c,row in zip(mt["confusion_matrix_labels"],mt["confusion_matrix"])],"compact")
def quotefail(qid,cid):
 row=qa[qid]
 return e(next(x["text"] for x in row["result"]["evidence"] if x["comment_id"]==cid))
def qa_count(cid,report):
 x=next(z for z in report["qa"] if z["case_id"]==cid)
 return x["metrics"]["direct"]
css_font='@font-face{font-family:Vazirmatn;src:url(data:font/woff2;base64,'+FONT+') format("woff2");font-weight:100 900;font-style:normal;font-display:swap}'
base_css=css_font+'''
:root{--navy:#152f6a;--blue:#315bb7;--pink:#b03986;--ink:#19294f;--muted:#485979;--line:#dce2f3;--paper:#fffafe;--light:#f5f5ff}
*{box-sizing:border-box}body{margin:0;background:#e9edf7;font-family:Vazirmatn,Tahoma,Arial,sans-serif;color:var(--ink);direction:rtl;line-height:1.65}
h1,h2,h3,p{margin:0}h1,h2{color:var(--navy);line-height:1.45}h3{color:var(--pink);line-height:1.35}
p{margin:.38em 0}ul{margin:.35em 0;padding-right:1.25em}li{margin:.24em 0}
.ltr,code,bdi{direction:ltr;unicode-bidi:isolate;font-feature-settings:"tnum";white-space:nowrap}code{font-family:Consolas,monospace;font-size:.88em;color:var(--blue);background:#edf1ff;padding:1px 4px;border-radius:3px}
table{width:100%;border-collapse:collapse;margin:.55em 0;line-height:1.43;font-size:.9em;table-layout:auto}
thead th{background:var(--navy);color:white;text-align:right;font-weight:600;padding:.46em .55em}
td{padding:.43em .55em;border-bottom:1px solid var(--line);vertical-align:top}tr:nth-child(even) td{background:#f5f5ff}
td:not(:first-child){text-align:center}.wide td:not(:first-child){text-align:right}
.note{margin:.65em 0;padding:.65em .85em;background:#edf1ff;border-right:4px solid var(--blue);border-radius:4px;color:#34486e;font-size:.91em}
.metrics{display:flex;align-items:stretch;gap:1em;margin:.9em 0 1em;justify-content:space-between}
.metric{flex:1;border-top:3px solid var(--pink);padding:.45em 0;color:var(--muted);font-size:.9em}
.value{color:var(--navy);font-weight:850;line-height:1.3;font-size:2.1em;margin-bottom:.12em}
.kicker{font-size:.83em;color:var(--pink);font-weight:750;letter-spacing:.02em}
footer{position:absolute;bottom:12mm;right:15mm;left:15mm;display:flex;justify-content:space-between;align-items:center;border-top:1px solid var(--line);padding-top:2mm;color:var(--muted);font-size:.76em}
@media print{body{background:white}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
'''
pages=[]
def add(title,body,kicker="بخش چهارم | ارزیابی سیستم نهایی"):
 pages.append((title,body,kicker))
add("ارزیابی سیستم نهایی فروهر",
    '<p class="lede">نتایج اجرای واقعی نسخهٔ ۱۰۰ هزار محصولی، مقایسهٔ پیش و پس از اصلاح و محدودیت‌های قابل دفاع</p>'+
    metrics(big(num(100000),"محصول واقعی"),big(num(1874457),"نظر واقعی"),big(num(94),"سناریوی اصلی"))+
    sub("خلاصهٔ نتیجه")+
    table(["معیار","یافتهٔ نسخهٔ نهایی"],[
    ["کیفیت پاسخ",f'{a["quality_full"]} پاسخ کامل، {a["quality_partial"]} جزئی و {a["quality_failed"]} ناموفق از ۲۹ پرسش؛ شامل ۳ امتناع درست'],
    ["Grounding",f'{q["exact_scoped_evidence"]}/{q["evidence_total"]} شاهد عیناً در منبع همان محصول؛ {a["semantic_grounding_full"]}/۲۹ ارزیابی معنایی کامل'],
    ["Retrieval Quality",f'جست‌وجوی شناسه‌محور: {ratio(s["known_hit_at_5"])}؛ بازیابی نظر مستقیم: {ratio(q["review_macro_recall_at_8"])}'],
    ["Recommendation Prediction",ltr("Macro F1")+" "+score(mt["macro_f1"],4)+" در آزمون ۱۵ هزار نظری؛ "+score(mr["macro_f1"],4)+" در بازاجرای نظرهای مشترک"],
    ["Latency",f'جست‌وجوی گرم: {num(lookup[("search_plain","warm_process")]["median_ms"],1)} ms؛ پاسخ ۲۰۰ نظر: {num(lookup[("qa_200_reviews","warm_process")]["median_ms"],1)} ms (میانه)'],
    ["Cost",ltr("$0")+" هزینهٔ API و صفر Token در حالت محلی"],
    ["Failure Analysis","درخواست خارج از دامنه، بازیابی ناقص عبارت فارسی، ابهام معنایی و خطای برچسب مدل"]
    ],"wide")+
    note("دامنهٔ نتیجه: حالت "+ltr("BM25 + extractive + sklearn + SQL")+"؛ نه ارزیابی مدل زبانی آنلاین یا کل دیتاست دیجی‌کالا.")+
    sub("روش و دادهٔ مرجع")+
    para("محصولات و نظرات از زیرمجموعهٔ هماهنگ نسخهٔ نهایی (کاتالوگ SQLite، نمایهٔ BM25 و فایل‌های Parquet) هستند. جست‌وجو و پاسخ‌دهی با همان مسیرهای سرویس رابط Streamlit اجرا شدند. همهٔ داده‌ها، کد ارزیابی و نتایج خام داخل پوشهٔ evaluation هستند.")+
    para("تاریخ اجرا: ۲۶ نوامبر ۲۰۲۶. مجموعهٔ مرجع: "+ltr("cases.json")+" و "+ltr("queries.json")+"."))
add("طراحی مجموعهٔ ارزیابی",
    table(["بخش","تعداد","مرجع و معیار"],[
    ["جست‌وجوی محصول", "۴۷", "۲۰ پرسش مشتق از عنوان، ۳ شناسه‌محور جدید، ۱۸ کشفِ دارای مصداق و ۶ درخواست بدون مصداق"],
    ["پرسش از نظرات","۲۹","۱۴ محصول و ۲۷۵ زوج پرسش-نظر؛ بررسی تمام نظرهای هر محصول انتخاب‌شده"],
    ["حالت مرزی و حجم بالاتر","۷","۴ حالت انتخاب محصول/شناسه و ۳ پرسش با ۱۵۷ تا ۲۰۰ نظر (صحت ساختاری)"],
    ["مقایسهٔ محصول","۳","۲، ۳ و ۵ محصول واقعی؛ صحت مشخصات و شواهد"],
    ["مدیر فروشگاه","۸","چهار نوع پرسش در دو دستهٔ کفش مردانه و کتاب شعر و ادبیات"],
    ["پیش‌بینی","۱۵٬۰۰۰ / ۴٬۵۳۷","آزمون ثبت‌شدهٔ مدل و بازاجرای نظرهای مشترک با دادهٔ بسته"],
    ["زمان پاسخ","۷۹","۱۴ سناریوی گرم × ۵ تکرار؛ ۳ سناریوی سرد × ۳ تکرار"]
    ],"wide")+
    sub("برچسب‌های مستقلِ مرجع")+
    para("جست‌وجوی کشف بر اساس شرایط ثابت عنوان، دسته، برند و قیمت ثبت‌شده، در تمام ۱۰۰ هزار محصول برچسب‌گذاری می‌شود. این معیار خودکار محافظه‌کارانه است و همهٔ ترجیح‌های ضمنی کاربر را پوشش نمی‌دهد.")+
    para("برای ۲۹ پرسش، هر نظر کامل بر اساس رابطه با سؤال نمرهٔ ۲ (مستقیم)، ۱ (زمینهٔ مرتبط) یا ۰ (نامرتبط) گرفت. به‌جای انتخاب چند خروجی موفق، همهٔ نظرهای محصولات انتخاب‌شده بررسی شدند. سپس خود نقل‌قول‌های نمایش داده‌شده جداگانه امتیاز گرفتند.")+
    note("تقسیم: ۱۵ پرسش توسعه و ۱۴ پرسش محصولات دیگر؛ در جست‌وجو ۱۳ توسعه، ۱۴ بخش کنارگذاشته و ۲۰ عنوان‌محور تاریخی. نمونه‌ها هدفمند و محدودند. بعضی خروجی‌های بخش کنارگذاشته پس از اجرای اولیه دیده شدند؛ بنابراین نتایج پیش/پس، تشخیصی هستند و آزمون کاملاً نادیدهٔ آینده‌نگر نیستند.")+
    sub("روش محاسبه")+
    bullet([ltr("Hit@5")+" = سهم پرسش‌هایی که هدف حداقل یک بار در پنج نتیجه دیده شد؛ "+ltr("MRR@5")+" = میانگین ۱/رتبهٔ نخستین هدف (صفر در صورت نبود).",
    ltr("Precision@5")+" = تعداد مورد مرتبط تقسیم بر ۵؛ دقت نتایج برگشتی مخرج تعداد واقعاً برگشتی دارد. "+ltr("Recall@8")+" = سهم نظرهای مستقیم مرجع که در ۸ شاهد آمده‌اند.",
    ltr("nDCG@k")+" = سود رتبه‌ای بر پایهٔ نمرهٔ ۰/۱/۲، نرمال‌شده نسبت به بهترین رتبه‌بندی ممکن."]))
add("کیفیت جست‌وجو و بازیابی",
    metrics(big(ratio(s["known_hit_at_5"]),"Hit@5؛ ۲۳ پرسش"),big(ratio(s["discovery_p_at_5"]),"Precision@5؛ ۱۸ کشف"),big(ratio(q["review_macro_recall_at_8"]),"Recall@8؛ ۲۶ پرسش پاسخ‌پذیر"))+
    table(["شاخص","قبل","بعد"],comparison_rows())+
    sub("تفسیر نتیجه")+
    para("اصلاح واژه‌های عمومی و حداقل پوشش واژگانی باعث کاهش نتایج نامرتبط شد. در آزمون میانی، شرط ۶۰٪ به‌تنهایی جست‌وجوی هدفون مانیتورینگ را به اشتباه خالی کرد. افزودن مسیر جایگزین برای دو واژهٔ آغازینِ حاضر در عنوان، سه نتیجهٔ درست را بازگرداند. در کشف محصول، "+ltr("Precision@5")+" ثابت ماند ولی سهم مرتبط بین نتایج برگشتی بهتر شد؛ سیستم به‌جای پر کردن اجباری پنج جایگاه، گاهی نتایج کمتر برمی‌گرداند.")+
    para("در بازیابی نظر، اضافه کردن صورت‌های محاوره‌ای فارسی و کاهش واژه‌های عمومی، بازیابی مستقیم از "+ratio(bq["review_macro_recall_at_8"])+" به "+ratio(q["review_macro_recall_at_8"])+" رسید. هر ۲۶ پرسش پاسخ‌پذیر حداقل یک نظر مستقیم دارند؛ نبود شاهد در ۳ پرسش درست اعلام شد.")+
    note("بهبود دقت نسبت به نتایج برگشتی به معنی افزایش تعداد همهٔ محصولات مرتبط نیست. "+ltr("Dense/Hybrid")+" روی این ۱۰۰ هزار محصول نمایهٔ آماده ندارد و در این گزارش اندازه‌گیری نشده است."))
add("کیفیت پاسخ و صحت ادعاها",
    metrics(big(ltr(f'{a["quality_mean_0_to_2"]:.2f} / 2'),"میانگین کیفیت پاسخ"),big(ltr(f'{a["semantic_grounding_mean_0_to_2"]:.2f} / 2'),"میانگین Grounding معنایی"),big(f'{q["exact_scoped_evidence"]}/{q["evidence_total"]}',"شاهد عینی و هم‌محصول"))+
    table(["داوری موردبه‌مورد","کامل (۲)","جزئی (۱)","ناموفق (۰)"],[
    ["کیفیت پاسخ (۲۹ پرسش)",a["quality_full"],a["quality_partial"],a["quality_failed"]],
    ["Grounding معنایی (۲۹ پرسش)",a["semantic_grounding_full"],a["semantic_grounding_partial"],a["semantic_grounding_failed"]],
    ["ارتباط نقل‌قول (۸۵ شاهد)",a["excerpt_grade2"],a["excerpt_grade1"],a["excerpt_count"]-a["excerpt_grade2"]-a["excerpt_grade1"]]
    ])+
    para("از ۲۶ پرسش پاسخ‌پذیر، ۱۵ کامل، ۱۰ جزئی و ۱ ناموفق است؛ سه امتناع درست جدا گزارش شدند. دقت ارتباطِ سخت‌گیرانهٔ نظر کامل "+ratio(q["review_strict_precision_micro"])+" است، اما خودِ نقل‌قول فقط "+ratio(a["excerpt_grade2"]/a["excerpt_count"])+" مستقیم است. انتخاب بخش اشتباه یک نظر می‌تواند نتیجه را ضعیف کند.")+
    sub("کنترل مستقل صحت ساختاری")+
    bullet([f'هر {q["grounding_structural_pass"]} پرسش از ۲۹، شناسهٔ معتبر، عدم اختلاط محصول، نقل‌قول اصلی و تطابق تعداد و نرخ برچسب را گذراند. {q["quotes_exact"]} نقل در متن پاسخ از {q["quote_total"]}، عیناً در منبع هستند.',
        '۳ آزمون حجم بالاتر: ۲۴/۲۴ شاهد واقعی. مقایسهٔ ۲، ۳ و ۵ محصول: ۵۰/۵۰ شاهد هم‌محصول و تطابق مشخصات/تعدادها. هر ۴ حالت مرزی انتخاب محصول درست.',
        '۸ پرسش مدیر در دو دسته: ۱۶۳/۱۶۳ نقل در متن پاک‌شدهٔ همان نظر و کنترل دامنه/نرخ صحیح. شمارش موضوع‌های شکایت فقط از نظر کران و مخرج بررسی شده؛ صحت معنایی همهٔ موضوع‌ها اثبات نشده.'])+
    sub("روش داوری و محدودیت")+
    para("ChatGPT هنگام تهیهٔ گزارش، پاسخ‌ها و نقل‌های ۲۹ سؤال را با متن مرجع ارزیابی کرد. این داوری مستقل انسانی یا API داور نیست. Rubric و دلیل هر نمره در "+ltr("judge_rubric_fa.md")+" و "+ltr("answer_assessments.json")+" ثبت شده؛ اگر پاسخ عوض شود نمره به آن منتقل نمی‌شود.")+
    note("واقعی بودن نقل‌قول و درست بودن شناسه، تضمین درستی تفسیر معنایی نیست. برای دفاع همین دو عدد باید جدا بیان شوند."))
add("پیش‌بینی recommendation_status",
    metrics(big(score(mt["macro_f1"],4),"Macro F1؛ آزمون ۱۵٬۰۰۰"),big(score(mr["macro_f1"],4),"Macro F1؛ بازاجرا"),big(score(mt["accuracy"],4),"Accuracy آزمون ثبت‌شده"))+
    para("مدل ذخیره‌شدهٔ LR با ویژگی‌های char TF-IDF اجرا شد. از ۱۵٬۰۰۰ ردیف فایل پیش‌بینی ثبت‌شده، ۴٬۵۳۷ شناسه در دادهٔ حاضر یافت شد؛ برچسب‌ها تطبیق داشتند. ۴٬۵۳۶ متن معتبر با عنوان و بدنهٔ خام دوباره به مدل داده شدند و ۱ مورد رد شد. F1 با احتساب رد به‌عنوان خطا نیز "+score(m["replay_macro_f1_with_rejections_counted_as_misses"],4)+" است.")+
    table(["کلاس","تعداد آزمون","Precision","Recall","F1 آزمون","F1 بازاجرا"],mlrows(),"compact")+
    sub("Confusion Matrix آزمون اصلی (۱۵٬۰۰۰ نظر)")+
    confmatrix()+
    para("ماتریس از فایل آزمون واقعی دوباره محاسبه شد. مدل اکثریت با پیش‌بینی همیشگی "+ltr(m["majority_class"])+" فقط "+score(m["majority_recorded_baseline"]["macro_f1"],4)+" Macro F1 دارد. کلاس "+ltr("no_idea")+" ضعیف‌ترین کلاس است؛ تعداد زیادی نظر مثبت به آن نسبت داده شده‌اند.")+
    sub("آزمایش تحلیل اطمینان، نه تغییر مدل")+
    table(["حداقل confidence","پوشش از ۴۵۳۷","Macro F1 پذیرفته‌ها"],[[score(x["threshold"],1),ratio(x["coverage_of_matched"]),score(x["macro_f1_on_accepted"],4)] for x in m["confidence_tradeoff"]])+
    note("F1 روی پذیرفته‌شده‌ها قابل مقایسهٔ مستقیم با F1 روی تمام ورودی‌ها نیست؛ افزایش آن با حذف نمونه‌های دشوار همراه است. هیچ آستانه‌ای روی آزمون تنظیم نشده و مدل آموزش مجدد ندیده است. دادهٔ دقیق آموزش/تقسیم اولیه داخل این بسته نیست."))
add("زمان پاسخ و هزینه",
    para("اندازه‌گیری مستقیم مسیر سرویس در Python، بدون زمان رندر مرورگر و شبکه. برای هر سناریوی گرم یک درخواست آماده‌سازی کنار گذاشته شد و ۵ تکرار ثبت شد. برای سرد سه فرآیند تازه اجرا شد (شامل بارگذاری وابستگی و مدل/نمایه؛ کش سیستم‌عامل پاک نشد). P95 با روش nearest-rank؛ در نمونهٔ اندک اغلب بیشینه است.")+
    table(["سناریو","حالت","N","Median (ms)","P95 (ms)"],[
       latrow("search_plain","جست‌وجوی ساده"),latrow("search_filter","جست‌وجوی فیلتر دار"),latrow("search_empty","درخواست بی‌نتیجه"),
       latrow("qa_6_reviews","پاسخ از ۶ نظر"),latrow("qa_200_reviews","پاسخ از ۲۰۰ نظر"),latrow("qa_no_evidence","پاسخ بدون شاهد"),
       latrow("compare_2","مقایسه ۲ محصول"),latrow("compare_5","مقایسه ۵ محصول"),
       latrow("predict_short","پیش‌بینی متن کوتاه"),latrow("predict_long","پیش‌بینی متن بلند"),
       latrow("manager_complaints","شکایت دسته کفش مردانه"),latrow("manager_low_recommendation","مدیر: نرخ پیشنهاد پایین"),
       latrow("manager_two_products","شکایت ۲ محصول"),latrow("analytics_distribution","توزیع دسته‌ها"),
       latrow("search_plain","جست‌وجوی ساده","cold_process"),latrow("qa_200_reviews","پاسخ از ۲۰۰ نظر","cold_process"),latrow("predict_short","پیش‌بینی متن کوتاه","cold_process")
    ],"latency")+
    sub("Cost و Token")+
    metrics(big(ltr("$0.00"),"هزینهٔ API"),big("0","Prompt Token"),big("0","Completion Token"))+
    para("هر ۹۴ سناریوی اصلی و تکرارهای زمانی/مدل محلی اجرا شدند؛ فراخوانی API و تلاش شبکه‌ای صفر است. مصرف CPU/RAM، برق و ذخیره‌سازی در هزینهٔ API لحاظ نشده. عملکرد درگاه LLM زنده به‌دلیل استفاده از حالت محلی اندازه‌گیری نشده است.")+
    para("برای اجرای آتی با درگاه: "+'<bdi dir="ltr" style="white-space:normal">Cost = (input_tokens × input_rate + output_tokens × output_rate) / 1,000,000</bdi>'+"؛ نرخ واقعی دلاری به ازای یک میلیون Token را از درگاه تیم وارد کنید. در صورت Token نامعلوم یا retry، هزینه کامل نامعلوم است. اسکریپت "+ltr("cost_from_usage.py")+" آن را صفر فرض نمی‌کند."))
add("تحلیل شکست و اصلاح‌های انجام‌شده",
    sub("۱. جست‌وجوی خارج از دامنه (S10، S11، S22)")+
    para("پرسش «بلیط هواپیما از تهران به پاریس»: کاتالوگ مرجع هیچ مصداقی ندارد، اما قبل از اصلاح «"+e(sbase["S10"]["results"][0]["title"])+"» برگرداند. جست‌وجو صرفاً واژه‌هایی مانند پاریس را تطبیق می‌داد. پس از حذف واژه‌های عمومی و شرط پوشش، خروجی خالی شد. از ۶ نمونهٔ بی‌مصداق، "+ltr("3/6 → 6/6")+".")+
    sub("۲. امتناع اشتباه در نسخهٔ میانی (S05)")+
    para("شرط ۶۰٪ به‌تنهایی، درخواست هدفون مانیتورینگ برای میکس را رد کرد؛ واژه‌های توصیف‌کننده هدف استفاده در عنوان کالا نیامده بودند. قاعدهٔ عنوان دوواژه‌ای، ۳ نتیجهٔ صحیح را بازگرداند. "+ltr("Precision@5")+" کشف "+ratio(s["discovery_p_at_5"])+" ماند؛ دقت نتایج واقعاً برگشتی "+ratio(bs["discovery_precision_returned_micro"])+" به "+ratio(s["discovery_precision_returned_micro"])+" رسید.")+
    sub("۳. صورت‌های محاوره‌ای فارسی (QA03، QA04، QA10، QA19)")+
    para("در سؤال نرم شدن دست، قبل فقط یک نظر مستقیم و بعد سه نظر مستقیم ارائه شد؛ نمونه نظر "+ltr("51816235")+" می‌گوید «دستهام خیلی نرم شدند». در پرسش نرم شدن مو، قبل پاسخ ناکافی و بعد چهار نظر مستقیم. در سؤال پارابن، شاهد عمومی «استفاده می‌کنم» حذف شد. با گسترش صورتِ مالکیتِ «جوهرش»، یک شاهد مستقیم دیگر پیدا شد.")+
    sub("۴. نفی و تعبیر شکایت (QA22)")+
    para("نظر "+ltr("35793980")+" در فیلد معایب نوشته بود «بد نبود». نسخهٔ اولیه آن را شکایت حساب کرد. اصلاح تشخیص نفی و بررسی آیتم‌های فهرست، این شاهد نامرتبط را از پرسش شکایت حذف کرد.")+
    note("تغییرات واقعی فقط در دو فایل "+ltr("src/bm25_index.py")+" و "+ltr("src/task2_evidence_selector.py")+" اعمال شدند. داده، نمایه، مدل ML و شش تب صورتی-آبی حفظ شدند. نسخهٔ اولیهٔ دو سیاست و نتایج پیش/پس ذخیره شده‌اند."))
add("شکست‌های باقی‌مانده و مسیر بهبود",
    table(["مورد واقعی","علت محتمل / اقدام بعدی"],[
    [ltr("S14")+"؛ گلس Galaxy Fit 2: ۶ مرجع، صفر نتیجهٔ مستقیم در ۵ رتبه. اولین خروجی Galaxy Active 2 است.","اختلاف فارسی/لاتین و مدل نزدیک. استخراج نام مدل و یکسان‌سازی واژه‌های دوزبانه، سپس آزمون روی پرسش‌های تازه."],
    [ltr("S19 / S24")+"؛ کوله مهد کودک یا هدفون بی‌سیم؛ نتیجهٔ مستقیم پیدا نشد.","نام‌های هم‌معنی و متن کوتاه کاتالوگ؛ ساخت Dense/Hybrid هم‌تراز و ارزیابی جدید. فعلاً انجام نشده."],
    [ltr("QA21")+"؛ لمس صفحه محافظ: فقط ۱ نقل مستقیم از ۳. موضوع «محتوا و چاپ» اشتباه است.","واژه چندمعنایی «صفحه»؛ تفکیک زمینهٔ کتاب و نمایشگر و رتبه‌بندی دقیق جنبهٔ سؤال."],
    [ltr("QA25")+"؛ وزن چای سفید: تنها «مغایرت مشخصات» از نظر "+ltr("53625458")+" نمایش داده شد.","متن همان نظر ۲۳ گرم و نظر 49732011 عدد ۲۲ گرم مقابل ۵۰ گرم را دارد. استخراج عدد و انتخاب بخش درستِ نظر، نه صرفاً عنوان."],
    [ltr("QA16")+"؛ شمارش منفی با ذکر خرابی کفش قبلی، در حالی که مدل فعلی چندماه سالم مانده.","حل مرجع زمان و محصول در جمله‌بندی متضاد؛ بازبینی جهت احساس. در QA07 نیز «بیس خوبی نداره» نیازمند نفی دقیق‌تر است."],
    [ltr("ML 47615607")+"؛ متن «عالیه بسیار عالی هستش»، برچسب واقعی no_idea و مدل recommended با 0.9999.","برچسب کاربر و احساس متن همیشه یکسان نیستند. بازبینی نمونه‌های مبهم و اعتبارسنجی کالیبراسیون/دادهٔ آموزشی مستقل."]
    ],"wide")+
    sub("محدودیت‌های ارزیابی")+
    bullet(["نمونه‌ها هدفمند هستند؛ ۱۴ محصول QA کوچک‌اند و به تمام ۱۰۰ هزار محصول قابل تعمیم نیستند. سه سناریوی پرنظر فقط کنترل ساختاری دارند.",
    "داوری معنا به‌صورت برچسب‌گذاری کمک‌گرفته از دستیار انجام شده است؛ مقایسهٔ انسانی هنوز صورت نگرفته. قالب داوری و اسکریپت توافق در بسته است.",
    "آزمون ثبت‌شدهٔ ML قبلی است؛ بازاجرای فعلی فقط روی شناسه‌های مشترک است. نبود متن پردازش آموزش، بررسی کامل نشت داده را محدود می‌کند.",
    "مقادیر Latency مخصوص محیط اجرای فعلی و پردازش ترتیبی هستند؛ SLA، توان بار همزمان و هزینهٔ زندهٔ API را تضمین نمی‌کنند."]) )
add("راهنمای تکرار و فایل‌های دفاع",
    sub("اجرای دوباره بدون API")+
    '<pre>python -m pip install -r requirements.txt\npython -X utf8 evaluation/run_evaluation.py\npython -X utf8 evaluation/run_evaluation.py --baseline --core-only --output-dir evaluation/results/baseline_rerun</pre>'+
    para("خروجی اجرای جدید به‌طور پیش‌فرض در "+ltr("evaluation/results/rerun")+" نوشته می‌شود و گزارش ثبت‌شدهٔ تحویل دست‌نخورده می‌ماند. درج نمرهٔ معنایی فقط با Fingerprint یکسان صورت می‌گیرد.")+
    table(["فایل","کاربرد"],[
    [ltr("cases.json"),"مجموعهٔ ثابت پرسش‌ها، معیار کشف و برچسب کامل نظرها"],
    [ltr("results/final/evaluation.json"),"همهٔ پاسخ‌ها، شواهد، داوری‌ها، نتایج و شاخص‌ها"],
    [ltr("results/baseline_core.json"),"نتیجهٔ نسخهٔ قبل (بازتولید شده)"],
    [ltr("results/search_ablation.json"),"مقایسهٔ شرط ۶۰٪ تنها و نسخهٔ نهایی"],
    [ltr("results/final/*_cases.csv"),"جدول پرسش‌ها و معیار هر پرسش"],
    [ltr("qa_evidence.csv / ml_replay.csv"),"شواهد با شناسه و نمره / بازاجرای مدل نظر"],
    [ltr("latency.csv / latency_summary.csv"),"۷۹ زمان خام و خلاصه میانگین، میانه و P95"],
    [ltr("api_usage.csv / cost_report.json"),"تعداد درخواست، Token و هزینهٔ ثبت‌شده"],
    [ltr("answer_assessments.json"),"نمره و دلیل داوری پاسخ، نقل و Grounding معنا"],
    [ltr("judge_rubric_fa.md"),"Rubric، دستور داوری و روش مقایسه انسانی"],
    [ltr("human_review_template.csv"),"ستون‌های خالی نمره انسانی در پوشهٔ final"],
    [ltr("presentation_fa.pdf"),"اسلاید آماده ارائهٔ همین بخش"]
    ],"wide compact")+
    sub("جمع‌بندی دفاع")+
    para("نسخهٔ محلی روی دادهٔ واقعی کار می‌کند؛ نسبت نتایج مرتبط افزایش یافته، موارد بدون شاهد بهتر تشخیص داده می‌شوند و صحت نقل‌قول‌ها تأیید شده است. هنوز تعبیر معنایی چند پرسش، جست‌وجوی نام مدل دوزبانه و کلاس no_idea جای بهبود دارند. نمره‌های واقعی، خطاها و محدودیت‌ها را با هم ارائه کنید.")+
    note("آزمون رگرسیون: ۷۴ آزمون قبلی + ۵ آزمون جدید، همگی موفق؛ آزمون‌های API واقعی عمداً اجرا نشدند."))

report_css=base_css+'''
@page{size:A4;margin:0}.page{background:white;width:210mm;height:297mm;margin:12px auto;position:relative;padding:12mm 15mm 20mm;overflow:hidden;box-shadow:0 1px 8px #bdc6da;font-size:11.6pt}
.page::before{content:"";height:4mm;background:linear-gradient(90deg,#f482b8,#b03986,#315bb7,#152f6a);position:absolute;left:0;right:0;top:0}
.page h1{font-size:26px;margin:.24em 0 .7em;border-bottom:1px solid #dce2f3;padding-bottom:.34em}
.page h3{font-size:13.1pt;margin:.76em 0 .24em}.lede{font-size:12pt;color:var(--muted)}
.page .compact{font-size:9.2pt}.page .compact th,.page .compact td{padding:.34em .45em}.page .latency{font-size:8.5pt;line-height:1.2}
.page .latency th,.page .latency td{padding:.27em .45em}pre{direction:ltr;text-align:left;font-size:8.5pt;line-height:1.55;background:#eef2ff;padding:8px;border-radius:6px;white-space:pre-wrap;overflow-wrap:anywhere}
@media print{.page{margin:0;box-shadow:none;break-after:page}.page:last-child{break-after:auto}}
'''
def doc(title,css,body):
 return '<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+e(title)+'</title><style>'+css+'</style></head><body>'+body+'</body></html>'
def report_body():
 return ''.join('<section class="page"><div class="kicker">'+k+'</div><h1>'+t+'</h1><main>'+content+'</main><footer><span>فروهر · دادهٔ واقعی ۱۰۰ هزار محصول</span><span dir="ltr">'+str(i)+' / '+str(len(pages))+'</span></footer></section>' for i,(t,content,k) in enumerate(pages,1))
(DIR/'report_fa.html').write_text(doc("گزارش بخش ۴ فروهر",report_css,report_body()),encoding='utf-8')
slides=[]
def sl(title,content): slides.append((title,content))
sl("ارزیابی سیستم نهایی فروهر",'<div class="cover"><div class="cover-sub">پروژهٔ دیجی‌کالا | بخش چهارم</div>'+
    metrics(big(num(100000),"محصول واقعی"),big(num(1874457),"نظر واقعی"),big(num(94),"سناریوی اصلی"))+
    para("عملکرد نسخهٔ محلی، پیش و پس از اصلاح؛ همراه با خطاهای واقعی، دادهٔ مرجع و امکان بازاجرا."))
sl("مجموعهٔ ارزیابی و روش",table(["مسئله","تعداد","مرجع"],[
    ["جست‌وجو","۴۷","۲۳ شناسه‌محور؛ ۱۸ کشف و ۶ بدون مصداق"],
    ["پرسش از نظرات","۲۹","۱۴ محصول؛ ۲۷۵ زوج پرسش-نظر با نمرهٔ ۰/۱/۲"],
    ["مرزی و حجم بالاتر / مقایسه / مدیر","۷ / ۳ / ۸","حالت‌های واقعی و کنترل مشخصات، شواهد و آمار"],
    ["پیش‌بینی پیشنهاد خرید","۱۵٬۰۰۰ / ۴٬۵۳۷","نتیجهٔ ثبت‌شده + بازاجرای شناسه‌های مشترک"],
    ["زمان پاسخ","۷۹","۵ تکرار گرم؛ ۳ تکرار سرد برای سناریوهای منتخب"]],"wide")+
    note("معیارهای ثابت کاتالوگ و برچسب معنایی کمک‌گرفته از ChatGPT؛ ارزیابی انسانی مستقل انجام نشده است. نمونه‌ها هدفمند و محدودند."))
sl("Retrieval Quality؛ پیش و پس از اصلاح",table(["شاخص","نسخهٔ قبلی","نسخهٔ نهایی"],comparison_rows()[:8])+
    note("دقت نتایج برگشتی بیشتر شد، اما Precision@5 کشف ثابت ماند. شرط ۶۰٪ به‌تنهایی هدفون مانیتورینگ را از دست داد؛ مسیر جایگزین عنوان مشکل را رفع کرد."))
sl("کیفیت پاسخ و Grounding",metrics(big(ltr(f'{a["quality_mean_0_to_2"]:.2f} / 2'),"کیفیت"),big(ltr(f'{a["semantic_grounding_mean_0_to_2"]:.2f} / 2'),"Grounding معنایی"),big("85 / 85","شاهد دقیق"))+
    table(["معیار","کامل (۲)","جزئی (۱)","ناموفق (۰)"],[
      ["کیفیت پاسخ (۲۹)",18,10,1],["Grounding معنایی (۲۹)",22,7,0],["ربط نقل‌قول (۸۵)",63,17,5]])+
    note("۸۵ شاهد عین منبع هستند، اما تنها ۶۳ نقل مستقیم به سؤال مربوط‌اند. ۳ امتناع درست، ۴ حالت مرزی درست و ۳ مقایسهٔ محصول با کنترل موفق."))
sl("Recommendation Prediction",metrics(big(score(mt["macro_f1"],4),"Macro F1 آزمون"),big(score(mr["macro_f1"],4),"Macro F1 بازاجرا"),big(score(m["majority_recorded_baseline"]["macro_f1"],4),"Baseline اکثریت"))+
    table(["کلاس","F1 آزمون","F1 بازاجرا"],[[ltr(c),score(mt["report"][c]["f1-score"],4),score(mr["report"][c]["f1-score"],4)] for c in mt["confusion_matrix_labels"]])+
    note("۱۵٬۰۰۰ پیش‌بینی ثبت‌شده؛ بازاجرای ۴٬۵۳۷ شناسه مشترک (۴٬۵۳۶ معتبر). ضعف اصلی: no_idea. بازاجرا روی متن خام است و آزمون مستقل تازه محسوب نمی‌شود."))
sl("Latency؛ سناریوهای واقعی",table(["سناریو","حالت","N","Median ms","P95 ms"],[
    latrow("search_plain","جست‌وجوی ساده"),latrow("search_plain","جست‌وجوی ساده","cold_process"),
    latrow("qa_6_reviews","پاسخ ۶ نظر"),latrow("qa_200_reviews","پاسخ ۲۰۰ نظر"),latrow("qa_200_reviews","پاسخ ۲۰۰ نظر","cold_process"),
    latrow("compare_5","مقایسه ۵ محصول"),latrow("predict_short","پیش‌بینی"),latrow("predict_short","پیش‌بینی","cold_process"),
    latrow("manager_complaints","تحلیل شکایت دسته")],"compact")+
    note("۷۹ اندازه‌گیری ترتیبی در محیط فعلی. سرد = فرآیند تازه، گرم = پس از آماده‌سازی. بدون رندر مرورگر یا شبکه."))
sl("Cost و مصرف Token",metrics(big(ltr("$0.00"),"هزینه واقعی API"),big("0","Prompt Tokens"),big("0","Completion Tokens"))+
    para("تمام سناریوها با BM25، پاسخ مستند محلی، DuckDB و مدل ML اجرا شدند. هیچ درخواست پولی و هیچ LLM-as-a-Judge آنلاین انجام نشد.")+
    '<div class="formula" dir="ltr">Cost = (input tokens × input rate + output tokens × output rate) / 1,000,000</div>'+
    note("قیمت واقعی درگاه برای حالت زنده لازم است. نرخ نامعلوم تخمین زده نشده؛ زمان و هزینهٔ LLM آنلاین هنوز اندازه‌گیری نشده است."))
sl("شکست‌ها و اصلاح‌های واقعی",table(["نمونه","قبل","بعد"],[
    [ltr("S10")+" بلیط هواپیما","میکرودرم پاریس و سایر کالای نامرتبط","نتیجهٔ خالی درست"],
    [ltr("QA03")+" نرمی دست","۱ شاهد مستقیم","۳ شاهد مستقیم"],
    [ltr("QA19")+" موهای خشک","عدم کفایت شاهد","۴ شاهد مستقیم"],
    [ltr("QA22")+" متن «بد نبود»","به‌عنوان شکایت برگشت","حذف شاهد نامعتبر"]
    ],"wide")+
    note("تغییر در شرط پوشش واژه‌های جست‌وجو، صورت‌های محاوره‌ای و نفی. داده و مدل آموزش‌دیده تغییر نکردند."))
sl("شکست‌های باقی‌مانده و اعتبار نتیجه",bullet([
    ltr("S14")+" گلس Galaxy Fit 2: مشکل مدل دوزبانه؛ ۰ نتیجهٔ مستقیم از ۵.",
    ltr("QA25")+" وزن چای: ۲۲/۲۳ گرم در متن خام وجود دارد، اما عنوان کلی بازیابی شد.",
    ltr("QA21")+" «صفحه» به اشتباه موضوع چاپ کتاب برداشت شد؛ معنای شاهد باید جدا بررسی شود.",
    ltr("ML")+" نمونهٔ «عالیه بسیار عالی هستش» با برچسب no_idea در برابر recommended با اطمینان 0.9999.",
    "قدم بعدی: مجموعهٔ نادیدهٔ بزرگ‌تر با داوری انسانی، بهبود عبارات دوزبانه و زمینهٔ جنبه‌ها، کالیبراسیون روی Validation."
    ])+note("نتایج و Prompt داوری در evaluation/ قابل مشاهده و تکرارند. نسخهٔ LLM زنده و Dense/Hybrid هم‌تراز در این تحویل آزمون نشده‌اند."))
slide_css=base_css+'''
@page{size:13.333in 7.5in;margin:0}.slide{background:white;width:13.333in;height:7.5in;margin:12px auto;position:relative;padding:.38in .55in .5in;overflow:hidden;box-shadow:0 1px 9px #bdc6da;font-size:16pt;line-height:1.5}
.slide::before{content:"";height:.075in;background:linear-gradient(90deg,#f482b8,#b03986,#315bb7,#152f6a);position:absolute;left:0;right:0;top:0}
.slide h1{font-size:30pt;margin:.12em 0 .55em;border-bottom:1px solid #dce2f3;padding-bottom:.24em}
.slide h3{font-size:20pt;color:var(--pink)}.slide .metrics{margin:.45em 0 .63em;gap:1.2em}.slide .value{font-size:2.1em}
.slide table{font-size:14pt;margin:.32em 0;line-height:1.38}.slide th,.slide td{padding:.34em .48em}
.slide .compact{font-size:12.6pt}.slide .compact td,.slide .compact th{padding:.25em .4em}.slide .note{font-size:13.3pt;margin:.4em 0}
.slide footer{bottom:.23in;right:.55in;left:.55in;font-size:10pt;padding-top:.07in}.slide li{margin:.5em 0}
.slide.cover::before{height:.13in}.cover-sub{font-size:23pt;color:var(--pink);font-weight:700}.cover{padding-top:.72in}
.slide:first-child h1{font-size:42pt;border-bottom:none;margin-top:.8em}
.formula{font-size:17pt;background:#eef2ff;padding:.7em;margin:1.3em 0;text-align:center;color:var(--navy);border-radius:8px}
@media print{.slide{margin:0;box-shadow:none;break-after:page}.slide:last-child{break-after:auto}}
'''
slidebody=''.join('<section class="slide"><div class="kicker">فروهر | ارزیابی نهایی</div><h1>'+t+'</h1><main>'+c+'</main><footer><span>نتایج اجرای واقعی نسخهٔ ۱۰۰ هزار محصولی</span><span dir="ltr">'+str(i)+' / '+str(len(slides))+'</span></footer></section>' for i,(t,c) in enumerate(slides,1))
(DIR/'presentation_fa.html').write_text(doc("ارائه بخش ۴ فروهر",slide_css,slidebody),encoding='utf-8')
print("HTML report pages",len(pages),"presentation slides",len(slides))
