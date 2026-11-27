"""Build a self-contained Persian slide deck from the measured bonus reports."""
from __future__ import annotations

import ast
import html
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/bonus"


def load(name):return json.loads((OUT/f"{name}.json").read_text(encoding="utf8"))
def pct(v):return f"{100*v:.1f}%"
def esc(v):return html.escape(str(v))
def ltr(v):return f'<bdi dir="ltr">{esc(v)}</bdi>'


def build():
    ret=load("retrieval");ml=load("adaptation");wc=load("router_cache");gap=load("quality_gap")
    original=ret["groups"]["original_discovery"];stress=ret["groups"]["typo_stress"];combined=ret["groups"]["all"]
    bm=original["bm25"];hy=original["hybrid_char"];auto=combined["auto"]
    before=ml["test"]["baseline"];after=ml["test"]["adapted"]
    g=gap["report"];cache=wc["cache"]
    full=combined["hybrid_char"]
    skipped=full["hybrid_runs"]-auto["hybrid_runs"]
    speed=cache["uncached_global_median_ms"]/cache["cached_global_median_ms"]
    font=next(ast.literal_eval(n.value) for n in ast.parse((ROOT/"app.py").read_text(encoding="utf8")).body
              if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="FONT_WOFF2" for t in n.targets))
    def slide(title,body,foot=""):
        return f'<section class="slide"><h1>{title}</h1>{body}<footer>{foot}</footer></section>'
    def metric(label,value,detail=""):
        return f'<div class="metric"><div>{label}</div><strong>{ltr(value)}</strong><small>{detail}</small></div>'
    def bars(title,items,scale=1):
        rows=[]
        for name,value in items:
            rows.append(f'<div class="barrow"><span>{name}</span><div class="track"><div class="fill" style="width:{min(100,100*value/scale):.2f}%"></div></div><bdi>{100*value:.1f}%</bdi></div>')
        return f'<div class="bars"><h3>{title}</h3>{"".join(rows)}</div>'
    slides=[]
    slides.append(slide("فروهر: از پاسخ مستند تا آزمایشگاه شواهد",
        '<p class="lead">دستیار خرید و تحلیل محصولات دیجی‌کالا</p><div class="metrics">'+
        metric("محصول واقعی","100,000")+metric("نظر واقعی","1,874,457")+metric("دامنه آزمون مدل","15,000")+
        '</div><div class="statement">پرسش دفاع: کدام تغییر، کیفیت یا زمان پاسخ را به‌صورت قابل اندازه‌گیری بهتر کرد؟</div>',
        "بوت‌کمپ هوش مصنوعی کوئرا، دوره ۱۳"))
    slides.append(slide("پنج قابلیت اصلی حفظ شدند",
        '<div class="columns"><div><h3>سیستم پایه</h3><ol><li>جست‌وجوی فارسی و فیلترهای محصول</li><li>پاسخ با شناسه و متن واقعی نظرات</li><li>مقایسه ۲ تا ۵ محصول</li><li>تحلیل مدیران در سطح دسته و برند</li><li>پیش‌بینی recommendation_status</li></ol></div>'
        '<div class="panel"><h3>خط پایه و قیدها</h3><p>کاتالوگ SQLite / BM25 و تحلیل Parquet</p><p>۱۰۰ هزار محصول انتخاب قطعی با نظر متنی</p><p>مدل کاراکتری TF-IDF + Logistic Regression</p><p>نسخه اصلی، داده و گزارش‌ها دست‌نخورده‌اند.</p></div></div>',
        "منبع: دستور رسمی پروژه و reports/sections_1_3_validation.json"))
    slides.append(slide("بازیابی ترکیبی، غلط تایپی و نگارش متفاوت را پوشش می‌دهد",
        '<div class="columns">'+
        bars("Hit@5 در ۴۰ پرسش غلط تایپی", [("BM25",stress["bm25"]["hit_at_5"]),("Hybrid",stress["hybrid_char"]["hit_at_5"])])+
        bars("Precision@5 در کشف طبیعی قبلی",[("BM25",bm["p_at_5"]),("Hybrid",hy["p_at_5"])])+
        '</div><div class="flow"><div>BM25</div><div>Character TF-IDF</div><div>RRF + پوشش واژه</div><div>نتایج دارای قید</div></div>'
        '<p>قید برند/عدد و فیلترهای قیمت حفظ می‌شوند. تطابق تقریبی ضمانت سازگاری محصول نیست.</p>',
        "آزمایش ۴۰ عنوان یکتا با دو خطای جابه‌جایی حرف. مجموعه قدیمی و آزمون تکرارشده تشخیصی‌اند."))
    slides.append(slide("Router فقط در صورت نیاز مدل دوم را اجرا می‌کند",
        '<div class="metrics">'+metric("همیشه Hybrid",str(full["hybrid_runs"])+" runs")+metric("Router",str(auto["hybrid_runs"])+" runs",f"{skipped} اجرای کمتر")+
        metric("Hit@5 هر دو",pct(auto["hit_at_5"]))+'</div>'
        '<div class="columns"><div><h3>منطق تصمیم</h3><ul><li>شناسه یا عنوان دقیق: BM25</li><li>درخواست پیچیده یا خروجی خالی: Hybrid</li><li>پرسش مستند: بازیابی شاهد پیش از مدل زبانی</li></ul></div>'
        '<div class="panel"><h3>نتیجه کمی</h3><p>تشخیص نوع درخواست: '+ltr(str(wc["router"]["correct"])+" / "+str(wc["router"]["total"]))+'</p>'
        f'<p>میانگین زمان ثابت: {ltr(f"{full["latency"]["mean_ms"]:.1f} ms")} در برابر {ltr(f"{auto["latency"]["mean_ms"]:.1f} ms")}</p></div></div>',
        "۲۴ سناریوی Intent و ۸۷ سناریوی بازیابی. اجرای آزمایش در Python fallback."))
    slides.append(slide("گراف، مسیر پاسخ را قابل مشاهده می‌کند",
        '<div class="flow"><div>درخواست</div><div>Intent</div><div>Cache</div><div>سرویس منتخب</div><div>شاهد و Trace</div></div>'
        '<div class="columns"><div><h3>چرا این مدل گراف؟</h3><p>جست‌وجو، SQL، طبقه‌بندی و RAG مسئولیت‌های جدا دارند. شاخه‌ها مسیر نامرتبط را اجرا نمی‌کنند. گره LLM فقط با شاهد کافی و اجازه کاربر فعال می‌شود.</p></div>'
        '<div class="panel"><h3>مانیتورینگ</h3><p>هر درخواست Trace ID، زمان گره‌ها، خطا، Cache hit و usage دارد.</p><p>LangGraph و Langfuse مسیر اختیاری واقعی دارند؛ در این محیط SDK نصب نشد. تاریخچه محلی آزمایش شده است.</p></div></div>',
        "API call واقعی در آزمایش آفلاین: ۰. هیچ تعرفه‌ای حدس زده نشده است."))
    slides.append(slide("آموزش تکمیلی Head عملکرد مدل را بهتر کرد",
        '<div class="columns">'+bars("Macro F1، آزمون ثابت ۱۵ هزار نظر",[("مدل اصلی",before["macro_f1"]),("Head تکمیلی",after["macro_f1"])])+
        bars("Accuracy",[("مدل اصلی",before["accuracy"]),("Head تکمیلی",after["accuracy"])])+'</div>'
        '<div class="metrics">'+metric("Train",f'{ml["audit"]["splits"]["train"]["rows"]:,}')+metric("Validation",f'{ml["audit"]["splits"]["validation"]["rows"]:,}')+
        metric("هم‌پوشانی نمونه‌ها","0")+'</div><p>بردارساز ثابت است. وزن‌های طبقه‌بندی از مدل تحویلی شروع و با داده train به‌روز شدند. انتخاب فقط روی validation بود.</p>',
        "این آزمایش fine-tuning مدل خطی است، نه LoRA یا fine-tuning ترنسفورمر."))
    slides.append(slide("Result Cache پاسخ تکراری را بدون تغییر محتوا برمی‌گرداند",
        '<div class="metrics">'+metric("بدون کش، میانه",f'{cache["uncached_global_median_ms"]:.1f} ms')+
        metric("کش گرم، میانه",f'{cache["cached_global_median_ms"]:.2f} ms')+
        metric("سرعت نسبی اندازه‌گیری‌شده",f'{speed:.0f}x')+'</div>'
        '<div class="statement">۲۰ از ۲۰ درخواست گرم، همان خروجی نسخه بدون کش را برگرداندند.</div>'
        '<p>کلید کش شامل هش کاتالوگ، نسخه روش، شناسه محصول، سؤال، فیلتر و واحد قیمت است. کش محدود و زمان‌دار است؛ درخواست زنده API و متن پیش‌بینی در آن ذخیره نمی‌شود.</p>',
        "۵ تکرار در هر یک از ۴ سناریوی واقعی. این نتیجه کش گرم محلی است، نه تضمین همه محیط‌ها."))
    slides.append(slide("ارزیابی انسانی از داوری دستیار جدا می‌شود",
        '<div class="columns"><div><h3>فرم داور</h3><ol><li>مشاهده سؤال، پاسخ، نظرهای اصلی و شناسه شاهد</li><li>نمره مستقل کیفیت و Grounding از ۰ تا ۲</li><li>ثبت دلیل و شناسه مستعار</li><li>توافق دقیق و Weighted Kappa بعد از ثبت</li></ol></div>'
        '<div class="panel"><h3>وضعیت واقعی</h3><div class="big">۲۹</div><p>نمونه ثابت و فرم آماده</p><div class="big">۰</div><p>داوری انسانی مستقل تا زمان ساخت این ارائه</p></div></div>',
        "هیچ نمره انسانی از طرف سیستم تولید نشده است."))
    slides.append(slide("مسئله جدید: اختلاف امتیاز و نرخ پیشنهاد خرید",
        '<div class="metrics">'+metric("پرچم ساده",f'{g["naive_flags"]:,}')+metric("فاقد حداقل شواهد",f'{g["naive_flags_insufficient_support"]:,}')+
        metric("صف بررسی محافظه‌کارانه",str(g["conservative_flags"]))+'</div>'
        '<p>حداقل ۲۰ نظر برچسب‌دار، ۱۰ رأی و امتیاز ۷۰ از ۱۰۰. محصول زمانی اولویت بررسی می‌گیرد که فاصله امتیاز با کران بالای Wilson نرخ پیشنهاد، حداقل ۱۵ واحد درصد باشد.</p>'
        '<p class="statement">هدف: تمرکز وقت مدیر روی اختلاف‌هایی با پشتیبانی آماری بهتر. این پرچم اثبات نقص یا رابطه علّی نیست.</p>',
        "شمارش‌های واقعی روی ۱۰۰ هزار محصول؛ صحت شمارش و شواهد نمونه بررسی شده. تأیید منتور هنوز لازم است."))
    slides.append(slide("Demo: آزمایشگاه در تب چهارم Streamlit",
        '<div class="columns"><div><h3>مسیر ارائه (۴ دقیقه)</h3><ol><li>مقایسه BM25 و Hybrid با یک پرسش غلط تایپی یا کوله کودک.</li><li>اجرای Router و بررسی مسیر و زمان گره؛ اجرای دوباره برای Cache HIT.</li><li>مقایسه مدل اصلی و Head تکمیلی با یک نظر تازه.</li><li>تحلیل شکاف رضایت و مشاهده نظرهای منبع.</li></ol></div>'
        '<div class="panel"><h3>مرز ادعا</h3><p>بهبود بازیابی، مدل، زمان کش و حسابرسی داده اندازه‌گیری شده‌اند.</p><p>داوری انسان، تأیید منتور و بررسی واقعی SDK / Langfuse هنوز باید انجام شوند.</p><p>اجرای زنده API در این آزمایش صفر است.</p></div></div>',
        "گزارش‌های JSON و فرم‌های قابل دانلود کنار همین Demo قرار دارند."))
    css=f"""
    @font-face{{font-family:Vazirmatn;src:url(data:font/woff2;base64,{font}) format('woff2');font-weight:100 900}}
    *{{box-sizing:border-box}} html,body{{margin:0;background:#eaf0fa;color:#152f6a;font-family:Vazirmatn,Arial,sans-serif}}
    .slide{{width:1280px;height:720px;position:relative;margin:24px auto;padding:52px 66px 60px;background:linear-gradient(135deg,#f5f9ff 0,#fff 58%,#fceffa 100%);border-top:9px solid #346aba;overflow:hidden;page-break-after:always}}
    h1{{font-size:36px;line-height:1.48;margin:0 0 25px;font-weight:850}}h3{{font-size:24px;margin:0 0 15px;line-height:1.4}}
    p,li{{font-size:23px;line-height:1.65;margin:10px 0}}ol,ul{{padding-right:34px;margin:12px 0}}
    .lead{{font-size:30px;color:#627f9d}}.metrics{{display:flex;gap:22px;margin:25px 0 20px;direction:rtl}}
    .metric{{flex:1;min-width:0;background:#fff;border-radius:14px;border:1px solid #ccd9f0;padding:22px 18px;box-shadow:0 7px 19px #20365d0b}}
    .metric div{{font-size:18px;color:#526889}}.metric strong{{display:block;font-size:38px;line-height:1.4;color:#2062ba;margin:8px 0;direction:ltr;text-align:right}}.metric small{{display:block;font-size:15px;color:#9962a4}}
    .statement{{margin-top:24px;padding:18px 26px;border-right:5px solid #e58bbb;background:#fff1fa;border-radius:10px;font-size:24px;line-height:1.55}}
    .columns{{display:grid;grid-template-columns:1fr 1fr;gap:40px;align-items:start;margin:16px 0}}
    .panel{{border:1px solid #cbd8f0;border-radius:14px;padding:24px;background:#f4f9ff}}.panel p{{font-size:21px}}
    .bars{{padding:8px 0}}.barrow{{display:grid;grid-template-columns:130px 1fr 76px;gap:13px; align-items:center;margin:24px 0;direction:rtl;font-size:18px}}
    .track{{height:24px;border-radius:10px;background:#e4eaf6;direction:ltr;overflow:hidden}}.fill{{background:linear-gradient(90deg,#a68ce2,#3575cb);height:100%;border-radius:10px}}
    .barrow bdi{{font-weight:bold;color:#1c4d92}}.flow{{display:flex;justify-content:space-between;gap:10px;margin:18px 0 25px}}
    .flow div{{flex:1;border:1px solid #b7caea;border-radius:11px;padding:16px 9px;text-align:center;font-size:19px;background:#eef4ff}}
    .big{{font-size:46px;font-weight:bold;color:#2468bb}}footer{{position:absolute;bottom:23px;right:66px;left:66px;border-top:1px solid #e0e5f0;padding-top:12px;color:#697d99;font-size:14px}}
    @page{{size:1280px 720px;margin:0}}@media print{{html,body{{background:white}}.slide{{margin:0;break-after:page;print-color-adjust:exact;-webkit-print-color-adjust:exact}}.slide:last-child{{break-after:auto;page-break-after:auto}}}}
    """
    path=OUT/"bonus_presentation_fa.html"
    path.write_text('<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8"><title>فروهر - دفاع امکانات امتیازی</title><style>'+css+'</style></head><body>'+''.join(slides)+'</body></html>',encoding="utf8")
    summary=f"""## مسیر دفاع امکانات امتیازی

۱. **Retrieval (۴):** مقایسه روی همان ۱۰۰ هزار محصول. در ۴۰ غلط تایپی کنترل‌شده، `Hit@5` از {pct(stress["bm25"]["hit_at_5"])} به {pct(stress["hybrid_char"]["hit_at_5"])} رسید. `Precision@5` در کشف طبیعی قبلی از {pct(bm["p_at_5"])} به {pct(hy["p_at_5"])} رسید. مجموعه‌ها تشخیصی‌اند.

۲. **Fine-tuning (۴):** آموزش تکمیلی وزن‌های Head روی {ml["audit"]["splits"]["train"]["rows"]:,} نظر با بردارساز ثابت؛ `Macro F1` آزمون ۱۵هزار نظر از {pct(before["macro_f1"])} به {pct(after["macro_f1"])} رسید. این روش LoRA ترنسفورمر نیست؛ پذیرش نمره با داور است.

۳. **Router (۳):** {wc["router"]["correct"]}/{wc["router"]["total"]} تشخیص Intent درست، {skipped} اجرای کمتر مدل دوم نسبت به همیشه Hybrid با کیفیت Hit@5 برابر. Graph state/edge و ثبت هر گره قابل مشاهده‌اند.

۴. **Human Evaluation (۳):** فرم مستقل ۲۹ نمونه و محاسبه توافق آماده؛ تا زمان ساخت گزارش، داوری انسانی واقعی انجام نشده است.

۵. **UI (۳):** آزمایشگاه شش‌زیرتب در تب چهارم رابط اصلی.

۶. **Caching (۳):** میانه {cache["uncached_global_median_ms"]:.2f} ms بدون کش در برابر {cache["cached_global_median_ms"]:.2f} ms کش گرم؛ ۲۰ پاسخ یکسان. زمان روی همین محیط و سناریوها.

۷. **مسئله جدید (۳):** {g["conservative_flags"]} محصول برای بررسی محافظه‌کارانه اختلاف rating و recommendation؛ شمارش و شواهد نمونه واقعی. تأیید منتور انجام نشده است.

۸. **ارائه (۲):** فایل اسلاید فارسی HTML/PDF و مسیر Demo.

### آنچه باید در دفاع صادقانه گفته شود

- موتور LangGraph در صورت نصب اجرا می‌شود؛ آزمایش فعلی با همان workflow و Python fallback اجرا شد. Langfuse SDK و سرور زنده اینجا آزموده نشده‌اند.
- هیچ هزینه واقعی API برای آزمایش‌های آفلاین انجام نشده؛ سیاست LLM فقط opt-in است.
- مدل اصلی، داده و سه تب قبلی حفظ شده‌اند؛ مدل تکمیلی صرفاً گزینه مقایسه است.
- داوری انسان را واقعاً در فرم انجام دهید؛ از امتیاز ساختگی و ادعای تأیید منتور پرهیز کنید.
"""
    (OUT/"defense_fa.md").write_text(summary,encoding="utf8")
    chromium=shutil.which("chromium") or shutil.which("google-chrome")
    try:
        if not chromium:raise FileNotFoundError("Chromium")
        pdf=OUT/"bonus_presentation_fa.pdf"
        subprocess.run([chromium,"--headless","--no-sandbox","--disable-gpu","--disable-dev-shm-usage","--no-pdf-header-footer",
                        "--print-to-pdf="+str(pdf),path.as_uri()],check=True,timeout=90,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except (OSError,subprocess.SubprocessError):
        from bonus.presentation_pdf import render
        render(ret,ml,wc,gap,OUT/"bonus_presentation_fa.pdf")
    return path


if __name__=="__main__":
    print(build())
