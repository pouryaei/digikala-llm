"""Raster PDF fallback with proper Persian shaping (Pillow + RAQM).

The HTML deck remains editable. This creates a portable widescreen PDF when
a local Chromium PDF renderer is unavailable.
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,features

ROOT=Path(__file__).resolve().parents[1]
W,H=1600,900
NAVY="#152f6a";BLUE="#2564b5";MUTED="#536b8c";PALE="#e9f2ff";PINK="#fff0fa";BORDER="#c6d7ec"
# This family covers Persian, Latin technical terms and punctuation in one
# file, so RAQM can shape mixed-direction text without missing-glyph boxes.
FONT_PATH=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
BOLD_PATH=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def font(size,bold=False):
    path=BOLD_PATH if bold else FONT_PATH
    if not path.is_file():path=Path("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(str(path),size)


def wrap(draw,value,f,width):
    words=value.split();lines=[];part=""
    for word in words:
        candidate=(part+" "+word).strip()
        if draw.textlength(candidate,font=f,direction="rtl",language="fa")>width and part:
            lines.append(part);part=word
        else:part=candidate
    if part:lines.append(part)
    return lines


def rtl(draw,value,right,y,width,size=30,bold=False,color=NAVY,line_gap=11):
    f=font(size,bold)
    lines=[]
    for paragraph in value.split("\n"):
        lines.extend(wrap(draw,paragraph,f,width) or [""])
    for line in lines:
        draw.text((right,y),line,font=f,fill=color,anchor="rt",direction="rtl",language="fa")
        y+=size+line_gap+6
    return y


def base(title,foot,index):
    img=Image.new("RGB",(W,H),"#f8fbff");d=ImageDraw.Draw(img)
    d.rectangle((0,0,W,12),fill=BLUE);d.rectangle((0,H-125,W,H),fill="#f8eef9")
    d.rounded_rectangle((W-300,37,W-78,83),radius=22,fill=PALE)
    rtl(d,"فروهر",W-104,43,175,23,True)
    rtl(d,title,W-80,112,W-160,43,True)
    d.line((80,220,W-80,220),fill=BORDER,width=2)
    rtl(d,foot,W-80,H-86,W-210,20,color=MUTED,line_gap=2)
    d.text((80,H-70),f"{index:02}",font=font(24,True),fill=BLUE,anchor="lt")
    return img,d


def cards(d,values,y=255):
    n=len(values);gap=24;left=80;width=(W-160-gap*(n-1))/n
    for i,(label,value,detail) in enumerate(values):
        x=W-80-(i+1)*width-i*gap
        d.rounded_rectangle((x,y,x+width,y+184),radius=18,fill="white",outline=BORDER,width=2)
        rtl(d,label,x+width-18,y+18,width-32,25,color=MUTED)
        d.text((x+width-18,y+65),str(value),font=font(40,True),fill=BLUE,anchor="rt",direction="ltr")
        if detail:rtl(d,detail,x+width-18,y+128,width-32,20,color=MUTED,line_gap=3)


def paragraphs(d,items,*,y=480,size=30):
    for text in items:
        y=rtl(d,"• "+text,W-90,y,W-190,size,line_gap=9)+18


def chart(d,title,labels,values,right,top,width):
    rtl(d,title,right,top,width,29,True)
    for ix,(label,value) in enumerate(zip(labels,values)):
        y=top+76+ix*87
        rtl(d,label,right,y,width,23,color=MUTED)
        end=right-182;begin=right-width+20
        d.rounded_rectangle((begin,y+12,end,y+42),radius=12,fill="#dbe7f8")
        d.rounded_rectangle((begin,y+12,begin+(end-begin)*value,y+42),radius=12,fill=BLUE if ix else "#b78ed5")
        d.text((begin,y+48),f"{value:.1%}",font=font(22,True),fill=BLUE,anchor="lt",direction="ltr")


def render(ret,ml,wc,gap,output):
    if not features.check_feature("raqm"):raise RuntimeError("Pillow RAQM needed for Persian PDF shaping")
    org=ret["groups"]["original_discovery"];stress=ret["groups"]["typo_stress"];all_=ret["groups"]["all"]
    before=ml["test"]["baseline"];after=ml["test"]["adapted"];c=wc["cache"];g=gap["report"]
    slides=[]
    def add(title,foot):
        im,d=base(title,foot,len(slides)+1);slides.append(im);return d
    d=add("فروهر: از دستیار مستند تا آزمایشگاه شواهد","پروژه نهایی کوئرا، دوره ۱۳")
    cards(d,[("محصول واقعی","100,000",""),("نظر واقعی","1,874,457",""),("نمونه آزمون مدل","15,000","")])
    paragraphs(d,["هدف: تغییراتی که نتیجه قابل اندازه‌گیری دارند.","نسخه اصلی و شواهد حفظ شده‌اند؛ آزمایشگاه امکانات امتیازی در کنار آن قرار دارد."],y=505,size=34)
    d=add("قابلیت‌های اصلی و محدودیت منابع","داده ثابت، انتخاب قطعی محصولات دارای نظر متنی")
    paragraphs(d,["جست‌وجوی فارسی و فیلترهای ساخت‌یافته محصول","پاسخ بر اساس نظرهای واقعی و Comment ID","مقایسه تا پنج محصول، تحلیل مدیران و پیش‌بینی سه‌کلاسه","خط پایه BM25 و مدل Character TF-IDF + Logistic Regression"],y=265,size=33)
    d=add("بازیابی ترکیبی روی همان مجموعه داده","۴۰ غلط تایپی کنترل‌شده؛ مجموعه قبلی تشخیصی و نه آزمون انسانی مستقل")
    chart(d,"Hit@5 در غلط تایپی",["BM25","Hybrid"],[stress["bm25"]["hit_at_5"],stress["hybrid_char"]["hit_at_5"]],W-85,255,680)
    chart(d,"Precision@5 در کشف طبیعی",["BM25","Hybrid"],[org["bm25"]["p_at_5"],org["hybrid_char"]["p_at_5"]],W//2-30,255,680)
    paragraphs(d,["ترکیب BM25 و شباهت کاراکتری، RRF و بازرتبه‌بندی با پوشش واژه.","فیلتر قیمت و برند و قیود عددی محفوظ می‌مانند؛ تطابق تقریبی تضمین سازگاری نیست."],y=565,size=29)
    d=add("Router؛ مدل دوم فقط هنگام نیاز","۸۷ سناریوی بازیابی و ۲۴ سناریوی Intent")
    cards(d,[("همیشه Hybrid",str(all_["hybrid_char"]["hybrid_runs"]),"اجرا"),("Router",str(all_["auto"]["hybrid_runs"]),"اجرا"),("Intent",f'{wc["router"]["correct"]}/{wc["router"]["total"]}',"مسیر درست")])
    paragraphs(d,["شناسه/عنوان دقیق و درخواست ساده با BM25؛ درخواست پیچیده یا بدون نتیجه با Hybrid.",
                  f'Hit@5 روتر و Hybrid هر دو {all_["auto"]["hit_at_5"]:.1%}؛ زمان میانگین {all_["auto"]["latency"]["mean_ms"]:.1f} در برابر {all_["hybrid_char"]["latency"]["mean_ms"]:.1f} میلی‌ثانیه.'],y=480,size=29)
    d=add("گراف و مانیتورینگ گره‌های اجرا","SDK واقعی LangGraph/Langfuse اختیاری است؛ آزمایش کنونی Python fallback")
    labels=["درخواست","Intent","Cache","سرویس منتخب","Trace و خروجی"]
    for i,l in enumerate(labels):
        x=W-80-(i+1)*272-i*18
        d.rounded_rectangle((x,270,x+272,382),radius=16,fill=PALE,outline=BORDER,width=2)
        rtl(d,l,x+253,308,235,26,True)
    paragraphs(d,["هر گره زمان و وضعیت مستقل دارد؛ خطا و Cache HIT ثبت می‌شود.",
                  "مسیر مدل زبانی فقط با انتخاب صریح و شاهد کافی. خروجی مانیتورینگ بیرونی متن درخواست و نظر را شامل نمی‌شود.",
                  "در این آزمایش هیچ درخواست پولی API ارسال نشد."],y=445,size=29)
    d=add("آموزش تکمیلی Head؛ مدل اصلی حفظ شد","مدل خطی با بردارساز ثابت، نه LoRA ترنسفورمر؛ test فقط بعد از انتخاب validation")
    chart(d,"Macro F1",["مدل اصلی","Head تکمیلی"],[before["macro_f1"],after["macro_f1"]],W-85,255,680)
    chart(d,"Accuracy",["مدل اصلی","Head تکمیلی"],[before["accuracy"],after["accuracy"]],W//2-30,255,680)
    paragraphs(d,[f'{ml["audit"]["splits"]["train"]["rows"]:,} نظر train و {ml["audit"]["splits"]["validation"]["rows"]:,} نظر validation؛ هم‌پوشانی محصول/کلید متن بین نمونه‌های این آزمایش صفر.',
                  f'آزمون اصلی ۱۵ هزار نظر: بهبود Macro F1 {ml["test"]["delta_macro_f1"]*100:.2f} واحد درصد.'],y=565,size=29)
    d=add("کش نتیجه؛ پاسخ یکسان و زمان کمتر","۲۰ درخواست گرم، چهار سناریو، اندازه‌گیری روی همین محیط")
    cards(d,[("میانه بدون کش",f'{c["uncached_global_median_ms"]:.1f} ms',""),("میانه کش گرم",f'{c["cached_global_median_ms"]:.2f} ms',""),("پاسخ یکسان","20/20","")])
    paragraphs(d,["کلید کش به نسخه داده، پرسش، روش، فیلتر و شناسه وابسته است.","کش محدود و زمان‌دار؛ نتیجه پولی و متن پیش‌بینی در آن قرار نمی‌گیرد."],y=505,size=33)
    d=add("داوری انسانی مستقل","فرم ۲۹ نمونه آماده؛ تا زمان ساخت ارائه هیچ داوری انسانی واقعی ثبت نشده است")
    cards(d,[("نمونه قابل داوری","29",""),("داوری واقعی ثبت‌شده","0","")])
    paragraphs(d,["داور پیش از دیدن نمرات دستیار، سؤال، پاسخ، منبع و شناسه شواهد را می‌خواند.","کیفیت و Grounding از ۰ تا ۲، دلیل کوتاه؛ سپس Exact agreement و Weighted Kappa."],y=475,size=30)
    d=add("مسئله جدید؛ اختلاف امتیاز و نرخ پیشنهاد","تأیید منتور هنوز لازم است؛ پرچم به معنی اثبات نقص کالا نیست")
    cards(d,[("پرچم ساده",f'{g["naive_flags"]:,}',""),("شواهد ناکافی",f'{g["naive_flags_insufficient_support"]:,}',""),("صف بررسی محافظه‌کارانه",str(g["conservative_flags"]),"")])
    paragraphs(d,["حداقل ۲۰ نظر برچسب‌دار، ۱۰ رأی و امتیاز ۷۰ از ۱۰۰.","شکاف امتیاز با کران بالای Wilson نرخ پیشنهاد حداقل ۱۵ واحد درصد. شواهد نمونه قابل مشاهده‌اند."],y=480,size=30)
    d=add("Demo و مرز ادعا","بخش آزمایشگاه شواهد و امتیازی در تب چهارم Streamlit")
    paragraphs(d,["مقایسه بازیابی با کوله کودک یا غلط تایپی؛ مشاهده اختلاف نتایج.",
                  "اجرای مسیریاب و بررسی Trace؛ اجرای دوباره برای Cache HIT.",
                  "مقایسه مدل اصلی و Head تکمیلی؛ تحلیل شکاف رضایت.",
                  "داوری انسانی واقعی، تأیید منتور و اجرای SDK/سرور Langfuse هنوز نیازمند بررسی هستند."],y=260,size=31)
    slides[0].save(output,"PDF",resolution=120.0,save_all=True,append_images=slides[1:])
