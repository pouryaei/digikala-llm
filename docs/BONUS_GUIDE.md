# آزمایشگاه شواهد و امکانات امتیازی فروهر

نسخه پایه ۱۰۰ هزار محصولی، سه تب قبلی و داده‌ها حفظ شده‌اند. مدل فعال پیش‌بینی LR با Head Fine-tuning است؛ هر چهار مدل برای مقایسه باقی مانده‌اند. به‌روزرسانی گراف QA و مانیتورینگ در همین نسخه ادغام شده است. تب چهارم «آزمایشگاه شواهد و امتیازی» شش زیرتب دارد.

## اجرا در Windows / VS Code

~~~powershell
python -m pip install -r requirements.txt
.\RUN_FOROUHAR_FINAL.bat
~~~

آدرس http://localhost:8505 را باز کنید. بسته کامل شامل SQLite/Parquet، نمایه کاراکتری و وزن‌های مدل تکمیلی است؛ برای Demo آموزش یا دانلود مدل لازم نیست. در بسته سبک، ابتدا داده را مطابق README بازیابی و نمایه را بسازید:

~~~powershell
python scripts/restore_100k_assets.py --archive "D:\path\to\Forouhar_100K_Final.zip"
python -m bonus.retrieval
~~~

## شش زیرتب قابل نمایش

| زیرتب | عملکرد |
|---|---|
| مسیریاب و مانیتورینگ | گراف، زمان گره‌ها، Cache HIT، API usage و تاریخچه Trace |
| مقایسه بازیابی | BM25، Hybrid کاراکتری و Auto Router با پرسش/فیلتر یکسان |
| مدل و آموزش تکمیلی | مدل منتخب Head Fine-tuning، مقایسه چهار مدل، اجرای نظر تازه، جدول‌ها و نمودارهای Hyperparameter Search |
| ارزیابی انسانی | پرسش، پاسخ، منبع، ثبت نمره مستقل، توافق و CSV |
| تحلیل جدید | فیلتر دسته و آستانه شواهد؛ شکاف امتیاز و نرخ پیشنهاد |
| دفاع و گزارش | مسیر Demo و گزارش‌های قابل دریافت، بدون نیاز به اسلاید |

### Demo پیشنهادی

۱. در مقایسه بازیابی عبارت «کوله پشتی بچه برای مهد کودک» را اجرا کنید.

۲. در مسیریاب سناریوی «پرسش مستند» را اجرا و یک‌بار تکرار کنید تا Cache HIT و گراف کوتاه‌تر را ببینید.

۳. سناریوی مقایسه (1696173 و 2896502) و پیش‌بینی را اجرا کنید.

۴. در زیرتب مدل، مقایسه چهار مدل بر اساس Validation و Test را نشان دهید؛ بهترین مدل Head Fine-tuning است. یک نظر تازه را با هر چهار مدل اجرا کنید و در بخش‌های بازشونده هشت ترکیب LR و هشت ترکیب XGBoost را نشان دهید.

۵. آستانه‌های تحلیل جدید را تغییر دهید و شاهد واقعی یک محصول را باز کنید.

۶. فرم ارزیابی انسانی را با داور واقعی تکمیل کنید.

## چرا این معماری؟

کارهای جست‌وجو، RAG، تحلیل SQL و طبقه‌بندی مستقل‌اند و به اجرای همه سرویس‌ها برای هر سؤال نیاز ندارند. گراف START، تشخیص Intent، بررسی Cache، سرویس منتخب و خروجی/مانیتورینگ را مدل می‌کند. شاخه QA شامل resolve، retrieve، policy، extractive/insufficient/generate، validate و fallback است. تولید LLM فقط با وجود شاهد کافی، پوشش معیار مورد سؤال، نیاز به ترکیب شواهد، اجازه صریح کاربر و کلید معتبر انجام می‌شود؛ شاهدهای انتخاب‌شده دوباره بازیابی نمی‌شوند. شکست مدل زبانی جواب مستند محلی را از بین نمی‌برد.

مسیریاب بازیابی درخواست ساده/شناسه/عنوان دقیق را به BM25 می‌دهد و برای درخواست پیچیده یا خروجی خالی از روش ترکیبی استفاده می‌کند. Character TF-IDF با ۲–۴ گرام روی عنوان/برند اجرا می‌شود. اجتماع نامزدها با RRF، پوشش واژه و شباهت بازرتبه‌بندی می‌شود. این نمایه sparse واژگانی/کاراکتری است، نه Dense معنایی. قیود قیمت، دسته، برند و عدد حفظ می‌شوند. شباهت تقریبی ضمانت سازگاری فیزیکی نیست.

## LangGraph و Langfuse (اختیاری)

~~~powershell
python -m pip install -r requirements-bonus-observability.txt
python -m bonus.check_frameworks --require-langgraph
~~~

LangGraph از StateGraph، START/END و conditional edges استفاده می‌کند. اگر SDK نصب نباشد، همان سیاست گره‌ها با Python اجرا می‌شود و موتور fallback صریحاً در رابط مشخص است. در نسخه نهایی SDK واقعی LangGraph، Langfuse، Streamlit و DuckDB نصب و آزمون‌های گراف مجدد اجرا شدند. گزارش qa_graph_ab.json تطابق 36/36 پاسخ آفلاین و کاهش تلاش شبیه‌سازی‌شده مدل از 3 به 1 را ثبت کرده است. این اعداد کیفیت LLM واقعی یا دریافت Trace توسط سرور زنده را ثابت نمی‌کنند. راهنمای راه‌اندازی و تأیید سرور در QA_GRAPH_MONITORING_FA.md است.

مانیتورینگ محلی همیشه فعال است و Traceها را در bonus_state/observability.sqlite3 با زمان، وضعیت، مسیر، Cache و usage ثبت می‌کند؛ متن درخواست خصوصی و نظرات ذخیره نمی‌شوند. برای نظارت اختیاری بیرونی، روی سرور موجود Langfuse پروژه بسازید و در فایل محلی .env تنظیم کنید:

~~~dotenv
FOROUHAR_LANGFUSE_ENABLED=1
LANGFUSE_BASE_URL=http://localhost:3000
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_TRACING_ENVIRONMENT=forouhar
~~~

آدرس نمونه را با آدرس سرور خود جایگزین کنید. فایل .env را منتشر نکنید. برای ارسال Traceهای واقعی آفلاین و بدون مدل زبانی پولی:

~~~powershell
python -m bonus.check_frameworks --require-langgraph --export-langfuse
~~~

Trace ID را در داشبورد واقعی بررسی کنید. ثبت SDK/flush به تنهایی به معنی تأیید دریافت سرور نیست. درگاه مدل زبانی فقط با تیک اجازه پرداخت و کلید گروه استفاده می‌شود. هزینه تنها با usage گزارش‌شده و نرخ واقعی FOROUHAR_INPUT_USD_PER_MILLION و FOROUHAR_OUTPUT_USD_PER_MILLION محاسبه می‌شود.

مستندات رسمی:

- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [Langfuse instrumentation](https://langfuse.com/docs/observability/sdk/instrumentation)
- [Langfuse compatibility](https://langfuse.com/docs/compatibility)
- [Self-host Docker Compose](https://langfuse.com/self-hosting/deployment/docker-compose)

SDK v5 با سرور v4 و SDK v4 با سرور v3 سازگار است (جدول سازگاری رسمی را ببینید). راهنمای v4 برای استقرار معمولی VM حداقل ۱۶ GiB RAM پیشنهاد می‌کند؛ برای لپ‌تاپ محدود الزام نکرده‌ایم سرویس‌های کامل را کنار Streamlit اجرا کنید.

## نتایج و بازتولید

~~~powershell
python -X utf8 -m bonus.evaluate_retrieval
python -X utf8 -m bonus.evaluate_workflow
python -X utf8 -m bonus.quality_gap
python -X utf8 -m unittest discover -s tests -p test_bonus_lab.py -v
~~~

گزارش‌ها در reports/bonus:

- retrieval.json: مقایسه سه روش، ۴۷ مورد قدیمی و ۴۰ مورد غلط تایپی عنوان‌محور.
- retrieval_initial.json: اولین آزمایش و تحلیل شکست/پشتوانه اصلاح.
- retrieval_stress_cases.json: مجموعه ثابت ساخته‌شده پیش از آزمایش.
- router_cache.json: ۲۴ سناریوی Intent، Traceهای اجرای واقعی و کش.
- adaptation.json: اجرای واقعی آموزش تکمیلی مدل طبقه‌بندی.
- hyperparameter_search/report.json و trials.csv: مقایسه ابرپارامترهای مدل طبقه‌بندی، انتخاب روی Validation و ارزیابی نهایی Test.
- quality_gap.json: تحلیل تمام دامنه، حساسیت آستانه و کنترل شواهد.
- core_regression.json: حفظ قابلیت‌های اصلی.
- defense_fa.md: مسیر دفاع با Demo مستقیم.

مجموعه قدیمی مشاهده شده بود و مجموعه غلط تایپی عنوان‌محور است. هر دو تشخیصی‌اند؛ Hit@5 بالا به معنی تعمیم کامل به درخواست‌های انسانی نیست.

### آموزش تکمیلی

Head مدل Logistic Regression ذخیره‌شده با بردارساز ثابت روی نمونه train به‌روزرسانی شد. تقسیم product_id با MD5 مطابق EDA، حذف متن تکراری، validation مستقل و ۱۵ هزار شناسه test ثبت‌شده استفاده شدند. انتخاب epoch، learning rate و bias فقط روی validation بود. مدل اصلی دست‌نخورده است. کنترل هم‌پوشانی برای نمونه‌های جدید این آزمایش و test ثبت‌شده انجام شد؛ validation جدید از گروه‌های validation اصلی نمونه‌گیری شده و لزوماً همان ردیف‌های validation قبلی نیست. فایل کامل ورودی آموزش تاریخی برای بررسی تکرار با همه متن‌های آموزشی قبلی موجود نبود.

~~~powershell
python -X utf8 -m bonus.adapt_classifier --source-archive "D:\path\to\Forouhar_M1_Complete.zip"
~~~

این fine-tuning واقعی Head مدل خطی است، نه LoRA یا آموزش تکمیلی ترنسفورمر pretrained. اگر منتور مورد دوم را شرط امتیاز می‌داند، اجرای آن باقی است.

### Hyperparameter Search مدل Logistic Regression (آزمایش تاریخی)

همان معماری و تنظیمات Character TF-IDF و solver قبلی حفظ شده است. هشت ترکیب
`C = [0.25, 1, 4, 10]` و `class_weight = [None, balanced]` با
`GridSearchCV` و یک `PredefinedSplit` مطابق Train/Validation ثابت آزمایش
می‌شوند. این یک Validation ثابت است، نه K-fold تصادفی. بردارساز فقط روی Train
fit می‌شود؛ همه ترکیب‌ها دقیقاً از همان ویژگی‌ها و نمونه‌های Train استفاده می‌کنند.
با توجه به عدم توازن کلاس‌ها، انتخاب با بیشترین Macro F1 روی Validation
است؛ در تساوی دقیق C کوچک‌تر و سپس balanced اولویت دارد.
تنظیمات برنده دوباره فقط روی Train آموزش می‌بینند و Test تنها پس از انتخاب
ارزیابی می‌شود. در نمونه منتشرشده، داده از آرشیو کامل M1 با سیاست تقسیم
product_id اصلی و حذف کلید متن تکراری نمونه‌گیری شده است.

نتیجه اجرای واقعی (53,716 نظر Train، 15,846 نظر Validation، 15,000 نظر Test ثبت‌شده):

| آزمایش | Macro F1 | Accuracy |
|---|---:|---:|
| تنظیمات پایه روی Validation همین نمونه (C=1, balanced) | 66.33% | 77.86% |
| تنظیمات منتخب روی Validation همین نمونه (C=1, None) | 67.79% | 80.11% |
| نامزد منتخب روی Test | 67.82% | 78.84% |
| مدل اصلی ذخیره‌شده روی همان Test ثبت‌شده | 67.91% | 78.31% |

بهبود Validation برابر 1.46 واحد درصد است. نامزد جدید در Test Accuracy
کمی بالاتر دارد اما Macro F1 آن 0.08 واحد درصد کمتر از مدل اصلی است.
در نسخه پیشین مدل اصلی و Head تکمیلی تغییر نکردند. این‌ها گزارش‌های تاریخی هستند؛ در نسخه نهایی Head Fine-tuning فعال است. نامزد جداگانه برای بررسی
و اجرای بعدی نگهداری شده است. مقایسه با مدل تاریخی به معنی برابری
داده‌های آموزشی آن با نمونه جدید نیست.

گزارش آماده در Streamlit: «آزمایشگاه شواهد و امتیازی → مدل و آموزش تکمیلی».
برای مشاهده جدول و نمودار نیاز به آموزش مجدد نیست. اجرای دوباره با منبع کامل:

~~~powershell
python -X utf8 -m recommendation_ml.hyperparameter_search --source-archive "D:\path\to\Forouhar_M1_Complete.zip" --output-dir reports/bonus/hyperparameter_rerun --threads 3
~~~

یا اگر سه فایل ثابت خروجی EDA موجود هستند، برای اجرای روی کل splitها:

~~~powershell
python -X utf8 -m recommendation_ml.hyperparameter_search --data-dir ".\digikala\processed_repo_based" --output-dir reports/bonus/hyperparameter_full_eda --threads 3
~~~

هر اجرا یک پوشه خالی جدید می‌خواهد و گزارش JSON، CSV همه ترکیب‌ها،
پیش‌بینی Test و `tuned_candidate.joblib` را داخل آن ثبت می‌کند.
`m2_outputs/best_model.joblib` و `bonus_artifacts/adapted_head.npz`
بازنویسی یا جایگزین نمی‌شوند. رابط به‌صورت پیش‌فرض گزارش همراه بسته را
نمایش می‌دهد. برای نمایش اجرای تازه، پس از بررسی نتایج و نگهداری گزارش
قبلی، گزارش و CSV اجرای تازه را در reports/bonus/hyperparameter_search
قرار دهید. مقایسه تنظیمات روی Validation درون همین آزمایش منصفانه است؛
مقایسه با مدل تاریخی روی Test یکسان، در صورت وجود، الزاماً مقایسه دو
اندازه آموزش یکسان نیست. Test ثبت‌شده نیز آزمون کور جدید محسوب نمی‌شود.

### XGBoost + Hyperparameter Search (نامزد مقایسه)

Character TF-IDF با همان ۲ تا ۵ گرام و حداکثر ۲۰۰ هزار ویژگی حفظ شده
است. به دلیل هزینه مدل درختی در فضای sparse بزرگ، SelectKBest(chi2)
فقط از Train تعداد ۴٬۰۹۶ ویژگی انتخاب می‌کند. XGBoost از objective=multi:softprob،
tree_method=hist، max_bin=32، subsample=0.85، colsample_bytree=0.5،
min_child_weight=3 و reg_lambda=2 با CPU استفاده می‌کند.

| Hyperparameter | مقادیر |
|---|---|
| max_depth | 4, 6 |
| learning_rate | 0.10, 0.20 |
| n_estimators | 120, 240 |

چهار مسیر آموزش تا ۲۴۰ boosting round اجرا شدند؛ خروجی هر مسیر در
checkpointهای ۱۲۰ و ۲۴۰ با Validation Macro F1 سنجیده شد (۸ ترکیب).
در تساوی، تعداد درخت کمتر و عمق کمتر اولویت دارد. برنده:
max_depth=6، learning_rate=0.2، n_estimators=240. انتخاب و بردارساز
هیچ اطلاعاتی از Test دریافت نکردند. تعداد نمونه‌ها 53,716/15,846/15,000
و هم‌پوشانی شناسه نظر، محصول و کلید متن صفر است.

| مدل روی 15,000 شناسه Test یکسان | Macro F1 | Accuracy |
|---|---:|---:|
| Logistic Regression تاریخی | 67.91% | 78.31% |
| LR با Hyperparameter Search قبلی | 67.82% | 78.84% |
| Head تکمیلی LR تاریخی | 69.23% | 81.63% |
| XGBoost | 66.98% | 77.96% |

Validation Macro F1 مدل منتخب XGBoost برابر 67.24% بود. XGBoost در
این آزمایش از مدل تاریخی بهتر نیست؛ پس در نسخه نهایی فقط نامزد مقایسه است و Head Fine-tuning فعال شده است. مقایسه با مدل تاریخی، مقایسه الزاماً هم‌اندازه Train نیست.
Test آزمون ثبت‌شده قبلی است و آزمون کور جدید نیست.

برای بازتولید:

~~~powershell
python -X utf8 -m recommendation_ml.train_xgboost --source-archive "D:\path\to\Forouhar_M1_Complete.zip" --output-dir m2_outputs/xgboost_rerun --threads 5
python -X utf8 -m recommendation_ml.replay_xgboost --source-archive "D:\path\to\Forouhar_M1_Complete.zip"
~~~

برای کل سه split ثابت EDA، گزینه --data-dir را به جای --source-archive
بدهید. مدل XGBoost از Native model.ubj، preprocessor.joblib و report.json
بارگذاری می‌شود؛ سه فایل همراه بسته حاضرند. نمایش گزارش و پیش‌بینی
در Streamlit به بازآموزی نیاز ندارد. گزارش‌های قدیمی LR فقط تاریخی‌اند.
هیچ هزینه API یا GPU برای این آزمایش لازم نیست.

### انتخاب نهایی و اجرا روی داده

گزارش reports/final_model/report.json پس از بازاجرای واقعی هر چهار مدل
روی همان Validation و Test به دست آمده است. رتبه‌بندی بر اساس Validation
Macro F1 و قبل از بازاجرای Test انجام شد. بهترین مدل Head Fine-tuning با
Validation Macro F1=70.52%، Test Macro F1=69.23% و Test Accuracy=81.63%
است. این Head روی وزن مدل خطی تاریخی و بردارساز ثابت ساخته شده؛
خود Test قبلاً مشاهده شده و مجموعه کور تازه نیست.

اسکریپت recommendation_ml.predict_dataset مدل منتخب را به صورت دسته‌ای
روی CSV یا Parquet اجرا می‌کند. راهنمای سریع در START_HERE_FA.md است.

## ارزیابی انسانی و تأیید منتور

فرم، نمرات دستیار را قبل از داوری پنهان می‌کند. از داورهای مستقل بخواهید ۲۹ سؤال را با خواندن نظرهای منبع ارزیابی کنند و نمره ۰ تا ۲ کیفیت/Grounding با دلیل ثبت کنند. توافق دقیق و Weighted Kappa از نمرات واقعی محاسبه می‌شود. تا زمان تحویل، تعداد داوری انسانی واقعی صفر است.

صورت مسئله جدید برای تأیید منتور:

**آیا می‌توان محصولاتی با امتیاز بالا اما نرخ پیشنهاد خرید پایین را، با در نظر گرفتن حجم شواهد، برای بررسی مدیر دسته اولویت‌بندی کرد؟**

روش محافظه‌کارانه از کران بالای Wilson نرخ پیشنهاد استفاده می‌کند. امتیاز تجمیعی و نظرهای برچسب‌دار دو جمعیت متفاوت‌اند؛ پرچم به معنی اثبات نقص کالا نیست. شمارش و شواهد بررسی شده‌اند، ولی تأیید منتور و سودمندی انسانی باید واقعی ثبت شوند.

## وضعیت بندهای امتیازی

| بند | وضعیت |
|---|---|
| Retrieval (۴) | روش ترکیبی و شواهد کمی واقعی |
| Fine-tuning (۴) | Head تکمیلی LR منتخب با بهبود؛ چهار مدل مقایسه شده‌اند؛ LoRA ترنسفورمر انجام نشده |
| Router (۳) | مسیر و گراف، آزمایش Intent و مقایسه اجرای مدل دوم |
| ارزیابی پیشرفته (۳) | فرم و توافق آماده؛ داوری مستقل باقی |
| UI (۳) | تب چهارم و شش زیرتب؛ تست مرورگر اینجا انجام نشده |
| Caching (۳) | آزمایش گرم/بدون کش و پاسخ یکسان |
| مسئله جدید (۳) | راه‌حل و ارزیابی آماده؛ تأیید منتور باقی |
| ارائه (۲) | مسیر Demo و گزارش‌ها؛ اسلاید جدید تولید نشده |

این جدول وضعیت کار است، نه نمره تضمینی داور.


## دستیار حرفه‌ای
ProfessionalWorkflow روی گراف فعلی، سیاست مشترک کش/محلی/مدل زبانی
و سقف دو تلاش را پیاده می‌کند. مدل منتخب و داده حفظ شده‌اند. گزارش
professional_policy.json آزمایش fixture است؛ برای کیفیت و هزینه واقعی
باید با کلید گروه ارزیابی شود. راهنما: PROFESSIONAL_IMPLEMENTATION_FA.md.
