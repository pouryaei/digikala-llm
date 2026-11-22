# QBC13 — Digikala AI Shopping & Review Intelligence
## Master Project Specification v2.2
**Project duration:** 8 days  
**Team size:** 5 members  
**Status:** Day 1 — Research, Architecture Freeze & Team Setup (Not Started)  
**Document role:** Single source of truth for project scope, architecture, workflow, evaluation, team ownership, and execution.

---

# 1. Project Overview

Build a Persian-language AI system for discovering Digikala products, answering questions grounded in customer reviews, comparing products, analyzing categories/brands, and predicting review recommendation status.

The project should demonstrate a practical end-to-end AI engineering workflow: data engineering, classical ML, Transformer experimentation, information retrieval, RAG, LLM integration, API development, UI, evaluation, deployment, and teamwork.

**Project name:** QBC13 — Digikala AI Shopping & Review Intelligence

## Primary outcomes
1. Product Discovery through Persian natural-language queries and filters.
2. Review QA / RAG answers grounded in actual review evidence.
3. Product Comparison using product facts and review evidence.
4. Category/Brand Analytics using deterministic calculations.
5. Review Recommendation Classification using classical ML and a Transformer experiment.

The project is an 8-day bootcamp deliverable. Prioritize a working, measurable, integrated system over infrastructure complexity or unfinished experimental features.

---

# 2. Problem Statement

Product catalogs and customer reviews contain substantial information, but users often cannot efficiently:
- Find products that match a natural-language need.
- Understand patterns across reviews.
- Identify strengths, weaknesses, and conflicting customer experiences.
- Compare alternatives using both product facts and review evidence.
- Explore category/brand-level trends.
- Automatically classify review recommendation status.

QBC13 provides a single interface to these capabilities and makes its conclusions traceable to data and evidence.

---

# 3. Scope and Core Features

## 3.1 Product Discovery
Support Persian natural-language search, product filters, and ranking.

Potential query dimensions:
- Product/category
- Brand
- Price range
- Rating
- Review count
- Recommendation/satisfaction signals
- Product relevance to user needs

The LLM may interpret the query and produce structured filters, but deterministic application code validates and applies filters and performs ranking.

## 3.2 Review QA / RAG
Answer questions about products using retrieved customer reviews.

Requirements:
- Retrieve relevant reviews.
- Provide review/product IDs or traceable references.
- Ground answers in retrieved evidence.
- Distinguish individual opinions from aggregate claims.
- Include conflicting evidence and uncertainty.
- State when evidence is insufficient.
- Do not fabricate quotations, reviews, IDs, or claims.

## 3.3 Product Comparison
Compare two or more products using:
- Product facts and metadata.
- Price.
- Category and brand.
- Rating and rating count.
- Review count.
- Recommendation rate when validly calculated.
- Review-based strengths and weaknesses.
- Evidence coverage and limitations.

Keep factual product attributes separate from review observations and recommendations. Recommendations should be conditional, not universal.

## 3.4 Category / Brand Analytics
Use deterministic SQL/DuckDB computations for:
- Product/review counts.
- Average and median ratings.
- Recommendation rate and its denominator.
- Rating/recommendation distributions.
- Brand/category comparisons.
- Trends where timestamps and coverage allow.
- Frequent positive/negative themes.

The LLM may explain or summarize calculated results. It must not invent metrics or replace deterministic calculations.

## 3.5 ML Classification
Predict `recommendation_status` from Persian review text, using the dataset labels:
- `recommended`
- `not_recommended`
- `no_idea`

Compare classical ML baselines with a pretrained Transformer experiment. Do not assume the Transformer is superior.

---

## 3.6 Scope Tiers

The eight-day timeline is highly constrained. Features are prioritized into delivery tiers to prevent scope overload and ensure a reliable, integrated submission. The tiers are delivery priorities, not claims of production readiness.

| Tier | Features |
|---|---|
| **Must** (without these, the project fails) | Product Discovery + Review QA with citations + Analytics + Classical ML (TF-IDF + LR) + FastAPI + simple Streamlit + Docker + basic evaluation |
| **Should** | Product Comparison + Hybrid Retrieval (RRF) + Transformer experiment (if VRAM allows) + Reranker |
| **Nice** | Query Router + advanced caching + advanced dashboard + embedding fine-tuning |

Implementation and time allocation must protect Tier Must first. Tier Should should be attempted only when Must is functional and time/resources permit. Tier Nice is explicitly optional and must not delay integration, testing, or delivery.

# 4. Dataset

Source dataset: `RadeAI/Digikala_comments_products` on Hugging Face.

Files:
- `digikala-products.csv`
- `digikala-comments.csv`

Dataset scale is approximately over one million products and over six million reviews (verify actual downloaded counts).

## 4.1 Product fields (available fields to audit)
- `id`
- `title_fa`
- `Rate`
- `Rate_cnt`
- `Category1`
- `Category2`
- `Brand`
- `Price`
- `Seller`
- `Is_Fake`
- `min_price_last_month`
- `sub_category`

## 4.2 Review fields (available fields to audit)
- `id`
- `title`
- `body`
- `created_at`
- `rate`
- `recommendation_status`
- `is_buyer`
- `product_id`
- `advantages`
- `disadvantages`
- `likes`
- `dislikes`
- `seller_title`
- `seller_code`
- `true_to_size_rate`

Audit actual schemas, data types, missingness, duplicates, label distribution, language/content quality, product-review join coverage, and data anomalies before making assumptions.

## 4.3 Data handling rules
- Preserve raw downloaded data unchanged.
- Use separate raw, processed, and serving layers.
- Use Parquet for processed storage.
- Use DuckDB for local analytical/query access.
- Do not load the full dataset blindly into Pandas.
- Do not embed all six million reviews without a measured feasibility justification.
- Choose sample/corpus sizes based on coverage, evaluation, memory, runtime, and index size.
- Keep data lineage and IDs for evidence traceability.
- Avoid committing raw datasets and oversized generated artifacts.

---

# 5. Architecture — Frozen Baseline v2.2

## 5.1 Architectural style
**Modular Monolith.** No microservices.

## 5.2 Main components
- UI: Streamlit
- Backend/API: FastAPI + Pydantic
- Data and analytics: DuckDB + Parquet + local filesystem
- Intelligence/orchestration: query router, parser, product search, retrieval, RAG, comparison, analytics, classifier inference
- Retrieval: lexical search + dense search + hybrid fusion (RRF) + optional reranker + evidence selection
- Embeddings: benchmark local embedding models and/or bootcamp-provided API
- LLM: bootcamp-provided OpenAI-compatible API
- ML: Scikit-learn + PyTorch + Hugging Face Transformers
- Vector index: FAISS candidate; local library/index, not a separate vector database service
- Deployment: Docker Compose
- GPU: local RTX 3050 where appropriate, subject to hardware and VRAM audit

## 5.3 High-level data flow

Streamlit UI  
→ FastAPI  
→ Application / Intelligence Orchestration  
→ Product Search / Review QA / Comparison / Analytics / ML Prediction  
→ DuckDB + Parquet data access  
→ Lexical and Dense Retrieval / Hybrid RRF / optional Reranker  
→ Local GPU and/or hosted LLM API  
→ Response, evidence, IDs, metrics, and traceable results

## 5.4 Architecture constraints
Do not add or use:
- Kafka
- Spark
- PostgreSQL
- Redis
- Kubernetes
- Microservices
- Multi-agent architecture
- Distributed infrastructure
- Heavy local LLM inference
- LLM training from scratch
- Mandatory large-model fine-tuning
- Mandatory embedding fine-tuning
- Complex authentication or multi-tenant production infrastructure
- Real-time event processing
- Unnecessary cloud services or databases

Architecture changes require a genuine blocker, a written explanation, and explicit project-owner approval. Do not repeatedly redesign the architecture.

---

# 6. Hardware and Runtime Audit

Laptop: Asus FX506HEB  
CPU: Intel Core i7-11800H  
GPU: NVIDIA RTX 3050 laptop GPU (exact VRAM must be verified)  
Python environment and CUDA compatibility: audit on Day 1.

Audit:
- OS and environment.
- Python version.
- GPU model and VRAM.
- NVIDIA driver.
- CUDA availability in installed PyTorch.
- Available disk space.
- RAM.
- GPU utilization and memory under a small benchmark.
- Bootcamp API connectivity, model list, limits, and response behavior.

Do not assume CUDA works merely because an NVIDIA GPU is present. Verify with a small reproducible test.

GPU usage is preferred where it gives practical value, but avoid wasting the limited GPU memory or blocking teammates. Coordinate GPU-intensive jobs.

---

# 7. LLM Provider and Model Selection

Bootcamp OpenAI-compatible endpoint:
`https://193.93.169.12/v1`

Environment variable:
`QBC13_API_KEY`

Never hardcode credentials, commit `.env`, print secrets, or place keys in notebooks/reports.

## Day-1 API audit
- Verify endpoint accessibility and TLS/network behavior.
- Verify authentication.
- Retrieve available model IDs if supported.
- Test chat completion.
- Test structured output/JSON behavior if supported.
- Test timeout/error behavior.
- Record latency, response quality, token usage, limits, and provider constraints.
- Do not assume any named model is available until verified.

Candidate model families may include GPT, Gemini, DeepSeek, Qwen, Llama, or other models exposed by the provider. Availability is not guaranteed.

## Benchmark protocol
Use a project-specific Persian query set (initially 20–30 representative tasks) and evaluate:
- Intent/query parsing accuracy.
- Persian quality and clarity.
- Structured output validity.
- Grounding and hallucination.
- Task success.
- Latency.
- Reliability and error rate.
- Token usage/cost or quota consumption.

No model is preselected as winner. Use task-specific models only if measured gains justify extra complexity. Keep model IDs and configurations in config files, not hardcoded across modules.

---

# 8. Embedding Model Research and Selection

Candidate models:
- `intfloat/multilingual-e5-small`
- `intfloat/multilingual-e5-base`
- `BAAI/bge-m3`
- Embedding models exposed by the bootcamp API, if available

Selection criteria:
- Persian retrieval quality.
- Query/document handling.
- Model size and RAM/VRAM.
- Embedding throughput.
- Index size.
- Retrieval metrics.
- Latency.
- License and deployment suitability.

For E5 models, use the appropriate `query:` and `passage:` prefixes for asymmetric retrieval.

Benchmark on a representative corpus and evaluation set. Do not assume the largest model is best for this time budget.

---

# 9. Retrieval and RAG Design

## 9.1 Retrieval stages
1. Query understanding and product resolution.
2. Query normalization.
3. Product/review filtering.
4. Dense retrieval (embedding model + FAISS; mandatory baseline).
5. Optional lexical retrieval (BM25) and Hybrid fusion using RRF, only if the Section 9.10 gates pass.
6. Optional reranker, only if Tier Should feasibility and evaluation justify it.
7. Evidence selection and context construction.
8. LLM answer generation.
9. Citation/evidence validation.
10. Response with references, uncertainty, and traceability.

## 9.2 Lexical retrieval
Benchmark BM25 or an appropriate lexical baseline. Keep retrieval metadata and original IDs.

**Persian normalization is mandatory before BM25 tokenization/indexing and query matching.** At minimum, normalize Arabic/Persian variants of Yeh (`ي`/`ی`) and Kaf (`ك`/`ک`), consistently handle/remove ZWNJ (half-space), normalize digits and punctuation, and apply the same normalization policy to both indexed text and queries. Preserve original text for display and evidence; normalization is for retrieval representations.

Use a lightweight, tested normalization pipeline. `persian-rag` normalization, `hazm`, or `Parsivar` may be evaluated; do not add a dependency unless it is compatible and useful. On Day 2, report BM25-only Hit Rate@5 separately. If BM25 Hit Rate@5 is more than 15 percentage points below Dense-only on the same query set, mark BM25 as a likely noise source and do not allow it into RRF. This threshold is a project gate, not a universal retrieval rule. Record the measured effect of spelling/normalization variants in the Experiment Log.

A public `persian-rag-bench` report describes 34 documents and 42 questions and reports Sparse Hit@1 of 0.881 versus 0.952 for Sparse + Rewrite (a +7.1 percentage-point difference). This is evidence that Persian spelling/query rewriting can matter on that benchmark; it is not evidence that normalization alone caused that gain or that the same gain will occur on Digikala reviews.

## 9.3 Dense retrieval
Generate embeddings for a selected review corpus. Store mapping from vector index positions to source review IDs/product IDs and metadata.

## 9.4 Hybrid retrieval
Retrieve lexical and dense candidates in parallel and fuse their rankings with RRF.

Formula:
`RRF(d) = Σ 1 / (k + rank(d))`

Use a documented `k` and implementation. The same document can appear in both lists; preserve deduplication and source metadata.

## 9.5 Reranking
A cross-encoder or other reranker is optional. Add it only if evaluation shows a meaningful retrieval-quality improvement and acceptable latency/resource cost.

## 9.6 Corpus sizing
Evaluate progressively, for example:
- 10K–20K reviews as the initial corpus
- 50K reviews only if the expansion gate below passes
- Larger/full relevant corpus only if feasibility evidence supports it and time remains; 100K and 500K are not planned targets for this eight-day project

These are experimental sizes, not fixed final requirements. Sample to maintain product/category coverage and avoid duplicates or severe distribution bias.

### VRAM-Aware Index Build Protocol
On Day 3, start index/embedding work with 10K–20K reviews, not 50K or 100K. Record build/embedding time, peak system RAM, peak GPU VRAM (if GPU is used), free memory after the run, and index file size. Do not start directly at 100K.

Expansion to 50K is allowed only if all three conditions are met: (a) the 10K–20K run completes in under 30 minutes; (b) more than 2GB of GPU VRAM remains free during/after the chosen GPU workload (record system RAM separately; if CPU-only, report RAM and do not treat VRAM as applicable); and (c) Hit@5 improves by more than 5% on the same 20-query Silver set, using the same retrieval configuration and comparable evaluation. If any condition fails or cannot be verified, remain at the initial corpus size and report: “Corpus size was not expanded because of local hardware/time constraints.”

For FAISS indexing, prefer a CPU index for the initial small corpus unless a measured GPU advantage justifies GPU indexing. CPU embedding/inference or CPU FAISS is the fallback if GPU memory is constrained. Batch size applies primarily to embedding generation, not as a universal FAISS index-build setting: start CPU embedding batches at 32 and benchmark; if GPU embedding is attempted on a GPU with 4GB VRAM or less, start conservatively at batch size 1–4 and increase only after memory checks (the candidate range 1–8 is a ceiling for testing, not a guaranteed safe value). Keep `faiss-cpu` as the practical fallback. The official FAISS GPU documentation notes that GPU resources reserve temporary memory by default and that batch processing is useful, but does not establish the specific CPU 32–64 / GPU 1–8 rule as a universal recommendation.

## 9.7 Retrieval evaluation
Evaluate retrieval using metrics supported by the selected ground-truth method in Section 20.1. Method A uses Hit Rate@1, Hit Rate@5, and MRR only; Method B uses Precision@10 only. Recall@K and NDCG@K must not be presented as Digikala retrieval metrics under those limited silver-label methods. If the optional `persian-rag-bench` dataset is used, its supported metrics may be reported separately with its dataset/domain clearly labeled.

Also record latency, memory use, and index size. Do not claim a retrieval system is good solely from a few manually observed examples.

## 9.8 RAG grounding rules
- Answers must be based on retrieved evidence.
- Review/product IDs must be real and exist in the selected evidence.
- Do not invent direct quotations.
- Distinguish review opinions from product facts.
- Avoid representing a retrieved sample as the complete review population.
- Mention conflicting evidence when material.
- State insufficient evidence when appropriate.
- Validate citations/references before returning the response.
- Use an evaluation set and manually inspect failure cases.

---

## 9.9 Caching Strategy

Caching is required from Day 3 to reduce repeated API calls, latency, and unnecessary recomputation during development, evaluation, debugging, and demo preparation.

Required cache categories:
- **LLM responses:** cache by a hash of the effective prompt/input and relevant configuration, including model identifier, prompt/version, and generation parameters.
- **Embeddings:** cache by a hash of the normalized text plus embedding model/version and relevant preprocessing settings.
- **Retrieval results:** cache by a hash of the query, filters, retrieval configuration, and index/corpus version so stale results are not reused after meaningful changes.

Simple local implementation options include `diskcache`, SQLite, or JSON files on disk. Choose one practical approach; do not build a caching service.

Caching must have a documented invalidation/versioning strategy. It must not reuse results when effective inputs, model/configuration, or index/corpus versions have changed. Keep cache storage local and exclude unnecessary cache files from Git.

## 9.10 Dense-only Baseline Gate
By the end of Day 2, M3 must implement and evaluate Dense-only retrieval (embedding + FAISS, without BM25 or RRF) on the same evaluation queries and using the same evaluation method/metrics intended for any Hybrid comparison. At minimum, record Hit Rate@5 and MRR.

The Day-4 decision to include Hybrid/RRF is conditional: Hybrid must improve both Hit Rate@5 and MRR by at least 5% relative to Dense-only on the same query set. If either metric fails to meet that threshold, remove BM25 and RRF from the deliverable and ship Dense-only. The 5% threshold is relative improvement, not percentage points.

Dense-only is a valid first-class final outcome, not a failure. Record Dense-only and Hybrid results query-by-query in the Experiment Log, including where one wins/loses and the actual metrics. Cite and discuss *The Fusion Tax: A Diagnostic Analysis of When Hybrid Retrieval Hurts* (IEEE Access, 2026) as external motivation for measuring fusion rather than assuming it helps; its results are on other corpora and are not project-specific proof.

# 10. Product Search and Ranking

## 10.1 Query processing
The LLM may parse Persian query text into a structured intent/filter object. Validate the output using Pydantic and deterministic code.

Possible fields:
- Query text
- Category
- Brand
- Minimum/maximum price
- Rating threshold
- Review count threshold
- Required/avoid attributes
- Sort/ranking preferences

Invalid or unsupported filters must be handled safely and transparently.

## 10.2 Ranking
Potential ranking signals:
- Query relevance
- Category match
- Brand match
- Price fit
- Rating
- Rating count
- Review count
- Recommendation rate
- Review evidence confidence/coverage

Weights must be documented and tested. Do not rank solely by rating. Missing values must not be silently treated as zero. Explain or expose ranking criteria where practical.

---

# 11. Product Comparison

Comparison output should include, where available:
- Product identity/title
- Brand/category
- Price
- Rating and rating count
- Review count
- Recommendation rate with valid denominator
- Review-derived strengths
- Review-derived weaknesses
- Evidence/review references
- Data coverage and limitations

Rules:
- Facts and opinions are separate.
- Use comparable definitions and periods where applicable.
- Missing data is not zero.
- Avoid unsupported winner declarations.
- Give conditional recommendations based on different user priorities.
- Cite review evidence by real IDs/traceable links or IDs.

---

# 12. Analytics

Use DuckDB and deterministic SQL or Python calculations over validated data.

Possible analytics:
- Product and review counts.
- Category/brand counts.
- Mean/median rating.
- Rating distribution.
- Recommendation distribution and rate.
- Time-based trend, if timestamps are reliable and coverage is sufficient.
- Positive/negative themes.
- Brand/category comparison.
- Filters and denominators.

LLM role: interpret and explain computed outputs only.

Theme analysis may use keyword methods, TF-IDF, clustering, embeddings, or LLM categorization on selected evidence. Validate themes. Counts and percentages must be deterministic and reproducible.

---

# 13. ML Classification Experiments

## 13.1 Target
`recommendation_status`:
- `recommended`
- `not_recommended`
- `no_idea`

Audit label frequencies, missing labels, invalid labels, and possible label noise.

## 13.2 Input experiments
Start with:
- `title + body`

Then compare adding:
- `advantages`
- `disadvantages`

Avoid target leakage. Inspect duplicates and near-duplicates. If feasible, use product-aware/group-aware splitting to reduce leakage from reviews of the same product. Document the split strategy and its limitations.

## 13.3 Baselines
- Majority-class baseline.
- TF-IDF + Logistic Regression.
- TF-IDF word n-grams + character n-grams + Linear SVM.
- Class weighting where justified.

## 13.4 Transformer experiment
Compare suitable Persian/multilingual pretrained Transformer candidates (e.g., ParsBERT-compatible models and XLM-R candidates after model suitability audit).

Audit:
- Persian suitability.
- License.
- Model/tokenizer size.
- Maximum sequence length.
- VRAM/RAM needs.
- Training time.
- Inference throughput.

Use PyTorch and Hugging Face Transformers. Start with a small feasibility benchmark before full training. Use GPU when available and verified. Tune batch size/sequence length based on measured memory; mixed precision only if supported and validated; early stopping and reproducible seeds.

Do not assume Transformer will outperform classical baselines. Do not tune against test set.

## 13.5 Metrics
Primary:
- Macro F1

Also report:
- Accuracy
- Weighted F1
- Per-class precision/recall/F1
- Confusion matrix
- Error analysis

Save a reproducible model artifact and expose a stable prediction interface.

---

## 13.6 Transformer Feasibility Gate

The Transformer experiment is conditional on feasibility and must not jeopardize the core deliverable. By the end of Day 4, stop the Transformer experiment and report it as an experiment with limitations if any of the following applies:
- It does not achieve Macro-F1 better than the TF-IDF + Logistic Regression baseline on the same validation split and comparable preprocessing.
- Training takes more than 2 hours.
- It encounters an out-of-memory (OOM) failure that cannot be resolved within the available time without compromising the core project.

When the gate is triggered, the Classical ML model is delivered as the final classifier. The Transformer result, including the reason for stopping, is documented in the final report and Failure Analysis where applicable. This is a valid experiment outcome, not a project failure. Do not use the test set to make this decision or tune models.

# 14. API Contract — Initial Baseline

Initial endpoints:
- `GET /health`
- `POST /api/search`
- `POST /api/reviews/ask`
- `POST /api/products/compare`
- `POST /api/analytics`
- `POST /api/predict`
- `GET /api/categories`
- `GET /api/products/{id}`

Requirements:
- FastAPI.
- Pydantic request/response schemas.
- Input validation.
- Consistent response shape.
- Appropriate HTTP status codes.
- Error handling and timeouts.
- Logging and traceability.
- Do not expose stack traces/secrets to users.
- Preserve product/review IDs and evidence.
- Stable interfaces for team integration.
- OpenAPI docs available through FastAPI where configured.

The contracts are frozen early. Any change must be communicated to affected members and reviewed for integration impact.

---

# 15. Streamlit UI

Required pages (Tier Must):
1. Product Discovery
2. Review QA
3. Analytics
4. ML Prediction

Should pages (Tier Should — only if the corresponding backend feature is implemented):
5. Product Comparison

Optional if time allows (Tier Nice):
6. Evaluation / Model Information

UI requirements:
- Persian user-facing text.
- Clear loading and error states.
- Display sources/evidence IDs where relevant.
- Explain limitations and missing data.
- Avoid misleading charts or unsupported claims.
- Use backend/API functions rather than duplicating business logic in UI.

---

# 16. Repository Structure

```text
digikala-ai-assistant/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── configs/
│   ├── app.yaml
│   ├── models.yaml
│   ├── retrieval.yaml
│   ├── evaluation.yaml
│   └── gpu.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── serving/
│   ├── indexes/
│   └── cache/
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_ml_baselines.ipynb
│   ├── 04_transformer_experiments.ipynb
│   └── 05_retrieval_experiments.ipynb
├── scripts/
│   ├── download_data.py
│   ├── audit_data.py
│   ├── process_data.py
│   ├── build_serving_data.py
│   ├── build_indexes.py
│   ├── train_model.py
│   └── evaluate.py
├── src/
│   └── digikala_ai/
│       ├── __init__.py
│       ├── config/
│       ├── data/
│       │   ├── loader.py
│       │   ├── audit.py
│       │   ├── cleaning.py
│       │   ├── normalization.py
│       │   ├── schemas.py
│       │   ├── sampling.py
│       │   └── validation.py
│       │   ├── storage/
│       │   │   ├── duckdb_manager.py
│       │   │   ├── parquet_manager.py
│       │   │   └── file_manager.py
│       │   ├── search/
│       │   │   ├── product_search.py
│       │   │   ├── query_filters.py
│       │   │   ├── product_ranking.py
│       │   │   ├── lexical_search.py
│       │   │   ├── dense_search.py
│       │   │   ├── hybrid_search.py
│       │   │   ├── fusion.py
│       │   │   ├── reranker.py
│       │   │   └── evidence_selection.py
│       │   ├── intelligence/
│       │   │   ├── router.py
│       │   │   ├── query_parser.py
│       │   │   ├── llm_client.py
│       │   │   ├── prompts.py
│       │   │   ├── review_qa.py
│       │   │   ├── comparison.py
│       │   │   └── analytics.py
│       │   ├── ml/
│       │   │   ├── dataset.py
│       │   │   ├── preprocessing.py
│       │   │   ├── split.py
│       │   │   ├── train_baseline.py
│       │   │   ├── train_transformer.py
│       │   │   ├── evaluate.py
│       │   │   ├── predict.py
│       │   │   └── artifacts.py
│       │   ├── evaluation/
│       │   │   ├── datasets.py
│       │   │   ├── retrieval_metrics.py
│       │   │   ├── rag_metrics.py
│       │   │   ├── ml_metrics.py
│       │   │   ├── system_metrics.py
│       │   │   └── run_eval.py
│       │   ├── api/
│       │   │   ├── main.py
│       │   │   ├── routes.py
│       │   │   ├── schemas.py
│       │   │   ├── dependencies.py
│       │   │   └── error_handlers.py
│       │   └── utils/
│       │       ├── logging.py
│       │       ├── text.py
│       │       ├── timing.py
│       │       └── device.py
├── app/
│   ├── streamlit_app.py
│   ├── pages/
│   └── components/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── api/
│   ├── evaluation/
│   └── data/
├── reports/
│   ├── data_quality/
│   ├── eda/
│   ├── ml_evaluation/
│   ├── transformer_evaluation/
│   ├── retrieval_evaluation/
│   ├── rag_evaluation/
│   ├── system_benchmarks/
│   ├── failure_analysis/
│   └── final_results/
├── artifacts/
│   ├── ml_models/
│   ├── embedding_models/
│   ├── vector_indexes/
│   └── metadata/
└── docs/
    ├── architecture/
    ├── api_contract/
    ├── data_dictionary/
    ├── project_plan/
    ├── decisions/
    ├── evaluation_protocol/
    └── team_workflow/
```

Structure is a target scaffold. Do not create every directory/file prematurely. Add components as needed while preserving agreed interfaces and module boundaries.

## Repository hygiene
- Do not commit raw datasets.
- Do not commit `.env` or credentials.
- Do not commit unnecessary large model/index artifacts.
- Keep notebooks for experiments; reusable logic belongs in `src/`.
- Use clear English filenames, identifiers, commit messages, and API names.
- Document how to reproduce experiments and run the system.

---

# 17. Team Roles (Five Members)

## Member 1 — Data Engineering & EDA
Ownership:
- Dataset download and raw preservation.
- Schema audit and data quality.
- Cleaning and Persian normalization.
- Parquet processing.
- DuckDB storage/query layer.
- Product/review join and coverage.
- Serving data preparation.
- Data dictionary.
- Analytics aggregates and correctness.

Deliverables:
- Audited dataset report.
- Reproducible data pipeline.
- Processed/serving dataset.
- DuckDB access functions.
- Data dictionary and quality report.

## Member 2 — ML & Transformer
Ownership:
- Label distribution and quality.
- Dataset construction and split strategy.
- Leakage/duplicate checks.
- Majority baseline.
- TF-IDF/LR/SVM baselines.
- Transformer feasibility and training experiment.
- GPU training if feasible.
- Metrics, confusion matrix, error analysis.
- Model artifact and inference interface.

Deliverables:
- Experiment notebook/scripts.
- Evaluation report.
- Saved candidate model.
- Stable prediction function/API interface.
- Honest comparison of baseline and Transformer.

## Member 3 — Search & Retrieval
Ownership:
- Persian query normalization.
- Product search.
- Lexical/BM25 retrieval.
- Embedding benchmark.
- Dense retrieval and FAISS.
- Hybrid search and RRF.
- Optional reranker experiment.
- Evidence selection.
- Retrieval evaluation.

Deliverables:
- Search/retrieval modules.
- Index build scripts and metadata mappings.
- Benchmark report.
- Stable retrieval interface.

## Member 4 — Backend & LLM
Ownership:
- FastAPI application.
- Pydantic contracts.
- LLM API client and model benchmark.
- Query parser and routing.
- RAG orchestration.
- Comparison logic.
- Analytics orchestration.
- Errors, timeouts, logging, citation validation.

Deliverables:
- API endpoints.
- LLM client/config.
- Intelligence modules.
- Integration-ready contracts.
- Tests and documentation.

## Member 5 — Integration Coordinator, Demo & Documentation
Ownership:
- Integration coordination and dependency tracking.
- Demo flow and rehearsal.
- README and final documentation coordination.
- Presentation and final packaging.
- Failure Analysis report coordination and consolidation.

Deliverables:
- Integration status and dependency/blocker tracking.
- Demo script and backup demo materials if permitted.
- README and final presentation.
- Consolidated Failure Analysis.

M5 is not the primary UI developer or Evaluation Framework implementer. UI ownership is assigned to M3 starting Day 3 (MVP scope only). Evaluation Framework ownership is assigned to M2, with M1 supporting data and analytics validation. M5 coordinates integration but does not rewrite teammates' code or become the sole developer.

# 18. Team Workflow and Integration Rules

## Interface-first
Freeze schemas/contracts before parallel work. Each module should have:
- Inputs.
- Outputs.
- Types/schema.
- Error behavior.
- Example/mock.
- Owner.
- Validation/test.

## Parallel development
- Use mock data and mock APIs when dependencies are unavailable.
- Build independent prototypes in parallel.
- Integrate incrementally.
- Avoid waiting until the final days for integration.
- Coordinate GPU-heavy experiments.
- Do not run multiple large GPU jobs simultaneously without coordination.

## Git workflow
- Use branches and pull requests/reviewed merges.
- Keep main branch stable.
- Small, focused commits.
- Never commit secrets, raw data, or unnecessary large artifacts.
- Resolve interface changes with the affected members before merging.

## Daily checkpoints
At start:
- Confirm priorities and blockers.

Midday:
- Check progress and interface compatibility.

End of day:
Each member reports:
1. Completed.
2. In progress.
3. Blockers.
4. Files/modules changed.
5. Tests/validation run and results.
6. Next task.
7. Deadline risk.

---

# 19. Eight-Day Execution Plan

## Day 1 — Research, Architecture Freeze & Team Setup
- M1: dataset reconnaissance, schema, data contract.
- M2/M3: GPU/CUDA and model feasibility audit.
- M2/M3/M4: research model candidates and define benchmark plan.
- M4: API contract and mock.
- M2: define a minimum 20-query evaluation set across the Must features; M1 supports data/analytics validation. M5 tracks the deliverable and integration dependencies.
- M5: repository, branches, task board, Docker skeleton, `.env.example`.
- Team: audit availability and workload.

**Day-1 exit criteria:** architecture frozen; actual dataset/hardware/API constraints recorded; interfaces agreed; repository ready. No major architecture changes afterward.

## Day 2 — Independent Prototypes
- M1: download/raw storage/audit.
- M2: labels, split, majority baseline, TF-IDF baseline.
- M3: product search; BM25-only prototype with Persian normalization; Dense-only baseline (embeddings + FAISS, no BM25/RRF); report BM25-only and Dense-only Hit Rate@5 and MRR on the shared query set.
- M4: FastAPI skeleton, LLM client, query parser prototype.
- M5: integration checklist, task/dependency tracking, demo outline, and documentation skeleton. M2 owns the evaluation framework; M3 prepares UI requirements and mocks.

**Exit criteria:** every member has an executable prototype or verified scaffold.

## Day 3 — Core Development
- M1: clean/normalize/Parquet/DuckDB/join.
- M2: word/character TF-IDF, LR/SVM, Transformer loading/tokenizer/GPU benchmark.
- M3: dense retrieval and embedding benchmark; build FAISS corpus starting at 10K–20K reviews (direct jump to 100K prohibited); record build time, peak memory and index size; begin simple Streamlit UI (MVP pages only); implement embedding and retrieval-result caching.
- M2: implement the evaluation framework and ML metrics; coordinate the shared query set.
- M4: LLM client, prompts, structured output, API error handling, and LLM-response caching; router only if in scope.
- M5: integration coordination, dependency checks, documentation/demo preparation.

**Exit criteria:** core modules execute independently on representative data.

## Day 4 — Feature Completion
- M1: serving data and analytics.
- M2: classical candidate, Transformer training/validation/comparison, error analysis, save model.
- M3: compare Dense-only against BM25-only and, only if Section 9.10 gates are met, Hybrid/RRF; decide whether Hybrid qualifies for Tier Should. Continue/integrate the simple Streamlit UI; validate embedding/retrieval caches. Reranker only if time allows.
- M4: RAG and Analytics; validate LLM-response cache; Comparison only if Tier Should time permits.
- M2: evaluation framework, baseline comparison, and evaluation execution support.
- M5: coordinate integration, track blockers, and maintain demo/documentation readiness.

**Exit criteria:** all Tier Must features have working MVP implementations. Tier Should features may remain incomplete and must be tracked explicitly.

## Day 5 — Full Integration
M3 owns UI integration for the simple Streamlit MVP; M4 owns backend/API integration; M5 coordinates dependencies and tracks integration status. M2 owns the evaluation framework, with M1 validating data and analytics calculations.

Connect:
- Data pipeline.
- DuckDB.
- Product search.
- Review retrieval.
- Embeddings/index.
- LLM.
- RAG.
- Comparison.
- Analytics.
- ML prediction.
- UI.

Run end-to-end tests for all Tier Must features and any Tier Should features already implemented. Handle:
- Empty data/results.
- Invalid inputs.
- Missing product.
- API/model failures.
- Retrieval failures.
- Timeout.
- Invalid structured output.
- Missing evidence.

**Exit criteria:** integrated system works through UI/API for happy paths and critical failure paths.

## Day 6 — Evaluation & Optimization
- M1: data and SQL/analytics correctness.
- M2: final ML comparison, Macro F1, error analysis.
- M3: retrieval/embedding/hybrid/reranker benchmarks.
- M4: LLM/RAG/prompt/citation evaluation.
- M5: coordinate evaluation runs, E2E/regression tracking, latency and failure-report consolidation; M2 owns the evaluation framework and ML metrics.

Only make evidence-based optimizations. Record actual measured results; never substitute example or estimated metrics as measured results.

**Exit criteria:** evaluation results recorded and critical issues prioritized.

## Day 7 — Feature Freeze
- Regression testing.
- Critical bug fixes only.
- Final model/config selection.
- Final evaluation.
- Architecture and workflow docs.
- README and setup instructions.
- API documentation.
- Data and model limitations.
- Docker build/run.
- Demo and presentation preparation.

No new features after end of Day 7.

## Day 8 — Buffer & Submission
- Fresh-environment setup test.
- Docker build and run.
- Data/artifact availability.
- API/UI/end-to-end smoke tests.
- Demo rehearsal.
- Backup demo/video/screenshots if permitted.
- Final critical fixes only.
- Submission and backup.

---

# 20. Evaluation Protocol

## 20.1 Query Set and Silver Ground Truth
Use a minimum of 20 Persian queries rather than 30. Cover the Tier Must features, with a practical distribution across Product Discovery, Review QA, and Analytics. Product Comparison queries may be included if Tier Should is implemented. Add more queries only when time permits.

For Retrieval evaluation, use Silver Ground Truth to reduce manual annotation effort:
- **Method A — Product-title queries (primary lightweight method):** create a query from a product title and use that source product as the known relevant target. This removes full manual annotation but can produce relatively easy or biased queries; document the limitation. Under this method, report only Hit Rate@1, Hit Rate@5, and MRR. Mark Recall@K and NDCG@K as **“Not computable with Silver Ground Truth — omitted”** in the results. Dense-only vs. Hybrid decisions use Hit Rate@5 and MRR only.
- **Method B — Top-10 relevance checks (complementary):** for selected queries, manually label the top 10 results Yes/No for relevance, after defining the relevance criterion. Under this method, report Precision@10 only. Do not report Recall or NDCG from this partial judging pool.

If the team elects to use `persian-rag-bench` (34 documents, 42 questions, with relevant IDs) to retain standard Recall@K and NDCG@K, use it only to validate the retrieval algorithm on its encyclopedia-style Persian texts; it is not a replacement for product-review retrieval evaluation. This benchmark can support Recall@K, MRR, and nDCG@K because it provides relevant IDs. For RAG grounding/citation on Digikala data, continue to use 10 manually reviewed examples plus LLM-as-Judge as a supporting signal. Do not claim automated Faithfulness ground truth from this benchmark or from LLM judging alone.

For RAG and Product Comparison, use qualitative evaluation with LLM-as-Judge as a supporting signal and manually review 10 examples. Manually verify grounding and citation validity for RAG. LLM-as-Judge is not treated as ground truth.

Keep the evaluation workload proportional to the eight-day schedule. Record query sets, labeling method, coverage, limitations, and actual results.

## 20.2 Product Search
Metrics:
- Precision@K.
- Recall@K.
- Hit Rate@K.
- Filter correctness.
- Ranking relevance.
- Latency.

## 20.3 Retrieval
Report metrics only when supported by the selected ground-truth method:
- **Method A — Product-title Silver Ground Truth:** Hit Rate@1, Hit Rate@5, and MRR. Recall@K and NDCG@K: **Not applicable under chosen Silver Ground Truth method / Not computable with Silver Ground Truth — omitted**. Dense-only vs. Hybrid comparison is limited to Hit Rate@5 and MRR.
- **Method B — Top-10 Yes/No labels:** Precision@10 only. Recall and NDCG are omitted because the pool is not exhaustively judged.
- **Optional `persian-rag-bench`:** Recall@K, MRR, and nDCG@K may be reported on that benchmark because it supplies relevant IDs, but results must be labeled as encyclopedia-style Persian retrieval validation and not as Digikala-review results.

Also record latency, memory, and index size. Do not claim a retrieval system is good solely from a few manually observed examples, and do not calculate or present metrics unsupported by the chosen labeling method.

## 20.4 RAG
Evaluate:
- Answer relevance.
- Groundedness.
- Citation correctness.
- Citation coverage.
- Completeness.
- Hallucination rate.
- Conflict handling.
- Insufficient-evidence behavior.

## 20.5 Analytics
Check:
- SQL/filter correctness.
- Aggregation correctness.
- Denominator.
- Percentages.
- Missing values.
- Category/brand comparisons.
- Time period interpretation.

## 20.6 ML
Report:
- Accuracy.
- Macro F1.
- Weighted F1.
- Per-class precision/recall/F1.
- Confusion matrix.
- Error analysis.
- Split strategy and leakage checks.

## 20.7 System
Measure:
- Latency.
- API success/failure.
- Timeouts.
- Memory.
- GPU memory/utilization.
- CPU utilization.
- Model loading time.
- Index loading time.
- Throughput where relevant.

## 20.8 Failure analysis
Document at least 10 real failure cases:
- Input/query.
- Expected behavior.
- Actual behavior.
- Failure category.
- Root cause.
- Severity.
- Proposed fix.
- Fix applied (or not).
- Before/after result.

Do not fabricate failures or evaluation numbers. If fewer than 10 real failures are found, report the actual number observed and explain the limitation; never invent failures to reach 10.

---

# 21. Testing Strategy

## Unit tests
- Normalization.
- Data validation.
- Filter parsing and validation.
- Product ranking.
- RRF.
- Metric calculations.
- Citation/evidence validation.
- ML preprocessing and prediction interface.

## Integration tests
- DuckDB queries.
- Data-to-search.
- Retrieval-to-RAG.
- LLM client.
- API routes.
- Model loading/inference.

## End-to-end tests
- Product discovery.
- Review QA.
- Product comparison.
- Analytics.
- ML prediction.

## Regression tests
Preserve representative queries and expected behaviors. Run after integration and critical changes.

Definition of Done is not merely “code written.” The feature must execute, expected output must be checked, critical tests must pass, errors handled, and documentation/results reproducible.

---

# 22. Security, Privacy and Reliability

- Keep API keys in environment variables.
- Use `.env.example` with placeholder values only.
- `.env` must be ignored by Git.
- Do not log credentials or sensitive request headers.
- Validate user input and model-generated structured output.
- Do not expose stack traces or internal secrets in UI/API responses.
- Use timeouts and error handling for external API calls.
- Citation validation must verify source IDs.
- Treat LLM output as untrusted until validated.
- Limit request size and avoid unnecessary data exposure.
- Avoid executing generated code or SQL from the LLM.
- Analytics SQL should be application-controlled and parameterized/validated.
- Document limitations and known failure modes.

---

# 23. Docker and Deployment

Docker Compose is the target local deployment approach for the integrated system.

Expected:
- Reproducible build.
- Backend service.
- UI service.
- Shared configuration/environment.
- Data/artifact mount or documented local paths.
- Health check.
- Startup instructions.
- Error visibility in logs.

Do not introduce Kubernetes or distributed deployment.

GPU in Docker is optional and depends on verified host/container configuration. The project should have a CPU-compatible fallback for core functionality where practical.

---

# 24. Documentation and Final Deliverables

Required:
- README with project overview, features, architecture, setup, environment variables, data acquisition, run commands, API/UI, evaluation, limitations, and team contributions.
- Architecture diagram.
- API contract/docs.
- Data dictionary.
- Data quality/EDA report.
- ML baseline and Transformer experiment report.
- Retrieval benchmark report.
- RAG evaluation report.
- System benchmark report.
- Failure analysis.
- Final results table with actual values.
- Demo/presentation.
- Reproduction instructions.
- Cache implementation and configuration documentation (LLM response, embeddings, and retrieval caches), including cache-key/versioning and invalidation behavior.
- Known limitations and future work.

Do not claim production readiness. Clearly state the project is a time-boxed bootcamp prototype and describe what was tested.

---

# 25. Definition of Done (Project-Level)

Only **Tier Must** is mandatory for project completion. **Tier Should** features are delivered when time, resources, and feasibility allow; Tier Nice features are optional and must not jeopardize the mandatory deliverables.

## Tier Must completion criteria
- Product Discovery is integrated and returns sensible results with basic evaluation.
- Dense-only Retrieval (embedding + FAISS) is built and retained as an evaluated baseline. If Hybrid fails the Section 9.10 improvement gate, Dense-only is the final Retrieval system; Hybrid/BM25/RRF are not required for Tier Must.
- Review QA is integrated, grounded in retrieved evidence, and returns citations/references that are validated.
- Analytics calculations are deterministic and verified.
- Classical ML (TF-IDF + Logistic Regression) trains/evaluates and its prediction interface works with a saved artifact.
- FastAPI runs with the required core endpoints/contracts.
- A simple Streamlit UI supports the mandatory features at MVP level.
- Docker or documented reproducible setup works in a fresh environment.
- Basic evaluation is executed and actual results/limitations recorded.
- Critical failures are fixed or clearly documented.
- README, architecture/API notes, evaluation summary, and demo are ready.
- Final results are based on real measurements, not invented metrics.

## Tier Should completion (when feasible)
- Product Comparison.
- Hybrid Retrieval (RRF).
- Transformer experiment, subject to the feasibility gate in Section 13.6.
- Reranker, only if time and evaluation justify it.

Completion of Tier Should is not required to declare the project delivered, provided all Tier Must criteria are satisfied and the omitted Should items are clearly documented.

# 26. Optional Experiments — Only After Core Features Work

Possible:
- More Transformer candidates.
- More embedding candidates.
- Reranker.
- Advanced hybrid retrieval.
- Query routing.
- Prompt optimization.
- Additional EDA.
- GPU optimization.
- Additional evaluation queries.

Optional work must not jeopardize core integration, evaluation, or delivery.

---

# 27. Change Control and Architecture Freeze

Architecture baseline is Version 2.2. The core architecture remains the same; this version clarifies delivery tiers, feasibility gates, evaluation workload, role ownership, caching, Dense-only baseline requirements, Persian BM25 normalization gates, VRAM-aware index building, and Silver Ground Truth metric alignment.

Do not repeatedly reconsider the architecture or add fashionable technologies. A change request must include:
1. What is the blocker/problem?
2. Evidence that the current design cannot meet the requirement.
3. Proposed change.
4. Impact on timeline, interfaces, team workload, resources, and evaluation.
5. Alternatives considered.
6. Explicit approval from project owner before implementation.

Version the document when approved changes are made (e.g. v2.1 or v3.0). Preserve prior versions.

---

# 28. Execution Controller — Operating Rules for ChatGPT

When the user asks to begin or continue a project phase, act as project execution controller, mentor, reviewer, and coordinator.

For each task:
1. Check current day/phase, prerequisites, deadline, artifacts, and owner.
2. For a solo user, give exactly one immediate next task unless the user explicitly requests a batch. For the five-person project team, up to five parallel tasks (one per member) may be assigned in a response, with dependencies stated explicitly.
3. Explain **WHY** before **HOW**.
4. Give precise commands/code when needed.
5. State expected output.
6. Define validation steps.
7. Define task Definition of Done.
8. Do not advance until results are verified, unless the user explicitly asks to proceed.
9. Maintain project state and deadline risk.
10. Do not invent results, files, test outcomes, or completion claims.
11. Keep instructions practical and avoid over-engineering.
12. Use Persian for explanations and English for code, filenames, identifiers, commits, and interfaces.
13. Be direct and critical; identify weak assumptions and risks.
14. Architecture changes require explanation and approval.
15. Prioritize integrated, measurable deliverables.

## Starting State
- Project version: 2.2
- Current day: Day 1
- Current phase: Research, Architecture Freeze & Team Setup
- Status: Not Started
- First task: audit dataset, laptop GPU/runtime, model API, team availability, and repository setup.
- Do not mark any audit complete without actual evidence.

---

# 29. Day-1 Task Checklist

## Dataset audit
- [ ] Verify dataset source/access.
- [ ] Download or inspect dataset metadata/sample.
- [ ] Confirm actual file names and schemas.
- [ ] Measure actual row counts.
- [ ] Inspect missing values and duplicates.
- [ ] Inspect recommendation labels.
- [ ] Inspect product-review join coverage.
- [ ] Record data quality risks.

## Hardware/runtime audit
- [ ] Record OS.
- [ ] Record Python version.
- [ ] Record RAM and free disk.
- [ ] Verify GPU model and VRAM.
- [ ] Verify NVIDIA driver.
- [ ] Verify CUDA/PyTorch GPU availability.
- [ ] Run a small GPU benchmark and record output.

## API/model audit
- [ ] Confirm endpoint connectivity.
- [ ] Confirm authentication without exposing the key.
- [ ] Retrieve available model list if possible.
- [ ] Test one small chat request.
- [ ] Test structured response behavior.
- [ ] Record latency/errors/limits.

## Team and repository
- [ ] Confirm team members and availability.
- [ ] Assign owners.
- [ ] Create repository.
- [ ] Create branch/PR workflow.
- [ ] Create project board/tasks.
- [ ] Create `.env.example`.
- [ ] Configure `.gitignore`.
- [ ] Create minimal Docker skeleton.
- [ ] Freeze initial API/interface contracts.

---

# 30. Decision Log Template

Use this template for important decisions:

**Decision ID:**  
**Date:**  
**Context/problem:**  
**Options considered:**  
**Decision:**  
**Why:**  
**Evidence/benchmark:**  
**Impact:**  
**Approved by:**  
**Status:**  

---

# 31. Experiment Log Template

For each experiment, record:
- Experiment ID/name.
- Date/time.
- Owner.
- Dataset/corpus size and sampling method.
- Train/validation/test strategy.
- Model and version.
- Configuration/hyperparameters.
- Hardware/environment.
- Metrics.
- Runtime.
- Memory/VRAM.
- Observations.
- Failure cases.
- Conclusion.
- Next action.

### Retrieval Baseline Records (separate rows required)
Maintain separate records for Dense-only, BM25-only, and Hybrid. For each, record Hit Rate@5, MRR (where supported by chosen Silver method), build/indexing time, embedding generation time, peak system RAM, peak GPU VRAM (if used), index file size, query-by-query results, and selected/omitted status. Metrics that are unsupported by the ground-truth method must be marked omitted/not applicable, not estimated.

| Retrieval variant | Hit Rate@5 | MRR | Build/embedding time | Peak RAM | Peak VRAM | Index size | Query-by-query record |
|---|---:|---:|---|---|---|---|---|
| Dense-only | | | | | | | |
| BM25-only | | | | | | | |
| Hybrid (RRF) | | | | | | | |

---

# 32. Final Results Table Template

Fill only with measured results.

| Component | Candidate/Version | Evaluation | Actual result | Runtime/resources | Selected? | Notes |
|---|---|---|---|---|---|---|
| Product search | | | | | | |
| Lexical retrieval | | | | | | |
| Embeddings | | | | | | |
| Hybrid retrieval | | | | | | |
| Reranker | | | | | | |
| RAG/LLM | | | | | | |
| ML baseline | | | | | | |
| Transformer | | | | | | |
| Analytics | | | | | | |
| API/UI system | | | | | | |

---

# 33. Final Demo Checklist

- [ ] Environment and services start reliably.
- [ ] Dataset and indexes are available.
- [ ] Product discovery demo query works.
- [ ] Review QA returns evidence and IDs.
- [ ] Comparison works for selected products.
- [ ] Analytics calculations are verifiable.
- [ ] ML prediction works.
- [ ] Error/empty-state behavior is demonstrated or tested.
- [ ] No credentials visible.
- [ ] Evaluation results and limitations are ready.
- [ ] Backup demo material exists if allowed.
- [ ] Each team member's contribution is documented.

---

# 34. Reference Links

- Dataset: https://huggingface.co/datasets/RadeAI/Digikala_comments_products
- Multilingual E5 small: https://huggingface.co/intfloat/multilingual-e5-small
- Multilingual E5 base: https://huggingface.co/intfloat/multilingual-e5-base
- BGE-M3: https://huggingface.co/BAAI/bge-m3
- Hugging Face Trainer: https://huggingface.co/docs/transformers/main/main_classes/trainer
- Microsoft RAG information retrieval: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-information-retrieval
- Azure hybrid search ranking/RRF: https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking
- The Fusion Tax (IEEE Access 2026; DOI 10.1109/ACCESS.2026.3743502): https://arxiv.org/abs/2610.30612
- Persian RAG benchmark/toolkit (34 documents, 42 questions; benchmark results and methodology): https://github.com/sadrmanesh/persian-rag
- FAISS GPU guidance (memory, temporary resource reservation, batching): https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
- FAISS index selection guidance: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

---

# 35. Master Project Execution Prompt

Use the following prompt in a new project chat together with this specification:

> You are the QBC13 project execution controller, senior AI/ML engineer, MLOps engineer, software architect, technical mentor, code reviewer, evaluator, and project coordinator.
>
> Treat **QBC13 — Digikala AI Shopping & Review Intelligence Master Project Specification v2.2** as the single source of truth. Do not silently change its architecture, scope, technologies, team roles, API contracts, timeline, or constraints.
>
> The project has five members and eight days. The current state is Day 1, Phase: Research, Architecture Freeze & Team Setup, status: Not Started. First task: audit dataset, hardware/GPU/runtime, model API, team availability, and repository setup.
>
> For each task, first check prerequisites and project state. For a solo user, give one immediate task. For the five-person team, up to five parallel tasks (one per member) may be assigned in a response, with explicit dependencies. Explain WHY before HOW. Provide exact commands/code, expected output, validation, and Definition of Done. Do not move to the next task until the current task is verified unless I explicitly ask you to proceed.
>
> Be direct and critical. Do not praise vaguely. Do not invent outputs, metrics, test results, files, or completed work. Use Persian for explanation and English for code, filenames, identifiers, commits, and interfaces. Avoid over-engineering and unnecessary architecture changes. Prioritize integration, evaluation, and delivery within the eight-day deadline.
>
> If a genuine blocker requires architecture or scope changes, explain the blocker, evidence, alternatives, impact, and request explicit approval before changing the baseline.
>
> Maintain a concise state ledger: current day/phase, task, owner, verified artifacts, blockers, deadline risk, and next step.
>
> Begin with Day 1, Task 1. Do not assume any audit has been completed.

---

# 36. Current Project State Ledger

| Field | Current value |
|---|---|
| Project | QBC13 — Digikala AI Shopping & Review Intelligence |
| Specification | v2.2 |
| Team size | 5 |
| Duration | 8 days |
| Current day | Day 1 |
| Phase | Research, Architecture Freeze & Team Setup |
| Status | Not Started |
| Architecture | Frozen baseline — Modular Monolith |
| First task | Audit dataset, hardware/runtime/GPU, API, team, repository |
| Verified completed work | None recorded in this specification |
| Blockers | Must be discovered during Day-1 audit |
| Next step | Execute Day-1 audit checklist |

---

# Changelog v2.1

- **Section 3.6 — Scope Tiers:** Added Must / Should / Nice priorities to make the eight-day scope realistic and protect mandatory delivery.
- **Section 9.9 — Caching Strategy:** Added mandatory Day-3 caching for LLM responses, embeddings, and retrieval results, with effective-input cache keys, versioning/invalidation, and simple local implementation options.
- **Section 13.6 — Transformer Feasibility Gate:** Added stop/report criteria by Day 4 for validation Macro-F1, a two-hour training limit, and OOM; Classical ML remains the deliverable model if the gate is triggered.
- **Section 17 — Team Roles:** Reassigned M5 to integration coordination, demo, documentation, presentation, and Failure Analysis; UI MVP ownership moves to M3 from Day 3, while M2 owns Evaluation Framework and M1 supports data/analytics validation.
- **Section 19 — Eight-Day Execution Plan:** Updated Day 1–5 responsibilities to reflect the revised UI/evaluation ownership and lower annotation burden.
- **Section 20.1 — Evaluation:** Reduced the minimum query set from 30 to 20 and added Silver Ground Truth methods (product-title queries and Top-10 Yes/No relevance checks), plus qualitative RAG/Comparison evaluation with LLM-as-Judge support and manual review of 10 examples.
- **Section 20.8 — Failure Analysis:** Clarified that fewer than 10 real failures must be reported as the actual number; no fabricated cases may be created to meet a target.
- **Section 24 — Deliverables:** Added caching implementation/configuration documentation and cache invalidation/versioning details.
- **Section 25 — Definition of Done:** Explicitly made Tier Must mandatory, Tier Should conditional on feasibility/time, and Tier Nice optional.
- **Section 26 — Optional Experiments:** Reorganized optional work to align with Tier Nice and the delivery priority.
- **Sections 27, 28, 35, and 36:** Updated baseline/version references to v2.1 and added the five-member parallel-task rule (up to five concurrent member tasks with dependencies; solo user receives one immediate task).


# 37. Changelog v2.2

**Version 2.2 is the final frozen baseline for the eight-day QBC13 project.** After this version, architecture or scope changes are permitted only through the Change Control process in Section 27, with blocker evidence, impact analysis, and explicit project-owner approval.

1. **Section 9.2 — Persian BM25 normalization and noise gate:** made normalization of Arabic/Persian Yeh/Kaf variants, ZWNJ, digits, and punctuation mandatory for both corpus and queries. Added Day-2 BM25-only Hit Rate@5 reporting and a project noise gate: if BM25 is more than 15 percentage points below Dense-only on the same queries, it cannot enter RRF. Added benchmark context from `persian-rag-bench`; its reported +7.1 percentage-point sparse-to-sparse+rewrite Hit@1 change is attributed to rewriting, not normalization alone.
2. **Section 9.6 — VRAM-aware corpus/index build protocol:** start Day 3 at 10K–20K reviews; record build time, RAM/VRAM peaks, remaining memory, and index size. Expansion to 50K requires all three stated gates. Direct start at 100K is prohibited. CPU FAISS is the default practical starting point; GPU embedding batches start conservatively and are increased only after measured memory checks. The exact CPU/GPU batch-size rules in the request were not verified as universal FAISS guidance and are therefore written as project-specific starting values, not authoritative FAISS recommendations. Reference: official FAISS GPU guidance.
3. **Sections 9.10, 19 Day 2/Day 4, 25, and 31 — Dense-only baseline and Hybrid gate:** M3 must benchmark Dense-only by end of Day 2; Hybrid must beat Dense-only by at least 5% relative on both Hit Rate@5 and MRR to remain in scope for Day 4. Otherwise BM25/RRF are removed and Dense-only is the final retrieval choice. Separate query-by-query records are required. External evidence: *The Fusion Tax* reports RRF hurting 15–33% of queries and net-negative versus Dense-only on five corpora in the study; those results motivate testing and do not predict QBC13’s own outcomes.
4. **Sections 20.1 and 20.3 — Silver Ground Truth and metrics alignment:** Method A supports reporting Hit Rate@1, Hit Rate@5, and MRR only; Recall@K and NDCG@K are explicitly omitted. Method B supports Precision@10 only; Recall/NDCG are omitted for partial judging. Standard Recall@K/MRR/nDCG@K may be reported on `persian-rag-bench` only as a separate Persian encyclopedia-style retrieval validation dataset (34 docs, 42 queries, relevant IDs), not as product-review evaluation. RAG grounding/citations still require 10 manually reviewed examples plus LLM-as-Judge as a supporting signal; no automated Faithfulness ground truth is claimed.

**Evidence and reference links:**
- *The Fusion Tax*: https://arxiv.org/abs/2610.30612 (IEEE Access 2026, DOI 10.1109/ACCESS.2026.3743502).
- `persian-rag` and benchmark: https://github.com/sadrmanesh/persian-rag (34 documents, 42 questions; reported Sparse Hit@1 0.881 and Sparse+Rewrite 0.952 in the benchmark).
- FAISS GPU guidance: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU (GPU resource memory, batching, CPU fallback considerations).
- FAISS index selection: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index.

No other project scope, architecture, team role, API contract, or day-plan item is intentionally changed beyond the modifications required by the four structural issues above and the explicitly listed alignment edits.

**End of Master Specification v2.2**
