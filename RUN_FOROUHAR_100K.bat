@echo off
setlocal
cd /d "%~dp0"
if not exist "storage\catalog.sqlite3" (
  echo ERROR: Extract the complete ZIP first.
  pause
  exit /b 1
)
if not exist "data\analytics_100000\reviews.parquet" (
  echo ERROR: The packaged review dataset is missing. Extract the entire ZIP.
  pause
  exit /b 1
)
set "DIGIKALA_MODEL_PATH=%~dp0bonus_artifacts\adapted_head.npz"
set "QBC13_PREDICT_FUNCTION=ml_service:predict_recommendation"
set "QBC13_BACKEND_URL="
set "QBC13_SEARCH_FUNCTION="
set "QBC13_REVIEW_FUNCTION="
set "FOROUHAR_COMMENTS_PATH="
set "QBC13_REVIEW_SOURCE=%~dp0storage\catalog.sqlite3"
set "QBC13_ANALYTICS_DIR=%~dp0data\analytics_100000"
set "QBC13_UI_SEARCH_MODE=bm25"
rem Generation defaults to extractive. A local .env or explicit environment can opt in to llm.
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
echo Forouhar 100K - http://localhost:8503
python -m streamlit run "%~dp0app.py" --server.port 8503 --server.headless true
if errorlevel 1 (
  echo If a dependency is missing, run: python -m pip install -r requirements.txt
  pause
  exit /b 1
)
endlocal
