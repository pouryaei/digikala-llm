@echo off
setlocal
cd /d "%~dp0"
if not exist "storage\catalog.sqlite3" (
  echo ERROR: Restore the complete 100K data first. See README.md.
  pause
  exit /b 1
)
if not exist "bonus_artifacts\retrieval\manifest.json" (
  echo Character retrieval index is missing. Run python -m bonus.retrieval after restoring data.
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
rem Local app and monitoring by default; .env may explicitly opt into paid/API or remote tracing.
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
echo Forouhar Bonus Lab - http://localhost:8505
python -m streamlit run "%~dp0app.py" --server.port 8505 --server.headless true
if errorlevel 1 (
  echo Run python -m pip install -r requirements.txt if a dependency is missing.
  pause
  exit /b 1
)
endlocal
