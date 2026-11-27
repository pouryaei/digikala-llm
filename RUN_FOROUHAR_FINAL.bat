@echo off
setlocal
cd /d "%~dp0"
if not exist "reports\final_model\report.json" (
  echo ERROR: Extract the complete final archive first.
  pause
  exit /b 1
)
echo Forouhar final - Logistic Regression with Head Fine-tuning
call "%~dp0RUN_FOROUHAR_BONUS.bat"
endlocal
