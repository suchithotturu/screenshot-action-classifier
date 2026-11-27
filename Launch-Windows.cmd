@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 first.
  pause
  exit /b 1
)
set "CLASSIFIER_OCR=rapidocr"
echo Open http://127.0.0.1:8514 in your browser.
".venv\Scripts\python.exe" -m classifier app --mode classify
pause
