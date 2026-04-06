@echo off
setlocal

cd /d "%~dp0.."

echo Opening daily portfolio panel...
py -3 -m streamlit run src\dashboard\app.py

if errorlevel 1 (
  echo.
  echo ERROR: could not start Streamlit panel.
  echo Make sure dependencies are installed.
  pause
  exit /b 1
)
