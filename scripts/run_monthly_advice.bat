@echo off
setlocal

REM Move to repo root (this .bat lives in scripts\)
cd /d "%~dp0.."

echo Running monthly rebalance advisor...
py -3 scripts\monthly_rebalance_advice.py --positions data\inputs\positions.csv --outdir data\processed\monthly_advice

if errorlevel 1 (
  echo.
  echo ERROR: advisor execution failed.
  pause
  exit /b 1
)

echo.
echo Done. Check outputs:
echo   data\processed\monthly_advice\summary.csv
echo   data\processed\monthly_advice\actions.csv
echo   data\processed\monthly_advice\portfolio_snapshot.csv
echo.
pause
