@echo off
echo ====================================================================
echo Starting Medical Insurance Fraud Detection Backend Server
echo Student: Ebal Kumar Reddy
echo ====================================================================
echo.
cd /d "%~dp0backend"
"%~dp0.venv\Scripts\python.exe" app.py
pause
