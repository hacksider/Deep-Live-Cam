@echo off
cd /d "%~dp0"
call venv\Scripts\activate
python run.py --execution-provider dml
pause
