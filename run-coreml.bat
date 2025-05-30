@echo off
REM run-coreml.bat - Run Deep-Live-Cam with CoreML (Apple Silicon) on Windows (for reference, not for actual use)
call venv\Scripts\activate
python run.py --execution-provider coreml
