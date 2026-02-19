@echo off
setlocal

set SCRIPT_DIR=%~dp0
set SCRIPT_PY=%SCRIPT_DIR%run_pipeline.py

python "%SCRIPT_PY%" %*
if %ERRORLEVEL% EQU 0 exit /b 0

py -3 "%SCRIPT_PY%" %*
exit /b %ERRORLEVEL%
