@echo off
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0open_knowledge_distillation.ps1"
set EXITCODE=%ERRORLEVEL%

echo.
echo ========================================
echo PowerShell finished with exit code: %EXITCODE%
echo Press any key to close...
echo ========================================
pause >nul

exit /b %EXITCODE%