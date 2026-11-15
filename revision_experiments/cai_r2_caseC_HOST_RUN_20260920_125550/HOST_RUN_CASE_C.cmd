@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0HOST_RUN_CASE_C.ps1"
set "HOST_RC=%ERRORLEVEL%"
echo.
echo HOST_RUN_CASE_C.ps1 exited with code %HOST_RC%.
exit /b %HOST_RC%

